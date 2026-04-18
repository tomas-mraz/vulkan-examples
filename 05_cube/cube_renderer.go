package main

import (
	"fmt"
	"time"

	vk "github.com/tomas-mraz/vulkan"
	ash "github.com/tomas-mraz/vulkan-ash"
)

const depthFormat = vk.FormatD16Unorm

// cubeRenderer implements ash.Renderer for the rotating-cube demo.
//
// Lifetime buckets:
//   - "once": texture, uniform buffers, descriptor sets. These depend on the
//     device and the swapchain length but not on the extent/orientation.
//   - "sized": depth image, render pass, framebuffers, graphics pipeline.
//     The pipeline's viewport is baked in, so a rotation or resize must
//     rebuild it.
//
// The view matrix is constant; the projection matrix is rebuilt from the
// swapchain extent + preTransform every time CreateSized runs.
type cubeRenderer struct {
	// "once"
	texture  ash.ImageResource
	uniforms ash.UniformBuffers
	desc     ash.DescriptorInfo

	// "sized"
	depth      ash.ImageResource
	rasterPass ash.RasterizationPass
	pipeline   ash.PipelineRasterization

	// frame state
	startTime         time.Time
	projMatrix        ash.Mat4x4
	viewMatrix        ash.Mat4x4
	modelMatrix       ash.Mat4x4
	onceBuilt         bool
	sizedBuilt        bool
}

// CreateOnce builds device-lifetime resources. Called once after the first
// swapchain is up, and again on Android if the device is torn down/rebuilt
// across surface-lost/created cycles.
func (r *cubeRenderer) CreateOnce(s *ash.Session) error {
	swapLen := s.Swapchain.DefaultSwapchainLen()

	pixels, texW, texH, err := ash.DecodePNG(gopherPng)
	if err != nil {
		return fmt.Errorf("DecodePNG: %w", err)
	}

	r.texture, err = ash.NewImageTexture(s.Manager.Device, s.Manager.Gpu, texW, texH, pixels)
	if err != nil {
		return fmt.Errorf("NewImageTexture: %w", err)
	}
	r.texture.TransitionLayout(s.Manager.Queue, s.CmdCtx.GetCmdPool(), vk.ImageLayoutShaderReadOnlyOptimal)

	r.uniforms, err = ash.NewUniformBuffers(s.Manager.Device, s.Manager.Gpu, swapLen, uniformDataSize)
	if err != nil {
		return fmt.Errorf("NewUniformBuffers: %w", err)
	}

	r.desc, err = ash.NewDescriptorSets(s.Manager.Device, swapLen, []ash.DescriptorBinding{
		&ash.BindingUniformBuffer{StageFlags: vk.ShaderStageFlags(vk.ShaderStageVertexBit), Uniforms: &r.uniforms},
		ash.NewBindingImageSampler(vk.ShaderStageFlags(vk.ShaderStageFragmentBit), &r.texture, []vk.Sampler{r.texture.GetSampler()}),
	})
	if err != nil {
		return fmt.Errorf("NewDescriptorSets: %w", err)
	}

	r.modelMatrix.Identity()
	// View matrix is derived from the swapchain shape in CreateSized so that
	// the camera can be pulled back on portrait displays where the cube would
	// otherwise be clipped horizontally.
	r.startTime = time.Now()
	r.onceBuilt = true
	return nil
}

// DestroyOnce releases device-lifetime resources. Idempotent.
func (r *cubeRenderer) DestroyOnce() {
	if !r.onceBuilt {
		return
	}
	r.desc.Destroy()
	r.uniforms.Destroy()
	r.texture.Destroy()
	r.onceBuilt = false
}

// CreateSized builds extent-dependent resources and (re)derives the projection.
func (r *cubeRenderer) CreateSized(s *ash.Session, extent vk.Extent2D) error {
	var err error
	r.depth, err = ash.NewImageDepth(s.Manager.Device, s.Manager.Gpu, extent.Width, extent.Height, depthFormat)
	if err != nil {
		return fmt.Errorf("NewImageDepth: %w", err)
	}
	r.rasterPass, err = ash.NewRasterPassWithDepth(s.Manager.Device, s.Swapchain.DisplayFormat, depthFormat)
	if err != nil {
		return fmt.Errorf("NewRasterPassWithDepth: %w", err)
	}
	if err := s.Swapchain.CreateFramebuffers(r.rasterPass.GetRenderPass(), r.depth.GetView()); err != nil {
		return fmt.Errorf("CreateFramebuffers: %w", err)
	}
	r.pipeline, err = ash.NewPipelineRasterization(s.Manager.Device, extent, r.rasterPass.GetRenderPass(), ash.PipelineOptions{
		VertShaderData:       vertShaderCode,
		FragShaderData:       fragShaderCode,
		DescriptorSetLayouts: []vk.DescriptorSetLayout{r.desc.GetLayout()},
		DepthTestEnable:      true,
	})
	if err != nil {
		return fmt.Errorf("NewPipelineRasterization: %w", err)
	}

	r.viewMatrix, r.projMatrix = computeCamera(s.Swapchain)
	r.sizedBuilt = true
	return nil
}

// DestroySized releases extent-dependent resources. Idempotent.
func (r *cubeRenderer) DestroySized() {
	if !r.sizedBuilt {
		return
	}
	r.pipeline.Destroy()
	r.rasterPass.Destroy()
	r.depth.Destroy()
	r.sizedBuilt = false
}

// Draw records a single rotating-cube frame. Session has already begun the
// command buffer; we record the render pass and draw call into f.Cmd.
func (r *cubeRenderer) Draw(s *ash.Session, f *ash.Frame) error {
	// Animate: 45 degrees per second around Y.
	elapsed := float32(time.Since(r.startTime).Seconds()) * 45.0
	var rotated ash.Mat4x4
	r.modelMatrix.Identity()
	rotated.Dup(&r.modelMatrix)
	r.modelMatrix.Rotate(&rotated, 0.0, 1.0, 0.0, ash.DegreesToRadians(elapsed))

	writeCubeUniforms(&r.uniforms, f.ImageIndex, &r.projMatrix, &r.viewMatrix, &r.modelMatrix)

	clearValues := []vk.ClearValue{makeClearColor(0.2, 0.2, 0.2, 1.0), makeClearDepth(1.0, 0)}
	vk.CmdBeginRenderPass(f.Cmd, &vk.RenderPassBeginInfo{
		SType:           vk.StructureTypeRenderPassBeginInfo,
		RenderPass:      r.rasterPass.GetRenderPass(),
		Framebuffer:     f.Swapchain.Framebuffers[f.ImageIndex],
		RenderArea:      vk.Rect2D{Extent: f.Extent},
		ClearValueCount: uint32(len(clearValues)),
		PClearValues:    clearValues,
	}, vk.SubpassContentsInline)

	vk.CmdBindPipeline(f.Cmd, vk.PipelineBindPointGraphics, r.pipeline.GetPipeline())
	vk.CmdBindDescriptorSets(f.Cmd, vk.PipelineBindPointGraphics, r.pipeline.GetLayout(), 0, 1,
		[]vk.DescriptorSet{r.desc.GetSets()[f.ImageIndex]}, 0, nil)
	vk.CmdDraw(f.Cmd, 36, 1, 0, 0)
	vk.CmdEndRenderPass(f.Cmd)
	return nil
}

// computeCamera derives the view and projection matrices for the current
// swapchain shape.
//
// The perspective uses a fixed 45° vertical FOV, which looks natural on
// landscape / square displays. On portrait displays the horizontal FOV at the
// same vertical setting becomes too narrow (≈21° at aspect 0.45), so the
// 2x2x2 cube — especially its rotating ≈2.83-unit diagonal — would clip on
// the sides. Rather than warping the perspective (which makes the cube look
// stretched), we pull the camera straight back by 1/aspect. The cube stays
// geometrically correct, just smaller on-screen. Matches the user-accepted
// trade-off "i za cenu že bude menší".
//
// When the device is physically rotated on Android, the swapchain keeps its
// native-panel extent (e.g. 1080x2400) and the scene is rotated into place by
// PreRotationMatrix. The *apparent* on-screen aspect after pre-rotation swaps
// when PreTransform is 90° or 270°, so we swap width/height for the aspect
// calculation to match what the user actually sees.
//
// Finally pre-rotation is concatenated onto the projection so the compositor
// doesn't have to rotate the swapchain image at present time.
func computeCamera(swap *ash.Swapchain) (view, proj ash.Mat4x4) {
	// Empirically on Moto 30 Neo (Adreno 619) the swapchain in landscape hold
	// reports currentExtent already in "user-visible" orientation (2300x1080),
	// and currentTransform = Rotate90. When we render a square cube directly
	// into that 2300x1080 framebuffer and let the driver scan it out with
	// preTransform = Rotate90 (bypass), the driver effectively stretches the
	// 2300x1080 buffer to fit the native 1080x2400 panel without a rotation
	// pass — a vendor quirk rather than what Vulkan's spec promises.
	//
	// Net effect of the stretch: panel_X = fb_X * (1080/2300), panel_Y = fb_Y
	// * (2400/1080). For the cube to look square to the user in landscape we
	// must pre-compensate by rendering it wide-and-short in framebuffer (about
	// 4.7x wider than tall). That falls out for free from using the "native"
	// landscape aspect (extent width/height = 2.13) in the perspective AND
	// applying PreRotationMatrix to rotate the content into the physical
	// scan-out direction. Orientation and proportion then align.
	//
	// Portrait: extent is 1080x2400 (native), preTransform = IDENTITY, the
	// scale-back and no-op PreRotationMatrix keep the old portrait behavior.
	aspect := float32(swap.DisplaySize.Width) / float32(swap.DisplaySize.Height)

	scale := float32(1.0)
	if aspect < 1.0 {
		scale = 1.0 / aspect
	}
	eye := ash.Vec3{0, 3 * scale, 5 * scale}
	view.LookAt(&eye, &ash.Vec3{0, 0, 0}, &ash.Vec3{0, 1, 0})

	proj.Perspective(ash.DegreesToRadians(45.0), aspect, 0.1, 100.0)
	proj[1][1] *= -1 // flip Y for Vulkan

	preRot := swap.PreRotationMatrix()
	var rotated ash.Mat4x4
	rotated.Mult(&preRot, &proj)
	proj = rotated
	return view, proj
}

func writeCubeUniforms(uniforms *ash.UniformBuffers, index uint32, proj, view, model *ash.Mat4x4) {
	var VP, MVP ash.Mat4x4
	VP.Mult(proj, view)
	MVP.Mult(&VP, model)

	data := uniformData{MVP: MVP}
	for i := 0; i < 36; i++ {
		data.Position[i] = [4]float32{gVertexBufferData[i*3], gVertexBufferData[i*3+1], gVertexBufferData[i*3+2], 1.0}
		data.Attr[i] = [4]float32{gUVBufferData[i*2], gUVBufferData[i*2+1], 0, 0}
	}
	uniforms.Update(index, data.Bytes())
}

func makeClearColor(r, g, b, a float32) vk.ClearValue {
	var cv vk.ClearValue
	cv.SetColor([]float32{r, g, b, a})
	return cv
}

func makeClearDepth(depth float32, stencil uint32) vk.ClearValue {
	var cv vk.ClearValue
	cv.SetDepthStencil(depth, stencil)
	return cv
}
