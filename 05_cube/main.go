package main

import (
	_ "embed"
	"log"
	"time"
	"unsafe"

	vk "github.com/tomas-mraz/vulkan"
	"github.com/tomas-mraz/vulkan-ash"
)

//go:embed shaders/cube.vert.spv
var vertShaderCode []byte

//go:embed shaders/cube.frag.spv
var fragShaderCode []byte

//go:embed textures/gopher.png
var gopherPng []byte

const appName = "VulkanCube"

// uniformData matches the shader's uniform buffer layout.
type uniformData struct {
	MVP      ash.Mat4x4
	Position [36][4]float32
	Attr     [36][4]float32
}

const uniformDataSize = int(unsafe.Sizeof(uniformData{}))

func (u *uniformData) Bytes() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(u)), uniformDataSize)
}

func main() {
	ash.SetDebug(false)
	ash.SetValidations(false)

	start()
}

func initVulkanResources(manager *ash.Manager, cleanup *ash.Cleanup, width, height uint32) (
	swapchain ash.Swapchain,
	rasterPass ash.RasterizationPass,
	cmdCtx ash.CommandContext,
	texture ash.ImageResource,
	uniforms ash.UniformBuffers,
	desc ash.DescriptorInfo,
	gfx ash.PipelineRasterization,
	sync ash.SyncInfo,
	err error,
) {
	windowSize := ash.NewExtentSize(int(width), int(height))
	swapchain, err = ash.NewSwapchain(manager, windowSize)
	if err != nil {
		return
	}
	cleanup.Add(&swapchain)
	swapchainLen := swapchain.DefaultSwapchainLen()

	depth, err := ash.NewImageDepth(manager.Device, manager.Gpu, width, height, vk.FormatD16Unorm)
	if err != nil {
		return
	}
	cleanup.Add(&depth)

	rasterPass, err = ash.NewRasterPassWithDepth(manager.Device, swapchain.DisplayFormat, depth.GetFormat())
	if err != nil {
		return
	}
	cleanup.Add(&rasterPass)

	if err = swapchain.CreateFramebuffers(rasterPass.GetRenderPass(), depth.GetView()); err != nil {
		return ash.Swapchain{}, ash.RasterizationPass{}, ash.CommandContext{}, ash.ImageResource{}, ash.UniformBuffers{}, ash.DescriptorInfo{}, ash.PipelineRasterization{}, ash.SyncInfo{}, err
	}
	cmdCtx, err = ash.NewCommandContext(manager.Device, 0, swapchainLen)
	if err != nil {
		return
	}
	cleanup.Add(&cmdCtx)

	pixels, texW, texH, err := ash.DecodePNG(gopherPng)
	if err != nil {
		return
	}

	texture, err = ash.NewImageTexture(manager.Device, manager.Gpu, texW, texH, pixels)
	if err != nil {
		return
	}
	cleanup.Add(&texture)
	texture.TransitionLayout(manager.Queue, cmdCtx.GetCmdPool(), vk.ImageLayoutShaderReadOnlyOptimal)

	uniforms, err = ash.NewUniformBuffers(manager.Device, manager.Gpu, swapchainLen, uniformDataSize)
	if err != nil {
		return
	}
	cleanup.Add(&uniforms)

	desc, err = ash.NewDescriptorSets(manager.Device, swapchainLen, []ash.DescriptorBinding{
		&ash.BindingUniformBuffer{StageFlags: vk.ShaderStageFlags(vk.ShaderStageVertexBit), Uniforms: &uniforms},
		ash.NewBindingImageSampler(vk.ShaderStageFlags(vk.ShaderStageFragmentBit), &texture, []vk.Sampler{texture.GetSampler()}),
	})
	if err != nil {
		return
	}
	cleanup.Add(&desc)

	gfx, err = ash.NewPipelineRasterization(manager.Device, swapchain.DisplaySize, rasterPass.GetRenderPass(), ash.PipelineOptions{
		VertShaderData:       vertShaderCode,
		FragShaderData:       fragShaderCode,
		VertexBindings:       []vk.VertexInputBindingDescription{},
		VertexAttributes:     []vk.VertexInputAttributeDescription{},
		DescriptorSetLayouts: []vk.DescriptorSetLayout{desc.GetLayout()},
		DepthTestEnable:      true,
	})
	if err != nil {
		return
	}
	cleanup.Add(&gfx)

	sync, err = ash.NewSyncObjects(manager.Device)
	if err != nil {
		return
	}
	cleanup.Add(&sync)

	return
}

func updateUniformBuffer(uniforms *ash.UniformBuffers, index uint32, proj, view, model *ash.Mat4x4) {
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

func drawCubeFrame(ctx *ash.SwapchainContext, cmdCtx *ash.CommandContext,
	rasterPass ash.RasterizationPass, gfx ash.PipelineRasterization,
	descSets []vk.DescriptorSet, uniforms *ash.UniformBuffers,
	syncObj ash.SyncInfo,
	proj, view, model *ash.Mat4x4,
) bool {
	s := ctx.GetSwapchain()

	nextIdx, acquired, err := ctx.AcquireNextImage(vk.MaxUint64, syncObj.Semaphore, vk.NullFence)
	if err != nil || !acquired {
		return false
	}

	updateUniformBuffer(uniforms, nextIdx, proj, view, model)

	cmd, err := ctx.BeginFrame(nextIdx, cmdCtx)
	if err != nil {
		log.Println("BeginFrame:", err)
		return false
	}

	clearValues := make([]vk.ClearValue, 2)
	clearValues[0].SetColor([]float32{0.2, 0.2, 0.2, 1.0})
	clearValues[1].SetDepthStencil(1.0, 0)

	vk.CmdBeginRenderPass(cmd, &vk.RenderPassBeginInfo{
		SType: vk.StructureTypeRenderPassBeginInfo, RenderPass: rasterPass.GetRenderPass(), Framebuffer: s.Framebuffers[nextIdx],
		RenderArea:      vk.Rect2D{Extent: s.DisplaySize},
		ClearValueCount: 2, PClearValues: clearValues,
	}, vk.SubpassContentsInline)

	vk.CmdBindPipeline(cmd, vk.PipelineBindPointGraphics, gfx.GetPipeline())
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointGraphics, gfx.GetLayout(), 0, 1, []vk.DescriptorSet{descSets[nextIdx]}, 0, nil)
	vk.CmdDraw(cmd, 36, 1, 0, 0)
	vk.CmdEndRenderPass(cmd)

	if err := ctx.EndFrame(cmd); err != nil {
		log.Println("EndFrame:", err)
		return false
	}

	if err := ctx.SubmitRender(cmd, syncObj.Fence, []vk.Semaphore{syncObj.Semaphore}); err != nil {
		log.Println("SubmitRender:", err)
		return false
	}

	presented, err := ctx.PresentImage(nextIdx, nil)
	if err != nil {
		log.Println("PresentImage:", err)
		return false
	}
	return presented
}

func renderLoop(ctx *ash.SwapchainContext, cmdCtx *ash.CommandContext,
	rasterPass ash.RasterizationPass, gfx ash.PipelineRasterization,
	desc ash.DescriptorInfo, uniforms *ash.UniformBuffers,
	syncObj ash.SyncInfo,
) {
	size := ctx.GetSwapchain().DisplaySize
	aspectRatio := float32(size.Width) / float32(size.Height)

	preRotation := ctx.GetSwapchain().PreRotationMatrix()

	var projMatrix, viewMatrix, modelMatrix ash.Mat4x4
	projMatrix.Perspective(ash.DegreesToRadians(45.0), aspectRatio, 0.1, 100.0)
	viewMatrix.LookAt(&ash.Vec3{0.0, 3.0, 5.0}, &ash.Vec3{0.0, 0.0, 0.0}, &ash.Vec3{0.0, 1.0, 0.0})
	modelMatrix.Identity()
	projMatrix[1][1] *= -1 // Flip for Vulkan

	// Apply pre-rotation to projection matrix (handles Android surface rotation)
	var rotatedProj ash.Mat4x4
	rotatedProj.Mult(&preRotation, &projMatrix)
	projMatrix = rotatedProj

	log.Println("Vulkan initialized, starting render loop")
	startTime := time.Now()

	for {
		// [desktop/android] specific breaking loop (and desktop event check)
		if !pollEvents() {
			break
		}

		elapsed := float32(time.Since(startTime).Seconds()) * 45.0 // 45 deg/sec
		var rotated ash.Mat4x4
		modelMatrix.Identity()
		rotated.Dup(&modelMatrix)
		modelMatrix.Rotate(&rotated, 0.0, 1.0, 0.0, ash.DegreesToRadians(elapsed))

		if !drawCubeFrame(ctx, cmdCtx, rasterPass, gfx, desc.GetSets(), uniforms, syncObj, &projMatrix, &viewMatrix, &modelMatrix) {
			break
		}
	}
}

// Cube vertex data (36 vertices = 12 triangles = 6 faces)
var gVertexBufferData = []float32{
	-1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1,
	-1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1,
	-1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1,
	-1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1,
	1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1,
	-1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1,
}

var gUVBufferData = []float32{
	0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1,
	1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
	1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
	1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
	1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0,
	0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0,
}
