package main

import (
	"bytes"
	_ "embed"
	"image"
	"image/draw"
	"image/png"
	"log"
	"runtime"
	"time"
	"unsafe"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/tomas-mraz/vulkan"
	"github.com/tomas-mraz/vulkan-ash"
)

//go:embed shaders/cube.vert.spv
var vertShaderCode []byte

//go:embed shaders/cube.frag.spv
var fragShaderCode []byte

//go:embed textures/gopher.png
var gopherPng []byte

const (
	windowWidth  = 500
	windowHeight = 500
	appName      = "VulkanCube"
)

func init() {
	runtime.LockOSThread()
}

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
	if err := glfw.Init(); err != nil {
		log.Fatal(err)
	}
	defer glfw.Terminate()

	vk.SetGetInstanceProcAddr(glfw.GetVulkanGetInstanceProcAddress())
	if err := vk.Init(); err != nil {
		log.Fatal(err)
	}

	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI)
	glfw.WindowHint(glfw.Resizable, glfw.False)
	window, err := glfw.CreateWindow(windowWidth, windowHeight, appName, nil, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer window.Destroy()

	extensions := window.GetRequiredInstanceExtensions()
	createSurfaceFn := func(instance vk.Instance) (vk.Surface, error) {
		return ash.NewDesktopSurface(instance, window)
	}
	manager, err := ash.NewManager(appName, extensions, createSurfaceFn, nil)
	if err != nil {
		log.Fatal(err)
	}
	cleanup := ash.NewCleanup(&manager)
	defer cleanup.Destroy()

	windowSize := ash.NewExtentSize(windowWidth, windowHeight)
	swapchain, err := ash.NewSwapchain(manager.Device, manager.Gpu, manager.Surface, windowSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&swapchain)
	swapchainLen := swapchain.DefaultSwapchainLen()

	// Depth buffer via framework
	depth, err := ash.NewImageDepth(manager.Device, manager.Gpu, windowWidth, windowHeight, vk.FormatD16Unorm)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&depth)

	rasterPass, err := ash.NewRasterPassWithDepth(manager.Device, swapchain.DisplayFormat, depth.GetFormat())
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&rasterPass)

	if err := swapchain.CreateFramebuffers(rasterPass.GetRenderPass(), depth.GetView()); err != nil {
		log.Fatal(err)
	}
	cmdCtx, err := ash.NewCommandContext(manager.Device, 0, swapchainLen)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&cmdCtx)

	// Texture via framework
	img, err := png.Decode(bytes.NewReader(gopherPng))
	if err != nil {
		log.Fatal(err)
	}
	rgba := image.NewRGBA(img.Bounds())
	draw.Draw(rgba, rgba.Bounds(), img, image.Point{}, draw.Src)

	texture, err := ash.NewImageTexture(manager.Device, manager.Gpu, uint32(rgba.Bounds().Dx()), uint32(rgba.Bounds().Dy()), rgba.Pix)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&texture)
	texture.TransitionLayout(manager.Queue, cmdCtx.GetCmdPool(), vk.ImageLayoutShaderReadOnlyOptimal)

	// Uniform buffers (one per swapchain image) via framework
	uniforms, err := ash.NewUniformBuffers(manager.Device, manager.Gpu, swapchainLen, uniformDataSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&uniforms)

	// Descriptor set layout + pool + sets
	desc, err := ash.NewDescriptorSets(manager.Device, swapchainLen, []ash.DescriptorBinding{
		&ash.BindingUniformBuffer{StageFlags: vk.ShaderStageFlags(vk.ShaderStageVertexBit), Uniforms: &uniforms},
		ash.NewBindingImageSampler(vk.ShaderStageFlags(vk.ShaderStageFragmentBit), &texture, []vk.Sampler{texture.GetSampler()}),
	})
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&desc)

	// Pipeline
	gfx, err := ash.NewPipelineRasterization(manager.Device, swapchain.DisplaySize, rasterPass.GetRenderPass(), ash.PipelineOptions{
		VertShaderData:       vertShaderCode,
		FragShaderData:       fragShaderCode,
		VertexBindings:       []vk.VertexInputBindingDescription{},
		VertexAttributes:     []vk.VertexInputAttributeDescription{},
		DescriptorSetLayouts: []vk.DescriptorSetLayout{desc.GetLayout()},
		DepthTestEnable:      true,
	})
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&gfx)

	// Sync objects
	sync, err := ash.NewSyncObjects(manager.Device)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&sync)

	// Camera matrices
	var projMatrix, viewMatrix, modelMatrix ash.Mat4x4
	projMatrix.Perspective(ash.DegreesToRadians(45.0), 1.0, 0.1, 100.0)
	viewMatrix.LookAt(&ash.Vec3{0.0, 3.0, 5.0}, &ash.Vec3{0.0, 0.0, 0.0}, &ash.Vec3{0.0, 1.0, 0.0})
	modelMatrix.Identity()
	projMatrix[1][1] *= -1 // Flip for Vulkan

	log.Println("Vulkan initialized, starting render loop")
	startTime := time.Now()

	for !window.ShouldClose() {
		glfw.PollEvents()

		// Rotate model
		elapsed := float32(time.Since(startTime).Seconds()) * 45.0 // 45 deg/sec
		var rotated ash.Mat4x4
		modelMatrix.Identity()
		rotated.Dup(&modelMatrix)
		modelMatrix.Rotate(&rotated, 0.0, 1.0, 0.0, ash.DegreesToRadians(elapsed))

		if !drawCubeFrame(manager.Device, manager.Queue, swapchain, rasterPass, cmdCtx,
			sync.Fence, sync.Semaphore,
			gfx, desc.GetSets(),
			&uniforms,
			&projMatrix, &viewMatrix, &modelMatrix) {
			break
		}
	}
}

func updateUniformBuffer(uniforms *ash.VulkanUniformBuffers, index uint32, proj, view, model *ash.Mat4x4) {
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

func drawCubeFrame(dev vk.Device, queue vk.Queue, s ash.Swapchain,
	rasterPass ash.RasterizationPass, cmdCtx ash.CommandContext,
	fence vk.Fence, semaphore vk.Semaphore,
	gfx ash.PipelineRasterization, descSets []vk.DescriptorSet,
	uniforms *ash.VulkanUniformBuffers,
	proj, view, model *ash.Mat4x4,
) bool {
	var nextIdx uint32
	ret := vk.AcquireNextImage(dev, s.DefaultSwapchain(), vk.MaxUint64, semaphore, vk.NullFence, &nextIdx)
	if ret != vk.Success && ret != vk.Suboptimal {
		return false
	}

	// Update uniform buffer for this frame
	updateUniformBuffer(uniforms, nextIdx, proj, view, model)

	cmd := cmdCtx.GetCmdBuffers()[nextIdx]
	vk.ResetCommandBuffer(cmd, 0)
	vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo})

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
	vk.EndCommandBuffer(cmd)

	vk.ResetFences(dev, 1, []vk.Fence{fence})
	if err := vk.Error(vk.QueueSubmit(queue, 1, []vk.SubmitInfo{{
		SType: vk.StructureTypeSubmitInfo, WaitSemaphoreCount: 1, PWaitSemaphores: []vk.Semaphore{semaphore},
		PWaitDstStageMask:  []vk.PipelineStageFlags{vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit)},
		CommandBufferCount: 1, PCommandBuffers: cmdCtx.GetCmdBuffers()[nextIdx:],
	}}, fence)); err != nil {
		log.Println("QueueSubmit:", err)
		return false
	}
	if err := vk.Error(vk.WaitForFences(dev, 1, []vk.Fence{fence}, vk.True, 10_000_000_000)); err != nil {
		log.Println("WaitForFences:", err)
		return false
	}

	ret = vk.QueuePresent(queue, &vk.PresentInfo{
		SType: vk.StructureTypePresentInfo, SwapchainCount: 1, PSwapchains: s.Swapchains, PImageIndices: []uint32{nextIdx},
	})
	return ret == vk.Success || ret == vk.Suboptimal
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
