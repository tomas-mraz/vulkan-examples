package main

import (
	_ "embed"
	"log"
	"runtime"
	"time"
	"unsafe"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/tomas-mraz/vulkan"
	"github.com/tomas-mraz/vulkan-ash"
)

//go:embed shaders/textured.vert.spv
var vertShaderCode []byte

//go:embed shaders/textured.frag.spv
var fragShaderCode []byte

const (
	windowWidth  = 800
	windowHeight = 600
	appName      = "glTF Textured Model"
)

type uboData struct {
	MVP   ash.Mat4x4
	Model ash.Mat4x4
}

const uboSize = int(unsafe.Sizeof(uboData{}))

func (u *uboData) Bytes() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(u)), uboSize)
}

func init() {
	runtime.LockOSThread()
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

	// Load GLB model
	model, err := ash.LoadGLBModel("DiffuseTransmissionTeacup.glb")
	if err != nil {
		log.Fatal("LoadGLBModel:", err)
	}
	log.Printf("Loaded model: %d vertices, %d indices, texture %dx%d", model.VertexCount(), model.IndexCount(), model.TextureWidth, model.TextureHeight)

	// Vulkan init
	var cleanup ash.Cleanup
	defer cleanup.Destroy()

	ash.SetDebug(false)
	extensions := window.GetRequiredInstanceExtensions()
	manager, err := ash.NewManager(appName, extensions, func(instance vk.Instance) (vk.Surface, error) {
		return ash.NewDesktopSurface(instance, window)
	}, nil)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&manager)

	windowSize := ash.NewExtentSize(windowWidth, windowHeight)
	swapchain, err := ash.NewSwapchain(manager.Device, manager.Gpu, manager.Surface, windowSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&swapchain)
	swapchainLen := swapchain.DefaultSwapchainLen()

	depth, err := ash.NewImageDepth(manager.Device, manager.Gpu, windowWidth, windowHeight, vk.FormatD32Sfloat)
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

	// Buffers
	vertexBuf, err := ash.NewBufferHostVisible(
		manager.Device,
		manager.Gpu,
		model.Vertices,
		false,
		vk.BufferUsageFlags(vk.BufferUsageVertexBufferBit),
	)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&vertexBuf)
	indexBuf, err := ash.NewBufferHostVisible(
		manager.Device,
		manager.Gpu,
		model.Indices,
		false,
		vk.BufferUsageFlags(vk.BufferUsageIndexBufferBit),
	)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&indexBuf)

	// Texture
	texture, err := ash.NewImageTexture(manager.Device, manager.Gpu, model.TextureWidth, model.TextureHeight, model.TextureRGBA)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&texture)
	ash.TransitionImageLayout(manager.Device, manager.Queue, cmdCtx.GetCmdPool(),
		texture.GetImage(), vk.ImageLayoutPreinitialized, vk.ImageLayoutShaderReadOnlyOptimal)

	// Uniform buffers
	uniforms, err := ash.NewUniformBuffers(manager.Device, manager.Gpu, swapchainLen, uboSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&uniforms)

	// Descriptors (UBO + texture)
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
		VertShaderData: vertShaderCode,
		FragShaderData: fragShaderCode,
		VertexBindings: []vk.VertexInputBindingDescription{{
			Binding: 0, Stride: 8 * 4, InputRate: vk.VertexInputRateVertex, // pos3+norm3+uv2
		}},
		VertexAttributes: []vk.VertexInputAttributeDescription{
			{Binding: 0, Location: 0, Format: vk.FormatR32g32b32Sfloat, Offset: 0},     // position
			{Binding: 0, Location: 1, Format: vk.FormatR32g32b32Sfloat, Offset: 3 * 4}, // normal
			{Binding: 0, Location: 2, Format: vk.FormatR32g32Sfloat, Offset: 6 * 4},    // texcoord
		},
		DescriptorSetLayouts: []vk.DescriptorSetLayout{desc.GetLayout()},
		DepthTestEnable:      true,
	})
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&gfx)

	sync, err := ash.NewSyncObjects(manager.Device)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&sync)

	// Camera
	var projMatrix, viewMatrix ash.Mat4x4
	projMatrix.Perspective(ash.DegreesToRadians(45.0), float32(windowWidth)/float32(windowHeight), 0.001, 10.0)
	viewMatrix.LookAt(&ash.Vec3{0.0, 0.08, 0.2}, &ash.Vec3{0.0, 0.03, 0.0}, &ash.Vec3{0.0, 1.0, 0.0})
	projMatrix[1][1] *= -1

	log.Println("Vulkan initialized, starting render loop")
	startTime := time.Now()

	for !window.ShouldClose() {
		glfw.PollEvents()

		elapsed := float32(time.Since(startTime).Seconds()) * 30.0
		var modelMatrix ash.Mat4x4
		modelMatrix.Identity()
		var rotated ash.Mat4x4
		rotated.Dup(&modelMatrix)
		modelMatrix.Rotate(&rotated, 0.0, 1.0, 0.0, ash.DegreesToRadians(elapsed))

		if !drawFrame(manager.Device, manager.Queue, swapchain, rasterPass, cmdCtx,
			sync.Fence, sync.Semaphore, gfx, desc.GetSets(), &uniforms,
			vertexBuf, indexBuf, uint32(len(model.Indices)),
			&projMatrix, &viewMatrix, &modelMatrix) {
			break
		}
	}
}

func drawFrame(dev vk.Device, queue vk.Queue, s ash.Swapchain,
	rasterPass ash.RasterizationPass, cmdCtx ash.CommandContext,
	fence vk.Fence, semaphore vk.Semaphore,
	gfx ash.PipelineRasterization, descSets []vk.DescriptorSet,
	uniforms *ash.VulkanUniformBuffers,
	vertexBuf ash.BufferResource, indexBuf ash.BufferResource, indexCount uint32,
	proj, view, model *ash.Mat4x4,
) bool {
	var nextIdx uint32
	ret := vk.AcquireNextImage(dev, s.DefaultSwapchain(), vk.MaxUint64, semaphore, vk.NullFence, &nextIdx)
	if ret != vk.Success && ret != vk.Suboptimal {
		return false
	}

	var VP, MVP ash.Mat4x4
	VP.Mult(proj, view)
	MVP.Mult(&VP, model)
	ubo := uboData{MVP: MVP, Model: *model}
	uniforms.Update(nextIdx, ubo.Bytes())

	cmd := cmdCtx.GetCmdBuffers()[nextIdx]
	vk.ResetCommandBuffer(cmd, 0)
	vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo})

	clearValues := make([]vk.ClearValue, 2)
	clearValues[0].SetColor([]float32{0.15, 0.15, 0.18, 1.0})
	clearValues[1].SetDepthStencil(1.0, 0)

	vk.CmdBeginRenderPass(cmd, &vk.RenderPassBeginInfo{
		SType: vk.StructureTypeRenderPassBeginInfo, RenderPass: rasterPass.GetRenderPass(), Framebuffer: s.Framebuffers[nextIdx],
		RenderArea: vk.Rect2D{Extent: s.DisplaySize}, ClearValueCount: 2, PClearValues: clearValues,
	}, vk.SubpassContentsInline)

	vk.CmdBindPipeline(cmd, vk.PipelineBindPointGraphics, gfx.GetPipeline())
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointGraphics, gfx.GetLayout(), 0, 1, []vk.DescriptorSet{descSets[nextIdx]}, 0, nil)
	vk.CmdBindVertexBuffers(cmd, 0, 1, []vk.Buffer{vertexBuf.Buffer}, []vk.DeviceSize{0})
	vk.CmdBindIndexBuffer(cmd, indexBuf.Buffer, 0, vk.IndexTypeUint32)
	vk.CmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0)
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
