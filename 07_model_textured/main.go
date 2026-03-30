package main

import (
	_ "embed"
	"log"
	"runtime"
	"time"
	"unsafe"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/tomas-mraz/vulkan"
	asch "github.com/tomas-mraz/vulkan-ash"
	lin "github.com/xlab/linmath"
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
	MVP   lin.Mat4x4
	Model lin.Mat4x4
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
	model, err := asch.LoadGLBModel("DiffuseTransmissionTeacup.glb")
	if err != nil {
		log.Fatal("LoadGLBModel:", err)
	}
	log.Printf("Loaded model: %d vertices, %d indices, texture %dx%d", model.VertexCount(), model.IndexCount(), model.TextureWidth, model.TextureHeight)

	// Vulkan init
	var cleanup asch.Destroyer

	asch.SetDebug(false)
	extensions := window.GetRequiredInstanceExtensions()
	device, err := asch.NewDevice(appName, extensions, func(instance vk.Instance, _ uintptr) (vk.Surface, error) {
		surfPtr, err := window.CreateWindowSurface(instance, nil)
		if err != nil {
			return vk.NullSurface, err
		}
		return vk.SurfaceFromPointer(surfPtr), nil
	}, 0)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(device.Destroy)

	windowSize := asch.NewExtentSize(windowWidth, windowHeight)
	swapchain, err := asch.NewSwapchain(device.Device, device.GpuDevice, device.Surface, windowSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(swapchain.Destroy)
	swapchainLen := swapchain.DefaultSwapchainLen()

	depth, err := asch.NewDepthBuffer(device.Device, device.GpuDevice, windowWidth, windowHeight, vk.FormatD32Sfloat)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(depth.Destroy)

	renderer, err := asch.NewRendererWithDepth(device.Device, swapchain.DisplayFormat, depth.GetFormat())
	if err != nil {
		log.Fatal(err)
	}
	if err := swapchain.CreateFramebuffers(renderer.RenderPass, depth.GetView()); err != nil {
		log.Fatal(err)
	}
	if err := renderer.CreateCommandBuffers(swapchainLen); err != nil {
		log.Fatal(err)
	}
	cleanup.Add(renderer.Destroy)

	// Buffers
	vertexBuf, err := asch.NewBufferWithData(device.Device, device.GpuDevice, model.Vertices)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(vertexBuf.Destroy)
	indexBuf, err := asch.NewIndexBuffer32(device.Device, device.GpuDevice, model.Indices)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(indexBuf.Destroy)

	// Texture
	texture, err := asch.NewTexture(device.Device, device.GpuDevice, model.TextureWidth, model.TextureHeight, model.TextureRGBA)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(texture.Destroy)
	asch.TransitionImageLayout(device.Device, device.Queue, renderer.GetCmdPool(),
		texture.GetImage(), vk.ImageLayoutPreinitialized, vk.ImageLayoutShaderReadOnlyOptimal)

	// Uniform buffers
	uniforms, err := asch.NewUniformBuffers(device.Device, device.GpuDevice, swapchainLen, uboSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(uniforms.Destroy)

	// Descriptors (UBO + texture)
	descLayout, descPool, descSets, err := createDescriptors(device.Device, swapchainLen, &uniforms, texture)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(func() { vk.DestroyDescriptorSetLayout(device.Device, descLayout, nil) })
	cleanup.Add(func() { vk.DestroyDescriptorPool(device.Device, descPool, nil) })

	// Pipeline
	gfx, err := asch.NewGraphicsPipelineWithOptions(device.Device, swapchain.DisplaySize, renderer.RenderPass, asch.PipelineOptions{
		VertShaderData: vertShaderCode,
		FragShaderData: fragShaderCode,
		VertexBindings: []vk.VertexInputBindingDescription{{
			Binding: 0, Stride: 8 * 4, InputRate: vk.VertexInputRateVertex, // pos3+norm3+uv2
		}},
		VertexAttributes: []vk.VertexInputAttributeDescription{
			{Binding: 0, Location: 0, Format: vk.FormatR32g32b32Sfloat, Offset: 0},      // position
			{Binding: 0, Location: 1, Format: vk.FormatR32g32b32Sfloat, Offset: 3 * 4},   // normal
			{Binding: 0, Location: 2, Format: vk.FormatR32g32Sfloat, Offset: 6 * 4},       // texcoord
		},
		DescriptorSetLayouts: []vk.DescriptorSetLayout{descLayout},
		DepthTestEnable:      true,
	})
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(gfx.Destroy)

	fence, semaphore, err := createSyncObjects(device.Device)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(func() { vk.DestroyFence(device.Device, fence, nil) })
	cleanup.Add(func() { vk.DestroySemaphore(device.Device, semaphore, nil) })

	// Camera
	var projMatrix, viewMatrix lin.Mat4x4
	projMatrix.Perspective(lin.DegreesToRadians(45.0), float32(windowWidth)/float32(windowHeight), 0.001, 10.0)
	viewMatrix.LookAt(&lin.Vec3{0.0, 0.08, 0.2}, &lin.Vec3{0.0, 0.03, 0.0}, &lin.Vec3{0.0, 1.0, 0.0})
	projMatrix[1][1] *= -1

	log.Println("Vulkan initialized, starting render loop")
	startTime := time.Now()

	for !window.ShouldClose() {
		glfw.PollEvents()

		elapsed := float32(time.Since(startTime).Seconds()) * 30.0
		var modelMatrix lin.Mat4x4
		modelMatrix.Identity()
		var rotated lin.Mat4x4
		rotated.Dup(&modelMatrix)
		modelMatrix.Rotate(&rotated, 0.0, 1.0, 0.0, lin.DegreesToRadians(elapsed))

		if !drawFrame(device.Device, device.Queue, swapchain, renderer,
			fence, semaphore, gfx, descSets, &uniforms,
			vertexBuf, indexBuf,
			&projMatrix, &viewMatrix, &modelMatrix) {
			break
		}
	}

	cleanup.Destroy()
}

// --- Vulkan helpers ---

func createDescriptors(dev vk.Device, count uint32, uniforms *asch.VulkanUniformBuffers, tex asch.VulkanTextureInfo) (vk.DescriptorSetLayout, vk.DescriptorPool, []vk.DescriptorSet, error) {
	var layout vk.DescriptorSetLayout
	if err := vk.Error(vk.CreateDescriptorSetLayout(dev, &vk.DescriptorSetLayoutCreateInfo{
		SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: 2,
		PBindings: []vk.DescriptorSetLayoutBinding{
			{Binding: 0, DescriptorType: vk.DescriptorTypeUniformBuffer, DescriptorCount: 1,
				StageFlags: vk.ShaderStageFlags(vk.ShaderStageVertexBit), PImmutableSamplers: []vk.Sampler{vk.NullSampler}},
			{Binding: 1, DescriptorType: vk.DescriptorTypeCombinedImageSampler, DescriptorCount: 1,
				StageFlags: vk.ShaderStageFlags(vk.ShaderStageFragmentBit), PImmutableSamplers: []vk.Sampler{tex.GetSampler()}},
		},
	}, nil, &layout)); err != nil {
		return layout, nil, nil, err
	}

	var pool vk.DescriptorPool
	if err := vk.Error(vk.CreateDescriptorPool(dev, &vk.DescriptorPoolCreateInfo{
		SType: vk.StructureTypeDescriptorPoolCreateInfo, MaxSets: count, PoolSizeCount: 2,
		PPoolSizes: []vk.DescriptorPoolSize{
			{Type: vk.DescriptorTypeUniformBuffer, DescriptorCount: count},
			{Type: vk.DescriptorTypeCombinedImageSampler, DescriptorCount: count},
		},
	}, nil, &pool)); err != nil {
		return layout, pool, nil, err
	}

	sets := make([]vk.DescriptorSet, count)
	for i := uint32(0); i < count; i++ {
		if err := vk.Error(vk.AllocateDescriptorSets(dev, &vk.DescriptorSetAllocateInfo{
			SType: vk.StructureTypeDescriptorSetAllocateInfo, DescriptorPool: pool,
			DescriptorSetCount: 1, PSetLayouts: []vk.DescriptorSetLayout{layout},
		}, &sets[i])); err != nil {
			return layout, pool, sets, err
		}
	}

	for i := uint32(0); i < count; i++ {
		vk.UpdateDescriptorSets(dev, 2, []vk.WriteDescriptorSet{
			{SType: vk.StructureTypeWriteDescriptorSet, DstSet: sets[i], DstBinding: 0, DescriptorCount: 1,
				DescriptorType: vk.DescriptorTypeUniformBuffer,
				PBufferInfo:    []vk.DescriptorBufferInfo{{Buffer: uniforms.GetBuffer(i), Offset: 0, Range: vk.DeviceSize(uboSize)}}},
			{SType: vk.StructureTypeWriteDescriptorSet, DstSet: sets[i], DstBinding: 1, DescriptorCount: 1,
				DescriptorType: vk.DescriptorTypeCombinedImageSampler,
				PImageInfo:     []vk.DescriptorImageInfo{{Sampler: tex.GetSampler(), ImageView: tex.GetView(), ImageLayout: vk.ImageLayoutShaderReadOnlyOptimal}}},
		}, 0, nil)
	}
	return layout, pool, sets, nil
}

func createSyncObjects(dev vk.Device) (vk.Fence, vk.Semaphore, error) {
	var fence vk.Fence
	var sem vk.Semaphore
	if err := vk.Error(vk.CreateFence(dev, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &fence)); err != nil {
		return fence, sem, err
	}
	if err := vk.Error(vk.CreateSemaphore(dev, &vk.SemaphoreCreateInfo{SType: vk.StructureTypeSemaphoreCreateInfo}, nil, &sem)); err != nil {
		return fence, sem, err
	}
	return fence, sem, nil
}

func drawFrame(dev vk.Device, queue vk.Queue, s asch.VulkanSwapchainInfo,
	r asch.VulkanRenderInfo,
	fence vk.Fence, semaphore vk.Semaphore,
	gfx asch.VulkanGfxPipelineInfo, descSets []vk.DescriptorSet,
	uniforms *asch.VulkanUniformBuffers,
	vertexBuf asch.VulkanBufferInfo, indexBuf asch.VulkanIndexBufferInfo,
	proj, view, model *lin.Mat4x4,
) bool {
	var nextIdx uint32
	ret := vk.AcquireNextImage(dev, s.DefaultSwapchain(), vk.MaxUint64, semaphore, vk.NullFence, &nextIdx)
	if ret != vk.Success && ret != vk.Suboptimal {
		return false
	}

	var VP, MVP lin.Mat4x4
	VP.Mult(proj, view)
	MVP.Mult(&VP, model)
	ubo := uboData{MVP: MVP, Model: *model}
	uniforms.Update(nextIdx, ubo.Bytes())

	cmd := r.GetCmdBuffers()[nextIdx]
	vk.ResetCommandBuffer(cmd, 0)
	vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo})

	clearValues := make([]vk.ClearValue, 2)
	clearValues[0].SetColor([]float32{0.15, 0.15, 0.18, 1.0})
	clearValues[1].SetDepthStencil(1.0, 0)

	vk.CmdBeginRenderPass(cmd, &vk.RenderPassBeginInfo{
		SType: vk.StructureTypeRenderPassBeginInfo, RenderPass: r.RenderPass, Framebuffer: s.Framebuffers[nextIdx],
		RenderArea: vk.Rect2D{Extent: s.DisplaySize}, ClearValueCount: 2, PClearValues: clearValues,
	}, vk.SubpassContentsInline)

	vk.CmdBindPipeline(cmd, vk.PipelineBindPointGraphics, gfx.GetPipeline())
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointGraphics, gfx.GetLayout(), 0, 1, []vk.DescriptorSet{descSets[nextIdx]}, 0, nil)
	vk.CmdBindVertexBuffers(cmd, 0, 1, []vk.Buffer{vertexBuf.DefaultVertexBuffer()}, []vk.DeviceSize{0})
	vk.CmdBindIndexBuffer(cmd, indexBuf.GetBuffer(), 0, vk.IndexTypeUint32)
	vk.CmdDrawIndexed(cmd, indexBuf.GetCount(), 1, 0, 0, 0)
	vk.CmdEndRenderPass(cmd)
	vk.EndCommandBuffer(cmd)

	vk.ResetFences(dev, 1, []vk.Fence{fence})
	if err := vk.Error(vk.QueueSubmit(queue, 1, []vk.SubmitInfo{{
		SType: vk.StructureTypeSubmitInfo, WaitSemaphoreCount: 1, PWaitSemaphores: []vk.Semaphore{semaphore},
		PWaitDstStageMask:  []vk.PipelineStageFlags{vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit)},
		CommandBufferCount: 1, PCommandBuffers: r.GetCmdBuffers()[nextIdx:],
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
