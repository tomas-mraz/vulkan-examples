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
	asch "github.com/tomas-mraz/vulkan-ash"
	lin "github.com/xlab/linmath"
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
	MVP      lin.Mat4x4
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

	windowSize := asch.NewExtentSize(windowWidth, windowHeight)
	swapchain, err := asch.NewSwapchain(device.Device, device.GpuDevice, device.Surface, windowSize)
	if err != nil {
		log.Fatal(err)
	}

	swapchainLen := swapchain.DefaultSwapchainLen()

	// Depth buffer via framework
	depth, err := asch.NewDepthBuffer(device.Device, device.GpuDevice, windowWidth, windowHeight, vk.FormatD16Unorm)
	if err != nil {
		log.Fatal(err)
	}

	// Renderer with depth (render pass + command pool)
	renderer, err := asch.NewRendererWithDepth(device.Device, swapchain.DisplayFormat, depth.GetFormat())
	if err != nil {
		log.Fatal(err)
	}

	// Framebuffers
	if err := swapchain.CreateFramebuffers(renderer.RenderPass, depth.GetView()); err != nil {
		log.Fatal(err)
	}
	if err := renderer.CreateCommandBuffers(swapchainLen); err != nil {
		log.Fatal(err)
	}

	// Texture via framework
	img, err := png.Decode(bytes.NewReader(gopherPng))
	if err != nil {
		log.Fatal(err)
	}
	rgba := image.NewRGBA(img.Bounds())
	draw.Draw(rgba, rgba.Bounds(), img, image.Point{}, draw.Src)

	texture, err := asch.NewTexture(device.Device, device.GpuDevice,
		uint32(rgba.Bounds().Dx()), uint32(rgba.Bounds().Dy()), rgba.Pix)
	if err != nil {
		log.Fatal(err)
	}
	asch.TransitionImageLayout(device.Device, device.Queue, renderer.GetCmdPool(),
		texture.GetImage(), vk.ImageLayoutPreinitialized, vk.ImageLayoutShaderReadOnlyOptimal)

	// Uniform buffers (one per swapchain image)
	uniformBuffers, uniformMemories, err := createUniformBuffers(device.Device, device.GpuDevice, swapchainLen)
	if err != nil {
		log.Fatal(err)
	}

	// Descriptor set layout + pool + sets
	descLayout, descPool, descSets, err := createDescriptors(device.Device, swapchainLen, uniformBuffers, texture.GetView(), texture.GetSampler())
	if err != nil {
		log.Fatal(err)
	}

	// Pipeline
	pipelineLayout, pipelineObj, pipelineCache, err := createCubePipeline(device.Device, renderer.RenderPass, descLayout)
	if err != nil {
		log.Fatal(err)
	}

	// Sync objects
	fence, semaphore, err := createSyncObjects(device.Device)
	if err != nil {
		log.Fatal(err)
	}

	// Camera matrices
	var projMatrix, viewMatrix, modelMatrix lin.Mat4x4
	projMatrix.Perspective(lin.DegreesToRadians(45.0), 1.0, 0.1, 100.0)
	viewMatrix.LookAt(&lin.Vec3{0.0, 3.0, 5.0}, &lin.Vec3{0.0, 0.0, 0.0}, &lin.Vec3{0.0, 1.0, 0.0})
	modelMatrix.Identity()
	projMatrix[1][1] *= -1 // Flip for Vulkan

	log.Println("Vulkan initialized, starting render loop")
	startTime := time.Now()

	for !window.ShouldClose() {
		glfw.PollEvents()

		// Rotate model
		elapsed := float32(time.Since(startTime).Seconds()) * 45.0 // 45 deg/sec
		var rotated lin.Mat4x4
		modelMatrix.Identity()
		rotated.Dup(&modelMatrix)
		modelMatrix.Rotate(&rotated, 0.0, 1.0, 0.0, lin.DegreesToRadians(elapsed))

		if !drawCubeFrame(device.Device, device.Queue, swapchain, renderer,
			fence, semaphore,
			pipelineLayout, pipelineObj, descSets,
			uniformBuffers, uniformMemories,
			&projMatrix, &viewMatrix, &modelMatrix) {
			break
		}
	}

	vk.DeviceWaitIdle(device.Device)

	// Cleanup
	vk.DestroySemaphore(device.Device, semaphore, nil)
	vk.DestroyFence(device.Device, fence, nil)
	vk.DestroyPipeline(device.Device, pipelineObj, nil)
	vk.DestroyPipelineCache(device.Device, pipelineCache, nil)
	vk.DestroyPipelineLayout(device.Device, pipelineLayout, nil)
	vk.DestroyDescriptorPool(device.Device, descPool, nil)
	vk.DestroyDescriptorSetLayout(device.Device, descLayout, nil)
	for i := uint32(0); i < swapchainLen; i++ {
		vk.FreeMemory(device.Device, uniformMemories[i], nil)
		vk.DestroyBuffer(device.Device, uniformBuffers[i], nil)
	}
	texture.Destroy()
	vk.FreeCommandBuffers(device.Device, renderer.GetCmdPool(), swapchainLen, renderer.GetCmdBuffers())
	vk.DestroyCommandPool(device.Device, renderer.GetCmdPool(), nil)
	vk.DestroyRenderPass(device.Device, renderer.RenderPass, nil)
	depth.Destroy()
	swapchain.Destroy()
	vk.DestroyDevice(device.Device, nil)
	if device.GetDebugCallback() != vk.NullDebugReportCallback {
		vk.DestroyDebugReportCallback(device.Instance, device.GetDebugCallback(), nil)
	}
	vk.DestroySurface(device.Instance, device.Surface, nil)
	vk.DestroyInstance(device.Instance, nil)
}


func createUniformBuffers(dev vk.Device, gpu vk.PhysicalDevice, count uint32) ([]vk.Buffer, []vk.DeviceMemory, error) {
	buffers := make([]vk.Buffer, count)
	memories := make([]vk.DeviceMemory, count)

	for i := uint32(0); i < count; i++ {
		if err := vk.Error(vk.CreateBuffer(dev, &vk.BufferCreateInfo{
			SType: vk.StructureTypeBufferCreateInfo, Size: vk.DeviceSize(uniformDataSize),
			Usage: vk.BufferUsageFlags(vk.BufferUsageUniformBufferBit), SharingMode: vk.SharingModeExclusive,
		}, nil, &buffers[i])); err != nil {
			return buffers, memories, err
		}

		var memReq vk.MemoryRequirements
		vk.GetBufferMemoryRequirements(dev, buffers[i], &memReq)
		memReq.Deref()

		memIdx, _ := vk.FindMemoryTypeIndex(gpu, memReq.MemoryTypeBits, vk.MemoryPropertyHostVisibleBit|vk.MemoryPropertyHostCoherentBit)
		if err := vk.Error(vk.AllocateMemory(dev, &vk.MemoryAllocateInfo{
			SType: vk.StructureTypeMemoryAllocateInfo, AllocationSize: memReq.Size, MemoryTypeIndex: memIdx,
		}, nil, &memories[i])); err != nil {
			return buffers, memories, err
		}
		vk.BindBufferMemory(dev, buffers[i], memories[i], 0)
	}
	return buffers, memories, nil
}

func createDescriptors(dev vk.Device, count uint32, uniformBuffers []vk.Buffer, texView vk.ImageView, sampler vk.Sampler) (vk.DescriptorSetLayout, vk.DescriptorPool, []vk.DescriptorSet, error) {
	var layout vk.DescriptorSetLayout
	// Use immutable sampler to work around vulkan binding bug where nil PImmutableSamplers
	// gets converted to a non-NULL C pointer.
	if err := vk.Error(vk.CreateDescriptorSetLayout(dev, &vk.DescriptorSetLayoutCreateInfo{
		SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: 2,
		PBindings: []vk.DescriptorSetLayoutBinding{
			{Binding: 0, DescriptorType: vk.DescriptorTypeUniformBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageFlags(vk.ShaderStageVertexBit), PImmutableSamplers: []vk.Sampler{vk.NullSampler}},
			{Binding: 1, DescriptorType: vk.DescriptorTypeCombinedImageSampler, DescriptorCount: 1, StageFlags: vk.ShaderStageFlags(vk.ShaderStageFragmentBit), PImmutableSamplers: []vk.Sampler{sampler}},
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

	layouts := make([]vk.DescriptorSetLayout, count)
	for i := range layouts {
		layouts[i] = layout
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
				PBufferInfo:    []vk.DescriptorBufferInfo{{Buffer: uniformBuffers[i], Offset: 0, Range: vk.DeviceSize(uniformDataSize)}}},
			{SType: vk.StructureTypeWriteDescriptorSet, DstSet: sets[i], DstBinding: 1, DescriptorCount: 1,
				DescriptorType: vk.DescriptorTypeCombinedImageSampler,
				PImageInfo:     []vk.DescriptorImageInfo{{Sampler: sampler, ImageView: texView, ImageLayout: vk.ImageLayoutShaderReadOnlyOptimal}}},
		}, 0, nil)
	}

	return layout, pool, sets, nil
}

func createCubePipeline(dev vk.Device, renderPass vk.RenderPass, descLayout vk.DescriptorSetLayout) (vk.PipelineLayout, vk.Pipeline, vk.PipelineCache, error) {
	var pipelineLayout vk.PipelineLayout
	if err := vk.Error(vk.CreatePipelineLayout(dev, &vk.PipelineLayoutCreateInfo{
		SType: vk.StructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1,
		PSetLayouts: []vk.DescriptorSetLayout{descLayout},
	}, nil, &pipelineLayout)); err != nil {
		return pipelineLayout, nil, nil, err
	}

	vertModule, err := asch.LoadShaderFromBytes(dev, vertShaderCode)
	if err != nil {
		return pipelineLayout, nil, nil, err
	}
	defer vk.DestroyShaderModule(dev, vertModule, nil)

	fragModule, err := asch.LoadShaderFromBytes(dev, fragShaderCode)
	if err != nil {
		return pipelineLayout, nil, nil, err
	}
	defer vk.DestroyShaderModule(dev, fragModule, nil)

	var cache vk.PipelineCache
	vk.CreatePipelineCache(dev, &vk.PipelineCacheCreateInfo{SType: vk.StructureTypePipelineCacheCreateInfo}, nil, &cache)

	pipelines := make([]vk.Pipeline, 1)
	if err := vk.Error(vk.CreateGraphicsPipelines(dev, cache, 1, []vk.GraphicsPipelineCreateInfo{{
		SType: vk.StructureTypeGraphicsPipelineCreateInfo, Layout: pipelineLayout, RenderPass: renderPass,
		StageCount: 2,
		PStages: []vk.PipelineShaderStageCreateInfo{
			{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageVertexBit, Module: vertModule, PName: []byte("main\x00")},
			{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageFragmentBit, Module: fragModule, PName: []byte("main\x00")},
		},
		PVertexInputState:   &vk.PipelineVertexInputStateCreateInfo{SType: vk.StructureTypePipelineVertexInputStateCreateInfo},
		PInputAssemblyState: &vk.PipelineInputAssemblyStateCreateInfo{SType: vk.StructureTypePipelineInputAssemblyStateCreateInfo, Topology: vk.PrimitiveTopologyTriangleList},
		PViewportState:      &vk.PipelineViewportStateCreateInfo{SType: vk.StructureTypePipelineViewportStateCreateInfo, ViewportCount: 1, ScissorCount: 1},
		PRasterizationState: &vk.PipelineRasterizationStateCreateInfo{SType: vk.StructureTypePipelineRasterizationStateCreateInfo, PolygonMode: vk.PolygonModeFill, CullMode: vk.CullModeFlags(vk.CullModeBackBit), FrontFace: vk.FrontFaceCounterClockwise, LineWidth: 1},
		PMultisampleState:   &vk.PipelineMultisampleStateCreateInfo{SType: vk.StructureTypePipelineMultisampleStateCreateInfo, RasterizationSamples: vk.SampleCount1Bit},
		PDepthStencilState: &vk.PipelineDepthStencilStateCreateInfo{
			SType: vk.StructureTypePipelineDepthStencilStateCreateInfo, DepthTestEnable: vk.True, DepthWriteEnable: vk.True, DepthCompareOp: vk.CompareOpLessOrEqual,
			Back:  vk.StencilOpState{FailOp: vk.StencilOpKeep, PassOp: vk.StencilOpKeep, CompareOp: vk.CompareOpAlways},
			Front: vk.StencilOpState{FailOp: vk.StencilOpKeep, PassOp: vk.StencilOpKeep, CompareOp: vk.CompareOpAlways},
		},
		PColorBlendState: &vk.PipelineColorBlendStateCreateInfo{SType: vk.StructureTypePipelineColorBlendStateCreateInfo, AttachmentCount: 1,
			PAttachments: []vk.PipelineColorBlendAttachmentState{{ColorWriteMask: 0xF}}},
		PDynamicState: &vk.PipelineDynamicStateCreateInfo{SType: vk.StructureTypePipelineDynamicStateCreateInfo, DynamicStateCount: 2,
			PDynamicStates: []vk.DynamicState{vk.DynamicStateViewport, vk.DynamicStateScissor}},
	}}, nil, pipelines)); err != nil {
		return pipelineLayout, nil, cache, err
	}
	return pipelineLayout, pipelines[0], cache, nil
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

func updateUniformBuffer(dev vk.Device, mem vk.DeviceMemory, proj, view, model *lin.Mat4x4) {
	var VP, MVP lin.Mat4x4
	VP.Mult(proj, view)
	MVP.Mult(&VP, model)

	data := uniformData{MVP: MVP}
	for i := 0; i < 36; i++ {
		data.Position[i] = [4]float32{gVertexBufferData[i*3], gVertexBufferData[i*3+1], gVertexBufferData[i*3+2], 1.0}
		data.Attr[i] = [4]float32{gUVBufferData[i*2], gUVBufferData[i*2+1], 0, 0}
	}

	var pData unsafe.Pointer
	vk.MapMemory(dev, mem, 0, vk.DeviceSize(uniformDataSize), 0, &pData)
	vk.Memcopy(pData, data.Bytes())
	vk.UnmapMemory(dev, mem)
}

func drawCubeFrame(dev vk.Device, queue vk.Queue, s asch.VulkanSwapchainInfo,
	r asch.VulkanRenderInfo,
	fence vk.Fence, semaphore vk.Semaphore,
	pipelineLayout vk.PipelineLayout, pipeline vk.Pipeline,
	descSets []vk.DescriptorSet,
	uniformBuffers []vk.Buffer, uniformMemories []vk.DeviceMemory,
	proj, view, model *lin.Mat4x4,
) bool {
	var nextIdx uint32
	ret := vk.AcquireNextImage(dev, s.DefaultSwapchain(), vk.MaxUint64, semaphore, vk.NullFence, &nextIdx)
	if ret != vk.Success && ret != vk.Suboptimal {
		return false
	}

	// Update uniform buffer for this frame
	updateUniformBuffer(dev, uniformMemories[nextIdx], proj, view, model)

	cmd := r.GetCmdBuffers()[nextIdx]
	vk.ResetCommandBuffer(cmd, 0)
	vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo})

	clearValues := make([]vk.ClearValue, 2)
	clearValues[0].SetColor([]float32{0.2, 0.2, 0.2, 1.0})
	clearValues[1].SetDepthStencil(1.0, 0)

	vk.CmdBeginRenderPass(cmd, &vk.RenderPassBeginInfo{
		SType: vk.StructureTypeRenderPassBeginInfo, RenderPass: r.RenderPass, Framebuffer: s.Framebuffers[nextIdx],
		RenderArea:      vk.Rect2D{Extent: s.DisplaySize},
		ClearValueCount: 2, PClearValues: clearValues,
	}, vk.SubpassContentsInline)

	vk.CmdBindPipeline(cmd, vk.PipelineBindPointGraphics, pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointGraphics, pipelineLayout, 0, 1, []vk.DescriptorSet{descSets[nextIdx]}, 0, nil)
	vk.CmdSetViewport(cmd, 0, 1, []vk.Viewport{{Width: float32(s.DisplaySize.Width), Height: float32(s.DisplaySize.Height), MaxDepth: 1.0}})
	vk.CmdSetScissor(cmd, 0, 1, []vk.Rect2D{{Extent: s.DisplaySize}})
	vk.CmdDraw(cmd, 36, 1, 0, 0)
	vk.CmdEndRenderPass(cmd)
	vk.EndCommandBuffer(cmd)

	vk.ResetFences(dev, 1, []vk.Fence{fence})
	if err := vk.Error(vk.QueueSubmit(queue, 1, []vk.SubmitInfo{{
		SType: vk.StructureTypeSubmitInfo, WaitSemaphoreCount: 1, PWaitSemaphores: []vk.Semaphore{semaphore},
		PWaitDstStageMask: []vk.PipelineStageFlags{vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit)},
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
