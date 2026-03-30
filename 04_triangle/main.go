package main

import (
	_ "embed"
	"log"
	"math"
	"runtime"
	"time"
	"unsafe"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/tomas-mraz/vulkan"
	asch "github.com/tomas-mraz/vulkan-ash"
)

//go:embed shaders/triangle.vert.spv
var vertShaderCode []byte

//go:embed shaders/triangle.frag.spv
var fragShaderCode []byte

const (
	windowWidth  = 800
	windowHeight = 600
	appName      = "Rotating Triangle"
)

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

	// Use vulkan-ash for device creation with GLFW surface
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

	// Use vulkan-ash for swapchain
	windowSize := asch.NewExtentSize(windowWidth, windowHeight)
	swapchain, err := asch.NewSwapchain(device.Device, device.GpuDevice, device.Surface, windowSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(swapchain.Destroy)

	// Use vulkan-ash for renderer (render pass + command pool)
	renderer, err := asch.NewRenderer(device.Device, swapchain.DisplayFormat)
	if err != nil {
		log.Fatal(err)
	}

	// Create framebuffers using vulkan-ash swapchain
	if err := swapchain.CreateFramebuffers(renderer.RenderPass, vk.NullImageView); err != nil {
		log.Fatal(err)
	}

	// Allocate command buffers via framework
	if err := renderer.CreateCommandBuffers(swapchain.DefaultSwapchainLen()); err != nil {
		log.Fatal(err)
	}
	cleanup.Add(renderer.Destroy)

	// Create vertex buffer with triangle data via framework
	r := float32(0.5)
	vertices := []float32{
		0, -r, 0, // top
		r * float32(math.Sin(2*math.Pi/3)), -r * float32(math.Cos(2*math.Pi/3)), 0, // bottom-left
		r * float32(math.Sin(4*math.Pi/3)), -r * float32(math.Cos(4*math.Pi/3)), 0, // bottom-right
	}
	buffer, err := asch.NewBufferWithData(device.Device, device.GpuDevice, vertices)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(buffer.Destroy)

	// Create graphics pipeline with push constants via framework
	gfx, err := asch.NewGraphicsPipelineWithOptions(device.Device, swapchain.DisplaySize, renderer.RenderPass, asch.PipelineOptions{
		VertShaderData: vertShaderCode,
		FragShaderData: fragShaderCode,
		PushConstantRanges: []vk.PushConstantRange{{
			StageFlags: vk.ShaderStageFlags(vk.ShaderStageVertexBit),
			Offset:     0,
			Size:       4, // float32
		}},
	})
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(gfx.Destroy)

	// Create sync objects
	fence, semaphore, err := createSyncObjects(device.Device)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(func() { vk.DestroyFence(device.Device, fence, nil) })
	cleanup.Add(func() { vk.DestroySemaphore(device.Device, semaphore, nil) })

	log.Println("Vulkan initialized, starting render loop")
	startTime := time.Now()

	// Main loop
	for !window.ShouldClose() {
		glfw.PollEvents()

		angle := float32(time.Since(startTime).Seconds())

		if !drawFrame(device.Device, device.Queue, swapchain, renderer, buffer,
			fence, semaphore, gfx, angle) {
			break
		}
	}

	cleanup.Destroy()
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
	r asch.VulkanRenderInfo, b asch.VulkanBufferInfo,
	fence vk.Fence, semaphore vk.Semaphore,
	gfx asch.VulkanGfxPipelineInfo, angle float32,
) bool {
	var nextIdx uint32
	ret := vk.AcquireNextImage(dev, s.DefaultSwapchain(), vk.MaxUint64, semaphore, vk.NullFence, &nextIdx)
	if ret != vk.Success && ret != vk.Suboptimal {
		return false
	}

	cmd := r.GetCmdBuffers()[nextIdx]
	vk.ResetCommandBuffer(cmd, 0)

	vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo})

	clearValues := []vk.ClearValue{vk.NewClearValue([]float32{0, 0, 0, 1})}
	vk.CmdBeginRenderPass(cmd, &vk.RenderPassBeginInfo{
		SType:           vk.StructureTypeRenderPassBeginInfo,
		RenderPass:      r.RenderPass,
		Framebuffer:     s.Framebuffers[nextIdx],
		RenderArea:      vk.Rect2D{Extent: s.DisplaySize},
		ClearValueCount: 1,
		PClearValues:    clearValues,
	}, vk.SubpassContentsInline)

	vk.CmdBindPipeline(cmd, vk.PipelineBindPointGraphics, gfx.GetPipeline())
	vk.CmdPushConstants(cmd, gfx.GetLayout(), vk.ShaderStageFlags(vk.ShaderStageVertexBit), 0, 4, unsafe.Pointer(&angle))
	vk.CmdBindVertexBuffers(cmd, 0, 1, []vk.Buffer{b.DefaultVertexBuffer()}, []vk.DeviceSize{0})
	vk.CmdDraw(cmd, 3, 1, 0, 0)
	vk.CmdEndRenderPass(cmd)
	vk.EndCommandBuffer(cmd)

	// Submit and wait
	vk.ResetFences(dev, 1, []vk.Fence{fence})
	submitInfo := []vk.SubmitInfo{{
		SType:              vk.StructureTypeSubmitInfo,
		WaitSemaphoreCount: 1,
		PWaitSemaphores:    []vk.Semaphore{semaphore},
		PWaitDstStageMask:  []vk.PipelineStageFlags{vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit)},
		CommandBufferCount: 1,
		PCommandBuffers:    r.GetCmdBuffers()[nextIdx:],
	}}
	if err := vk.Error(vk.QueueSubmit(queue, 1, submitInfo, fence)); err != nil {
		log.Println("QueueSubmit:", err)
		return false
	}
	if err := vk.Error(vk.WaitForFences(dev, 1, []vk.Fence{fence}, vk.True, 10_000_000_000)); err != nil {
		log.Println("WaitForFences:", err)
		return false
	}

	// Present
	ret = vk.QueuePresent(queue, &vk.PresentInfo{
		SType:          vk.StructureTypePresentInfo,
		SwapchainCount: 1,
		PSwapchains:    s.Swapchains,
		PImageIndices:  []uint32{nextIdx},
	})
	return ret == vk.Success || ret == vk.Suboptimal
}
