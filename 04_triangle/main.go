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
	var cleanup asch.Cleanup
	defer cleanup.Destroy()

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
	cleanup.Add(&device)

	// Use vulkan-ash for swapchain
	windowSize := asch.NewExtentSize(windowWidth, windowHeight)
	swapchain, err := asch.NewSwapchain(device.Device, device.GpuDevice, device.Surface, windowSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&swapchain)

	rasterPass, err := asch.NewRasterPass(device.Device, swapchain.DisplayFormat)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&rasterPass)

	if err := swapchain.CreateFramebuffers(rasterPass.GetRenderPass(), vk.NullImageView); err != nil {
		log.Fatal(err)
	}

	cmdCtx, err := asch.NewCommandContext(device.Device, 0, swapchain.DefaultSwapchainLen())
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&cmdCtx)

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
	cleanup.Add(&buffer)

	// Create graphics pipeline with push constants via framework
	gfx, err := asch.NewGraphicsPipelineWithOptions(device.Device, swapchain.DisplaySize, rasterPass.GetRenderPass(), asch.PipelineOptions{
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
	cleanup.Add(&gfx)

	// Create sync objects
	sync, err := asch.NewSyncObjects(device.Device)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&sync)

	log.Println("Vulkan initialized, starting render loop")
	startTime := time.Now()

	// Main loop
	for !window.ShouldClose() {
		glfw.PollEvents()

		angle := float32(time.Since(startTime).Seconds())

		if !drawFrame(device.Device, device.Queue, swapchain, rasterPass, cmdCtx, buffer,
			sync.Fence, sync.Semaphore, gfx, angle) {
			break
		}
	}
}

func drawFrame(dev vk.Device, queue vk.Queue, s asch.VulkanSwapchainInfo,
	rasterPass asch.VulkanRasterPassInfo, cmdCtx asch.VulkanCommandContext, b asch.VulkanBufferInfo,
	fence vk.Fence, semaphore vk.Semaphore,
	gfx asch.VulkanGfxPipelineInfo, angle float32,
) bool {
	var nextIdx uint32
	ret := vk.AcquireNextImage(dev, s.DefaultSwapchain(), vk.MaxUint64, semaphore, vk.NullFence, &nextIdx)
	if ret != vk.Success && ret != vk.Suboptimal {
		return false
	}

	cmd := cmdCtx.GetCmdBuffers()[nextIdx]
	vk.ResetCommandBuffer(cmd, 0)

	vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo})

	clearValues := []vk.ClearValue{vk.NewClearValue([]float32{0, 0, 0, 1})}
	vk.CmdBeginRenderPass(cmd, &vk.RenderPassBeginInfo{
		SType:           vk.StructureTypeRenderPassBeginInfo,
		RenderPass:      rasterPass.GetRenderPass(),
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
		PCommandBuffers:    cmdCtx.GetCmdBuffers()[nextIdx:],
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
