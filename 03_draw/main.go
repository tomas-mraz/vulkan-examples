package main

import (
	_ "embed"
	"log"
	"runtime"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/tomas-mraz/vulkan"
	asch "github.com/tomas-mraz/vulkan-ash"
)

//go:embed shaders/tri-vert.spv
var vertShaderCode []byte

//go:embed shaders/tri-frag.spv
var fragShaderCode []byte

const (
	windowWidth  = 640
	windowHeight = 480
	appName      = "VulkanDraw"
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

	sync, err := asch.NewSyncObjects(device.Device)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&sync)

	vertices := []float32{
		-1, -1, 0,
		1, -1, 0,
		0, 1, 0,
	}
	buffer, err := asch.NewBufferHostVisible(
		device.Device,
		device.GpuDevice,
		vertices,
		false,
		vk.BufferUsageFlags(vk.BufferUsageVertexBufferBit),
	)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&buffer)

	gfx, err := asch.NewGraphicsPipelineWithOptions(device.Device, swapchain.DisplaySize, rasterPass.GetRenderPass(), asch.PipelineOptions{
		VertShaderData: vertShaderCode,
		FragShaderData: fragShaderCode,
	})
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&gfx)

	if err := recordCommandBuffers(swapchain, rasterPass, cmdCtx, buffer, gfx); err != nil {
		log.Fatal(err)
	}

	log.Println("Vulkan initialized, starting render loop")

	for !window.ShouldClose() {
		glfw.PollEvents()
		if window.GetAttrib(glfw.Iconified) != 1 {
			if !drawFrame(device.Device, device.Queue, swapchain, cmdCtx, sync.Fence, sync.Semaphore) {
				break
			}
		}
	}
}

func recordCommandBuffers(s asch.VulkanSwapchainInfo, rasterPass asch.VulkanRasterPassInfo, cmdCtx asch.VulkanCommandContext,
	buffer asch.VulkanBufferResource, gfx asch.PipelineRasterizationInfo,
) error {
	clearValues := []vk.ClearValue{
		vk.NewClearValue([]float32{0.098, 0.71, 0.996, 1}),
	}
	cmdBuffers := cmdCtx.GetCmdBuffers()
	for i := range cmdBuffers {
		if err := vk.Error(vk.BeginCommandBuffer(cmdBuffers[i], &vk.CommandBufferBeginInfo{
			SType: vk.StructureTypeCommandBufferBeginInfo,
		})); err != nil {
			return err
		}
		vk.CmdBeginRenderPass(cmdBuffers[i], &vk.RenderPassBeginInfo{
			SType:       vk.StructureTypeRenderPassBeginInfo,
			RenderPass:  rasterPass.GetRenderPass(),
			Framebuffer: s.Framebuffers[i],
			RenderArea: vk.Rect2D{
				Extent: s.DisplaySize,
			},
			ClearValueCount: 1,
			PClearValues:    clearValues,
		}, vk.SubpassContentsInline)
		vk.CmdBindPipeline(cmdBuffers[i], vk.PipelineBindPointGraphics, gfx.GetPipeline())
		vk.CmdBindVertexBuffers(cmdBuffers[i], 0, 1, []vk.Buffer{buffer.Buffer}, []vk.DeviceSize{0})
		vk.CmdDraw(cmdBuffers[i], 3, 1, 0, 0)
		vk.CmdEndRenderPass(cmdBuffers[i])
		if err := vk.Error(vk.EndCommandBuffer(cmdBuffers[i])); err != nil {
			return err
		}
	}
	return nil
}

func drawFrame(device vk.Device, queue vk.Queue, s asch.VulkanSwapchainInfo,
	cmdCtx asch.VulkanCommandContext, fence vk.Fence, semaphore vk.Semaphore,
) bool {
	var nextIdx uint32

	ret := vk.AcquireNextImage(device, s.DefaultSwapchain(), vk.MaxUint64, semaphore, vk.NullFence, &nextIdx)
	if ret == vk.Suboptimal || ret == vk.ErrorOutOfDate {
		log.Println("AcquireNextImage returned Suboptimal or ErrorOutOfDate")
	}
	if ret != vk.Success && ret != vk.Suboptimal {
		if err := vk.Error(ret); err != nil {
			log.Println("AcquireNextImage:", err)
		}
		return false
	}

	waitStages := []vk.PipelineStageFlags{vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit)}
	fences := []vk.Fence{fence}
	semaphores := []vk.Semaphore{semaphore}
	cmdBuffers := cmdCtx.GetCmdBuffers()

	vk.ResetFences(device, 1, fences)
	submitInfo := []vk.SubmitInfo{{
		SType:              vk.StructureTypeSubmitInfo,
		WaitSemaphoreCount: 1,
		PWaitSemaphores:    semaphores,
		PWaitDstStageMask:  waitStages,
		CommandBufferCount: 1,
		PCommandBuffers:    cmdBuffers[nextIdx:],
	}}
	if err := vk.Error(vk.QueueSubmit(queue, 1, submitInfo, fence)); err != nil {
		log.Println("QueueSubmit:", err)
		return false
	}

	const timeoutNano = 10 * 1000 * 1000 * 1000
	if err := vk.Error(vk.WaitForFences(device, 1, fences, vk.True, timeoutNano)); err != nil {
		log.Println("WaitForFences:", err)
		return false
	}

	ret = vk.QueuePresent(queue, &vk.PresentInfo{
		SType:          vk.StructureTypePresentInfo,
		SwapchainCount: 1,
		PSwapchains:    s.Swapchains,
		PImageIndices:  []uint32{nextIdx},
	})
	if ret == vk.Suboptimal || ret == vk.ErrorOutOfDate {
		log.Println("QueuePresent returned Suboptimal or ErrorOutOfDate")
	}
	if ret != vk.Success {
		if err := vk.Error(ret); err != nil {
			log.Println("QueuePresent:", err)
		}
		return false
	}
	return true
}
