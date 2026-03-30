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

	renderer, err := asch.NewRenderer(device.Device, swapchain.DisplayFormat)
	if err != nil {
		log.Fatal(err)
	}
	if err := swapchain.CreateFramebuffers(renderer.RenderPass, vk.NullImageView); err != nil {
		log.Fatal(err)
	}
	if err := renderer.CreateCommandBuffers(swapchain.DefaultSwapchainLen()); err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&renderer)

	buffer, err := asch.NewBuffer(device.Device, device.GpuDevice)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&buffer)

	gfx, err := asch.NewGraphicsPipelineWithOptions(device.Device, swapchain.DisplaySize, renderer.RenderPass, asch.PipelineOptions{
		VertShaderData: vertShaderCode,
		FragShaderData: fragShaderCode,
	})
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&gfx)

	// Record command buffers (static, no per-frame updates needed)
	asch.VulkanStart(device.Device, &swapchain, &renderer, &buffer, &gfx)

	log.Println("Vulkan initialized, starting render loop")

	for !window.ShouldClose() {
		glfw.PollEvents()
		if window.GetAttrib(glfw.Iconified) != 1 {
			if !drawFrame(device.Device, device.Queue, swapchain, renderer) {
				break
			}
		}
	}
}

func drawFrame(device vk.Device, queue vk.Queue, s asch.VulkanSwapchainInfo, r asch.VulkanRenderInfo) bool {
	var nextIdx uint32

	ret := vk.AcquireNextImage(device, s.DefaultSwapchain(), vk.MaxUint64, r.DefaultSemaphore(), vk.NullFence, &nextIdx)
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
	fences := []vk.Fence{r.DefaultFence()}
	semaphores := []vk.Semaphore{r.DefaultSemaphore()}
	cmdBuffers := r.GetCmdBuffers()

	vk.ResetFences(device, 1, fences)
	submitInfo := []vk.SubmitInfo{{
		SType:              vk.StructureTypeSubmitInfo,
		WaitSemaphoreCount: 1,
		PWaitSemaphores:    semaphores,
		PWaitDstStageMask:  waitStages,
		CommandBufferCount: 1,
		PCommandBuffers:    cmdBuffers[nextIdx:],
	}}
	if err := vk.Error(vk.QueueSubmit(queue, 1, submitInfo, r.DefaultFence())); err != nil {
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
