package main

import (
	_ "embed"
	"log"
	"runtime"
	"time"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/tomas-mraz/vulkan"
	ash "github.com/tomas-mraz/vulkan-ash"
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
	glfw.WindowHint(glfw.Resizable, glfw.True)
	window, err := glfw.CreateWindow(windowWidth, windowHeight, appName, nil, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer window.Destroy()

	ash.SetDebug(false)
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

	windowSize := waitForFramebufferSize(window)
	swapchain, err := ash.NewSwapchain(manager.Device, manager.Gpu, manager.Surface, windowSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&swapchain)
	swapchainCtx := ash.NewSwapchainContext(&manager, &swapchain)

	window.SetFramebufferSizeCallback(func(_ *glfw.Window, width int, height int) {
		if width == 0 || height == 0 {
			return
		}
		windowSize = ash.NewExtentSize(width, height)
		swapchainCtx.RequestRecreate()
	})

	rasterPass, err := ash.NewRasterPass(manager.Device, swapchain.DisplayFormat)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&rasterPass)
	if err := swapchain.CreateFramebuffers(rasterPass.GetRenderPass(), vk.NullImageView); err != nil {
		log.Fatal(err)
	}
	cmdCtx, err := ash.NewCommandContext(manager.Device, 0, swapchain.DefaultSwapchainLen())
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&cmdCtx)

	sync, err := ash.NewSyncObjects(manager.Device)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&sync)

	vertices := []float32{
		-1, -1, 0,
		1, -1, 0,
		0, 1, 0,
	}
	buffer, err := ash.NewBufferHostVisible(
		manager.Device,
		manager.Gpu,
		vertices,
		false,
		vk.BufferUsageFlags(vk.BufferUsageVertexBufferBit),
	)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&buffer)

	pipelineOptions := ash.PipelineOptions{
		VertShaderData: vertShaderCode,
		FragShaderData: fragShaderCode,
	}
	pipeline, err := ash.NewPipelineRasterization(manager.Device, swapchain.DisplaySize, rasterPass.GetRenderPass(), pipelineOptions)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&pipeline)

	rasterCfg := ash.RasterizationRecreateConfig{
		QueueFamilyIndex: 0,
		PipelineOptions:  pipelineOptions,
	}
	clearValues := []vk.ClearValue{
		vk.NewClearValue([]float32{0.098, 0.71, 0.996, 1}),
	}

	log.Println("Vulkan initialized, starting render loop")

	// main loop
	for !window.ShouldClose() {
		glfw.PollEvents()
		if window.GetAttrib(glfw.Iconified) == 1 {
			time.Sleep(100 * time.Millisecond)
			continue
		}
		imageIndex, ok, err := swapchainCtx.AcquireNextImageRasterization(windowSize, &rasterPass, &cmdCtx, &pipeline, rasterCfg, sync.Semaphore)
		if err != nil {
			log.Println("AcquireNextImage:", err)
			break
		}
		if !ok {
			continue
		}

		cmdBuffer, err := swapchainCtx.BeginRenderPass(imageIndex, &rasterPass, &cmdCtx, clearValues)
		if err != nil {
			log.Println("BeginRenderPass:", err)
			break
		}
		cmdCtx.BindRasterPipeline(cmdBuffer, pipeline)
		cmdCtx.BindVertexBuffers(cmdBuffer, 0, []vk.Buffer{buffer.Buffer}, []vk.DeviceSize{0})
		cmdCtx.Draw(cmdBuffer, 3, 1, 0, 0)
		if err := swapchainCtx.EndRenderPass(cmdBuffer); err != nil {
			log.Println("EndRenderPass:", err)
			break
		}

		if err := swapchainCtx.SubmitRender(cmdBuffer, sync.Fence, []vk.Semaphore{sync.Semaphore}); err != nil {
			log.Println("SubmitRender:", err)
			break
		}
		if err := swapchainCtx.PresentImageRasterization(windowSize, &rasterPass, &cmdCtx, &pipeline, rasterCfg, imageIndex); err != nil {
			log.Println("PresentImage:", err)
			break
		}
	}
}

func waitForFramebufferSize(window *glfw.Window) vk.Extent2D {
	width, height := window.GetFramebufferSize()
	for width == 0 || height == 0 {
		glfw.WaitEvents()
		width, height = window.GetFramebufferSize()
	}
	return ash.NewExtentSize(width, height)
}
