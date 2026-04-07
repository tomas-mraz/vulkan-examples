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
	"github.com/tomas-mraz/vulkan-ash"
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
	glfw.WindowHint(glfw.Resizable, glfw.True)
	window, err := glfw.CreateWindow(windowWidth, windowHeight, appName, nil, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer window.Destroy()

	var cleanup ash.Cleanup
	defer cleanup.Destroy()

	ash.SetDebug(false)
	extensions := window.GetRequiredInstanceExtensions()

	newSurfaceFn := func(instance vk.Instance) (vk.Surface, error) {
		return ash.NewDesktopSurface(instance, window)
	}

	manager, err := ash.NewManager(appName, extensions, newSurfaceFn, nil)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&manager)

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

	r := float32(0.5)
	vertices := []float32{
		0, -r, 0,
		r * float32(math.Sin(2*math.Pi/3)), -r * float32(math.Cos(2*math.Pi/3)), 0,
		r * float32(math.Sin(4*math.Pi/3)), -r * float32(math.Cos(4*math.Pi/3)), 0,
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
		PushConstantRanges: []vk.PushConstantRange{{
			StageFlags: vk.ShaderStageFlags(vk.ShaderStageVertexBit),
			Offset:     0,
			Size:       4,
		}},
	}
	pipeline, err := ash.NewPipelineRasterization(manager.Device, swapchain.DisplaySize, rasterPass.GetRenderPass(), pipelineOptions)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&pipeline)

	sync, err := ash.NewSyncObjects(manager.Device)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&sync)

	rasterCfg := ash.RasterizationRecreateConfig{
		QueueFamilyIndex: 0,
		PipelineOptions:  pipelineOptions,
	}
	clearValues := []vk.ClearValue{vk.NewClearValue([]float32{0, 0, 0, 1})}

	log.Println("Vulkan initialized, starting render loop")
	startTime := time.Now()

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

		cmd, err := swapchainCtx.BeginRenderPass(imageIndex, &rasterPass, &cmdCtx, clearValues)
		if err != nil {
			log.Println("BeginRenderPass:", err)
			break
		}

		angle := float32(time.Since(startTime).Seconds())
		cmdCtx.BindRasterPipeline(cmd, pipeline)
		vk.CmdPushConstants(cmd, pipeline.GetLayout(), vk.ShaderStageFlags(vk.ShaderStageVertexBit), 0, 4, unsafe.Pointer(&angle))
		cmdCtx.BindVertexBuffers(cmd, 0, []vk.Buffer{buffer.Buffer}, []vk.DeviceSize{0})
		cmdCtx.Draw(cmd, 3, 1, 0, 0)

		if err := swapchainCtx.EndRenderPass(cmd); err != nil {
			log.Println("EndRenderPass:", err)
			break
		}
		if err := swapchainCtx.SubmitRender(cmd, sync.Fence, []vk.Semaphore{sync.Semaphore}); err != nil {
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
