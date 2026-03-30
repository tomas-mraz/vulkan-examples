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
			asch.DrawFrame(device.Device, device.Queue, swapchain, renderer)
		}
	}
}
