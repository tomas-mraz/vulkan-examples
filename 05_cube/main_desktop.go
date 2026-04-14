//go:build !android

package main

import (
	"log"
	"runtime"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/tomas-mraz/vulkan"
	"github.com/tomas-mraz/vulkan-ash"
)

const (
	windowWidth  = 500
	windowHeight = 500
)

func init() {
	runtime.LockOSThread()
}

var pollEventsWindow *glfw.Window

func pollEvents() bool {
	glfw.PollEvents()
	return !pollEventsWindow.ShouldClose()
}

func start() {
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
	pollEventsWindow = window

	createSurfaceFn := func(instance vk.Instance) (vk.Surface, error) {
		return ash.NewDesktopSurface(instance, window)
	}

	manager, err := ash.NewManager(appName, createSurfaceFn, &ash.DeviceOptions{
		InstanceExtensions: window.GetRequiredInstanceExtensions(),
	})
	if err != nil {
		log.Fatal(err)
	}

	var cleanup ash.Cleanup
	defer cleanup.Destroy()
	cleanup.Add(&manager)

	swapchain, rasterPass, cmdCtx, _, uniforms, desc, gfx, syncObj :=
		initVulkanResources(&manager, &cleanup, windowWidth, windowHeight)

	ctx := ash.NewSwapchainContext(&manager, &swapchain)

	renderLoop(&ctx, &cmdCtx, rasterPass, gfx, desc, &uniforms, syncObj)
}
