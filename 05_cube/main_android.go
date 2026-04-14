//go:build android

package main

import (
	"log"
	"sync/atomic"

	vk "github.com/tomas-mraz/vulkan"
	"github.com/tomas-mraz/vulkan-ash"

	"github.com/tomas-mraz/android-go/android"
	"github.com/tomas-mraz/android-go/app"
)

var renderRunning atomic.Bool

func pollEvents() bool { return renderRunning.Load() }

func start() {
	nativeWindowEvents := make(chan app.NativeWindowEvent)
	inputQueueEvents := make(chan app.InputQueueEvent, 1)
	inputQueueChan := make(chan *android.InputQueue, 1)

	app.Main(func(a app.NativeActivity) {
		if err := vk.Init(); err != nil {
			log.Fatal(err)
		}

		a.HandleNativeWindowEvents(nativeWindowEvents)
		a.HandleInputQueueEvents(inputQueueEvents)
		go app.HandleInputQueues(inputQueueChan, func() {
			a.InputQueueHandled()
		}, app.SkipInputEvents)
		a.InitDone()

		var (
			manager ash.Manager
			cleanup ash.Cleanup
			window  *android.NativeWindow
		)

		stopRender := func() {
			if renderRunning.Load() {
				renderRunning.Store(false)
				vk.DeviceWaitIdle(manager.Device)
			}
		}

		startRender := func() {
			width := uint32(android.NativeWindowGetWidth(window))
			height := uint32(android.NativeWindowGetHeight(window))
			if width == 0 || height == 0 {
				width, height = 640, 480
			}

			cleanup.Destroy()
			cleanup = ash.NewCleanup(&manager)

			swapchain, rasterPass, cmdCtx, _, uniforms, desc, gfx, syncObj := initVulkanResources(&manager, &cleanup, width, height)

			dt := ash.NewDisplayTiming(manager.Device, swapchain.DefaultSwapchain())
			ctx := ash.NewSwapchainContext(&manager, &swapchain)
			ctx.SetDisplayTiming(&dt)

			renderRunning.Store(true)

			go renderLoop(&ctx, &cmdCtx, rasterPass, gfx, desc, &uniforms, syncObj)
		}

		for {
			select {
			case event := <-a.LifecycleEvents():
				switch event.Kind {
				case app.OnDestroy:
					stopRender()
					cleanup.Destroy()
					return
				}

			case event := <-inputQueueEvents:
				switch event.Kind {
				case app.QueueCreated:
					inputQueueChan <- event.Queue
				case app.QueueDestroyed:
					inputQueueChan <- nil
				}

			case event := <-nativeWindowEvents:
				switch event.Kind {
				case app.NativeWindowCreated:
					var err error
					window = event.Window
					windowPtr := window.Ptr()

					createSurfaceFn := func(instance vk.Instance) (vk.Surface, error) {
						return ash.NewAndroidSurface(instance, windowPtr)
					}
					manager, err = ash.NewManager(appName, createSurfaceFn, nil)
					if err != nil {
						log.Fatal(err)
					}

					cleanup = ash.NewCleanup(&manager)
					log.Println("Vulkan initialized on Android")
					startRender()

				case app.NativeWindowDestroyed:
					stopRender()
					cleanup.Destroy()
					window = nil

				case app.NativeWindowRedrawNeeded:
					stopRender()
					startRender()
					a.NativeWindowRedrawDone()
				}
			}
		}
	})
}
