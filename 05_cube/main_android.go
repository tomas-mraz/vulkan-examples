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
		a.HandleNativeWindowEvents(nativeWindowEvents)
		a.HandleInputQueueEvents(inputQueueEvents)
		go app.HandleInputQueues(inputQueueChan, func() {
			a.InputQueueHandled()
		}, app.SkipInputEvents)
		a.InitDone()

		var (
			manager        ash.Manager
			managerCleanup ash.Cleanup
			renderCleanup  ash.Cleanup
			window         *android.NativeWindow
			hasManager     bool
			err            error
		)

		stopRender := func() {
			if renderRunning.Load() {
				renderRunning.Store(false)
				if hasManager {
					vk.DeviceWaitIdle(manager.Device)
				}
			}
		}

		destroyRenderResources := func() {
			stopRender()
			renderCleanup.Destroy()
			renderCleanup = ash.NewCleanup()
		}

		destroyManager := func() {
			destroyRenderResources()
			if hasManager {
				managerCleanup.Destroy()
				managerCleanup = ash.NewCleanup()
				manager = ash.Manager{}
				hasManager = false
			}
			window = nil
		}

		startRender := func() {
			if window == nil || !hasManager {
				return
			}

			destroyRenderResources()

			width := uint32(android.NativeWindowGetWidth(window))
			height := uint32(android.NativeWindowGetHeight(window))
			if width == 0 || height == 0 {
				width, height = 640, 480
			}

			swapchain, rasterPass, cmdCtx, _, uniforms, desc, gfx, syncObj := initVulkanResources(&manager, &renderCleanup, width, height)

			// Some Android Vulkan stacks advertise the extension but expose an invalid
			// vkGetRefreshCycleDurationGOOGLE entry point, which crashes inside cgo.
			// Keep frame pacing disabled here until the wrapper can verify proc availability.
			ctx := ash.NewSwapchainContext(&manager, &swapchain)

			renderRunning.Store(true)

			go renderLoop(&ctx, &cmdCtx, rasterPass, gfx, desc, &uniforms, syncObj)
		}

		for {
			select {
			case event := <-a.LifecycleEvents():
				switch event.Kind {
				case app.OnDestroy:
					destroyManager()
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
					destroyManager()

					err = vk.SetDefaultGetInstanceProcAddr()
					if err != nil {
						log.Fatal(err)
					}
					err = vk.Init()
					if err != nil {
						log.Fatal(err)
					}

					window = event.Window
					windowPtr := window.Ptr()

					createSurfaceFn := func(instance vk.Instance) (vk.Surface, error) {
						return ash.NewAndroidSurface(instance, windowPtr)
					}
					manager, err = ash.NewManager(appName, createSurfaceFn, nil)
					if err != nil {
						log.Fatal(err)
					}

					hasManager = true
					managerCleanup = ash.NewCleanup()
					managerCleanup.Add(&manager)
					renderCleanup = ash.NewCleanup()
					log.Println("Vulkan initialized on Android")
					startRender()

				case app.NativeWindowDestroyed:
					destroyManager()

				case app.NativeWindowRedrawNeeded:
					window = event.Window
					startRender()
					a.NativeWindowRedrawDone()
				}
			}
		}
	})
}
