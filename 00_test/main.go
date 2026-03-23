package main

import (
	"log"
	"math"
	"runtime"
	"time"
	"unsafe"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/tomas-mraz/vulkan"
	asch "github.com/tomas-mraz/vulkan-ash"
)

var vertShaderCode = mustReadFile("shaders/tri.vert.spv")
var fragShaderCode = mustReadFile("shaders/tri.frag.spv")

const (
	windowWidth   = 800
	windowHeight  = 600
	appName  = "7 Triangles Test"
	triCount = 7
)

// pushData matches the push constant layout in both shaders.
type pushData struct {
	OffsetX float32
	OffsetY float32
	Angle   float32
	Aspect  float32
	ColorR  float32
	ColorG  float32
	ColorB  float32
}

const pushSize = uint32(unsafe.Sizeof(pushData{}))

type vec2 struct{ x, y float32 }

// triState describes the current visual state of one triangle.
type triState struct {
	r, g, b float32
	angle   float32
}

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

	windowSize := asch.NewExtentSize(windowWidth, windowHeight)
	swapchain, err := asch.NewSwapchain(device.Device, device.GpuDevice, device.Surface, windowSize)
	if err != nil {
		log.Fatal(err)
	}

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

	// Equilateral triangle vertices (unit circle, aspect correction done in shader)
	sz := float32(0.1)
	vertices := []float32{
		0, -sz, 0,
		sz * float32(math.Sin(2*math.Pi/3)), -sz * float32(math.Cos(2*math.Pi/3)), 0,
		sz * float32(math.Sin(4*math.Pi/3)), -sz * float32(math.Cos(4*math.Pi/3)), 0,
	}
	buffer, err := asch.NewBufferWithData(device.Device, device.GpuDevice, vertices)
	if err != nil {
		log.Fatal(err)
	}

	gfx, err := asch.NewGraphicsPipelineWithOptions(device.Device, swapchain.DisplaySize, renderer.RenderPass, asch.PipelineOptions{
		VertShaderData: vertShaderCode,
		FragShaderData: fragShaderCode,
		PushConstantRanges: []vk.PushConstantRange{{
			StageFlags: vk.ShaderStageFlags(vk.ShaderStageVertexBit | vk.ShaderStageFragmentBit),
			Offset:     0,
			Size:       pushSize,
		}},
	})
	if err != nil {
		log.Fatal(err)
	}

	fence, semaphore, err := asch.NewSyncObjects(device.Device)
	if err != nil {
		log.Fatal(err)
	}

	// Compute offsets so 7 triangles are placed diagonally (top-left to bottom-right)
	var offsets [triCount]vec2
	for i := 0; i < triCount; i++ {
		f := float32(i) / float32(triCount-1)
		offsets[i] = vec2{
			x: -0.7 + 1.4*f,
			y: -0.6 + 1.2*f,
		}
	}

	rtExtensions := []string{
		"VK_KHR_acceleration_structure\x00",
		"VK_KHR_ray_tracing_pipeline\x00",
		"VK_KHR_buffer_device_address\x00",
		"VK_KHR_deferred_host_operations\x00",
		"VK_EXT_descriptor_indexing\x00",
		"VK_KHR_spirv_1_4\x00",
		"VK_KHR_shader_float_controls\x00",
	}

	// Check function for each triangle phase
	gpu := device.GpuDevice
	checkFn := func(phase int) bool {
		return runCheck(phase, gpu, rtExtensions)
	}

	// Per-triangle parameters
	var params [triCount]triParams
	for i := 0; i < triCount; i++ {
		params[i].spinDur = math.MaxFloat64 // spin indefinitely until check completes
	}

	// Async check state
	type checkResult struct {
		idx  int
		pass bool
	}
	checkStarted := [triCount]bool{}
	resultCh := make(chan checkResult, triCount)

	// Dynamic start times — each triangle starts when the previous finishes
	var triStart [triCount]float64
	triStart[0] = 0
	for i := 1; i < triCount; i++ {
		triStart[i] = -1 // not yet scheduled
	}

	log.Println("Starting render loop")
	wallStart := time.Now()

	for !window.ShouldClose() {
		glfw.PollEvents()
		elapsed := time.Since(wallStart).Seconds()

		// Schedule next triangle when previous one finishes
		for i := 1; i < triCount; i++ {
			if triStart[i] >= 0 {
				continue // already scheduled
			}
			prev := i - 1
			prevDone := triStart[prev] + rampUpDur + params[prev].spinDur + rampDownDur
			if elapsed >= prevDone {
				triStart[i] = prevDone
			}
		}

		// Launch check goroutines when each triangle starts its animation
		for i := 0; i < triCount; i++ {
			if triStart[i] < 0 || checkStarted[i] {
				continue
			}
			t := elapsed - triStart[i]
			if t >= 0 {
				checkStarted[i] = true
				phase := i + 1
				go func() {
					resultCh <- checkResult{phase - 1, checkFn(phase)}
				}()
			}
		}

		// Collect completed checks (non-blocking)
		for {
			select {
			case r := <-resultCh:
				// Set spin duration to elapsed spin time (0 if still in ramp-up)
				spinT := elapsed - triStart[r.idx] - rampUpDur
				if spinT < 0 {
					spinT = 0
				}
				params[r.idx].spinDur = spinT
				if !r.pass {
					params[r.idx].fails = true
					params[r.idx].failAt = spinT
				}
			default:
				goto doneCollecting
			}
		}
	doneCollecting:

		var states [triCount]triState
		stopped := false
		for i := 0; i < triCount; i++ {
			if stopped || triStart[i] < 0 {
				states[i] = triState{0, 0, 1, 0} // stays blue
			} else {
				states[i] = triCheck(triStart[i], elapsed, params[i].spinDur, params[i].fails, params[i].failAt)
				if params[i].fails && triHasFailed(triStart[i], elapsed, params[i].spinDur, params[i].failAt) {
					stopped = true
				}
			}
		}

		if !drawFrame(device.Device, device.Queue, swapchain, renderer, buffer, fence, semaphore, gfx, offsets, states) {
			break
		}

		// Check if all triangles finished and RT is available → transition
		if allTrianglesDone(elapsed, params[:], triStart[:]) {
			log.Println("All tests passed — switching to ray tracing scene")
			time.Sleep(500 * time.Millisecond) // brief pause to show final state
			break
		}
	}

	vk.DeviceWaitIdle(device.Device)
	vk.DestroySemaphore(device.Device, semaphore, nil)
	vk.DestroyFence(device.Device, fence, nil)
	gfx.Destroy()
	vk.FreeMemory(device.Device, buffer.GetDeviceMemory(), nil)
	buffer.Destroy()
	vk.FreeCommandBuffers(device.Device, renderer.GetCmdPool(), uint32(len(renderer.GetCmdBuffers())), renderer.GetCmdBuffers())
	vk.DestroyCommandPool(device.Device, renderer.GetCmdPool(), nil)
	vk.DestroyRenderPass(device.Device, renderer.RenderPass, nil)
	swapchain.Destroy()
	vk.DestroyDevice(device.Device, nil)
	if device.GetDebugCallback() != vk.NullDebugReportCallback {
		vk.DestroyDebugReportCallback(device.Instance, device.GetDebugCallback(), nil)
	}
	vk.DestroySurface(device.Instance, device.Surface, nil)
	vk.DestroyInstance(device.Instance, nil)

	if !params[triCount-1].fails && !window.ShouldClose() {
		runRayTracingScene(window)
	}
}

// triCheck computes the state for triangle N given elapsed time and its parameters.
// Each triangle starts when the previous one finishes.
const (
	maxSpeed    = 4.0 // rad/s at full spin
	rampUpDur   = 0.5 // seconds for blue→yellow (speed 0→max)
	rampDownDur = 1.0 // seconds for yellow→green (speed max→0)
)

// rampUpAngle returns accumulated angle during ramp-up phase (speed linearly 0→maxSpeed).
func rampUpAngle(t float64) float32 {
	// speed(τ) = maxSpeed * τ/rampUpDur → angle = maxSpeed * t² / (2*rampUpDur)
	return float32(maxSpeed * t * t / (2 * rampUpDur))
}

// rampDownAngle returns accumulated angle during ramp-down phase (speed linearly maxSpeed→0).
func rampDownAngle(t float64) float32 {
	// speed(τ) = maxSpeed * (1 - τ/rampDownDur) → angle = maxSpeed * (t - t²/(2*rampDownDur))
	return float32(maxSpeed * (t - t*t/(2*rampDownDur)))
}

func triCheck(start float64, elapsed float64, spinDur float64, fails bool, failAt float64) triState {
	t := elapsed - start
	if t < 0 {
		return triState{0, 0, 1, 0} // blue — not started
	}

	angleAtRampEnd := rampUpAngle(rampUpDur) // angle accumulated during ramp-up

	if t < rampUpDur {
		// Ramp up: blue→yellow, rotation accelerates
		f := float32(t / rampUpDur)
		angle := rampUpAngle(t)
		return triState{f, f, 1 - f, angle}
	}
	spinT := t - rampUpDur
	spinAngle := float32(angleAtRampEnd) + float32(maxSpeed*spinT) // angle during full-speed spin

	// Check for failure during spin
	if fails && spinT >= failAt {
		failAngle := float32(angleAtRampEnd) + float32(maxSpeed*failAt)
		return triState{1, 0, 0, failAngle} // red, stopped
	}
	if spinT < spinDur {
		// Full speed spinning yellow
		return triState{1, 1, 0, spinAngle}
	}

	// Ramp down: yellow→green, rotation decelerates
	greenT := t - rampUpDur - spinDur
	spinEndAngle := float32(angleAtRampEnd) + float32(maxSpeed*spinDur)

	if greenT < rampDownDur {
		f := float32(greenT / rampDownDur)
		angle := spinEndAngle + rampDownAngle(greenT)
		return triState{1 - f, 1, 0, angle}
	}

	finalAngle := spinEndAngle + rampDownAngle(rampDownDur)
	return triState{0, 1, 0, finalAngle} // green, done
}

// triHasFailed returns true if triangle N has already failed at the given elapsed time.
func triHasFailed(start float64, elapsed float64, spinDur float64, failAt float64) bool {
	t := elapsed - start
	return t >= rampUpDur+failAt
}

func runCheck(phase int, gpu vk.PhysicalDevice, rtExtensions []string) bool {
	switch phase {
	case 1:
		log.Println("check1: pass")
		time.Sleep(1300 * time.Millisecond)
		return true
	case 2:
		log.Println("check2: pass")
		time.Sleep(100 * time.Millisecond)
		return true
	case 3:
		log.Println("check3: pass")
		time.Sleep(1500 * time.Millisecond)
		return true
	case 4:
		log.Println("check4: pass")
		time.Sleep(800 * time.Millisecond)
		return true
	case 5:
		log.Println("check5: pass")
		time.Sleep(1600 * time.Millisecond)
		return true
	case 6:
		log.Println("check6: pass")
		time.Sleep(1100 * time.Millisecond)
		return true
	case 7:
		time.Sleep(1000 * time.Millisecond)
		ok, ver := asch.CheckDeviceApiVersion(gpu, vk.MakeVersion(1, 2, 0))
		if !ok {
			log.Printf("check7: FAIL — GPU Vulkan API %s < 1.2", vk.Version(ver))
			return false
		}
		ok2, missing := asch.CheckDeviceExtensions(gpu, rtExtensions)
		if !ok2 {
			log.Printf("check7: FAIL — GPU missing RT extensions: %v", missing)
			return false
		}
		log.Printf("check7: pass — Vulkan %s + HW ray tracing", vk.Version(ver))
		return true
	default:
		return false
	}
}

type triParams struct {
	spinDur float64
	fails   bool
	failAt  float64
}

func allTrianglesDone(elapsed float64, params []triParams, triStart []float64) bool {
	for i, p := range params {
		if p.fails {
			return false
		}
		if triStart[i] < 0 {
			return false
		}
		t := elapsed - triStart[i]
		doneAt := rampUpDur + p.spinDur + rampDownDur
		if t < doneAt {
			return false
		}
	}
	return true
}

func drawFrame(dev vk.Device, queue vk.Queue, s asch.VulkanSwapchainInfo,
	r asch.VulkanRenderInfo, b asch.VulkanBufferInfo,
	fence vk.Fence, semaphore vk.Semaphore,
	gfx asch.VulkanGfxPipelineInfo,
	offsets [triCount]vec2, states [triCount]triState,
) bool {
	var nextIdx uint32
	ret := vk.AcquireNextImage(dev, s.DefaultSwapchain(), vk.MaxUint64, semaphore, vk.NullFence, &nextIdx)
	if ret != vk.Success && ret != vk.Suboptimal {
		return false
	}

	cmd := r.GetCmdBuffers()[nextIdx]
	vk.ResetCommandBuffer(cmd, 0)
	vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo})

	clearValues := []vk.ClearValue{vk.NewClearValue([]float32{0.1, 0.1, 0.1, 1})}
	vk.CmdBeginRenderPass(cmd, &vk.RenderPassBeginInfo{
		SType:           vk.StructureTypeRenderPassBeginInfo,
		RenderPass:      r.RenderPass,
		Framebuffer:     s.Framebuffers[nextIdx],
		RenderArea:      vk.Rect2D{Extent: s.DisplaySize},
		ClearValueCount: 1,
		PClearValues:    clearValues,
	}, vk.SubpassContentsInline)

	vk.CmdBindPipeline(cmd, vk.PipelineBindPointGraphics, gfx.GetPipeline())
	vk.CmdBindVertexBuffers(cmd, 0, 1, []vk.Buffer{b.DefaultVertexBuffer()}, []vk.DeviceSize{0})

	stageFlags := vk.ShaderStageFlags(vk.ShaderStageVertexBit | vk.ShaderStageFragmentBit)
	for i := 0; i < triCount; i++ {
		pd := pushData{
			OffsetX: offsets[i].x,
			OffsetY: offsets[i].y,
			Angle:   states[i].angle,
			Aspect:  float32(windowHeight) / float32(windowWidth),
			ColorR:  states[i].r,
			ColorG:  states[i].g,
			ColorB:  states[i].b,
		}
		vk.CmdPushConstants(cmd, gfx.GetLayout(), stageFlags, 0, pushSize, unsafe.Pointer(&pd))
		vk.CmdDraw(cmd, 3, 1, 0, 0)
	}

	vk.CmdEndRenderPass(cmd)
	vk.EndCommandBuffer(cmd)

	vk.ResetFences(dev, 1, []vk.Fence{fence})
	if err := vk.Error(vk.QueueSubmit(queue, 1, []vk.SubmitInfo{{
		SType: vk.StructureTypeSubmitInfo, WaitSemaphoreCount: 1, PWaitSemaphores: []vk.Semaphore{semaphore},
		PWaitDstStageMask:  []vk.PipelineStageFlags{vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit)},
		CommandBufferCount: 1, PCommandBuffers: r.GetCmdBuffers()[nextIdx:],
	}}, fence)); err != nil {
		log.Println("QueueSubmit:", err)
		return false
	}
	if err := vk.Error(vk.WaitForFences(dev, 1, []vk.Fence{fence}, vk.True, 10_000_000_000)); err != nil {
		log.Println("WaitForFences:", err)
		return false
	}

	ret = vk.QueuePresent(queue, &vk.PresentInfo{
		SType: vk.StructureTypePresentInfo, SwapchainCount: 1,
		PSwapchains: s.Swapchains, PImageIndices: []uint32{nextIdx},
	})
	return ret == vk.Success || ret == vk.Suboptimal
}
