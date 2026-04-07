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
	ash "github.com/tomas-mraz/vulkan-ash"
)

//go:embed shaders/raygen.rgen.spv
var raygenShaderCode []byte

//go:embed shaders/miss.rmiss.spv
var missShaderCode []byte

//go:embed shaders/shadow.rmiss.spv
var shadowMissShaderCode []byte

//go:embed shaders/closesthit.rchit.spv
var closestHitShaderCode []byte

//go:embed shaders/anyhit.rahit.spv
var anyHitShaderCode []byte

const (
	windowWidth  = 800
	windowHeight = 600
	appName      = "Ray Tracing glTF"
)

var frameCounter uint32

type uniformData struct {
	ViewInverse ash.Mat4x4
	ProjInverse ash.Mat4x4
	Frame       uint32
	Pad         [3]uint32
	LightPos    [4]float32
}

const uniformSize = int(unsafe.Sizeof(uniformData{}))

func (u *uniformData) Bytes() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(u)), uniformSize)
}

func init() {
	runtime.LockOSThread()
}

func main() {
	ash.StartPrintGCPauses(10 * time.Second)

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

	// Create device with ray tracing extensions
	ash.SetDebug(false)
	instanceExtensions := window.GetRequiredInstanceExtensions()

	// Enable required features via pNext chain
	bufferDeviceAddressFeatures := vk.PhysicalDeviceBufferDeviceAddressFeatures{
		SType:               vk.StructureTypePhysicalDeviceBufferDeviceAddressFeatures,
		BufferDeviceAddress: vk.True,
	}
	rtPipelineFeatures := vk.PhysicalDeviceRayTracingPipelineFeatures{
		SType:              vk.StructureTypePhysicalDeviceRayTracingPipelineFeatures,
		RayTracingPipeline: vk.True,
		PNext:              unsafe.Pointer(&bufferDeviceAddressFeatures),
	}
	asFeatures := vk.PhysicalDeviceAccelerationStructureFeatures{
		SType:                 vk.StructureTypePhysicalDeviceAccelerationStructureFeatures,
		AccelerationStructure: vk.True,
		PNext:                 unsafe.Pointer(&rtPipelineFeatures),
	}
	descriptorIndexingFeatures := vk.PhysicalDeviceDescriptorIndexingFeatures{
		SType: vk.StructureTypePhysicalDeviceDescriptorIndexingFeatures,
		ShaderSampledImageArrayNonUniformIndexing: vk.True,
		DescriptorBindingVariableDescriptorCount:  vk.True,
		RuntimeDescriptorArray:                    vk.True,
		PNext:                                     unsafe.Pointer(&asFeatures),
	}
	enabledFeatures := vk.PhysicalDeviceFeatures{
		ShaderInt64:                          vk.True,
		ShaderStorageImageReadWithoutFormat:  vk.True,
		ShaderStorageImageWriteWithoutFormat: vk.True,
	}

	newSurfaceFn := func(instance vk.Instance) (vk.Surface, error) {
		return ash.NewDesktopSurface(instance, window)
	}

	deviceOptions := &ash.DeviceOptions{
		DeviceExtensions: ash.RaytracingExtensions(),
		PNextChain:       unsafe.Pointer(&descriptorIndexingFeatures),
		EnabledFeatures:  &enabledFeatures,
		ApiVersion:       vk.MakeVersion(1, 2, 0),
	}

	manager, err := ash.NewManager(appName, instanceExtensions, newSurfaceFn, deviceOptions)
	if err != nil {
		log.Fatal(err)
	}
	cleanup := ash.NewCleanup(&manager)
	defer cleanup.Destroy()
	cleanup.Add(&manager)
	dev := manager.Device
	gpu := manager.Gpu
	queue := manager.Queue

	// Check Vulkan 1.2 and HW ray tracing support
	requiredVersion := vk.MakeVersion(1, 2, 0)
	if ok, ver := ash.CheckDeviceApiVersion(gpu, requiredVersion); !ok {
		log.Fatalf("GPU supports Vulkan %s, but %s is required", vk.Version(ver), vk.Version(requiredVersion))
	}
	if ok, missing := ash.CheckDeviceExtensions(gpu, ash.RaytracingExtensions()); !ok {
		log.Fatalf("GPU does not support HW accelerated ray tracing, missing extensions: %v", missing)
	}

	// Query RT pipeline properties (use hardcoded defaults, standard on all GPUs)
	const shaderGroupHandleSize = 32
	const shaderGroupHandleAlignment = 32

	// Create swapchain
	windowSize := ash.NewExtentSize(windowWidth, windowHeight)
	swapchain, err := ash.NewSwapchain(dev, gpu, manager.Surface, windowSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&swapchain)
	swapchainLen := swapchain.DefaultSwapchainLen()

	cmdCtx, err := ash.NewCommandContext(dev, 0, swapchainLen)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&cmdCtx)

	rtContext := ash.NewRaytracingContext(dev, gpu, queue, &cmdCtx)
	cleanup.Add(&rtContext)

	// --- Load glTF model ---
	model, err := ash.LoadGLTFModel(&rtContext, "assets/FlightHelmet/FlightHelmet.gltf")
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Loaded %d primitives into one BLAS", len(model.Primitives))

	// --- Build TLAS with one instance for the model BLAS ---
	tlas, err := rtContext.NewTopLevelAccelerationStructure([]ash.TLASInstance{{
		Transform:           [12]float32{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0},
		InstanceCustomIndex: 0,
		Mask:                0xFF,
		SBTRecordOffset:     0,
		Flags:               vk.GeometryInstanceFlags(vk.GeometryInstanceTriangleFacingCullDisableBit),
		BLAS:                &model.BLAS,
	}}, vk.BuildAccelerationStructureFlags(vk.BuildAccelerationStructurePreferFastTraceBit))
	if err != nil {
		log.Fatal("NewTopLevelAccelerationStructure:", err)
	}

	// --- Create storage image ---
	storageImg, err := ash.NewImageStorage(dev, gpu, queue, cmdCtx.GetCmdPool(), windowWidth, windowHeight, swapchain.DisplayFormat)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&storageImg)

	// --- Uniform buffers ---
	uniforms, err := ash.NewUniformBuffers(dev, gpu, swapchainLen, uniformSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&uniforms)

	// --- Descriptors ---
	immutableFallbackSampler := []vk.Sampler{model.Textures[0].GetSampler()}
	immutableTextureSamplers := make([]vk.Sampler, len(model.Textures))
	textureInfos := make([]vk.DescriptorImageInfo, len(model.Textures))
	for i, tex := range model.Textures {
		immutableTextureSamplers[i] = tex.GetSampler()
		textureInfos[i] = tex.SampledDescriptorInfo()
	}
	rayHitStages := vk.ShaderStageFlags(vk.ShaderStageClosestHitBit | vk.ShaderStageAnyHitBit)
	desc, err := ash.NewDescriptorSets(dev, swapchainLen, []ash.DescriptorBinding{
		&ash.BindingAccelerationStructure{StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit | vk.ShaderStageClosestHitBit), AccelerationStructure: tlas.AccelerationStructure},
		ash.NewBindingStorageImage(vk.ShaderStageFlags(vk.ShaderStageRaygenBit), &storageImg),
		&ash.BindingUniformBuffer{StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit | vk.ShaderStageClosestHitBit | vk.ShaderStageMissBit), Uniforms: &uniforms},
		ash.NewBindingImageSampler(rayHitStages, &model.Textures[0], immutableFallbackSampler),
		&ash.BindingStorageBuffer{StageFlags: rayHitStages, Buffer: model.GeometryBuffer.Buffer},
		&ash.BindingImageSamplerArray{StageFlags: rayHitStages, ImageInfos: textureInfos, ImmutableSamplers: immutableTextureSamplers},
	})
	if err != nil {
		log.Fatal("NewDescriptorSets:", err)
	}
	cleanup.Add(&desc)
	// --- RT Pipeline ---
	rtPipeline, err := ash.NewRTPipeline(dev, ash.RTPipelineOptions{
		Groups: []ash.RTShaderGroup{
			{RaygenShader: raygenShaderCode},
			{MissShader: missShaderCode},
			{MissShader: shadowMissShaderCode},
			{ClosestHitShader: closestHitShaderCode, AnyHitShader: anyHitShaderCode},
		},
		DescriptorSetLayouts: []vk.DescriptorSetLayout{desc.GetLayout()},
	})
	if err != nil {
		log.Fatal("NewRTPipeline:", err)
	}
	cleanup.Add(&rtPipeline)
	// --- Shader Binding Table ---
	sbt, err := ash.NewShaderBindingTable(dev, gpu, rtPipeline.GetPipeline(), shaderGroupHandleSize, shaderGroupHandleAlignment, 1, 2, 1, 0)
	if err != nil {
		log.Fatal("NewShaderBindingTable:", err)
	}
	cleanup.Add(&sbt)

	// --- Sync objects ---
	sync, err := ash.NewSyncObjects(dev)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&sync)

	// Match Sascha Willems raytracinggltf camera setup.
	var projMatrix, viewMatrix ash.Mat4x4
	setPerspectiveZO(&projMatrix, ash.DegreesToRadians(60.0), float32(windowWidth)/float32(windowHeight), 0.1, 512.0)
	viewMatrix.Translate(0.0, -0.1, -1.0)

	log.Println("Ray tracing initialized, starting render loop")
	startTime := time.Now()

	for !window.ShouldClose() {
		glfw.PollEvents()

		// Red light orbiting above the model
		elapsed := float32(time.Since(startTime).Seconds())
		lightAngle := elapsed * 0.672 // radians per second
		lightRadius := float32(1.0)
		lightX := lightRadius * float32(math.Cos(float64(lightAngle)))
		lightZ := lightRadius * float32(math.Sin(float64(lightAngle)))
		lightPos := [4]float32{lightX, 0.15, lightZ, 0.0}
		frameCounter = 0 // reset accumulation — light moved

		if !drawFrame(dev, queue, swapchain, &cmdCtx, sync.Fence, sync.Semaphore,
			&rtPipeline, desc.GetSets(), &uniforms,
			storageImg.GetImage(), &sbt,
			&projMatrix, &viewMatrix, lightPos) {
			break
		}
	}

}
func setPerspectiveZO(m *ash.Mat4x4, yFov, aspect, near, far float32) {
	f := float32(1.0 / math.Tan(float64(yFov)/2.0))

	m[0][0] = f / aspect
	m[0][1] = 0
	m[0][2] = 0
	m[0][3] = 0

	m[1][0] = 0
	m[1][1] = f
	m[1][2] = 0
	m[1][3] = 0

	m[2][0] = 0
	m[2][1] = 0
	m[2][2] = far / (near - far)
	m[2][3] = -1

	m[3][0] = 0
	m[3][1] = 0
	m[3][2] = (near * far) / (near - far)
	m[3][3] = 0
}

// --- Helper functions ---

func drawFrame(dev vk.Device, queue vk.Queue, s ash.Display, cmdCtx *ash.CommandContext,
	fence vk.Fence, semaphore vk.Semaphore,
	rtPipeline *ash.PipelineRaytracing,
	descSets []vk.DescriptorSet, uniforms *ash.VulkanUniformBuffers,
	storageImage vk.Image,
	sbt *ash.ShaderBindingTable,
	proj, view *ash.Mat4x4, lightPos [4]float32,
) bool {
	var nextIdx uint32
	cmdBuffers := cmdCtx.GetCmdBuffers()
	ret := vk.AcquireNextImage(dev, s.DefaultSwapchain(), vk.MaxUint64, semaphore, vk.NullFence, &nextIdx)
	if ret != vk.Success && ret != vk.Suboptimal {
		return false
	}

	// Update uniform buffer with inverse matrices
	projInv := ash.InvertMatrix(proj)
	viewInv := ash.InvertMatrix(view)
	ubo := uniformData{ViewInverse: viewInv, ProjInverse: projInv, Frame: frameCounter, LightPos: lightPos}
	uniforms.Update(nextIdx, ubo.Bytes())
	frameCounter++

	cmd := cmdBuffers[nextIdx]
	vk.ResetCommandBuffer(cmd, 0)
	vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo})

	// Bind RT pipeline and descriptors
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointRayTracing, rtPipeline.GetPipeline())
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointRayTracing, rtPipeline.GetLayout(), 0, 1, []vk.DescriptorSet{descSets[nextIdx]}, 0, nil)

	// Trace rays
	vk.CmdTraceRays(cmd, &sbt.Raygen, &sbt.Miss, &sbt.Hit, &sbt.Callable, windowWidth, windowHeight, 1)

	// Copy storage image to swapchain
	s.CmdCopyToSwapchain(cmd, storageImage, nextIdx)

	vk.EndCommandBuffer(cmd)

	vk.ResetFences(dev, 1, []vk.Fence{fence})
	if err := vk.Error(vk.QueueSubmit(queue, 1, []vk.SubmitInfo{{
		SType: vk.StructureTypeSubmitInfo, WaitSemaphoreCount: 1, PWaitSemaphores: []vk.Semaphore{semaphore},
		PWaitDstStageMask:  []vk.PipelineStageFlags{vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit)},
		CommandBufferCount: 1, PCommandBuffers: cmdBuffers[nextIdx:],
	}}, fence)); err != nil {
		log.Println("QueueSubmit:", err)
		return false
	}
	if err := vk.Error(vk.WaitForFences(dev, 1, []vk.Fence{fence}, vk.True, 10_000_000_000)); err != nil {
		log.Println("WaitForFences:", err)
		return false
	}
	ret = vk.QueuePresent(queue, &vk.PresentInfo{
		SType: vk.StructureTypePresentInfo, SwapchainCount: 1, PSwapchains: s.Swapchains, PImageIndices: []uint32{nextIdx},
	})
	return ret == vk.Success || ret == vk.Suboptimal
}
