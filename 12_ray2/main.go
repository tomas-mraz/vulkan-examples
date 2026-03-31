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
}

const uniformSize = int(unsafe.Sizeof(uniformData{}))

func (u *uniformData) Bytes() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(u)), uniformSize)
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

	var cleanup ash.Cleanup
	defer cleanup.Destroy()

	// Create device with ray tracing extensions
	ash.SetDebug(false)
	extensions := window.GetRequiredInstanceExtensions()

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

	device, err := ash.NewDeviceWithOptions(appName, extensions, func(instance vk.Instance, _ uintptr) (vk.Surface, error) {
		surfPtr, err := window.CreateWindowSurface(instance, nil)
		if err != nil {
			return vk.NullSurface, err
		}
		return vk.SurfaceFromPointer(surfPtr), nil
	}, 0, &ash.DeviceOptions{
		DeviceExtensions: ash.RaytracingExtensions(),
		PNextChain:       unsafe.Pointer(&descriptorIndexingFeatures),
		EnabledFeatures:  &enabledFeatures,
		ApiVersion:       vk.MakeVersion(1, 2, 0),
	})
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&device)
	dev := device.Device
	gpu := device.GpuDevice
	queue := device.Queue

	// Query RT pipeline properties (use hardcoded defaults, standard on all GPUs)
	const shaderGroupHandleSize = 32
	const shaderGroupHandleAlignment = 32

	// Create swapchain
	windowSize := ash.NewExtentSize(windowWidth, windowHeight)
	swapchain, err := ash.NewSwapchain(dev, gpu, device.Surface, windowSize)
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

	// --- Load glTF model ---
	model, err := ash.LoadGLTFModel(dev, gpu, queue, &cmdCtx, "assets/FlightHelmet/FlightHelmet.gltf")
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Loaded %d primitives into one BLAS", len(model.Primitives))
	cleanup.Add(&model)

	// --- Build TLAS with one instance for the model BLAS ---
	tlas := buildTLAS(dev, gpu, queue, &cmdCtx, model.BLAS)
	cleanup.Add(&tlas)

	// --- Create storage image ---
	storageImg, err := ash.NewImageStorage(dev, gpu, queue, cmdCtx.GetCmdPool(), windowWidth, windowHeight, swapchain.DisplayFormat)
	if err != nil {
		log.Fatal("NewImageStorage:", err)
	}
	cleanup.Add(&storageImg)

	// --- Uniform buffers ---
	uniforms, err := ash.NewUniformBuffers(dev, gpu, swapchainLen, uniformSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&uniforms)

	// --- Descriptors ---
	// Build texture descriptor data
	immutableFallbackSampler := []vk.Sampler{model.Textures[0].Sampler}
	immutableTextureSamplers := make([]vk.Sampler, len(model.Textures))
	textureInfos := make([]vk.DescriptorImageInfo, len(model.Textures))
	for i, tex := range model.Textures {
		immutableTextureSamplers[i] = tex.Sampler
		textureInfos[i] = vk.DescriptorImageInfo{Sampler: tex.Sampler, ImageView: tex.View, ImageLayout: vk.ImageLayoutShaderReadOnlyOptimal}
	}
	rayHitStages := vk.ShaderStageFlags(vk.ShaderStageClosestHitBit | vk.ShaderStageAnyHitBit)
	desc, err := ash.NewDescriptorSets(dev, swapchainLen, []ash.DescriptorBinding{
		&ash.BindingAccelerationStructure{StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit | vk.ShaderStageClosestHitBit), AccelerationStructure: tlas.AccelerationStructure},
		&ash.BindingStorageImage{StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit), ImageView: storageImg.GetView()},
		&ash.BindingUniformBuffer{StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit | vk.ShaderStageClosestHitBit | vk.ShaderStageMissBit), Uniforms: &uniforms},
		&ash.BindingImageSampler{StageFlags: rayHitStages, ImageView: textureInfos[0].ImageView, Sampler: textureInfos[0].Sampler, ImmutableSamplers: immutableFallbackSampler},
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

		// Rotate the view around Y axis (like 06_model)
		elapsed := float32(time.Since(startTime).Seconds()) * 45.0
		var rotatedView ash.Mat4x4
		rotatedView.Rotate(&viewMatrix, 0.0, 1.0, 0.0, ash.DegreesToRadians(elapsed))
		frameCounter = 0 // reset accumulation — camera changed

		if !drawFrame(dev, queue, swapchain, &cmdCtx, sync.Fence, sync.Semaphore,
			&rtPipeline, desc.GetSets(), &uniforms,
			storageImg.GetImage(), &sbt,
			&projMatrix, &rotatedView) {
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

func createBufferWithAddress[T any](dev vk.Device, gpu vk.PhysicalDevice, usage vk.BufferUsageFlags, data []T) ash.VulkanBufferResource {
	buf, err := ash.NewBufferHostVisible(dev, gpu, data, true, usage)
	if err != nil {
		log.Fatal("NewBufferHostVisible:", err)
	}
	return buf
}

func createDeviceLocalBuffer(dev vk.Device, gpu vk.PhysicalDevice, usage vk.BufferUsageFlags, size uint64) ash.VulkanBufferResource {
	buf, err := ash.NewBufferDeviceLocal(dev, gpu, size, true, usage)
	if err != nil {
		log.Fatal("NewBufferDeviceLocal:", err)
	}
	return buf
}

// setDeviceAddressConst writes a DeviceAddress into a DeviceOrHostAddressConst byte array
func setDeviceAddressConst(addr *vk.DeviceOrHostAddressConst, da vk.DeviceAddress) {
	*(*vk.DeviceAddress)(unsafe.Pointer(&addr[0])) = da
}

// setDeviceAddress writes a DeviceAddress into a DeviceOrHostAddress byte array
func setDeviceAddress(addr *vk.DeviceOrHostAddress, da vk.DeviceAddress) {
	*(*vk.DeviceAddress)(unsafe.Pointer(&addr[0])) = da
}

// setGeometryInstances writes AccelerationStructureGeometryInstancesData into the union
func setGeometryInstances(data *vk.AccelerationStructureGeometryData, inst *vk.AccelerationStructureGeometryInstancesData) {
	cInst, _ := inst.PassRef()
	src := unsafe.Slice((*byte)(unsafe.Pointer(cInst)), len(*data))
	copy((*data)[:], src)
}

// buildTLAS creates a TLAS with one instance that references the model BLAS.
func buildTLAS(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdCtx *ash.CommandContext, blas ash.AccelerationStructure) ash.AccelerationStructure {
	instanceData := make([]byte, 64)
	transform := [12]float32{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0}
	blasAddr := blas.GetDeviceAddress()

	copy(instanceData[:48], unsafe.Slice((*byte)(unsafe.Pointer(&transform[0])), 48))
	instanceData[48] = 0
	instanceData[49] = 0
	instanceData[50] = 0
	instanceData[51] = 0xFF
	instanceData[52] = 0
	instanceData[53] = 0
	instanceData[54] = 0
	instanceData[55] = 0x01 // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR
	*(*uint64)(unsafe.Pointer(&instanceData[56])) = uint64(blasAddr)

	instanceBuf := createBufferWithAddress(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageShaderDeviceAddressBit|vk.BufferUsageAccelerationStructureBuildInputReadOnlyBit), instanceData)
	instanceAddr := instanceBuf.DeviceAddress

	var instancesData vk.AccelerationStructureGeometryInstancesData
	instancesData.SType = vk.StructureTypeAccelerationStructureGeometryInstancesData
	setDeviceAddressConst(&instancesData.Data, instanceAddr)

	var geometry vk.AccelerationStructureGeometry
	geometry.SType = vk.StructureTypeAccelerationStructureGeometry
	geometry.GeometryType = vk.GeometryTypeInstances
	geometry.Flags = vk.GeometryFlags(vk.GeometryOpaqueBit)
	setGeometryInstances(&geometry.Geometry, &instancesData)

	buildInfo := vk.AccelerationStructureBuildGeometryInfo{
		SType:         vk.StructureTypeAccelerationStructureBuildGeometryInfo,
		Type:          vk.AccelerationStructureTypeTopLevel,
		Flags:         vk.BuildAccelerationStructureFlags(vk.BuildAccelerationStructurePreferFastTraceBit),
		GeometryCount: 1,
		PGeometries:   []vk.AccelerationStructureGeometry{geometry},
	}

	primitiveCount := uint32(1)
	var sizeInfo vk.AccelerationStructureBuildSizesInfo
	sizeInfo.SType = vk.StructureTypeAccelerationStructureBuildSizesInfo
	vk.GetAccelerationStructureBuildSizes(dev, vk.AccelerationStructureBuildTypeDevice, &buildInfo, &primitiveCount, &sizeInfo)
	sizeInfo.Deref()
	log.Printf("TLAS size: AS=%d, scratch=%d (instances=1)", sizeInfo.AccelerationStructureSize, sizeInfo.BuildScratchSize)

	asBuf := createDeviceLocalBuffer(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageAccelerationStructureStorageBit|vk.BufferUsageShaderDeviceAddressBit), uint64(sizeInfo.AccelerationStructureSize))

	var as vk.AccelerationStructure
	if err := vk.Error(vk.CreateAccelerationStructure(dev, &vk.AccelerationStructureCreateInfo{
		SType: vk.StructureTypeAccelerationStructureCreateInfo, Buffer: asBuf.Buffer,
		Size: sizeInfo.AccelerationStructureSize, Type: vk.AccelerationStructureTypeTopLevel,
	}, nil, &as)); err != nil {
		log.Fatal("CreateAccelerationStructure (TLAS):", err)
	}

	scratchBuf := createDeviceLocalBuffer(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageStorageBufferBit|vk.BufferUsageShaderDeviceAddressBit),
		uint64(sizeInfo.BuildScratchSize))
	scratchAddr := scratchBuf.DeviceAddress

	buildInfo2 := vk.AccelerationStructureBuildGeometryInfo{
		SType:                    vk.StructureTypeAccelerationStructureBuildGeometryInfo,
		Type:                     vk.AccelerationStructureTypeTopLevel,
		Flags:                    vk.BuildAccelerationStructureFlags(vk.BuildAccelerationStructurePreferFastTraceBit),
		Mode:                     vk.BuildAccelerationStructureModeBuild,
		DstAccelerationStructure: as,
		GeometryCount:            1,
		PGeometries:              []vk.AccelerationStructureGeometry{geometry},
	}
	setDeviceAddress(&buildInfo2.ScratchData, scratchAddr)

	rangeInfos := []vk.AccelerationStructureBuildRangeInfo{{PrimitiveCount: primitiveCount}}

	cmd, err := cmdCtx.BeginOneTime()
	if err != nil {
		log.Fatal("BeginOneTime:", err)
	}
	vk.CmdBuildAccelerationStructures(cmd, 1, &buildInfo2, [][]vk.AccelerationStructureBuildRangeInfo{rangeInfos})
	if err := cmdCtx.EndOneTime(queue, cmd); err != nil {
		log.Fatal("EndOneTime:", err)
	}

	scratchBuf.Destroy()
	instanceBuf.Destroy()
	return ash.AccelerationStructure{
		AccelerationStructure: as,
		Buffer:                asBuf,
		Type:                  vk.AccelerationStructureTypeTopLevel,
	}
}

func alignUp(size, alignment uint32) uint32 {
	return (size + alignment - 1) &^ (alignment - 1)
}

func drawFrame(dev vk.Device, queue vk.Queue, s ash.VulkanSwapchainInfo, cmdCtx *ash.CommandContext,
	fence vk.Fence, semaphore vk.Semaphore,
	rtPipeline *ash.PipelineRaytracing,
	descSets []vk.DescriptorSet, uniforms *ash.VulkanUniformBuffers,
	storageImage vk.Image,
	sbt *ash.ShaderBindingTable,
	proj, view *ash.Mat4x4,
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
	ubo := uniformData{ViewInverse: viewInv, ProjInverse: projInv, Frame: frameCounter}
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
