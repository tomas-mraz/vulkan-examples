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

	_ = []string{
		"VK_KHR_acceleration_structure\x00",
		"VK_KHR_ray_tracing_pipeline\x00",
		"VK_KHR_buffer_device_address\x00",
		"VK_KHR_deferred_host_operations\x00",
		"VK_EXT_descriptor_indexing\x00",
		"VK_KHR_spirv_1_4\x00",
		"VK_KHR_shader_float_controls\x00",
	}

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

	ooo := ash.DeviceOptions{
		DeviceExtensions: ash.RaytracingExtensions(),
		PNextChain:       unsafe.Pointer(&descriptorIndexingFeatures),
		EnabledFeatures:  &enabledFeatures,
		ApiVersion:       vk.MakeVersion(1, 2, 0),
	}

	fff := func(instance vk.Instance, _ uintptr) (vk.Surface, error) {
		surfPtr, err := window.CreateWindowSurface(instance, nil)
		if err != nil {
			return vk.NullSurface, err
		}
		return vk.SurfaceFromPointer(surfPtr), nil
	}

	device, err := ash.NewDeviceWithOptions(appName, extensions, fff, 0, &ooo)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&device)
	dev := device.Device
	gpu := device.GpuDevice
	queue := device.Queue

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
	descLayout, descPool, descSets := createDescriptorSets(dev, swapchainLen, tlas.AccelerationStructure, storageImg.GetView(), model.GeometryBuffer.Buffer, model.Textures, &uniforms)
	cleanup.Add(ash.DestroyerFunc(func() {
		vk.DestroyDescriptorPool(dev, descPool, nil)
		vk.DestroyDescriptorSetLayout(dev, descLayout, nil)
	}))
	// --- RT Pipeline ---
	pipelineLayout, pipeline := createRTPipeline(dev, descLayout)
	cleanup.Add(ash.DestroyerFunc(func() {
		vk.DestroyPipeline(dev, pipeline, nil)
		vk.DestroyPipelineLayout(dev, pipelineLayout, nil)
	}))
	// --- Shader Binding Table ---
	sbt, err := ash.NewSBT(dev, gpu, pipeline, shaderGroupHandleSize, shaderGroupHandleAlignment, 1, 2, 1, 0)
	if err != nil {
		log.Fatal("NewSBT:", err)
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
			pipeline, pipelineLayout, descSets, &uniforms,
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
func buildTLAS(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdCtx *ash.VulkanCommandContext, blas ash.VulkanAccelerationStructure) ash.VulkanAccelerationStructure {
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

	instanceBuf, err := ash.NewBufferHostVisible(dev, gpu, instanceData, true, vk.BufferUsageFlags(vk.BufferUsageAccelerationStructureBuildInputReadOnlyBit))
	if err != nil {
		log.Fatal(err)
	}
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

	asBuf, err := ash.NewBufferDeviceLocal(dev, gpu, uint64(sizeInfo.AccelerationStructureSize), true,
		vk.BufferUsageFlags(vk.BufferUsageAccelerationStructureStorageBit))
	if err != nil {
		log.Fatal(err)
	}

	var as vk.AccelerationStructure
	if err := vk.Error(vk.CreateAccelerationStructure(dev, &vk.AccelerationStructureCreateInfo{
		SType: vk.StructureTypeAccelerationStructureCreateInfo, Buffer: asBuf.Buffer,
		Size: sizeInfo.AccelerationStructureSize, Type: vk.AccelerationStructureTypeTopLevel,
	}, nil, &as)); err != nil {
		log.Fatal("CreateAccelerationStructure (TLAS):", err)
	}

	scratchBuf, err := ash.NewBufferDeviceLocal(dev, gpu, uint64(sizeInfo.BuildScratchSize), true,
		vk.BufferUsageFlags(vk.BufferUsageStorageBufferBit))
	if err != nil {
		log.Fatal(err)
	}
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
	return ash.VulkanAccelerationStructure{
		AccelerationStructure: as,
		Buffer:                asBuf,
		Type:                  vk.AccelerationStructureTypeTopLevel,
	}
}

func createDescriptorSets(dev vk.Device, count uint32, tlas vk.AccelerationStructure, storageImageView vk.ImageView, geometryBuf vk.Buffer, textures []ash.VulkanImageResource, uniforms *ash.VulkanUniformBuffers) (vk.DescriptorSetLayout, vk.DescriptorPool, []vk.DescriptorSet) {
	textureCount := uint32(len(textures))
	if textureCount == 0 {
		log.Fatal("createDescriptorSets: texture array must contain at least the fallback texture")
	}

	immutableFallbackSampler := []vk.Sampler{textures[0].GetSampler()}
	immutableTextureSamplers := make([]vk.Sampler, 0, len(textures))
	for _, texture := range textures {
		immutableTextureSamplers = append(immutableTextureSamplers, texture.GetSampler())
	}

	var layout vk.DescriptorSetLayout
	vk.CreateDescriptorSetLayout(dev, &vk.DescriptorSetLayoutCreateInfo{
		SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: 6,
		PBindings: []vk.DescriptorSetLayoutBinding{
			{Binding: 0, DescriptorType: vk.DescriptorTypeAccelerationStructure, DescriptorCount: 1, StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit | vk.ShaderStageClosestHitBit)},
			{Binding: 1, DescriptorType: vk.DescriptorTypeStorageImage, DescriptorCount: 1, StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit)},
			{Binding: 2, DescriptorType: vk.DescriptorTypeUniformBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit | vk.ShaderStageClosestHitBit | vk.ShaderStageMissBit)},
			{Binding: 3, DescriptorType: vk.DescriptorTypeCombinedImageSampler, DescriptorCount: 1, StageFlags: vk.ShaderStageFlags(vk.ShaderStageClosestHitBit | vk.ShaderStageAnyHitBit), PImmutableSamplers: immutableFallbackSampler},
			{Binding: 4, DescriptorType: vk.DescriptorTypeStorageBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageFlags(vk.ShaderStageClosestHitBit | vk.ShaderStageAnyHitBit)},
			{Binding: 5, DescriptorType: vk.DescriptorTypeCombinedImageSampler, DescriptorCount: textureCount, StageFlags: vk.ShaderStageFlags(vk.ShaderStageClosestHitBit | vk.ShaderStageAnyHitBit), PImmutableSamplers: immutableTextureSamplers},
		},
	}, nil, &layout)

	var pool vk.DescriptorPool
	vk.CreateDescriptorPool(dev, &vk.DescriptorPoolCreateInfo{
		SType: vk.StructureTypeDescriptorPoolCreateInfo, MaxSets: count, PoolSizeCount: 6,
		PPoolSizes: []vk.DescriptorPoolSize{
			{Type: vk.DescriptorTypeAccelerationStructure, DescriptorCount: count},
			{Type: vk.DescriptorTypeStorageImage, DescriptorCount: count},
			{Type: vk.DescriptorTypeUniformBuffer, DescriptorCount: count},
			{Type: vk.DescriptorTypeCombinedImageSampler, DescriptorCount: count},
			{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: count},
			{Type: vk.DescriptorTypeCombinedImageSampler, DescriptorCount: count * textureCount},
		},
	}, nil, &pool)

	sets := make([]vk.DescriptorSet, count)
	for i := uint32(0); i < count; i++ {
		vk.AllocateDescriptorSets(dev, &vk.DescriptorSetAllocateInfo{
			SType: vk.StructureTypeDescriptorSetAllocateInfo, DescriptorPool: pool,
			DescriptorSetCount: 1, PSetLayouts: []vk.DescriptorSetLayout{layout},
		}, &sets[i])
	}

	textureInfos := make([]vk.DescriptorImageInfo, 0, len(textures))
	for _, texture := range textures {
		textureInfos = append(textureInfos, vk.DescriptorImageInfo{
			Sampler:     texture.GetSampler(),
			ImageView:   texture.GetView(),
			ImageLayout: vk.ImageLayoutShaderReadOnlyOptimal,
		})
	}

	for i := uint32(0); i < count; i++ {
		// Acceleration structure write uses pNext
		asWriteInfo := vk.WriteDescriptorSetAccelerationStructure{
			SType:                      vk.StructureTypeWriteDescriptorSetAccelerationStructure,
			AccelerationStructureCount: 1, PAccelerationStructures: []vk.AccelerationStructure{tlas},
		}
		fallbackTextureInfo := textureInfos[0]
		geometryInfo := vk.DescriptorBufferInfo{Buffer: geometryBuf, Offset: 0, Range: vk.DeviceSize(vk.WholeSize)}
		vk.UpdateDescriptorSets(dev, 6, []vk.WriteDescriptorSet{
			{SType: vk.StructureTypeWriteDescriptorSet, DstSet: sets[i], DstBinding: 0, DescriptorCount: 1,
				DescriptorType: vk.DescriptorTypeAccelerationStructure, PNext: unsafe.Pointer(&asWriteInfo)},
			{SType: vk.StructureTypeWriteDescriptorSet, DstSet: sets[i], DstBinding: 1, DescriptorCount: 1,
				DescriptorType: vk.DescriptorTypeStorageImage,
				PImageInfo:     []vk.DescriptorImageInfo{{ImageView: storageImageView, ImageLayout: vk.ImageLayoutGeneral}}},
			{SType: vk.StructureTypeWriteDescriptorSet, DstSet: sets[i], DstBinding: 2, DescriptorCount: 1,
				DescriptorType: vk.DescriptorTypeUniformBuffer,
				PBufferInfo:    []vk.DescriptorBufferInfo{{Buffer: uniforms.GetBuffer(i), Offset: 0, Range: vk.DeviceSize(uniformSize)}}},
			{SType: vk.StructureTypeWriteDescriptorSet, DstSet: sets[i], DstBinding: 3, DescriptorCount: 1,
				DescriptorType: vk.DescriptorTypeCombinedImageSampler,
				PImageInfo:     []vk.DescriptorImageInfo{fallbackTextureInfo}},
			{SType: vk.StructureTypeWriteDescriptorSet, DstSet: sets[i], DstBinding: 4, DescriptorCount: 1,
				DescriptorType: vk.DescriptorTypeStorageBuffer,
				PBufferInfo:    []vk.DescriptorBufferInfo{geometryInfo}},
			{SType: vk.StructureTypeWriteDescriptorSet, DstSet: sets[i], DstBinding: 5, DescriptorCount: textureCount,
				DescriptorType: vk.DescriptorTypeCombinedImageSampler,
				PImageInfo:     textureInfos},
		}, 0, nil)
	}
	return layout, pool, sets
}

func createRTPipeline(dev vk.Device, descLayout vk.DescriptorSetLayout) (vk.PipelineLayout, vk.Pipeline) {
	var pipelineLayout vk.PipelineLayout
	vk.CreatePipelineLayout(dev, &vk.PipelineLayoutCreateInfo{
		SType: vk.StructureTypePipelineLayoutCreateInfo, SetLayoutCount: 1,
		PSetLayouts: []vk.DescriptorSetLayout{descLayout},
	}, nil, &pipelineLayout)

	raygenModule, _ := ash.LoadShaderFromBytes(dev, raygenShaderCode)
	missModule, _ := ash.LoadShaderFromBytes(dev, missShaderCode)
	shadowMissModule, _ := ash.LoadShaderFromBytes(dev, shadowMissShaderCode)
	closestHitModule, _ := ash.LoadShaderFromBytes(dev, closestHitShaderCode)
	anyHitModule, _ := ash.LoadShaderFromBytes(dev, anyHitShaderCode)

	stages := []vk.PipelineShaderStageCreateInfo{
		{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageFlagBits(vk.ShaderStageRaygenBit), Module: raygenModule, PName: []byte("main\x00")},
		{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageFlagBits(vk.ShaderStageMissBit), Module: missModule, PName: []byte("main\x00")},
		{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageFlagBits(vk.ShaderStageMissBit), Module: shadowMissModule, PName: []byte("main\x00")},
		{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageFlagBits(vk.ShaderStageClosestHitBit), Module: closestHitModule, PName: []byte("main\x00")},
		{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageFlagBits(vk.ShaderStageAnyHitBit), Module: anyHitModule, PName: []byte("main\x00")},
	}
	groups := []vk.RayTracingShaderGroupCreateInfo{
		// Raygen
		{SType: vk.StructureTypeRayTracingShaderGroupCreateInfo, Type: vk.RayTracingShaderGroupTypeGeneral,
			GeneralShader: 0, ClosestHitShader: vk.ShaderUnused, AnyHitShader: vk.ShaderUnused, IntersectionShader: vk.ShaderUnused},
		// Miss
		{SType: vk.StructureTypeRayTracingShaderGroupCreateInfo, Type: vk.RayTracingShaderGroupTypeGeneral,
			GeneralShader: 1, ClosestHitShader: vk.ShaderUnused, AnyHitShader: vk.ShaderUnused, IntersectionShader: vk.ShaderUnused},
		// Shadow miss
		{SType: vk.StructureTypeRayTracingShaderGroupCreateInfo, Type: vk.RayTracingShaderGroupTypeGeneral,
			GeneralShader: 2, ClosestHitShader: vk.ShaderUnused, AnyHitShader: vk.ShaderUnused, IntersectionShader: vk.ShaderUnused},
		// Closest hit + any hit
		{SType: vk.StructureTypeRayTracingShaderGroupCreateInfo, Type: vk.RayTracingShaderGroupTypeTrianglesHitGroup,
			GeneralShader: vk.ShaderUnused, ClosestHitShader: 3, AnyHitShader: 4, IntersectionShader: vk.ShaderUnused},
	}

	createInfo := vk.RayTracingPipelineCreateInfo{
		SType:      vk.StructureTypeRayTracingPipelineCreateInfo,
		StageCount: uint32(len(stages)), PStages: stages,
		GroupCount: uint32(len(groups)), PGroups: groups,
		MaxPipelineRayRecursionDepth: 1, Layout: pipelineLayout,
	}
	var pip vk.Pipeline
	if err := vk.Error(vk.CreateRayTracingPipelines(dev, vk.DeferredOperation(vk.NullHandle), vk.NullPipelineCache, 1, &createInfo, nil, &pip)); err != nil {
		log.Fatal("CreateRayTracingPipelines:", err)
	}

	vk.DestroyShaderModule(dev, raygenModule, nil)
	vk.DestroyShaderModule(dev, missModule, nil)
	vk.DestroyShaderModule(dev, shadowMissModule, nil)
	vk.DestroyShaderModule(dev, closestHitModule, nil)
	vk.DestroyShaderModule(dev, anyHitModule, nil)
	return pipelineLayout, pip
}

func drawFrame(dev vk.Device, queue vk.Queue, s ash.VulkanSwapchainInfo, cmdCtx *ash.VulkanCommandContext,
	fence vk.Fence, semaphore vk.Semaphore,
	pipeline vk.Pipeline, pipelineLayout vk.PipelineLayout,
	descSets []vk.DescriptorSet, uniforms *ash.VulkanUniformBuffers,
	storageImage vk.Image,
	sbt *ash.VulkanSBT,
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
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointRayTracing, pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointRayTracing, pipelineLayout, 0, 1, []vk.DescriptorSet{descSets[nextIdx]}, 0, nil)

	// Trace rays
	vk.CmdTraceRays(cmd, &sbt.Raygen, &sbt.Miss, &sbt.Hit, &sbt.Callable, windowWidth, windowHeight, 1)

	// Copy storage image to swapchain image
	subresourceRange := vk.ImageSubresourceRange{AspectMask: vk.ImageAspectFlags(vk.ImageAspectColorBit), LevelCount: 1, LayerCount: 1}

	// Get swapchain images
	var imgCount uint32
	vk.GetSwapchainImages(dev, s.DefaultSwapchain(), &imgCount, nil)
	swapImages := make([]vk.Image, imgCount)
	vk.GetSwapchainImages(dev, s.DefaultSwapchain(), &imgCount, swapImages)

	// Transition swapchain image to transfer dst
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit), vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit),
		0, 0, nil, 0, nil, 1, []vk.ImageMemoryBarrier{{
			SType: vk.StructureTypeImageMemoryBarrier, OldLayout: vk.ImageLayoutUndefined, NewLayout: vk.ImageLayoutTransferDstOptimal,
			Image: swapImages[nextIdx], SubresourceRange: subresourceRange, DstAccessMask: vk.AccessFlags(vk.AccessTransferWriteBit),
			SrcQueueFamilyIndex: vk.QueueFamilyIgnored, DstQueueFamilyIndex: vk.QueueFamilyIgnored,
		}})
	// Transition storage image to transfer src
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit), vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit),
		0, 0, nil, 0, nil, 1, []vk.ImageMemoryBarrier{{
			SType: vk.StructureTypeImageMemoryBarrier, OldLayout: vk.ImageLayoutGeneral, NewLayout: vk.ImageLayoutTransferSrcOptimal,
			Image: storageImage, SubresourceRange: subresourceRange,
			SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessTransferReadBit),
			SrcQueueFamilyIndex: vk.QueueFamilyIgnored, DstQueueFamilyIndex: vk.QueueFamilyIgnored,
		}})

	// Copy
	vk.CmdCopyImage(cmd, storageImage, vk.ImageLayoutTransferSrcOptimal, swapImages[nextIdx], vk.ImageLayoutTransferDstOptimal,
		1, []vk.ImageCopy{{
			SrcSubresource: vk.ImageSubresourceLayers{AspectMask: vk.ImageAspectFlags(vk.ImageAspectColorBit), LayerCount: 1},
			DstSubresource: vk.ImageSubresourceLayers{AspectMask: vk.ImageAspectFlags(vk.ImageAspectColorBit), LayerCount: 1},
			Extent:         vk.Extent3D{Width: windowWidth, Height: windowHeight, Depth: 1},
		}})

	// Transition swapchain image to present
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit), vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit),
		0, 0, nil, 0, nil, 1, []vk.ImageMemoryBarrier{{
			SType: vk.StructureTypeImageMemoryBarrier, OldLayout: vk.ImageLayoutTransferDstOptimal, NewLayout: vk.ImageLayoutPresentSrc,
			Image: swapImages[nextIdx], SubresourceRange: subresourceRange,
			SrcAccessMask:       vk.AccessFlags(vk.AccessTransferWriteBit),
			SrcQueueFamilyIndex: vk.QueueFamilyIgnored, DstQueueFamilyIndex: vk.QueueFamilyIgnored,
		}})
	// Transition storage image back to general
	vk.CmdPipelineBarrier(cmd, vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit), vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit),
		0, 0, nil, 0, nil, 1, []vk.ImageMemoryBarrier{{
			SType: vk.StructureTypeImageMemoryBarrier, OldLayout: vk.ImageLayoutTransferSrcOptimal, NewLayout: vk.ImageLayoutGeneral,
			Image: storageImage, SubresourceRange: subresourceRange,
			SrcAccessMask: vk.AccessFlags(vk.AccessTransferReadBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit),
			SrcQueueFamilyIndex: vk.QueueFamilyIgnored, DstQueueFamilyIndex: vk.QueueFamilyIgnored,
		}})

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
