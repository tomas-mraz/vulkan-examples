package main

import (
	_ "embed"
	"log"
	"runtime"
	"unsafe"

	"github.com/go-gl/glfw/v3.3/glfw"
	vk "github.com/tomas-mraz/vulkan"
	ash "github.com/tomas-mraz/vulkan-ash"
)

//go:embed shaders/raygen.rgen.spv
var raygenShaderCode []byte

//go:embed shaders/miss.rmiss.spv
var missShaderCode []byte

//go:embed shaders/closesthit.rchit.spv
var closestHitShaderCode []byte

const (
	windowWidth  = 800
	windowHeight = 600
	appName      = "Ray Tracing Triangle"
)

type uniformData struct {
	ViewInverse ash.Mat4x4
	ProjInverse ash.Mat4x4
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

	device, err := ash.NewDeviceWithOptions(appName, extensions, func(instance vk.Instance, _ uintptr) (vk.Surface, error) {
		surfPtr, err := window.CreateWindowSurface(instance, nil)
		if err != nil {
			return vk.NullSurface, err
		}
		return vk.SurfaceFromPointer(surfPtr), nil
	}, 0, &ash.DeviceOptions{
		DeviceExtensions: ash.RaytracingExtensions(),
		PNextChain:       unsafe.Pointer(&asFeatures),
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

	// --- Create scene geometry (triangle) ---
	// Triangle vertices: position (xyz)
	vertices := []float32{
		1.0, 1.0, 0.0,
		-1.0, 1.0, 0.0,
		0.0, -1.0, 0.0,
	}
	indices := []uint32{0, 1, 2}

	// Create vertex buffer with device address
	vertexBuf := createBufferWithAddress(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageShaderDeviceAddressBit|vk.BufferUsageAccelerationStructureBuildInputReadOnlyBit|vk.BufferUsageStorageBufferBit),
		uint64(len(vertices)*4), unsafe.Pointer(&vertices[0]))
	cleanup.Add(&vertexBuf)
	vertexAddr := vertexBuf.DeviceAddress

	indexBuf := createBufferWithAddress(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageShaderDeviceAddressBit|vk.BufferUsageAccelerationStructureBuildInputReadOnlyBit|vk.BufferUsageStorageBufferBit),
		uint64(len(indices)*4), unsafe.Pointer(&indices[0]))
	cleanup.Add(&indexBuf)
	indexAddr := indexBuf.DeviceAddress

	// --- Build BLAS ---
	blasBuf, blas := buildBLAS(dev, gpu, queue, &cmdCtx, vertexAddr, indexAddr, 3, 1)
	cleanup.Add(ash.DestroyerFunc(func() {
		vk.DestroyAccelerationStructure(dev, blas, nil)
		blasBuf.Destroy()
	}))
	// --- Build TLAS ---
	tlasBuf, tlas := buildTLAS(dev, gpu, queue, &cmdCtx, blas)
	cleanup.Add(ash.DestroyerFunc(func() {
		vk.DestroyAccelerationStructure(dev, tlas, nil)
		tlasBuf.Destroy()
	}))
	// --- Create storage image ---
	storageImage, storageImageMem, storageImageView := createStorageImage(dev, gpu, queue, &cmdCtx, windowWidth, windowHeight, swapchain.DisplayFormat)
	cleanup.Add(ash.DestroyerFunc(func() {
		vk.DestroyImageView(dev, storageImageView, nil)
		vk.DestroyImage(dev, storageImage, nil)
		vk.FreeMemory(dev, storageImageMem, nil)
	}))

	// --- Uniform buffers ---
	uniforms, err := ash.NewUniformBuffers(dev, gpu, swapchainLen, uniformSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&uniforms)

	// --- Descriptors ---
	descLayout, descPool, descSets := createDescriptorSets(dev, swapchainLen, tlas, storageImageView, &uniforms)
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
	raygenSBT, missSBT, hitSBT, sbtBuf := createSBT(dev, gpu, pipeline, shaderGroupHandleSize, shaderGroupHandleAlignment)
	cleanup.Add(&sbtBuf)

	// --- Sync objects ---
	sync, err := ash.NewSyncObjects(dev)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&sync)

	// Camera matrices
	var projMatrix, viewMatrix ash.Mat4x4
	projMatrix.Perspective(ash.DegreesToRadians(60.0), float32(windowWidth)/float32(windowHeight), 0.1, 512.0)
	viewMatrix.LookAt(&ash.Vec3{0, 0, -2.5}, &ash.Vec3{0, 0, 0}, &ash.Vec3{0, 1, 0})
	projMatrix[1][1] *= -1

	log.Println("Ray tracing initialized, starting render loop")

	for !window.ShouldClose() {
		glfw.PollEvents()
		if !drawFrame(dev, queue, swapchain, &cmdCtx, sync.Fence, sync.Semaphore,
			pipeline, pipelineLayout, descSets, &uniforms,
			storageImage, &raygenSBT, &missSBT, &hitSBT,
			&projMatrix, &viewMatrix) {
			break
		}
	}

}

// --- Helper functions ---

func createBufferWithAddress(dev vk.Device, gpu vk.PhysicalDevice, usage vk.BufferUsageFlags, size uint64, data unsafe.Pointer) ash.VulkanBufferResource {
	buf, err := ash.NewHostVisibleBufferResource(dev, gpu, usage, size, data, true)
	if err != nil {
		log.Fatal("NewHostVisibleBufferResource:", err)
	}
	return buf
}

func createDeviceLocalBuffer(dev vk.Device, gpu vk.PhysicalDevice, usage vk.BufferUsageFlags, size uint64) ash.VulkanBufferResource {
	buf, err := ash.NewDeviceLocalBufferResource(dev, gpu, usage, size, true)
	if err != nil {
		log.Fatal("NewDeviceLocalBufferResource:", err)
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

// setGeometryTriangles writes AccelerationStructureGeometryTrianglesData into the union
func setGeometryTriangles(data *vk.AccelerationStructureGeometryData, tri *vk.AccelerationStructureGeometryTrianglesData) {
	cTri, _ := tri.PassRef()
	src := unsafe.Slice((*byte)(unsafe.Pointer(cTri)), len(*data))
	copy((*data)[:], src)
}

// setGeometryInstances writes AccelerationStructureGeometryInstancesData into the union
func setGeometryInstances(data *vk.AccelerationStructureGeometryData, inst *vk.AccelerationStructureGeometryInstancesData) {
	cInst, _ := inst.PassRef()
	src := unsafe.Slice((*byte)(unsafe.Pointer(cInst)), len(*data))
	copy((*data)[:], src)
}

func buildBLAS(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdCtx *ash.VulkanCommandContext, vertexAddr, indexAddr vk.DeviceAddress, maxVertex, triangleCount uint32) (ash.VulkanBufferResource, vk.AccelerationStructure) {
	var trianglesData vk.AccelerationStructureGeometryTrianglesData
	trianglesData.SType = vk.StructureTypeAccelerationStructureGeometryTrianglesData
	trianglesData.VertexFormat = vk.FormatR32g32b32Sfloat
	setDeviceAddressConst(&trianglesData.VertexData, vertexAddr)
	trianglesData.VertexStride = 12 // 3 floats * 4 bytes
	trianglesData.MaxVertex = maxVertex
	trianglesData.IndexType = vk.IndexTypeUint32
	setDeviceAddressConst(&trianglesData.IndexData, indexAddr)

	var geometry vk.AccelerationStructureGeometry
	geometry.SType = vk.StructureTypeAccelerationStructureGeometry
	geometry.GeometryType = vk.GeometryTypeTriangles
	geometry.Flags = vk.GeometryFlags(vk.GeometryOpaqueBit)
	setGeometryTriangles(&geometry.Geometry, &trianglesData)

	buildInfo := vk.AccelerationStructureBuildGeometryInfo{
		SType:         vk.StructureTypeAccelerationStructureBuildGeometryInfo,
		Type:          vk.AccelerationStructureTypeBottomLevel,
		Flags:         vk.BuildAccelerationStructureFlags(vk.BuildAccelerationStructurePreferFastTraceBit),
		GeometryCount: 1,
		PGeometries:   []vk.AccelerationStructureGeometry{geometry},
	}

	var sizeInfo vk.AccelerationStructureBuildSizesInfo
	sizeInfo.SType = vk.StructureTypeAccelerationStructureBuildSizesInfo
	vk.GetAccelerationStructureBuildSizes(dev, vk.AccelerationStructureBuildTypeDevice, &buildInfo, &triangleCount, &sizeInfo)
	sizeInfo.Deref()
	log.Printf("BLAS size: AS=%d, scratch=%d", sizeInfo.AccelerationStructureSize, sizeInfo.BuildScratchSize)

	// Create AS buffer
	asBuf := createDeviceLocalBuffer(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageAccelerationStructureStorageBit|vk.BufferUsageShaderDeviceAddressBit),
		uint64(sizeInfo.AccelerationStructureSize))

	var as vk.AccelerationStructure
	if err := vk.Error(vk.CreateAccelerationStructure(dev, &vk.AccelerationStructureCreateInfo{
		SType: vk.StructureTypeAccelerationStructureCreateInfo, Buffer: asBuf.Buffer,
		Size: sizeInfo.AccelerationStructureSize, Type: vk.AccelerationStructureTypeBottomLevel,
	}, nil, &as)); err != nil {
		log.Fatal("CreateAccelerationStructure (BLAS):", err)
	}

	// Scratch buffer
	scratchBuf := createDeviceLocalBuffer(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageStorageBufferBit|vk.BufferUsageShaderDeviceAddressBit),
		uint64(sizeInfo.BuildScratchSize))
	scratchAddr := scratchBuf.DeviceAddress

	// Create a fresh struct for the build call (PassRef caches the C struct)
	buildInfo2 := vk.AccelerationStructureBuildGeometryInfo{
		SType:                    vk.StructureTypeAccelerationStructureBuildGeometryInfo,
		Type:                     vk.AccelerationStructureTypeBottomLevel,
		Flags:                    vk.BuildAccelerationStructureFlags(vk.BuildAccelerationStructurePreferFastTraceBit),
		Mode:                     vk.BuildAccelerationStructureModeBuild,
		DstAccelerationStructure: as,
		GeometryCount:            1,
		PGeometries:              []vk.AccelerationStructureGeometry{geometry},
	}
	setDeviceAddress(&buildInfo2.ScratchData, scratchAddr)

	rangeInfos := []vk.AccelerationStructureBuildRangeInfo{{PrimitiveCount: triangleCount}}

	cmd, err := cmdCtx.BeginOneTime()
	if err != nil {
		log.Fatal("BeginOneTime:", err)
	}
	vk.CmdBuildAccelerationStructures(cmd, 1, &buildInfo2, [][]vk.AccelerationStructureBuildRangeInfo{rangeInfos})
	if err := cmdCtx.EndOneTime(queue, cmd); err != nil {
		log.Fatal("EndOneTime:", err)
	}

	scratchBuf.Destroy()
	return asBuf, as
}

func buildTLAS(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdCtx *ash.VulkanCommandContext, blas vk.AccelerationStructure) (ash.VulkanBufferResource, vk.AccelerationStructure) {
	// VkAccelerationStructureInstanceKHR is 64 bytes
	// Layout: transformMatrix (48 bytes), instanceCustomIndex+mask (4 bytes),
	//         instanceShaderBindingTableRecordOffset+flags (4 bytes), accelerationStructureReference (8 bytes)
	blasAddr := vk.GetAccelerationStructureDeviceAddress(dev, &vk.AccelerationStructureDeviceAddressInfo{
		SType: vk.StructureTypeAccelerationStructureDeviceAddressInfo, AccelerationStructure: blas,
	})

	// VkAccelerationStructureInstanceKHR: 64 bytes
	instanceData := make([]byte, 64)
	// Identity transform matrix (3x4 row-major)
	identity := [12]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}
	copy(instanceData[:48], unsafe.Slice((*byte)(unsafe.Pointer(&identity[0])), 48))
	// instanceCustomIndex (24 bit) + mask (8 bit) = 0x000000FF
	instanceData[48] = 0    // customIndex low
	instanceData[49] = 0    // customIndex mid
	instanceData[50] = 0    // customIndex high
	instanceData[51] = 0xFF // mask
	// instanceShaderBindingTableRecordOffset (24 bit) + flags (8 bit)
	instanceData[52] = 0    // offset low
	instanceData[53] = 0    // offset mid
	instanceData[54] = 0    // offset high
	instanceData[55] = 0x01 // VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR
	// accelerationStructureReference (uint64)
	*(*uint64)(unsafe.Pointer(&instanceData[56])) = uint64(blasAddr)

	instanceBuf := createBufferWithAddress(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageShaderDeviceAddressBit|vk.BufferUsageAccelerationStructureBuildInputReadOnlyBit),
		64, unsafe.Pointer(&instanceData[0]))
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
	log.Printf("TLAS size: AS=%d, scratch=%d", sizeInfo.AccelerationStructureSize, sizeInfo.BuildScratchSize)

	asBuf := createDeviceLocalBuffer(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageAccelerationStructureStorageBit|vk.BufferUsageShaderDeviceAddressBit),
		uint64(sizeInfo.AccelerationStructureSize))

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

	rangeInfos := []vk.AccelerationStructureBuildRangeInfo{{PrimitiveCount: 1}}

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
	return asBuf, as
}

func createStorageImage(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdCtx *ash.VulkanCommandContext, width, height uint32, format vk.Format) (vk.Image, vk.DeviceMemory, vk.ImageView) {
	var img vk.Image
	vk.CreateImage(dev, &vk.ImageCreateInfo{
		SType: vk.StructureTypeImageCreateInfo, ImageType: vk.ImageType2d, Format: format,
		Extent:    vk.Extent3D{Width: width, Height: height, Depth: 1},
		MipLevels: 1, ArrayLayers: 1, Samples: vk.SampleCount1Bit,
		Tiling: vk.ImageTilingOptimal,
		Usage:  vk.ImageUsageFlags(vk.ImageUsageTransferSrcBit | vk.ImageUsageStorageBit),
	}, nil, &img)

	var memReqs vk.MemoryRequirements
	vk.GetImageMemoryRequirements(dev, img, &memReqs)
	memReqs.Deref()
	memIdx, _ := vk.FindMemoryTypeIndex(gpu, memReqs.MemoryTypeBits, vk.MemoryPropertyDeviceLocalBit)
	var mem vk.DeviceMemory
	vk.AllocateMemory(dev, &vk.MemoryAllocateInfo{
		SType: vk.StructureTypeMemoryAllocateInfo, AllocationSize: memReqs.Size,
		MemoryTypeIndex: memIdx,
	}, nil, &mem)
	vk.BindImageMemory(dev, img, mem, 0)

	var view vk.ImageView
	vk.CreateImageView(dev, &vk.ImageViewCreateInfo{
		SType: vk.StructureTypeImageViewCreateInfo, Image: img,
		ViewType: vk.ImageViewType2d, Format: format,
		SubresourceRange: vk.ImageSubresourceRange{
			AspectMask: vk.ImageAspectFlags(vk.ImageAspectColorBit), LevelCount: 1, LayerCount: 1,
		},
	}, nil, &view)

	// Transition to general layout
	cmd, err := cmdCtx.BeginOneTime()
	if err != nil {
		log.Fatal("BeginOneTime:", err)
	}
	vk.CmdPipelineBarrier(cmd,
		vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit), vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit),
		0, 0, nil, 0, nil, 1, []vk.ImageMemoryBarrier{{
			SType: vk.StructureTypeImageMemoryBarrier, OldLayout: vk.ImageLayoutUndefined, NewLayout: vk.ImageLayoutGeneral,
			Image: img, SubresourceRange: vk.ImageSubresourceRange{AspectMask: vk.ImageAspectFlags(vk.ImageAspectColorBit), LevelCount: 1, LayerCount: 1},
			DstAccessMask:       vk.AccessFlags(vk.AccessShaderWriteBit),
			SrcQueueFamilyIndex: vk.QueueFamilyIgnored, DstQueueFamilyIndex: vk.QueueFamilyIgnored,
		}})
	if err := cmdCtx.EndOneTime(queue, cmd); err != nil {
		log.Fatal("EndOneTime:", err)
	}
	return img, mem, view
}

func createDescriptorSets(dev vk.Device, count uint32, tlas vk.AccelerationStructure, storageImageView vk.ImageView, uniforms *ash.VulkanUniformBuffers) (vk.DescriptorSetLayout, vk.DescriptorPool, []vk.DescriptorSet) {
	var layout vk.DescriptorSetLayout
	vk.CreateDescriptorSetLayout(dev, &vk.DescriptorSetLayoutCreateInfo{
		SType: vk.StructureTypeDescriptorSetLayoutCreateInfo, BindingCount: 3,
		PBindings: []vk.DescriptorSetLayoutBinding{
			{Binding: 0, DescriptorType: vk.DescriptorTypeAccelerationStructure, DescriptorCount: 1, StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit)},
			{Binding: 1, DescriptorType: vk.DescriptorTypeStorageImage, DescriptorCount: 1, StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit)},
			{Binding: 2, DescriptorType: vk.DescriptorTypeUniformBuffer, DescriptorCount: 1, StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit)},
		},
	}, nil, &layout)

	var pool vk.DescriptorPool
	vk.CreateDescriptorPool(dev, &vk.DescriptorPoolCreateInfo{
		SType: vk.StructureTypeDescriptorPoolCreateInfo, MaxSets: count, PoolSizeCount: 3,
		PPoolSizes: []vk.DescriptorPoolSize{
			{Type: vk.DescriptorTypeAccelerationStructure, DescriptorCount: count},
			{Type: vk.DescriptorTypeStorageImage, DescriptorCount: count},
			{Type: vk.DescriptorTypeUniformBuffer, DescriptorCount: count},
		},
	}, nil, &pool)

	sets := make([]vk.DescriptorSet, count)
	for i := uint32(0); i < count; i++ {
		vk.AllocateDescriptorSets(dev, &vk.DescriptorSetAllocateInfo{
			SType: vk.StructureTypeDescriptorSetAllocateInfo, DescriptorPool: pool,
			DescriptorSetCount: 1, PSetLayouts: []vk.DescriptorSetLayout{layout},
		}, &sets[i])
	}

	for i := uint32(0); i < count; i++ {
		// Acceleration structure write uses pNext
		asWriteInfo := vk.WriteDescriptorSetAccelerationStructure{
			SType:                      vk.StructureTypeWriteDescriptorSetAccelerationStructure,
			AccelerationStructureCount: 1, PAccelerationStructures: []vk.AccelerationStructure{tlas},
		}
		vk.UpdateDescriptorSets(dev, 3, []vk.WriteDescriptorSet{
			{SType: vk.StructureTypeWriteDescriptorSet, DstSet: sets[i], DstBinding: 0, DescriptorCount: 1,
				DescriptorType: vk.DescriptorTypeAccelerationStructure, PNext: unsafe.Pointer(&asWriteInfo)},
			{SType: vk.StructureTypeWriteDescriptorSet, DstSet: sets[i], DstBinding: 1, DescriptorCount: 1,
				DescriptorType: vk.DescriptorTypeStorageImage,
				PImageInfo:     []vk.DescriptorImageInfo{{ImageView: storageImageView, ImageLayout: vk.ImageLayoutGeneral}}},
			{SType: vk.StructureTypeWriteDescriptorSet, DstSet: sets[i], DstBinding: 2, DescriptorCount: 1,
				DescriptorType: vk.DescriptorTypeUniformBuffer,
				PBufferInfo:    []vk.DescriptorBufferInfo{{Buffer: uniforms.GetBuffer(i), Offset: 0, Range: vk.DeviceSize(uniformSize)}}},
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
	closestHitModule, _ := ash.LoadShaderFromBytes(dev, closestHitShaderCode)

	stages := []vk.PipelineShaderStageCreateInfo{
		{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageFlagBits(vk.ShaderStageRaygenBit), Module: raygenModule, PName: []byte("main\x00")},
		{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageFlagBits(vk.ShaderStageMissBit), Module: missModule, PName: []byte("main\x00")},
		{SType: vk.StructureTypePipelineShaderStageCreateInfo, Stage: vk.ShaderStageFlagBits(vk.ShaderStageClosestHitBit), Module: closestHitModule, PName: []byte("main\x00")},
	}
	groups := []vk.RayTracingShaderGroupCreateInfo{
		// Raygen
		{SType: vk.StructureTypeRayTracingShaderGroupCreateInfo, Type: vk.RayTracingShaderGroupTypeGeneral,
			GeneralShader: 0, ClosestHitShader: vk.ShaderUnused, AnyHitShader: vk.ShaderUnused, IntersectionShader: vk.ShaderUnused},
		// Miss
		{SType: vk.StructureTypeRayTracingShaderGroupCreateInfo, Type: vk.RayTracingShaderGroupTypeGeneral,
			GeneralShader: 1, ClosestHitShader: vk.ShaderUnused, AnyHitShader: vk.ShaderUnused, IntersectionShader: vk.ShaderUnused},
		// Closest hit
		{SType: vk.StructureTypeRayTracingShaderGroupCreateInfo, Type: vk.RayTracingShaderGroupTypeTrianglesHitGroup,
			GeneralShader: vk.ShaderUnused, ClosestHitShader: 2, AnyHitShader: vk.ShaderUnused, IntersectionShader: vk.ShaderUnused},
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
	vk.DestroyShaderModule(dev, closestHitModule, nil)
	return pipelineLayout, pip
}

func alignUp(size, alignment uint32) uint32 {
	return (size + alignment - 1) &^ (alignment - 1)
}

func createSBT(dev vk.Device, gpu vk.PhysicalDevice, pipeline vk.Pipeline, handleSize, handleAlignment uint32) (vk.StridedDeviceAddressRegion, vk.StridedDeviceAddressRegion, vk.StridedDeviceAddressRegion, ash.VulkanBufferResource) {
	groupCount := uint32(3) // raygen, miss, hit
	handleSizeAligned := alignUp(handleSize, handleAlignment)
	// Read all shader group handles
	sbtSize := groupCount * handleSizeAligned
	handleStorage := make([]byte, sbtSize)
	if err := vk.Error(vk.GetRayTracingShaderGroupHandles(dev, pipeline, 0, groupCount, uint64(sbtSize), unsafe.Pointer(&handleStorage[0]))); err != nil {
		log.Fatal("GetRayTracingShaderGroupHandles:", err)
	}

	// Create single SBT buffer containing all groups
	sbtBuf := createBufferWithAddress(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageShaderBindingTableBit|vk.BufferUsageShaderDeviceAddressBit),
		uint64(sbtSize), unsafe.Pointer(&handleStorage[0]))
	sbtAddr := sbtBuf.DeviceAddress

	raygenSBT := vk.StridedDeviceAddressRegion{DeviceAddress: sbtAddr, Stride: vk.DeviceSize(handleSizeAligned), Size: vk.DeviceSize(handleSizeAligned)}
	missSBT := vk.StridedDeviceAddressRegion{DeviceAddress: sbtAddr + vk.DeviceAddress(handleSizeAligned), Stride: vk.DeviceSize(handleSizeAligned), Size: vk.DeviceSize(handleSizeAligned)}
	hitSBT := vk.StridedDeviceAddressRegion{DeviceAddress: sbtAddr + vk.DeviceAddress(2*handleSizeAligned), Stride: vk.DeviceSize(handleSizeAligned), Size: vk.DeviceSize(handleSizeAligned)}
	return raygenSBT, missSBT, hitSBT, sbtBuf
}

func drawFrame(dev vk.Device, queue vk.Queue, s ash.VulkanSwapchainInfo, cmdCtx *ash.VulkanCommandContext,
	fence vk.Fence, semaphore vk.Semaphore,
	pipeline vk.Pipeline, pipelineLayout vk.PipelineLayout,
	descSets []vk.DescriptorSet, uniforms *ash.VulkanUniformBuffers,
	storageImage vk.Image,
	raygenSBT, missSBT, hitSBT *vk.StridedDeviceAddressRegion,
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
	ubo := uniformData{ViewInverse: viewInv, ProjInverse: projInv}
	uniforms.Update(nextIdx, ubo.Bytes())

	cmd := cmdBuffers[nextIdx]
	vk.ResetCommandBuffer(cmd, 0)
	vk.BeginCommandBuffer(cmd, &vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo})

	// Bind RT pipeline and descriptors
	vk.CmdBindPipeline(cmd, vk.PipelineBindPointRayTracing, pipeline)
	vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointRayTracing, pipelineLayout, 0, 1, []vk.DescriptorSet{descSets[nextIdx]}, 0, nil)

	// Trace rays
	emptySBT := vk.StridedDeviceAddressRegion{}
	vk.CmdTraceRays(cmd, raygenSBT, missSBT, hitSBT, &emptySBT, windowWidth, windowHeight, 1)

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
