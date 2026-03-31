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
	rtx := ash.NewRaytracingContext(dev, gpu, queue, &cmdCtx)

	// --- Create scene geometry (triangle) ---
	// Triangle vertices: position (xyz)
	vertices := []float32{
		1.0, 1.0, 0.0,
		-1.0, 1.0, 0.0,
		0.0, -1.0, 0.0,
	}
	indices := []uint32{0, 1, 2}

	// Create vertex buffer with device address
	rtUsage := vk.BufferUsageFlags(vk.BufferUsageShaderDeviceAddressBit | vk.BufferUsageAccelerationStructureBuildInputReadOnlyBit | vk.BufferUsageStorageBufferBit)
	vertexBuf, err := ash.NewBufferHostVisible(dev, gpu, vertices, true, rtUsage)
	if err != nil {
		log.Fatal("NewBufferHostVisible:", err)
	}
	cleanup.Add(&vertexBuf)
	vertexAddr := vertexBuf.DeviceAddress

	indexBuf, err := ash.NewBufferHostVisible(dev, gpu, indices, true, rtUsage)
	if err != nil {
		log.Fatal("NewBufferHostVisible:", err)
	}
	cleanup.Add(&indexBuf)
	indexAddr := indexBuf.DeviceAddress

	// --- Build BLAS ---
	blas := buildBLAS(dev, gpu, queue, &cmdCtx, vertexAddr, indexAddr, 3, 1)
	cleanup.Add(&blas)
	// --- Build TLAS ---
	tlas, err := rtx.NewTopLevelAccelerationStructure([]ash.TLASInstance{{
		Transform:           [12]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
		InstanceCustomIndex: 0,
		Mask:                0xFF,
		SBTRecordOffset:     0,
		Flags:               vk.GeometryInstanceFlags(vk.GeometryInstanceTriangleFacingCullDisableBit),
		BLAS:                &blas,
	}}, vk.BuildAccelerationStructureFlags(vk.BuildAccelerationStructurePreferFastTraceBit))
	if err != nil {
		log.Fatal("NewTopLevelAccelerationStructure:", err)
	}
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
	desc, err := ash.NewDescriptorSets(dev, swapchainLen, []ash.DescriptorBinding{
		&ash.BindingAccelerationStructure{StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit), AccelerationStructure: tlas.AccelerationStructure},
		&ash.BindingStorageImage{StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit), ImageView: storageImg.GetView()},
		&ash.BindingUniformBuffer{StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit), Uniforms: &uniforms},
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
			{ClosestHitShader: closestHitShaderCode},
		},
		DescriptorSetLayouts: []vk.DescriptorSetLayout{desc.GetLayout()},
	})
	if err != nil {
		log.Fatal("NewRTPipeline:", err)
	}
	cleanup.Add(&rtPipeline)
	// --- Shader Binding Table ---
	sbt, err := ash.NewShaderBindingTable(dev, gpu, rtPipeline.GetPipeline(), shaderGroupHandleSize, shaderGroupHandleAlignment, 1, 1, 1, 0)
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

	// Camera matrices
	var projMatrix, viewMatrix ash.Mat4x4
	projMatrix.Perspective(ash.DegreesToRadians(60.0), float32(windowWidth)/float32(windowHeight), 0.1, 512.0)
	viewMatrix.LookAt(&ash.Vec3{0, 0, -2.5}, &ash.Vec3{0, 0, 0}, &ash.Vec3{0, 1, 0})
	projMatrix[1][1] *= -1

	log.Println("Ray tracing initialized, starting render loop")

	for !window.ShouldClose() {
		glfw.PollEvents()
		if !drawFrame(dev, queue, swapchain, &cmdCtx, sync.Fence, sync.Semaphore,
			&rtPipeline, desc.GetSets(), &uniforms,
			storageImg.GetImage(), &sbt,
			&projMatrix, &viewMatrix) {
			break
		}
	}

}

// --- Helper functions ---

func createDeviceLocalBuffer(dev vk.Device, gpu vk.PhysicalDevice, usage vk.BufferUsageFlags, size uint64) ash.VulkanBufferResource {
	buf, err := ash.NewBufferDeviceLocal(dev, gpu, size, true, usage)
	if err != nil {
		log.Fatal("NewBufferResourceDeviceLocal:", err)
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

func buildBLAS(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdCtx *ash.CommandContext, vertexAddr, indexAddr vk.DeviceAddress, maxVertex, triangleCount uint32) ash.AccelerationStructure {
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
	return ash.AccelerationStructure{
		AccelerationStructure: as,
		Buffer:                asBuf,
		Type:                  vk.AccelerationStructureTypeBottomLevel,
	}
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
	ubo := uniformData{ViewInverse: viewInv, ProjInverse: projInv}
	uniforms.Update(nextIdx, ubo.Bytes())

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
