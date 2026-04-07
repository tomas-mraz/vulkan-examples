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

	newSurfaceFn := func(instance vk.Instance) (vk.Surface, error) {
		return ash.NewDesktopSurface(instance, window)
	}

	deviceOptions := &ash.DeviceOptions{
		DeviceExtensions: ash.RaytracingExtensions(),
		PNextChain:       unsafe.Pointer(&asFeatures),
		ApiVersion:       vk.MakeVersion(1, 2, 0),
	}

	manager, err := ash.NewManager(appName, extensions, newSurfaceFn, deviceOptions)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&manager)

	// Query RT pipeline properties (use hardcoded defaults, standard on all GPUs)
	const shaderGroupHandleSize = 32
	const shaderGroupHandleAlignment = 32

	// Create swapchain
	windowSize := ash.NewExtentSize(windowWidth, windowHeight)
	swapchain, err := ash.NewSwapchain(manager.Device, manager.Gpu, manager.Surface, windowSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&swapchain)
	swapchainCtx := ash.NewSwapchainContext(&manager, &swapchain)
	swapchainLen := swapchain.DefaultSwapchainLen()

	cmdCtx, err := ash.NewCommandContext(manager.Device, 0, swapchainLen)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&cmdCtx)
	rtx := ash.NewRaytracingContext(manager.Device, manager.Gpu, manager.Queue, &cmdCtx)

	// --- Create scene geometry (triangle) ---
	vertices := []float32{
		1.0, 1.0, 0.0,
		-1.0, 1.0, 0.0,
		0.0, -1.0, 0.0,
	}
	indices := []uint32{0, 1, 2}

	rtUsage := vk.BufferUsageFlags(vk.BufferUsageShaderDeviceAddressBit | vk.BufferUsageAccelerationStructureBuildInputReadOnlyBit | vk.BufferUsageStorageBufferBit)
	vertexBuf, err := ash.NewBufferHostVisible(manager.Device, manager.Gpu, vertices, true, rtUsage)
	if err != nil {
		log.Fatal("NewBufferHostVisible:", err)
	}
	cleanup.Add(&vertexBuf)

	indexBuf, err := ash.NewBufferHostVisible(manager.Device, manager.Gpu, indices, true, rtUsage)
	if err != nil {
		log.Fatal("NewBufferHostVisible:", err)
	}
	cleanup.Add(&indexBuf)

	// --- Build BLAS ---
	blas := buildBLAS(&rtx, vertexBuf.DeviceAddress, indexBuf.DeviceAddress, 3, 1)
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
	storageImg, err := ash.NewImageStorage(manager.Device, manager.Gpu, manager.Queue, cmdCtx.GetCmdPool(), windowWidth, windowHeight, swapchain.DisplayFormat)
	if err != nil {
		log.Fatal("NewImageStorage:", err)
	}
	cleanup.Add(&storageImg)

	// --- Uniform buffers ---
	uniforms, err := ash.NewUniformBuffers(manager.Device, manager.Gpu, swapchainLen, uniformSize)
	if err != nil {
		log.Fatal(err)
	}
	cleanup.Add(&uniforms)

	// --- Descriptors ---
	desc, err := ash.NewDescriptorSets(manager.Device, swapchainLen, []ash.DescriptorBinding{
		&ash.BindingAccelerationStructure{StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit), AccelerationStructure: tlas.AccelerationStructure},
		ash.NewBindingStorageImage(vk.ShaderStageFlags(vk.ShaderStageRaygenBit), &storageImg),
		&ash.BindingUniformBuffer{StageFlags: vk.ShaderStageFlags(vk.ShaderStageRaygenBit), Uniforms: &uniforms},
	})
	if err != nil {
		log.Fatal("NewDescriptorSets:", err)
	}
	cleanup.Add(&desc)

	// --- RT Pipeline ---
	rtPipeline, err := ash.NewRTPipeline(manager.Device, ash.RTPipelineOptions{
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
	sbt, err := ash.NewShaderBindingTable(manager.Device, manager.Gpu, rtPipeline.GetPipeline(), shaderGroupHandleSize, shaderGroupHandleAlignment, 1, 1, 1, 0)
	if err != nil {
		log.Fatal("NewShaderBindingTable:", err)
	}
	cleanup.Add(&sbt)

	// --- Sync objects ---
	sync, err := ash.NewSyncObjects(manager.Device)
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

		// Update uniform buffer with inverse matrices
		projInv := ash.InvertMatrix(&projMatrix)
		viewInv := ash.InvertMatrix(&viewMatrix)

		imageIndex, acquired, err := swapchainCtx.AcquireNextImage(vk.MaxUint64, sync.Semaphore, vk.NullFence)
		if err != nil {
			log.Println("AcquireNextImage:", err)
			break
		}
		if !acquired {
			continue
		}

		ubo := uniformData{ViewInverse: viewInv, ProjInverse: projInv}
		uniforms.Update(imageIndex, ubo.Bytes())

		cmd, err := swapchainCtx.BeginFrame(imageIndex, &cmdCtx)
		if err != nil {
			log.Println("BeginFrame:", err)
			break
		}

		// Bind RT pipeline and descriptors
		vk.CmdBindPipeline(cmd, vk.PipelineBindPointRayTracing, rtPipeline.GetPipeline())
		vk.CmdBindDescriptorSets(cmd, vk.PipelineBindPointRayTracing, rtPipeline.GetLayout(), 0, 1, []vk.DescriptorSet{desc.GetSets()[imageIndex]}, 0, nil)

		// Trace rays
		vk.CmdTraceRays(cmd, &sbt.Raygen, &sbt.Miss, &sbt.Hit, &sbt.Callable, windowWidth, windowHeight, 1)

		// Copy storage image to swapchain
		swapchainCtx.GetSwapchain().CmdCopyToSwapchain(cmd, storageImg.GetImage(), imageIndex)

		if err := swapchainCtx.EndFrame(cmd); err != nil {
			log.Println("EndFrame:", err)
			break
		}
		if err := swapchainCtx.SubmitRender(cmd, sync.Fence, []vk.Semaphore{sync.Semaphore}); err != nil {
			log.Println("SubmitRender:", err)
			break
		}
		if _, err := swapchainCtx.PresentImage(imageIndex, nil); err != nil {
			log.Println("PresentImage:", err)
			break
		}
	}
}

// --- Helper functions ---

func setDeviceAddressConst(addr *vk.DeviceOrHostAddressConst, da vk.DeviceAddress) {
	*(*vk.DeviceAddress)(unsafe.Pointer(&addr[0])) = da
}

func setDeviceAddress(addr *vk.DeviceOrHostAddress, da vk.DeviceAddress) {
	*(*vk.DeviceAddress)(unsafe.Pointer(&addr[0])) = da
}

func setGeometryTriangles(data *vk.AccelerationStructureGeometryData, tri *vk.AccelerationStructureGeometryTrianglesData) {
	cTri, _ := tri.PassRef()
	src := unsafe.Slice((*byte)(unsafe.Pointer(cTri)), len(*data))
	copy((*data)[:], src)
}

func buildBLAS(rtx *ash.RaytracingContext, vertexAddr, indexAddr vk.DeviceAddress, maxVertex, triangleCount uint32) ash.AccelerationStructure {
	var trianglesData vk.AccelerationStructureGeometryTrianglesData
	trianglesData.SType = vk.StructureTypeAccelerationStructureGeometryTrianglesData
	trianglesData.VertexFormat = vk.FormatR32g32b32Sfloat
	setDeviceAddressConst(&trianglesData.VertexData, vertexAddr)
	trianglesData.VertexStride = 12
	trianglesData.MaxVertex = maxVertex
	trianglesData.IndexType = vk.IndexTypeUint32
	setDeviceAddressConst(&trianglesData.IndexData, indexAddr)

	var geometry vk.AccelerationStructureGeometry
	geometry.SType = vk.StructureTypeAccelerationStructureGeometry
	geometry.GeometryType = vk.GeometryTypeTriangles
	geometry.Flags = vk.GeometryFlags(vk.GeometryOpaqueBit)
	setGeometryTriangles(&geometry.Geometry, &trianglesData)

	blas, err := rtx.NewBottomLevelAccelerationStructure(
		[]vk.AccelerationStructureGeometry{geometry},
		[]uint32{triangleCount},
		vk.BuildAccelerationStructureFlags(vk.BuildAccelerationStructurePreferFastTraceBit))
	if err != nil {
		log.Fatal("NewBottomLevelAccelerationStructure:", err)
	}
	return blas
}
