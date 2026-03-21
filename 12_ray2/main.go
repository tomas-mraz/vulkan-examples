package main

import (
	"bytes"
	_ "embed"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"math"
	"os"
	"time"
	"path/filepath"
	"runtime"
	"unsafe"

	"github.com/go-gl/glfw/v3.3/glfw"
	"github.com/qmuntal/gltf"
	"github.com/qmuntal/gltf/modeler"
	vk "github.com/tomas-mraz/vulkan"
	asch "github.com/tomas-mraz/vulkan-ash"
	lin "github.com/xlab/linmath"
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
	ViewInverse lin.Mat4x4
	ProjInverse lin.Mat4x4
	Frame       uint32
}

const uniformSize = int(unsafe.Sizeof(uniformData{}))

func (u *uniformData) Bytes() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(u)), uniformSize)
}

// primitiveData holds per-primitive geometry resources.
type primitiveData struct {
	vertexBuf     vk.Buffer
	vertexMem     vk.DeviceMemory
	indexBuf      vk.Buffer
	indexMem      vk.DeviceMemory
	vertexCount   uint32
	triangleCount uint32
	transform     [12]float32
	baseColorTex  int32
	occlusionTex  int32
}

type textureData struct {
	image   vk.Image
	memory  vk.DeviceMemory
	view    vk.ImageView
	sampler vk.Sampler
}

// modelData owns primitive buffers plus a single multi-geometry BLAS for the model.
type modelData struct {
	primitives  []primitiveData
	geometryBuf vk.Buffer
	geometryMem vk.DeviceMemory
	blasBuf     vk.Buffer
	blasMem     vk.DeviceMemory
	blas        vk.AccelerationStructure
	textures    []textureData
}

type geometryNode struct {
	VertexBufferDeviceAddress uint64
	IndexBufferDeviceAddress  uint64
	TextureIndexBaseColor     int32
	TextureIndexOcclusion     int32
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

	// Create device with ray tracing extensions
	asch.SetDebug(false)
	extensions := window.GetRequiredInstanceExtensions()

	rtExtensions := []string{
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

	device, err := asch.NewDeviceWithOptions(appName, extensions, func(instance vk.Instance, _ uintptr) (vk.Surface, error) {
		surfPtr, err := window.CreateWindowSurface(instance, nil)
		if err != nil {
			return vk.NullSurface, err
		}
		return vk.SurfaceFromPointer(surfPtr), nil
	}, 0, &asch.DeviceOptions{
		DeviceExtensions: rtExtensions,
		PNextChain:       unsafe.Pointer(&descriptorIndexingFeatures),
		EnabledFeatures:  &enabledFeatures,
		ApiVersion:       vk.MakeVersion(1, 2, 0),
	})
	if err != nil {
		log.Fatal(err)
	}
	dev := device.Device
	gpu := device.GpuDevice
	queue := device.Queue

	// Query RT pipeline properties (use hardcoded defaults, standard on all GPUs)
	const shaderGroupHandleSize = 32
	const shaderGroupHandleAlignment = 32

	// Create swapchain
	windowSize := asch.NewExtentSize(windowWidth, windowHeight)
	swapchain, err := asch.NewSwapchain(dev, gpu, device.Surface, windowSize)
	if err != nil {
		log.Fatal(err)
	}
	swapchainLen := swapchain.DefaultSwapchainLen()

	// Create command pool
	var cmdPool vk.CommandPool
	if err := vk.Error(vk.CreateCommandPool(dev, &vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		Flags:            vk.CommandPoolCreateFlags(vk.CommandPoolCreateResetCommandBufferBit),
		QueueFamilyIndex: 0,
	}, nil, &cmdPool)); err != nil {
		log.Fatal("CreateCommandPool:", err)
	}

	// Create command buffers
	cmdBuffers := make([]vk.CommandBuffer, swapchainLen)
	if err := vk.Error(vk.AllocateCommandBuffers(dev, &vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        cmdPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: swapchainLen,
	}, cmdBuffers)); err != nil {
		log.Fatal("AllocateCommandBuffers:", err)
	}

	// --- Load glTF model ---
	model := loadGLTFModel(dev, gpu, queue, cmdPool, "assets/FlightHelmet/FlightHelmet.gltf")
	log.Printf("Loaded %d primitives into one BLAS", len(model.primitives))

	// --- Build TLAS with one instance for the model BLAS ---
	tlasBuf, tlasMem, tlas := buildTLAS(dev, gpu, queue, cmdPool, model.blas)

	// --- Create storage image ---
	storageImage, storageImageMem, storageImageView := createStorageImage(dev, gpu, queue, cmdPool, windowWidth, windowHeight, swapchain.DisplayFormat)

	// --- Uniform buffers ---
	uniforms, err := asch.NewUniformBuffers(dev, gpu, swapchainLen, uniformSize)
	if err != nil {
		log.Fatal(err)
	}

	// --- Descriptors ---
	descLayout, descPool, descSets := createDescriptorSets(dev, swapchainLen, tlas, storageImageView, model.geometryBuf, model.textures, &uniforms)
	// --- RT Pipeline ---
	pipelineLayout, pipeline := createRTPipeline(dev, descLayout)
	// --- Shader Binding Table ---
	raygenSBT, missSBT, hitSBT, sbtBuf, sbtMem := createSBT(dev, gpu, pipeline, shaderGroupHandleSize, shaderGroupHandleAlignment)

	// --- Sync objects ---
	fence, semaphore, err := createSyncObjects(dev)
	if err != nil {
		log.Fatal(err)
	}

	// Match Sascha Willems raytracinggltf camera setup.
	var projMatrix, viewMatrix lin.Mat4x4
	setPerspectiveZO(&projMatrix, lin.DegreesToRadians(60.0), float32(windowWidth)/float32(windowHeight), 0.1, 512.0)
	viewMatrix.Translate(0.0, -0.1, -1.0)

	log.Println("Ray tracing initialized, starting render loop")
	startTime := time.Now()

	for !window.ShouldClose() {
		glfw.PollEvents()

		// Rotate the view around Y axis (like 06_model)
		elapsed := float32(time.Since(startTime).Seconds()) * 45.0
		var rotatedView lin.Mat4x4
		rotatedView.Rotate(&viewMatrix, 0.0, 1.0, 0.0, lin.DegreesToRadians(elapsed))
		frameCounter = 0 // reset accumulation — camera changed

		if !drawFrame(dev, queue, swapchain, cmdBuffers, fence, semaphore,
			pipeline, pipelineLayout, descSets, &uniforms,
			storageImage, &raygenSBT, &missSBT, &hitSBT,
			&projMatrix, &rotatedView) {
			break
		}
	}

	// Cleanup
	vk.DeviceWaitIdle(dev)
	vk.DestroySemaphore(dev, semaphore, nil)
	vk.DestroyFence(dev, fence, nil)
	vk.DestroyPipeline(dev, pipeline, nil)
	vk.DestroyPipelineLayout(dev, pipelineLayout, nil)
	vk.DestroyDescriptorPool(dev, descPool, nil)
	vk.DestroyDescriptorSetLayout(dev, descLayout, nil)
	uniforms.Destroy()
	vk.DestroyBuffer(dev, sbtBuf, nil)
	vk.FreeMemory(dev, sbtMem, nil)
	vk.DestroyImageView(dev, storageImageView, nil)
	vk.DestroyImage(dev, storageImage, nil)
	vk.FreeMemory(dev, storageImageMem, nil)
	for i := range model.textures {
		destroyTexture(dev, model.textures[i])
	}
	vk.DestroyAccelerationStructure(dev, tlas, nil)
	vk.DestroyBuffer(dev, tlasBuf, nil)
	vk.FreeMemory(dev, tlasMem, nil)
	vk.DestroyAccelerationStructure(dev, model.blas, nil)
	vk.DestroyBuffer(dev, model.blasBuf, nil)
	vk.FreeMemory(dev, model.blasMem, nil)
	vk.DestroyBuffer(dev, model.geometryBuf, nil)
	vk.FreeMemory(dev, model.geometryMem, nil)
	for i := range model.primitives {
		vk.DestroyBuffer(dev, model.primitives[i].indexBuf, nil)
		vk.FreeMemory(dev, model.primitives[i].indexMem, nil)
		vk.DestroyBuffer(dev, model.primitives[i].vertexBuf, nil)
		vk.FreeMemory(dev, model.primitives[i].vertexMem, nil)
	}
	vk.FreeCommandBuffers(dev, cmdPool, swapchainLen, cmdBuffers)
	vk.DestroyCommandPool(dev, cmdPool, nil)
	swapchain.Destroy()
	vk.DestroyDevice(dev, nil)
	if device.GetDebugCallback() != vk.NullDebugReportCallback {
		vk.DestroyDebugReportCallback(device.Instance, device.GetDebugCallback(), nil)
	}
	vk.DestroySurface(device.Instance, device.Surface, nil)
	vk.DestroyInstance(device.Instance, nil)
}

// --- glTF loading ---

func loadGLTFModel(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdPool vk.CommandPool, path string) modelData {
	doc, err := gltf.Open(path)
	if err != nil {
		log.Fatal("gltf.Open:", err)
	}
	if len(doc.Scenes) == 0 {
		log.Fatal("gltf model has no scenes")
	}

	var prims []primitiveData
	activeScene := 0
	if doc.Scene != nil {
		activeScene = *doc.Scene
	}
	if activeScene < 0 || activeScene >= len(doc.Scenes) {
		log.Fatalf("gltf scene index %d out of range", activeScene)
	}

	textures := loadGLTFTextures(dev, gpu, queue, cmdPool, doc, filepath.Dir(path))
	var visitNode func(nodeIndex int, parentTransform [16]float32)
	visitNode = func(nodeIndex int, parentTransform [16]float32) {
		node := doc.Nodes[nodeIndex]
		worldTransform := multiplyMat4(parentTransform, gltfNodeTransform(node))

		if node.Mesh != nil {
			meshIndex := *node.Mesh
			mesh := doc.Meshes[meshIndex]
			for pi, prim := range mesh.Primitives {
				positions, err := modeler.ReadPosition(doc, doc.Accessors[prim.Attributes[gltf.POSITION]], nil)
				if err != nil {
					log.Fatalf("Node %d mesh %d prim %d ReadPosition: %v", nodeIndex, meshIndex, pi, err)
				}
				normals, err := modeler.ReadNormal(doc, doc.Accessors[prim.Attributes[gltf.NORMAL]], nil)
				if err != nil {
					log.Fatalf("Node %d mesh %d prim %d ReadNormal: %v", nodeIndex, meshIndex, pi, err)
				}
				uvs, err := modeler.ReadTextureCoord(doc, doc.Accessors[prim.Attributes[gltf.TEXCOORD_0]], nil)
				if err != nil {
					log.Fatalf("Node %d mesh %d prim %d ReadTextureCoord: %v", nodeIndex, meshIndex, pi, err)
				}
				indices, err := modeler.ReadIndices(doc, doc.Accessors[*prim.Indices], nil)
				if err != nil {
					log.Fatalf("Node %d mesh %d prim %d ReadIndices: %v", nodeIndex, meshIndex, pi, err)
				}

				// Interleave: pos3 + normal3 + uv2 = 8 floats per vertex
				vertices := make([]float32, 0, len(positions)*8)
				for i := range positions {
					vertices = append(vertices,
						positions[i][0], positions[i][1], positions[i][2],
						normals[i][0], normals[i][1], normals[i][2],
						uvs[i][0], uvs[i][1],
					)
				}

				log.Printf("  Node %d mesh %d prim %d: %d verts, %d indices", nodeIndex, meshIndex, pi, len(positions), len(indices))

				// Create GPU buffers with device address
				rtUsage := vk.BufferUsageFlags(vk.BufferUsageShaderDeviceAddressBit | vk.BufferUsageAccelerationStructureBuildInputReadOnlyBit | vk.BufferUsageStorageBufferBit)
				vertexBuf, vertexMem := createBufferWithAddress(dev, gpu, rtUsage,
					uint64(len(vertices)*4), unsafe.Pointer(&vertices[0]))
				indexBuf, indexMem := createBufferWithAddress(dev, gpu, rtUsage,
					uint64(len(indices)*4), unsafe.Pointer(&indices[0]))

				baseColorTex := int32(0)
				occlusionTex := int32(-1)
				if prim.Material != nil && *prim.Material >= 0 && *prim.Material < len(doc.Materials) {
					material := doc.Materials[*prim.Material]
					if material != nil {
						if material.PBRMetallicRoughness != nil && material.PBRMetallicRoughness.BaseColorTexture != nil {
							baseColorTex = int32(material.PBRMetallicRoughness.BaseColorTexture.Index + 1)
						}
						if material.OcclusionTexture != nil && material.OcclusionTexture.Index != nil {
							occlusionTex = int32(*material.OcclusionTexture.Index + 1)
						}
					}
				}

				prims = append(prims, primitiveData{
					vertexBuf: vertexBuf, vertexMem: vertexMem,
					indexBuf: indexBuf, indexMem: indexMem,
					vertexCount:   uint32(len(positions)),
					triangleCount: uint32(len(indices) / 3),
					transform:     vkTransformMatrix(worldTransform),
					baseColorTex:  baseColorTex,
					occlusionTex:  occlusionTex,
				})
			}
		}
		for _, childIndex := range node.Children {
			visitNode(childIndex, worldTransform)
		}
	}

	for _, rootNode := range doc.Scenes[activeScene].Nodes {
		visitNode(rootNode, identityMat4())
	}

	if len(prims) == 0 {
		log.Fatal("gltf model has no primitives")
	}

	blasBuf, blasMem, blas := buildBLAS(dev, gpu, queue, cmdPool, prims)
	geometryBuf, geometryMem := createGeometryNodesBuffer(dev, gpu, prims)
	return modelData{
		primitives:  prims,
		geometryBuf: geometryBuf,
		geometryMem: geometryMem,
		blasBuf:     blasBuf,
		blasMem:     blasMem,
		blas:        blas,
		textures:    textures,
	}
}

func identityMat4() [16]float32 {
	return [16]float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
}

func multiplyMat4(a, b [16]float32) [16]float32 {
	var out [16]float32
	for col := 0; col < 4; col++ {
		for row := 0; row < 4; row++ {
			var sum float32
			for k := 0; k < 4; k++ {
				sum += a[k*4+row] * b[col*4+k]
			}
			out[col*4+row] = sum
		}
	}
	return out
}

func gltfNodeTransform(node *gltf.Node) [16]float32 {
	if node.Matrix != gltf.DefaultMatrix && node.Matrix != [16]float64{} {
		return gltfMatrixToArray(node.Matrix)
	}

	translation := node.TranslationOrDefault()
	rotation := node.RotationOrDefault()
	scale := node.ScaleOrDefault()

	var t lin.Mat4x4
	t.Translate(float32(translation[0]), float32(translation[1]), float32(translation[2]))

	var r lin.Mat4x4
	r.FromQuat(&lin.Quat{
		float32(rotation[0]),
		float32(rotation[1]),
		float32(rotation[2]),
		float32(rotation[3]),
	})

	var rs lin.Mat4x4
	rs.ScaleAniso(&r, float32(scale[0]), float32(scale[1]), float32(scale[2]))

	var trs lin.Mat4x4
	trs.Mult(&t, &rs)
	return mat4ToArray(&trs)
}

func vkTransformMatrix(m [16]float32) [12]float32 {
	return [12]float32{
		m[0], m[4], m[8], m[12],
		m[1], m[5], m[9], m[13],
		m[2], m[6], m[10], m[14],
	}
}

func gltfMatrixToArray(m [16]float64) [16]float32 {
	var out [16]float32
	for i := range out {
		out[i] = float32(m[i])
	}
	return out
}

func mat4ToArray(m *lin.Mat4x4) [16]float32 {
	var out [16]float32
	for col := 0; col < 4; col++ {
		for row := 0; row < 4; row++ {
			out[col*4+row] = m[col][row]
		}
	}
	return out
}

func setPerspectiveZO(m *lin.Mat4x4, yFov, aspect, near, far float32) {
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

func createGeometryNodesBuffer(dev vk.Device, gpu vk.PhysicalDevice, prims []primitiveData) (vk.Buffer, vk.DeviceMemory) {
	nodes := make([]geometryNode, len(prims))
	for i := range prims {
		nodes[i] = geometryNode{
			VertexBufferDeviceAddress: uint64(getBufferAddress(dev, prims[i].vertexBuf)),
			IndexBufferDeviceAddress:  uint64(getBufferAddress(dev, prims[i].indexBuf)),
			TextureIndexBaseColor:     prims[i].baseColorTex,
			TextureIndexOcclusion:     prims[i].occlusionTex,
		}
	}
	return createBufferWithAddress(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageStorageBufferBit|vk.BufferUsageShaderDeviceAddressBit),
		uint64(len(nodes))*uint64(unsafe.Sizeof(nodes[0])),
		unsafe.Pointer(&nodes[0]))
}

// --- Helper functions ---

func findMemoryType(gpu vk.PhysicalDevice, typeBits uint32, properties vk.MemoryPropertyFlags) uint32 {
	var memProps vk.PhysicalDeviceMemoryProperties
	vk.GetPhysicalDeviceMemoryProperties(gpu, &memProps)
	memProps.Deref()
	for i := uint32(0); i < memProps.MemoryTypeCount; i++ {
		memProps.MemoryTypes[i].Deref()
		if typeBits&(1<<i) != 0 && (vk.MemoryPropertyFlags(memProps.MemoryTypes[i].PropertyFlags)&properties) == properties {
			return i
		}
	}
	log.Fatal("Failed to find suitable memory type")
	return 0
}

func createBufferWithAddress(dev vk.Device, gpu vk.PhysicalDevice, usage vk.BufferUsageFlags, size uint64, data unsafe.Pointer) (vk.Buffer, vk.DeviceMemory) {
	var buf vk.Buffer
	if err := vk.Error(vk.CreateBuffer(dev, &vk.BufferCreateInfo{
		SType: vk.StructureTypeBufferCreateInfo,
		Size:  vk.DeviceSize(size),
		Usage: usage,
	}, nil, &buf)); err != nil {
		log.Fatal("CreateBuffer:", err)
	}
	var memReqs vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(dev, buf, &memReqs)
	memReqs.Deref()
	allocFlags := vk.MemoryAllocateFlagsInfo{
		SType: vk.StructureTypeMemoryAllocateFlagsInfo,
		Flags: vk.MemoryAllocateFlags(vk.MemoryAllocateDeviceAddressBit),
	}
	var mem vk.DeviceMemory
	if err := vk.Error(vk.AllocateMemory(dev, &vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  memReqs.Size,
		MemoryTypeIndex: findMemoryType(gpu, memReqs.MemoryTypeBits, vk.MemoryPropertyFlags(vk.MemoryPropertyHostVisibleBit|vk.MemoryPropertyHostCoherentBit)),
		PNext:           unsafe.Pointer(&allocFlags),
	}, nil, &mem)); err != nil {
		log.Fatal("AllocateMemory:", err)
	}
	vk.BindBufferMemory(dev, buf, mem, 0)
	if data != nil {
		var mapped unsafe.Pointer
		vk.MapMemory(dev, mem, 0, vk.DeviceSize(size), 0, &mapped)
		vk.Memcopy(mapped, unsafe.Slice((*byte)(data), int(size)))
		vk.UnmapMemory(dev, mem)
	}
	return buf, mem
}

func createDeviceLocalBuffer(dev vk.Device, gpu vk.PhysicalDevice, usage vk.BufferUsageFlags, size uint64) (vk.Buffer, vk.DeviceMemory) {
	var buf vk.Buffer
	if err := vk.Error(vk.CreateBuffer(dev, &vk.BufferCreateInfo{
		SType: vk.StructureTypeBufferCreateInfo,
		Size:  vk.DeviceSize(size),
		Usage: usage,
	}, nil, &buf)); err != nil {
		log.Fatal("CreateBuffer:", err)
	}
	var memReqs vk.MemoryRequirements
	vk.GetBufferMemoryRequirements(dev, buf, &memReqs)
	memReqs.Deref()
	allocFlags := vk.MemoryAllocateFlagsInfo{
		SType: vk.StructureTypeMemoryAllocateFlagsInfo,
		Flags: vk.MemoryAllocateFlags(vk.MemoryAllocateDeviceAddressBit),
	}
	var mem vk.DeviceMemory
	if err := vk.Error(vk.AllocateMemory(dev, &vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  memReqs.Size,
		MemoryTypeIndex: findMemoryType(gpu, memReqs.MemoryTypeBits, vk.MemoryPropertyFlags(vk.MemoryPropertyDeviceLocalBit)),
		PNext:           unsafe.Pointer(&allocFlags),
	}, nil, &mem)); err != nil {
		log.Fatal("AllocateMemory:", err)
	}
	vk.BindBufferMemory(dev, buf, mem, 0)
	return buf, mem
}

func getBufferAddress(dev vk.Device, buf vk.Buffer) vk.DeviceAddress {
	return vk.GetBufferDeviceAddress(dev, &vk.BufferDeviceAddressInfo{
		SType:  vk.StructureTypeBufferDeviceAddressInfo,
		Buffer: buf,
	})
}

func beginOneTimeCmd(dev vk.Device, cmdPool vk.CommandPool) vk.CommandBuffer {
	cmds := make([]vk.CommandBuffer, 1)
	vk.AllocateCommandBuffers(dev, &vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        cmdPool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}, cmds)
	vk.BeginCommandBuffer(cmds[0], &vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
		Flags: vk.CommandBufferUsageFlags(vk.CommandBufferUsageOneTimeSubmitBit),
	})
	return cmds[0]
}

func endOneTimeCmd(dev vk.Device, queue vk.Queue, cmdPool vk.CommandPool, cmd vk.CommandBuffer) {
	vk.EndCommandBuffer(cmd)
	var fence vk.Fence
	vk.CreateFence(dev, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &fence)
	vk.QueueSubmit(queue, 1, []vk.SubmitInfo{{
		SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1,
		PCommandBuffers: []vk.CommandBuffer{cmd},
	}}, fence)
	vk.WaitForFences(dev, 1, []vk.Fence{fence}, vk.True, 10_000_000_000)
	vk.DestroyFence(dev, fence, nil)
	vk.FreeCommandBuffers(dev, cmdPool, 1, []vk.CommandBuffer{cmd})
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

// buildBLAS creates one BLAS containing one geometry per glTF primitive.
func buildBLAS(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdPool vk.CommandPool, prims []primitiveData) (vk.Buffer, vk.DeviceMemory, vk.AccelerationStructure) {
	geometries := make([]vk.AccelerationStructureGeometry, 0, len(prims))
	primitiveCounts := make([]uint32, 0, len(prims))
	rangeInfos := make([]vk.AccelerationStructureBuildRangeInfo, 0, len(prims))
	transformMatrices := make([][12]float32, len(prims))

	for i := range prims {
		transformMatrices[i] = prims[i].transform
	}
	transformBuf, transformMem := createBufferWithAddress(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageShaderDeviceAddressBit|vk.BufferUsageAccelerationStructureBuildInputReadOnlyBit),
		uint64(len(transformMatrices))*uint64(unsafe.Sizeof(transformMatrices[0])),
		unsafe.Pointer(&transformMatrices[0]))
	transformAddr := getBufferAddress(dev, transformBuf)
	transformStride := vk.DeviceAddress(unsafe.Sizeof(transformMatrices[0]))

	for i := range prims {
		vertexAddr := getBufferAddress(dev, prims[i].vertexBuf)
		indexAddr := getBufferAddress(dev, prims[i].indexBuf)

		var trianglesData vk.AccelerationStructureGeometryTrianglesData
		trianglesData.SType = vk.StructureTypeAccelerationStructureGeometryTrianglesData
		trianglesData.VertexFormat = vk.FormatR32g32b32Sfloat
		setDeviceAddressConst(&trianglesData.VertexData, vertexAddr)
		trianglesData.VertexStride = 32 // pos3 + normal3 + uv2 = 8 floats * 4 bytes
		trianglesData.MaxVertex = prims[i].vertexCount - 1
		trianglesData.IndexType = vk.IndexTypeUint32
		setDeviceAddressConst(&trianglesData.IndexData, indexAddr)
		setDeviceAddressConst(&trianglesData.TransformData, transformAddr+vk.DeviceAddress(i)*transformStride)

		var geometry vk.AccelerationStructureGeometry
		geometry.SType = vk.StructureTypeAccelerationStructureGeometry
		geometry.GeometryType = vk.GeometryTypeTriangles
		geometry.Flags = vk.GeometryFlags(vk.GeometryOpaqueBit)
		setGeometryTriangles(&geometry.Geometry, &trianglesData)

		geometries = append(geometries, geometry)
		primitiveCounts = append(primitiveCounts, prims[i].triangleCount)
		rangeInfos = append(rangeInfos, vk.AccelerationStructureBuildRangeInfo{
			PrimitiveCount: prims[i].triangleCount,
		})
	}

	buildInfo := vk.AccelerationStructureBuildGeometryInfo{
		SType:         vk.StructureTypeAccelerationStructureBuildGeometryInfo,
		Type:          vk.AccelerationStructureTypeBottomLevel,
		Flags:         vk.BuildAccelerationStructureFlags(vk.BuildAccelerationStructurePreferFastTraceBit),
		GeometryCount: uint32(len(geometries)),
		PGeometries:   geometries,
	}

	var sizeInfo vk.AccelerationStructureBuildSizesInfo
	sizeInfo.SType = vk.StructureTypeAccelerationStructureBuildSizesInfo
	vk.GetAccelerationStructureBuildSizes(dev, vk.AccelerationStructureBuildTypeDevice, &buildInfo, &primitiveCounts[0], &sizeInfo)
	sizeInfo.Deref()
	log.Printf("BLAS size: AS=%d, scratch=%d (geometries=%d)", sizeInfo.AccelerationStructureSize, sizeInfo.BuildScratchSize, len(geometries))

	// Create AS buffer
	asBuf, asMem := createDeviceLocalBuffer(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageAccelerationStructureStorageBit|vk.BufferUsageShaderDeviceAddressBit),
		uint64(sizeInfo.AccelerationStructureSize))

	var as vk.AccelerationStructure
	if err := vk.Error(vk.CreateAccelerationStructure(dev, &vk.AccelerationStructureCreateInfo{
		SType: vk.StructureTypeAccelerationStructureCreateInfo, Buffer: asBuf,
		Size: sizeInfo.AccelerationStructureSize, Type: vk.AccelerationStructureTypeBottomLevel,
	}, nil, &as)); err != nil {
		log.Fatal("CreateAccelerationStructure (BLAS):", err)
	}

	// Scratch buffer
	scratchBuf, scratchMem := createDeviceLocalBuffer(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageStorageBufferBit|vk.BufferUsageShaderDeviceAddressBit),
		uint64(sizeInfo.BuildScratchSize))
	scratchAddr := getBufferAddress(dev, scratchBuf)

	// Create a fresh struct for the build call (PassRef caches the C struct)
	buildInfo2 := vk.AccelerationStructureBuildGeometryInfo{
		SType:                    vk.StructureTypeAccelerationStructureBuildGeometryInfo,
		Type:                     vk.AccelerationStructureTypeBottomLevel,
		Flags:                    vk.BuildAccelerationStructureFlags(vk.BuildAccelerationStructurePreferFastTraceBit),
		Mode:                     vk.BuildAccelerationStructureModeBuild,
		DstAccelerationStructure: as,
		GeometryCount:            uint32(len(geometries)),
		PGeometries:              geometries,
	}
	setDeviceAddress(&buildInfo2.ScratchData, scratchAddr)

	cmd := beginOneTimeCmd(dev, cmdPool)
	vk.CmdBuildAccelerationStructures(cmd, 1, &buildInfo2, [][]vk.AccelerationStructureBuildRangeInfo{rangeInfos})
	endOneTimeCmd(dev, queue, cmdPool, cmd)

	vk.DestroyBuffer(dev, transformBuf, nil)
	vk.FreeMemory(dev, transformMem, nil)
	vk.DestroyBuffer(dev, scratchBuf, nil)
	vk.FreeMemory(dev, scratchMem, nil)
	return asBuf, asMem, as
}

// buildTLAS creates a TLAS with one instance that references the model BLAS.
func buildTLAS(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdPool vk.CommandPool, blas vk.AccelerationStructure) (vk.Buffer, vk.DeviceMemory, vk.AccelerationStructure) {
	instanceData := make([]byte, 64)
	transform := [12]float32{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0}
	blasAddr := vk.GetAccelerationStructureDeviceAddress(dev, &vk.AccelerationStructureDeviceAddressInfo{
		SType: vk.StructureTypeAccelerationStructureDeviceAddressInfo, AccelerationStructure: blas,
	})

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

	instanceBuf, instanceMem := createBufferWithAddress(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageShaderDeviceAddressBit|vk.BufferUsageAccelerationStructureBuildInputReadOnlyBit),
		uint64(len(instanceData)), unsafe.Pointer(&instanceData[0]))
	instanceAddr := getBufferAddress(dev, instanceBuf)

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

	asBuf, asMem := createDeviceLocalBuffer(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageAccelerationStructureStorageBit|vk.BufferUsageShaderDeviceAddressBit),
		uint64(sizeInfo.AccelerationStructureSize))

	var as vk.AccelerationStructure
	if err := vk.Error(vk.CreateAccelerationStructure(dev, &vk.AccelerationStructureCreateInfo{
		SType: vk.StructureTypeAccelerationStructureCreateInfo, Buffer: asBuf,
		Size: sizeInfo.AccelerationStructureSize, Type: vk.AccelerationStructureTypeTopLevel,
	}, nil, &as)); err != nil {
		log.Fatal("CreateAccelerationStructure (TLAS):", err)
	}

	scratchBuf, scratchMem := createDeviceLocalBuffer(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageStorageBufferBit|vk.BufferUsageShaderDeviceAddressBit),
		uint64(sizeInfo.BuildScratchSize))
	scratchAddr := getBufferAddress(dev, scratchBuf)

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

	cmd := beginOneTimeCmd(dev, cmdPool)
	vk.CmdBuildAccelerationStructures(cmd, 1, &buildInfo2, [][]vk.AccelerationStructureBuildRangeInfo{rangeInfos})
	endOneTimeCmd(dev, queue, cmdPool, cmd)

	vk.DestroyBuffer(dev, scratchBuf, nil)
	vk.FreeMemory(dev, scratchMem, nil)
	vk.DestroyBuffer(dev, instanceBuf, nil)
	vk.FreeMemory(dev, instanceMem, nil)
	return asBuf, asMem, as
}

func createStorageImage(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdPool vk.CommandPool, width, height uint32, format vk.Format) (vk.Image, vk.DeviceMemory, vk.ImageView) {
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
	var mem vk.DeviceMemory
	vk.AllocateMemory(dev, &vk.MemoryAllocateInfo{
		SType: vk.StructureTypeMemoryAllocateInfo, AllocationSize: memReqs.Size,
		MemoryTypeIndex: findMemoryType(gpu, memReqs.MemoryTypeBits, vk.MemoryPropertyFlags(vk.MemoryPropertyDeviceLocalBit)),
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
	cmd := beginOneTimeCmd(dev, cmdPool)
	vk.CmdPipelineBarrier(cmd,
		vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit), vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit),
		0, 0, nil, 0, nil, 1, []vk.ImageMemoryBarrier{{
			SType: vk.StructureTypeImageMemoryBarrier, OldLayout: vk.ImageLayoutUndefined, NewLayout: vk.ImageLayoutGeneral,
			Image: img, SubresourceRange: vk.ImageSubresourceRange{AspectMask: vk.ImageAspectFlags(vk.ImageAspectColorBit), LevelCount: 1, LayerCount: 1},
			DstAccessMask:       vk.AccessFlags(vk.AccessShaderWriteBit),
			SrcQueueFamilyIndex: vk.QueueFamilyIgnored, DstQueueFamilyIndex: vk.QueueFamilyIgnored,
		}})
	endOneTimeCmd(dev, queue, cmdPool, cmd)
	return img, mem, view
}

func destroyTexture(dev vk.Device, texture textureData) {
	vk.DestroySampler(dev, texture.sampler, nil)
	vk.DestroyImageView(dev, texture.view, nil)
	vk.DestroyImage(dev, texture.image, nil)
	vk.FreeMemory(dev, texture.memory, nil)
}

func createTexture(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdPool vk.CommandPool, width, height uint32, pixels []byte, samplerInfo vk.SamplerCreateInfo) textureData {
	stagingBuf, stagingMem := createBufferWithAddress(dev, gpu, vk.BufferUsageFlags(vk.BufferUsageTransferSrcBit), uint64(len(pixels)), unsafe.Pointer(&pixels[0]))

	var img vk.Image
	if err := vk.Error(vk.CreateImage(dev, &vk.ImageCreateInfo{
		SType:     vk.StructureTypeImageCreateInfo,
		ImageType: vk.ImageType2d,
		Format:    vk.FormatR8g8b8a8Unorm,
		Extent:    vk.Extent3D{Width: width, Height: height, Depth: 1},
		MipLevels: 1, ArrayLayers: 1, Samples: vk.SampleCount1Bit,
		Tiling: vk.ImageTilingOptimal,
		Usage:  vk.ImageUsageFlags(vk.ImageUsageTransferDstBit | vk.ImageUsageSampledBit),
	}, nil, &img)); err != nil {
		log.Fatal("CreateImage (texture):", err)
	}

	var memReqs vk.MemoryRequirements
	vk.GetImageMemoryRequirements(dev, img, &memReqs)
	memReqs.Deref()
	var mem vk.DeviceMemory
	if err := vk.Error(vk.AllocateMemory(dev, &vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  memReqs.Size,
		MemoryTypeIndex: findMemoryType(gpu, memReqs.MemoryTypeBits, vk.MemoryPropertyFlags(vk.MemoryPropertyDeviceLocalBit)),
	}, nil, &mem)); err != nil {
		log.Fatal("AllocateMemory (texture):", err)
	}
	vk.BindImageMemory(dev, img, mem, 0)

	cmd := beginOneTimeCmd(dev, cmdPool)
	rangeColor := vk.ImageSubresourceRange{AspectMask: vk.ImageAspectFlags(vk.ImageAspectColorBit), LevelCount: 1, LayerCount: 1}
	vk.CmdPipelineBarrier(cmd,
		vk.PipelineStageFlags(vk.PipelineStageTopOfPipeBit), vk.PipelineStageFlags(vk.PipelineStageTransferBit),
		0, 0, nil, 0, nil, 1, []vk.ImageMemoryBarrier{{
			SType:     vk.StructureTypeImageMemoryBarrier,
			OldLayout: vk.ImageLayoutUndefined, NewLayout: vk.ImageLayoutTransferDstOptimal,
			Image: img, SubresourceRange: rangeColor,
			DstAccessMask:       vk.AccessFlags(vk.AccessTransferWriteBit),
			SrcQueueFamilyIndex: vk.QueueFamilyIgnored, DstQueueFamilyIndex: vk.QueueFamilyIgnored,
		}})
	vk.CmdCopyBufferToImage(cmd, stagingBuf, img, vk.ImageLayoutTransferDstOptimal, 1, []vk.BufferImageCopy{{
		ImageSubresource: vk.ImageSubresourceLayers{AspectMask: vk.ImageAspectFlags(vk.ImageAspectColorBit), LayerCount: 1},
		ImageExtent:      vk.Extent3D{Width: width, Height: height, Depth: 1},
	}})
	vk.CmdPipelineBarrier(cmd,
		vk.PipelineStageFlags(vk.PipelineStageTransferBit), vk.PipelineStageFlags(vk.PipelineStageFragmentShaderBit|vk.PipelineStageRayTracingShaderBit),
		0, 0, nil, 0, nil, 1, []vk.ImageMemoryBarrier{{
			SType:     vk.StructureTypeImageMemoryBarrier,
			OldLayout: vk.ImageLayoutTransferDstOptimal, NewLayout: vk.ImageLayoutShaderReadOnlyOptimal,
			Image: img, SubresourceRange: rangeColor,
			SrcAccessMask: vk.AccessFlags(vk.AccessTransferWriteBit), DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit),
			SrcQueueFamilyIndex: vk.QueueFamilyIgnored, DstQueueFamilyIndex: vk.QueueFamilyIgnored,
		}})
	endOneTimeCmd(dev, queue, cmdPool, cmd)

	vk.DestroyBuffer(dev, stagingBuf, nil)
	vk.FreeMemory(dev, stagingMem, nil)

	var view vk.ImageView
	if err := vk.Error(vk.CreateImageView(dev, &vk.ImageViewCreateInfo{
		SType:            vk.StructureTypeImageViewCreateInfo,
		Image:            img,
		ViewType:         vk.ImageViewType2d,
		Format:           vk.FormatR8g8b8a8Unorm,
		SubresourceRange: rangeColor,
	}, nil, &view)); err != nil {
		log.Fatal("CreateImageView (texture):", err)
	}

	var sampler vk.Sampler
	if err := vk.Error(vk.CreateSampler(dev, &samplerInfo, nil, &sampler)); err != nil {
		log.Fatal("CreateSampler (texture):", err)
	}

	return textureData{image: img, memory: mem, view: view, sampler: sampler}
}

func defaultSamplerCreateInfo() vk.SamplerCreateInfo {
	return vk.SamplerCreateInfo{
		SType:        vk.StructureTypeSamplerCreateInfo,
		MagFilter:    vk.FilterLinear,
		MinFilter:    vk.FilterLinear,
		MipmapMode:   vk.SamplerMipmapModeLinear,
		AddressModeU: vk.SamplerAddressModeRepeat,
		AddressModeV: vk.SamplerAddressModeRepeat,
		AddressModeW: vk.SamplerAddressModeRepeat,
		MaxLod:       0,
		BorderColor:  vk.BorderColorIntOpaqueWhite,
	}
}

func createDummyTexture(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdPool vk.CommandPool) textureData {
	return createTexture(dev, gpu, queue, cmdPool, 1, 1, []byte{255, 255, 255, 255}, defaultSamplerCreateInfo())
}

func loadGLTFTextures(dev vk.Device, gpu vk.PhysicalDevice, queue vk.Queue, cmdPool vk.CommandPool, doc *gltf.Document, baseDir string) []textureData {
	textures := make([]textureData, 0, len(doc.Textures)+1)
	textures = append(textures, createDummyTexture(dev, gpu, queue, cmdPool))
	for i, tex := range doc.Textures {
		if tex == nil || tex.Source == nil {
			log.Printf("Texture %d has no source, using fallback texture", i)
			textures = append(textures, createDummyTexture(dev, gpu, queue, cmdPool))
			continue
		}

		pixels, width, height, err := decodeGLTFTexture(doc, baseDir, *tex.Source)
		if err != nil {
			log.Fatalf("decodeGLTFTexture %d: %v", i, err)
		}
		textures = append(textures, createTexture(dev, gpu, queue, cmdPool, width, height, pixels, samplerCreateInfoForTexture(doc, tex)))
	}
	return textures
}

func decodeGLTFTexture(doc *gltf.Document, baseDir string, imageIndex int) ([]byte, uint32, uint32, error) {
	if imageIndex < 0 || imageIndex >= len(doc.Images) {
		return nil, 0, 0, fmt.Errorf("image index %d out of range", imageIndex)
	}
	imageDef := doc.Images[imageIndex]
	if imageDef == nil {
		return nil, 0, 0, fmt.Errorf("image %d is nil", imageIndex)
	}

	var raw []byte
	var err error
	switch {
	case imageDef.IsEmbeddedResource():
		raw, err = imageDef.MarshalData()
	case imageDef.URI != "":
		raw, err = os.ReadFile(filepath.Join(baseDir, imageDef.URI))
	case imageDef.BufferView != nil:
		return nil, 0, 0, fmt.Errorf("bufferView-backed images are not supported")
	default:
		return nil, 0, 0, fmt.Errorf("image %d has no data source", imageIndex)
	}
	if err != nil {
		return nil, 0, 0, err
	}

	decoded, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, 0, 0, err
	}
	bounds := decoded.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, decoded, bounds.Min, draw.Src)
	return rgba.Pix, uint32(bounds.Dx()), uint32(bounds.Dy()), nil
}

func samplerCreateInfoForTexture(doc *gltf.Document, tex *gltf.Texture) vk.SamplerCreateInfo {
	info := defaultSamplerCreateInfo()
	if tex == nil || tex.Sampler == nil || *tex.Sampler < 0 || *tex.Sampler >= len(doc.Samplers) {
		return info
	}
	sampler := doc.Samplers[*tex.Sampler]
	if sampler == nil {
		return info
	}

	info.MagFilter = magFilterFromGLTF(sampler.MagFilter)
	info.MinFilter, info.MipmapMode = minFilterFromGLTF(sampler.MinFilter)
	info.AddressModeU = addressModeFromGLTF(sampler.WrapS)
	info.AddressModeV = addressModeFromGLTF(sampler.WrapT)
	return info
}

func magFilterFromGLTF(filter gltf.MagFilter) vk.Filter {
	switch filter {
	case gltf.MagNearest:
		return vk.FilterNearest
	default:
		return vk.FilterLinear
	}
}

func minFilterFromGLTF(filter gltf.MinFilter) (vk.Filter, vk.SamplerMipmapMode) {
	switch filter {
	case gltf.MinNearest, gltf.MinNearestMipMapNearest, gltf.MinNearestMipMapLinear:
		return vk.FilterNearest, vk.SamplerMipmapModeNearest
	case gltf.MinLinearMipMapNearest:
		return vk.FilterLinear, vk.SamplerMipmapModeNearest
	default:
		return vk.FilterLinear, vk.SamplerMipmapModeLinear
	}
}

func addressModeFromGLTF(mode gltf.WrappingMode) vk.SamplerAddressMode {
	switch mode {
	case gltf.WrapClampToEdge:
		return vk.SamplerAddressModeClampToEdge
	case gltf.WrapMirroredRepeat:
		return vk.SamplerAddressModeMirroredRepeat
	default:
		return vk.SamplerAddressModeRepeat
	}
}

func createDescriptorSets(dev vk.Device, count uint32, tlas vk.AccelerationStructure, storageImageView vk.ImageView, geometryBuf vk.Buffer, textures []textureData, uniforms *asch.VulkanUniformBuffers) (vk.DescriptorSetLayout, vk.DescriptorPool, []vk.DescriptorSet) {
	textureCount := uint32(len(textures))
	if textureCount == 0 {
		log.Fatal("createDescriptorSets: texture array must contain at least the fallback texture")
	}

	immutableFallbackSampler := []vk.Sampler{textures[0].sampler}
	immutableTextureSamplers := make([]vk.Sampler, 0, len(textures))
	for _, texture := range textures {
		immutableTextureSamplers = append(immutableTextureSamplers, texture.sampler)
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
			Sampler:     texture.sampler,
			ImageView:   texture.view,
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

	raygenModule, _ := asch.LoadShaderFromBytes(dev, raygenShaderCode)
	missModule, _ := asch.LoadShaderFromBytes(dev, missShaderCode)
	shadowMissModule, _ := asch.LoadShaderFromBytes(dev, shadowMissShaderCode)
	closestHitModule, _ := asch.LoadShaderFromBytes(dev, closestHitShaderCode)
	anyHitModule, _ := asch.LoadShaderFromBytes(dev, anyHitShaderCode)

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

func alignUp(size, alignment uint32) uint32 {
	return (size + alignment - 1) &^ (alignment - 1)
}

func createSBT(dev vk.Device, gpu vk.PhysicalDevice, pipeline vk.Pipeline, handleSize, handleAlignment uint32) (vk.StridedDeviceAddressRegion, vk.StridedDeviceAddressRegion, vk.StridedDeviceAddressRegion, vk.Buffer, vk.DeviceMemory) {
	groupCount := uint32(4) // raygen, miss, shadow miss, hit
	handleSizeAligned := alignUp(handleSize, handleAlignment)
	// Read all shader group handles
	sbtSize := groupCount * handleSizeAligned
	handleStorage := make([]byte, sbtSize)
	if err := vk.Error(vk.GetRayTracingShaderGroupHandles(dev, pipeline, 0, groupCount, uint64(sbtSize), unsafe.Pointer(&handleStorage[0]))); err != nil {
		log.Fatal("GetRayTracingShaderGroupHandles:", err)
	}

	// Create single SBT buffer containing all groups
	sbtBuf, sbtMem := createBufferWithAddress(dev, gpu,
		vk.BufferUsageFlags(vk.BufferUsageShaderBindingTableBit|vk.BufferUsageShaderDeviceAddressBit),
		uint64(sbtSize), unsafe.Pointer(&handleStorage[0]))
	sbtAddr := getBufferAddress(dev, sbtBuf)

	raygenSBT := vk.StridedDeviceAddressRegion{DeviceAddress: sbtAddr, Stride: vk.DeviceSize(handleSizeAligned), Size: vk.DeviceSize(handleSizeAligned)}
	missSBT := vk.StridedDeviceAddressRegion{DeviceAddress: sbtAddr + vk.DeviceAddress(handleSizeAligned), Stride: vk.DeviceSize(handleSizeAligned), Size: vk.DeviceSize(2 * handleSizeAligned)}
	hitSBT := vk.StridedDeviceAddressRegion{DeviceAddress: sbtAddr + vk.DeviceAddress(3*handleSizeAligned), Stride: vk.DeviceSize(handleSizeAligned), Size: vk.DeviceSize(handleSizeAligned)}
	return raygenSBT, missSBT, hitSBT, sbtBuf, sbtMem
}

func createSyncObjects(dev vk.Device) (vk.Fence, vk.Semaphore, error) {
	var fence vk.Fence
	var sem vk.Semaphore
	if err := vk.Error(vk.CreateFence(dev, &vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}, nil, &fence)); err != nil {
		return fence, sem, err
	}
	if err := vk.Error(vk.CreateSemaphore(dev, &vk.SemaphoreCreateInfo{SType: vk.StructureTypeSemaphoreCreateInfo}, nil, &sem)); err != nil {
		return fence, sem, err
	}
	return fence, sem, nil
}

// Simple 4x4 matrix inverse using cofactor expansion
func invertMatrix(m *lin.Mat4x4) lin.Mat4x4 {
	var inv lin.Mat4x4
	s := [6]float32{
		m[0][0]*m[1][1] - m[1][0]*m[0][1], m[0][0]*m[1][2] - m[1][0]*m[0][2], m[0][0]*m[1][3] - m[1][0]*m[0][3],
		m[0][1]*m[1][2] - m[1][1]*m[0][2], m[0][1]*m[1][3] - m[1][1]*m[0][3], m[0][2]*m[1][3] - m[1][2]*m[0][3],
	}
	c := [6]float32{
		m[2][0]*m[3][1] - m[3][0]*m[2][1], m[2][0]*m[3][2] - m[3][0]*m[2][2], m[2][0]*m[3][3] - m[3][0]*m[2][3],
		m[2][1]*m[3][2] - m[3][1]*m[2][2], m[2][1]*m[3][3] - m[3][1]*m[2][3], m[2][2]*m[3][3] - m[3][2]*m[2][3],
	}
	det := s[0]*c[5] - s[1]*c[4] + s[2]*c[3] + s[3]*c[2] - s[4]*c[1] + s[5]*c[0]
	if math.Abs(float64(det)) < 1e-10 {
		inv.Identity()
		return inv
	}
	d := 1.0 / det
	inv[0][0] = (m[1][1]*c[5] - m[1][2]*c[4] + m[1][3]*c[3]) * d
	inv[0][1] = (-m[0][1]*c[5] + m[0][2]*c[4] - m[0][3]*c[3]) * d
	inv[0][2] = (m[3][1]*s[5] - m[3][2]*s[4] + m[3][3]*s[3]) * d
	inv[0][3] = (-m[2][1]*s[5] + m[2][2]*s[4] - m[2][3]*s[3]) * d
	inv[1][0] = (-m[1][0]*c[5] + m[1][2]*c[2] - m[1][3]*c[1]) * d
	inv[1][1] = (m[0][0]*c[5] - m[0][2]*c[2] + m[0][3]*c[1]) * d
	inv[1][2] = (-m[3][0]*s[5] + m[3][2]*s[2] - m[3][3]*s[1]) * d
	inv[1][3] = (m[2][0]*s[5] - m[2][2]*s[2] + m[2][3]*s[1]) * d
	inv[2][0] = (m[1][0]*c[4] - m[1][1]*c[2] + m[1][3]*c[0]) * d
	inv[2][1] = (-m[0][0]*c[4] + m[0][1]*c[2] - m[0][3]*c[0]) * d
	inv[2][2] = (m[3][0]*s[4] - m[3][1]*s[2] + m[3][3]*s[0]) * d
	inv[2][3] = (-m[2][0]*s[4] + m[2][1]*s[2] - m[2][3]*s[0]) * d
	inv[3][0] = (-m[1][0]*c[3] + m[1][1]*c[1] - m[1][2]*c[0]) * d
	inv[3][1] = (m[0][0]*c[3] - m[0][1]*c[1] + m[0][2]*c[0]) * d
	inv[3][2] = (-m[3][0]*s[3] + m[3][1]*s[1] - m[3][2]*s[0]) * d
	inv[3][3] = (m[2][0]*s[3] - m[2][1]*s[1] + m[2][2]*s[0]) * d
	return inv
}

func drawFrame(dev vk.Device, queue vk.Queue, s asch.VulkanSwapchainInfo, cmdBuffers []vk.CommandBuffer,
	fence vk.Fence, semaphore vk.Semaphore,
	pipeline vk.Pipeline, pipelineLayout vk.PipelineLayout,
	descSets []vk.DescriptorSet, uniforms *asch.VulkanUniformBuffers,
	storageImage vk.Image,
	raygenSBT, missSBT, hitSBT *vk.StridedDeviceAddressRegion,
	proj, view *lin.Mat4x4,
) bool {
	var nextIdx uint32
	ret := vk.AcquireNextImage(dev, s.DefaultSwapchain(), vk.MaxUint64, semaphore, vk.NullFence, &nextIdx)
	if ret != vk.Success && ret != vk.Suboptimal {
		return false
	}

	// Update uniform buffer with inverse matrices
	projInv := invertMatrix(proj)
	viewInv := invertMatrix(view)
	ubo := uniformData{ViewInverse: viewInv, ProjInverse: projInv, Frame: frameCounter}
	uniforms.Update(nextIdx, ubo.Bytes())
	frameCounter++

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
