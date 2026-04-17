package main

import (
	_ "embed"
	"log"
	"runtime"
	"unsafe"

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

// uniformData matches the raygen shader's UBO layout. Both matrices are
// inverted on the CPU and consumed directly in the shader to un-project
// ray-gen screen samples into world space.
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
	ash.SetDebug(false)

	// Chain the ray tracing / acceleration structure / buffer-device-address
	// feature toggles as pNext into the device create info. The order of
	// chaining doesn't matter to Vulkan but we keep AS at the head since it
	// is what the Session ultimately hands to vkCreateDevice via PNextChain.
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

	host := ash.NewDesktopHost(windowWidth, windowHeight, appName)
	sess := ash.NewSession(host, appName, &ash.SessionOptions{
		// VK_GOOGLE_display_timing is Android-only; nothing to gain on desktop
		// RT, and the CheckDeviceExtensions guard in Session would skip it
		// anyway — keep the signal explicit here.
		EnableTiming: false,
		DeviceOptions: &ash.DeviceOptions{
			DeviceExtensions: ash.RaytracingDeviceExtensions(),
			PNextChain:       unsafe.Pointer(&asFeatures),
			ApiVersion:       vk.MakeVersion(1, 2, 0),
		},
	})

	if err := sess.Run(&ray1Renderer{}); err != nil {
		log.Fatal(err)
	}
}
