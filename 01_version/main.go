package main

import (
	"fmt"
	"log"
	"log/slog"

	vk "github.com/tomas-mraz/vulkan"
)

func main() {
	slog.SetLogLoggerLevel(slog.LevelDebug)

	if err := vk.SetDefaultGetInstanceProcAddr(); err != nil {
		log.Fatal(err)
	}
	if err := vk.Init(); err != nil {
		log.Fatal(err)
	}

	var loaderVersion uint32
	if err := vk.Error(vk.EnumerateInstanceVersion(&loaderVersion)); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Vulkan loader API version: %s\n", vk.Version(loaderVersion))

	// Create minimal instance (no extensions, no layers)
	var instance vk.Instance
	if err := vk.Error(vk.CreateInstance(&vk.InstanceCreateInfo{
		SType: vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo: &vk.ApplicationInfo{
			SType:      vk.StructureTypeApplicationInfo,
			ApiVersion: vk.MakeVersion(1, 0, 0),
		},
	}, nil, &instance)); err != nil {
		log.Fatal(err)
	}
	defer vk.DestroyInstance(instance, nil)
	if err := vk.InitInstance(instance); err != nil {
		log.Fatal(err)
	}

	// Enumerate physical devices
	var count uint32
	if err := vk.Error(vk.EnumeratePhysicalDevices(instance, &count, nil)); err != nil {
		log.Fatal(err)
	}
	if count == 0 {
		fmt.Println("No Vulkan physical devices found")
		return
	}
	gpus := make([]vk.PhysicalDevice, count)
	if err := vk.Error(vk.EnumeratePhysicalDevices(instance, &count, gpus)); err != nil {
		log.Fatal(err)
	}

	for i, gpu := range gpus[:count] {
		var props vk.PhysicalDeviceProperties
		vk.GetPhysicalDeviceProperties(gpu, &props)
		props.Deref()
		fmt.Printf("GPU %d: %s, API %s, driver=%d, vendor=%d, device=%d\n",
			i,
			vk.ToString(props.DeviceName[:]),
			vk.Version(props.ApiVersion),
			props.DriverVersion,
			props.VendorID,
			props.DeviceID,
		)
	}
}
