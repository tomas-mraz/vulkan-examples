package main

import (
	"fmt"
	"log"

	vk "github.com/tomas-mraz/vulkan"
)

func main() {
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

	instance, err := createInstance(loaderVersion)
	if err != nil {
		log.Fatal(err)
	}
	defer vk.DestroyInstance(instance, nil)

	if err := vk.InitInstance(instance); err != nil {
		log.Fatal(err)
	}

	devices, err := enumeratePhysicalDevices(instance)
	if err != nil {
		log.Fatal(err)
	}
	if len(devices) == 0 {
		fmt.Println("No Vulkan physical devices found")
		return
	}

	for i, dev := range devices {
		fmt.Printf(
			"GPU %d: %s, API %s, driver=%d, vendor=%d, device=%d\n",
			i,
			dev.Name,
			dev.APIVersion,
			dev.DriverVersion,
			dev.VendorID,
			dev.DeviceID,
		)
	}
}

type deviceInfo struct {
	Name          string
	APIVersion    vk.Version
	DriverVersion uint32
	VendorID      uint32
	DeviceID      uint32
}

func createInstance(apiVersion uint32) (vk.Instance, error) {
	appInfo := &vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   []byte("api_version_example\x00"),
		ApplicationVersion: vk.MakeVersion(1, 0, 0),
		PEngineName:        []byte("none\x00"),
		EngineVersion:      vk.MakeVersion(1, 0, 0),
		ApiVersion:         apiVersion,
	}

	createInfo := &vk.InstanceCreateInfo{
		SType:            vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo: appInfo,
	}

	var instance vk.Instance
	if err := vk.Error(vk.CreateInstance(createInfo, nil, &instance)); err != nil {
		return instance, err
	}
	return instance, nil
}

func enumeratePhysicalDevices(instance vk.Instance) ([]deviceInfo, error) {
	var deviceCount uint32
	if err := vk.Error(vk.EnumeratePhysicalDevices(instance, &deviceCount, nil)); err != nil {
		return nil, err
	}
	if deviceCount == 0 {
		return nil, nil
	}

	physicalDevices := make([]vk.PhysicalDevice, deviceCount)
	if err := vk.Error(vk.EnumeratePhysicalDevices(instance, &deviceCount, physicalDevices)); err != nil {
		return nil, err
	}

	devices := make([]deviceInfo, 0, deviceCount)
	for _, physicalDevice := range physicalDevices[:deviceCount] {
		var props vk.PhysicalDeviceProperties
		vk.GetPhysicalDeviceProperties(physicalDevice, &props)
		props.Deref()

		devices = append(devices, deviceInfo{
			Name:          vk.ToString(props.DeviceName[:]),
			APIVersion:    vk.Version(props.ApiVersion),
			DriverVersion: props.DriverVersion,
			VendorID:      props.VendorID,
			DeviceID:      props.DeviceID,
		})
	}

	return devices, nil
}
