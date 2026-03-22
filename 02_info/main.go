package main

import (
	"fmt"
	"log"

	vk "github.com/tomas-mraz/vulkan"
	asch "github.com/tomas-mraz/vulkan-ash"
	"github.com/xlab/tablewriter"
)

func main() {
	if err := vk.SetDefaultGetInstanceProcAddr(); err != nil {
		log.Fatal(err)
	}
	if err := vk.Init(); err != nil {
		log.Fatal(err)
	}

	asch.SetDebug(false)
	device, err := asch.NewDevice("VulkanInfo", nil, createHeadlessSurface, 0)
	if err != nil {
		log.Fatal(err)
	}
	defer destroyDevice(device)

	gpuDevices, err := getPhysicalDevices(device.Instance)
	if err != nil {
		log.Fatal(err)
	}

	var loaderVersion uint32
	if err := vk.Error(vk.EnumerateInstanceVersion(&loaderVersion)); err != nil {
		log.Fatal(err)
	}

	printInfo(device, gpuDevices, loaderVersion)
}

func createHeadlessSurface(vk.Instance, uintptr) (vk.Surface, error) {
	return vk.NullSurface, nil
}

func destroyDevice(device asch.Vulkan) {
	vk.DestroyDevice(device.Device, nil)
	if device.Surface != vk.NullSurface {
		vk.DestroySurface(device.Instance, device.Surface, nil)
	}
	vk.DestroyInstance(device.Instance, nil)
}

func getPhysicalDevices(instance vk.Instance) ([]vk.PhysicalDevice, error) {
	var count uint32
	if err := vk.Error(vk.EnumeratePhysicalDevices(instance, &count, nil)); err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, fmt.Errorf("no GPUs found")
	}
	devices := make([]vk.PhysicalDevice, count)
	if err := vk.Error(vk.EnumeratePhysicalDevices(instance, &count, devices)); err != nil {
		return nil, err
	}
	return devices, nil
}

func getInstanceExtensions() []string {
	var count uint32
	vk.EnumerateInstanceExtensionProperties("", &count, nil)
	props := make([]vk.ExtensionProperties, count)
	vk.EnumerateInstanceExtensionProperties("", &count, props)
	names := make([]string, 0, count)
	for _, p := range props {
		p.Deref()
		names = append(names, vk.ToString(p.ExtensionName[:]))
	}
	return names
}


func getInstanceLayers() []string {
	var count uint32
	vk.EnumerateInstanceLayerProperties(&count, nil)
	props := make([]vk.LayerProperties, count)
	vk.EnumerateInstanceLayerProperties(&count, props)
	names := make([]string, 0, count)
	for _, p := range props {
		p.Deref()
		names = append(names, vk.ToString(p.LayerName[:]))
	}
	return names
}

func getDeviceLayers(gpu vk.PhysicalDevice) []string {
	var count uint32
	vk.EnumerateDeviceLayerProperties(gpu, &count, nil)
	props := make([]vk.LayerProperties, count)
	vk.EnumerateDeviceLayerProperties(gpu, &count, props)
	names := make([]string, 0, count)
	for _, p := range props {
		p.Deref()
		names = append(names, vk.ToString(p.LayerName[:]))
	}
	return names
}

func physicalDeviceType(t vk.PhysicalDeviceType) string {
	switch t {
	case vk.PhysicalDeviceTypeIntegratedGpu:
		return "Integrated GPU"
	case vk.PhysicalDeviceTypeDiscreteGpu:
		return "Discrete GPU"
	case vk.PhysicalDeviceTypeVirtualGpu:
		return "Virtual GPU"
	case vk.PhysicalDeviceTypeCpu:
		return "CPU"
	default:
		return "Other"
	}
}

func printInfo(device asch.Vulkan, gpuDevices []vk.PhysicalDevice, loaderVersion uint32) {
	var props vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(device.GpuDevice, &props)
	props.Deref()

	table := tablewriter.CreateTable()
	table.UTF8Box()
	table.AddTitle("VULKAN PROPERTIES")
	table.AddRow("Physical Device Name", vk.ToString(props.DeviceName[:]))
	table.AddRow("Physical Device Vendor", fmt.Sprintf("%x", props.VendorID))
	if props.DeviceType != vk.PhysicalDeviceTypeOther {
		table.AddRow("Physical Device Type", physicalDeviceType(props.DeviceType))
	}
	table.AddRow("Physical GPUs", len(gpuDevices))
	table.AddRow("Vulkan Loader Version", vk.Version(loaderVersion))
	table.AddRow("API Version", vk.Version(props.ApiVersion))
	table.AddRow("Driver Version", props.DriverVersion)

	if device.Surface != vk.NullSurface {
		var caps vk.SurfaceCapabilities
		vk.GetPhysicalDeviceSurfaceCapabilities(device.GpuDevice, device.Surface, &caps)
		caps.Deref()
		caps.CurrentExtent.Deref()
		caps.MinImageExtent.Deref()
		caps.MaxImageExtent.Deref()

		table.AddSeparator()
		table.AddRow("Image count", fmt.Sprintf("%d - %d", caps.MinImageCount, caps.MaxImageCount))
		table.AddRow("Array layers", caps.MaxImageArrayLayers)
		table.AddRow("Image size (current)", fmt.Sprintf("%dx%d", caps.CurrentExtent.Width, caps.CurrentExtent.Height))
		table.AddRow("Image size (extent)", fmt.Sprintf("%dx%d - %dx%d",
			caps.MinImageExtent.Width, caps.MinImageExtent.Height,
			caps.MaxImageExtent.Width, caps.MaxImageExtent.Height))
		table.AddRow("Usage flags", fmt.Sprintf("%02x", caps.SupportedUsageFlags))
		table.AddRow("Current transform", fmt.Sprintf("%02x", caps.CurrentTransform))
		table.AddRow("Allowed transforms", fmt.Sprintf("%02x", caps.SupportedTransforms))
		var formatCount uint32
		vk.GetPhysicalDeviceSurfaceFormats(device.GpuDevice, device.Surface, &formatCount, nil)
		table.AddRow("Surface formats", formatCount)
		table.AddSeparator()
	}

	table.AddSeparator()
	table.AddRow("INSTANCE EXTENSIONS", "")
	for i, name := range getInstanceExtensions() {
		table.AddRow(i+1, name)
	}

	table.AddSeparator()
	table.AddRow("DEVICE EXTENSIONS", "")
	for i, name := range asch.GetDeviceExtensions(device.GpuDevice) {
		table.AddRow(i+1, name)
	}

	if layers := getInstanceLayers(); len(layers) > 0 {
		table.AddSeparator()
		table.AddRow("INSTANCE LAYERS", "")
		for i, name := range layers {
			table.AddRow(i+1, name)
		}
	}

	if layers := getDeviceLayers(device.GpuDevice); len(layers) > 0 {
		table.AddSeparator()
		table.AddRow("DEVICE LAYERS", "")
		for i, name := range layers {
			table.AddRow(i+1, name)
		}
	}

	fmt.Println("\n" + table.Render())
}
