package main

import (
	"fmt"

	vk "github.com/tomas-mraz/vulkan"
	ash "github.com/tomas-mraz/vulkan-ash"
)

// ray1Renderer implements ash.Renderer for the RT triangle demo.
//
// Lifetime buckets:
//   - "once" (device lifetime): ray tracing context, static triangle geometry
//     buffers (vertex, index), and the two acceleration structures built over
//     them (BLAS + TLAS). None of these depend on the swapchain.
//   - "sized" (swapchain generation): storage image matched to the swapchain
//     extent, per-frame uniform buffers, the descriptor set that binds the
//     storage image / uniforms / TLAS, and the RT pipeline + shader binding
//     table. Pipeline and SBT go here because the pipeline layout is built
//     from the descriptor set layout — when descriptors rebuild, pipeline
//     rebuilds too. The cost is one vkCreateRayTracingPipelines per resize,
//     which for this demo is a few ms at most.
type ray1Renderer struct {
	// once
	rtContext ash.RaytracingContext
	vertexBuf ash.BufferResource
	indexBuf  ash.BufferResource
	blas      ash.AccelerationStructure
	tlas      ash.AccelerationStructure

	// sized
	storageImg ash.ImageResource
	uniforms   ash.UniformBuffers
	desc       ash.DescriptorInfo
	rtPipeline ash.PipelineRaytracing
	sbt        ash.ShaderBindingTable

	// camera state
	viewMatrix ash.Mat4x4

	onceBuilt  bool
	sizedBuilt bool
}

// Shader group handle sizes are standard on every RT-capable GPU today; the
// specification permits variation but in practice these are the de-facto
// constants, so we hard-code them to avoid a query round-trip.
const (
	shaderGroupHandleSize      = 32
	shaderGroupHandleAlignment = 32
)

// Triangle vertex data — three vertices in NDC-ish space, traced once.
var triangleVertices = []float32{
	1.0, 1.0, 0.0,
	-1.0, 1.0, 0.0,
	0.0, -1.0, 0.0,
}
var triangleIndices = []uint32{0, 1, 2}

func (r *ray1Renderer) CreateOnce(s *ash.Session) error {
	// RaytracingContext stores a pointer to s.CmdCtx. Session.recreateSwapchain
	// mutates CmdCtx through that pointer (*s.CmdCtx = newCmd), so rtContext
	// stays valid across swapchain rebuilds without needing a rebind.
	r.rtContext = ash.NewRaytracingContext(s.Manager, s.CmdCtx)

	rtUsage := vk.BufferUsageFlags(
		vk.BufferUsageShaderDeviceAddressBit |
			vk.BufferUsageAccelerationStructureBuildInputReadOnlyBit |
			vk.BufferUsageStorageBufferBit,
	)

	var err error
	r.vertexBuf, err = ash.NewBufferHostVisible(s.Manager.Device, s.Manager.Gpu, triangleVertices, true, rtUsage)
	if err != nil {
		return fmt.Errorf("vertex buffer: %w", err)
	}
	r.indexBuf, err = ash.NewBufferHostVisible(s.Manager.Device, s.Manager.Gpu, triangleIndices, true, rtUsage)
	if err != nil {
		return fmt.Errorf("index buffer: %w", err)
	}

	geom := ash.NewTriangleGeometry(ash.TriangleGeometryDesc{
		VertexAddress: r.vertexBuf.DeviceAddress,
		VertexFormat:  vk.FormatR32g32b32Sfloat,
		VertexStride:  12,
		MaxVertex:     3,
		IndexAddress:  r.indexBuf.DeviceAddress,
		IndexType:     vk.IndexTypeUint32,
		Flags:         vk.GeometryFlags(vk.GeometryOpaqueBit),
	})
	fastTrace := vk.BuildAccelerationStructureFlags(vk.BuildAccelerationStructurePreferFastTraceBit)

	r.blas, err = r.rtContext.NewBottomLevelAccelerationStructure(
		[]vk.AccelerationStructureGeometry{geom},
		[]uint32{1},
		fastTrace,
	)
	if err != nil {
		return fmt.Errorf("BLAS: %w", err)
	}

	r.tlas, err = r.rtContext.NewTopLevelAccelerationStructure([]ash.TLASInstance{{
		Transform:           [12]float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
		InstanceCustomIndex: 0,
		Mask:                0xFF,
		SBTRecordOffset:     0,
		Flags:               vk.GeometryInstanceFlags(vk.GeometryInstanceTriangleFacingCullDisableBit),
		BLAS:                &r.blas,
	}}, fastTrace)
	if err != nil {
		return fmt.Errorf("TLAS: %w", err)
	}

	r.viewMatrix.LookAt(&ash.Vec3{0, 0, -2.5}, &ash.Vec3{0, 0, 0}, &ash.Vec3{0, 1, 0})
	r.onceBuilt = true
	return nil
}

func (r *ray1Renderer) DestroyOnce() {
	if !r.onceBuilt {
		return
	}
	// Acceleration structures created through RaytracingContext are owned by the
	// context and tracked internally, so destroy them exactly once there.
	r.rtContext.Destroy()
	r.indexBuf.Destroy()
	r.vertexBuf.Destroy()
	r.onceBuilt = false
}

func (r *ray1Renderer) CreateSized(s *ash.Session, extent vk.Extent2D) error {
	swapLen := s.Swapchain.DefaultSwapchainLen()
	rtStage := vk.ShaderStageFlags(vk.ShaderStageRaygenBit)

	var err error
	r.storageImg, err = ash.NewImageStorage(
		s.Manager.Device, s.Manager.Gpu, s.Manager.Queue, s.CmdCtx.GetCmdPool(),
		extent.Width, extent.Height, s.Swapchain.DisplayFormat,
	)
	if err != nil {
		return fmt.Errorf("storage image: %w", err)
	}

	r.uniforms, err = ash.NewUniformBuffers(s.Manager.Device, s.Manager.Gpu, swapLen, uniformSize)
	if err != nil {
		return fmt.Errorf("uniforms: %w", err)
	}

	r.desc, err = ash.NewDescriptorSets(s.Manager.Device, swapLen, []ash.DescriptorBinding{
		&ash.BindingAccelerationStructure{StageFlags: rtStage, AccelerationStructure: r.tlas.AccelerationStructure},
		ash.NewBindingStorageImage(rtStage, &r.storageImg),
		&ash.BindingUniformBuffer{StageFlags: rtStage, Uniforms: &r.uniforms},
	})
	if err != nil {
		return fmt.Errorf("descriptors: %w", err)
	}

	r.rtPipeline, err = ash.NewRTPipeline(s.Manager.Device, ash.RTPipelineOptions{
		Groups: []ash.RTShaderGroup{
			{RaygenShader: raygenShaderCode},
			{MissShader: missShaderCode},
			{ClosestHitShader: closestHitShaderCode},
		},
		DescriptorSetLayouts: []vk.DescriptorSetLayout{r.desc.GetLayout()},
	})
	if err != nil {
		return fmt.Errorf("RT pipeline: %w", err)
	}

	r.sbt, err = ash.NewShaderBindingTable(
		s.Manager.Device, s.Manager.Gpu, r.rtPipeline.GetPipeline(),
		shaderGroupHandleSize, shaderGroupHandleAlignment,
		1, 1, 1, 0,
	)
	if err != nil {
		return fmt.Errorf("SBT: %w", err)
	}

	r.sizedBuilt = true
	return nil
}

func (r *ray1Renderer) DestroySized() {
	if !r.sizedBuilt {
		return
	}
	r.sbt.Destroy()
	r.rtPipeline.Destroy()
	r.desc.Destroy()
	r.uniforms.Destroy()
	r.storageImg.Destroy()
	r.sizedBuilt = false
}

func (r *ray1Renderer) Draw(s *ash.Session, f *ash.Frame) error {
	// Recompute the projection from the live extent so resize doesn't stretch
	// the rendered image. Cheap enough to do per-frame; could be cached on
	// CreateSized if this ever grew hot.
	aspect := float32(f.Extent.Width) / float32(f.Extent.Height)
	var proj ash.Mat4x4
	proj.Perspective(ash.DegreesToRadians(60.0), aspect, 0.1, 512.0)
	proj[1][1] *= -1 // flip Y for Vulkan clip space

	ubo := uniformData{
		ViewInverse: ash.InvertMatrix(&r.viewMatrix),
		ProjInverse: ash.InvertMatrix(&proj),
	}
	r.uniforms.Update(f.ImageIndex, ubo.Bytes())

	// Trace rays into the storage image, then blit to the swapchain image.
	s.CmdCtx.BindRTPipeline(f.Cmd, r.rtPipeline, r.desc.GetSets()[f.ImageIndex])
	s.CmdCtx.TraceRays(f.Cmd, &r.sbt, f.Extent)
	f.Swapchain.CmdCopyToSwapchain(f.Cmd, r.storageImg.GetImage(), f.ImageIndex)
	return nil
}
