package main

import (
	"fmt"
	"log"
	"time"

	vk "github.com/tomas-mraz/vulkan"
	"github.com/tomas-mraz/vulkan-ash"
)

const depthFormat = vk.FormatD32Sfloat

// modelRenderer implements ash.Renderer for the glTF teapot demo.
//
// Lifetime buckets:
//   - "once": loaded model, vertex + index buffers, uniform buffers, descriptor
//     sets. None depend on the swapchain extent.
//   - "sized": depth image, render pass, framebuffers, graphics pipeline.
//     Pipeline bakes in viewport/scissor and uses a depth attachment, so a
//     resize rebuilds the lot.
type modelRenderer struct {
	// once
	model     ash.Model
	vertexBuf ash.BufferResource
	indexBuf  ash.BufferResource
	uniforms  ash.UniformBuffers
	desc      ash.DescriptorInfo

	// sized
	depth      ash.ImageResource
	rasterPass ash.RasterizationPass
	pipeline   ash.PipelineRasterization

	// frame state
	projMatrix ash.Mat4x4
	viewMatrix ash.Mat4x4
	startTime  time.Time

	onceBuilt  bool
	sizedBuilt bool
}

func (r *modelRenderer) CreateOnce(s *ash.Session) error {
	model, err := ash.LoadModel(modelPath)
	if err != nil {
		return fmt.Errorf("LoadModel: %w", err)
	}
	r.model = model
	log.Printf("Loaded model: %d vertices, %d indices", model.VertexCount(), model.IndexCount())

	swapLen := s.Swapchain.DefaultSwapchainLen()

	r.vertexBuf, err = ash.NewBufferHostVisible(
		s.Manager.Device, s.Manager.Gpu,
		model.Vertices, false,
		vk.BufferUsageFlags(vk.BufferUsageVertexBufferBit),
	)
	if err != nil {
		return fmt.Errorf("vertex buffer: %w", err)
	}
	r.indexBuf, err = ash.NewBufferHostVisible(
		s.Manager.Device, s.Manager.Gpu,
		model.Indices, false,
		vk.BufferUsageFlags(vk.BufferUsageIndexBufferBit),
	)
	if err != nil {
		return fmt.Errorf("index buffer: %w", err)
	}
	r.uniforms, err = ash.NewUniformBuffers(s.Manager.Device, s.Manager.Gpu, swapLen, uboSize)
	if err != nil {
		return fmt.Errorf("uniforms: %w", err)
	}
	r.desc, err = ash.NewDescriptorSets(s.Manager.Device, swapLen, []ash.DescriptorBinding{
		&ash.BindingUniformBuffer{StageFlags: vk.ShaderStageFlags(vk.ShaderStageVertexBit), Uniforms: &r.uniforms},
	})
	if err != nil {
		return fmt.Errorf("descriptors: %w", err)
	}

	r.viewMatrix.LookAt(&ash.Vec3{0.0, 3.0, 8.0}, &ash.Vec3{0.0, 0.0, 0.0}, &ash.Vec3{0.0, 1.0, 0.0})
	r.startTime = time.Now()
	r.onceBuilt = true
	return nil
}

func (r *modelRenderer) DestroyOnce() {
	if !r.onceBuilt {
		return
	}
	r.desc.Destroy()
	r.uniforms.Destroy()
	r.indexBuf.Destroy()
	r.vertexBuf.Destroy()
	r.onceBuilt = false
}

func (r *modelRenderer) CreateSized(s *ash.Session, extent vk.Extent2D) error {
	var err error
	r.depth, err = ash.NewImageDepth(s.Manager.Device, s.Manager.Gpu, extent.Width, extent.Height, depthFormat)
	if err != nil {
		return fmt.Errorf("NewImageDepth: %w", err)
	}
	r.rasterPass, err = ash.NewRasterPassWithDepth(s.Manager.Device, s.Swapchain.DisplayFormat, depthFormat)
	if err != nil {
		return fmt.Errorf("NewRasterPassWithDepth: %w", err)
	}
	if err := s.Swapchain.CreateFramebuffers(r.rasterPass.GetRenderPass(), r.depth.GetView()); err != nil {
		return fmt.Errorf("CreateFramebuffers: %w", err)
	}
	r.pipeline, err = ash.NewPipelineRasterization(s.Manager.Device, extent, r.rasterPass.GetRenderPass(), ash.PipelineOptions{
		VertShaderData: vertShaderCode,
		FragShaderData: fragShaderCode,
		VertexBindings: []vk.VertexInputBindingDescription{{
			Binding: 0, Stride: 6 * 4, InputRate: vk.VertexInputRateVertex,
		}},
		VertexAttributes: []vk.VertexInputAttributeDescription{
			{Binding: 0, Location: 0, Format: vk.FormatR32g32b32Sfloat, Offset: 0},
			{Binding: 0, Location: 1, Format: vk.FormatR32g32b32Sfloat, Offset: 3 * 4},
		},
		DescriptorSetLayouts: []vk.DescriptorSetLayout{r.desc.GetLayout()},
		DepthTestEnable:      true,
	})
	if err != nil {
		return fmt.Errorf("NewPipelineRasterization: %w", err)
	}

	aspect := float32(extent.Width) / float32(extent.Height)
	r.projMatrix.Perspective(ash.DegreesToRadians(45.0), aspect, 0.1, 100.0)
	r.projMatrix[1][1] *= -1
	r.sizedBuilt = true
	return nil
}

func (r *modelRenderer) DestroySized() {
	if !r.sizedBuilt {
		return
	}
	r.pipeline.Destroy()
	r.rasterPass.Destroy()
	r.depth.Destroy()
	r.sizedBuilt = false
}

func (r *modelRenderer) Draw(s *ash.Session, f *ash.Frame) error {
	// Animate: 45°/s around Y.
	elapsed := float32(time.Since(r.startTime).Seconds()) * 45.0
	var model, rotated ash.Mat4x4
	model.Identity()
	rotated.Dup(&model)
	model.Rotate(&rotated, 0.0, 1.0, 0.0, ash.DegreesToRadians(elapsed))

	var VP, MVP ash.Mat4x4
	VP.Mult(&r.projMatrix, &r.viewMatrix)
	MVP.Mult(&VP, &model)
	ubo := uboData{MVP: MVP, Model: model}
	r.uniforms.Update(f.ImageIndex, ubo.Bytes())

	clearValues := make([]vk.ClearValue, 2)
	clearValues[0].SetColor([]float32{0.1, 0.1, 0.12, 1.0})
	clearValues[1].SetDepthStencil(1.0, 0)

	vk.CmdBeginRenderPass(f.Cmd, &vk.RenderPassBeginInfo{
		SType:           vk.StructureTypeRenderPassBeginInfo,
		RenderPass:      r.rasterPass.GetRenderPass(),
		Framebuffer:     f.Swapchain.Framebuffers[f.ImageIndex],
		RenderArea:      vk.Rect2D{Extent: f.Extent},
		ClearValueCount: 2,
		PClearValues:    clearValues,
	}, vk.SubpassContentsInline)

	vk.CmdBindPipeline(f.Cmd, vk.PipelineBindPointGraphics, r.pipeline.GetPipeline())
	vk.CmdBindDescriptorSets(f.Cmd, vk.PipelineBindPointGraphics, r.pipeline.GetLayout(), 0, 1,
		[]vk.DescriptorSet{r.desc.GetSets()[f.ImageIndex]}, 0, nil)
	vk.CmdBindVertexBuffers(f.Cmd, 0, 1, []vk.Buffer{r.vertexBuf.Buffer}, []vk.DeviceSize{0})
	vk.CmdBindIndexBuffer(f.Cmd, r.indexBuf.Buffer, 0, vk.IndexTypeUint32)
	vk.CmdDrawIndexed(f.Cmd, uint32(len(r.model.Indices)), 1, 0, 0, 0)
	vk.CmdEndRenderPass(f.Cmd)
	return nil
}
