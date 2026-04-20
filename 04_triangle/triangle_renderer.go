package main

import (
	"fmt"
	"time"
	"unsafe"

	vk "github.com/tomas-mraz/vulkan"
	"github.com/tomas-mraz/vulkan-ash"
)

// triangleRenderer implements ash.Renderer for the rotating-triangle demo.
//
// Lifetime buckets:
//   - "once": static vertex buffer.
//   - "sized": raster pass, framebuffers, graphics pipeline. Pipeline bakes in
//     viewport/scissor, so a resize rebuilds it.
type triangleRenderer struct {
	// once
	vertexBuf ash.BufferResource

	// sized
	rasterPass ash.RasterizationPass
	pipeline   ash.PipelineRasterization

	startTime  time.Time
	onceBuilt  bool
	sizedBuilt bool
}

func (r *triangleRenderer) CreateOnce(s *ash.Session) error {
	buf, err := ash.NewBufferHostVisible(
		s.Manager.Device,
		s.Manager.Gpu,
		triangleVertices,
		false,
		vk.BufferUsageFlags(vk.BufferUsageVertexBufferBit),
	)
	if err != nil {
		return fmt.Errorf("vertex buffer: %w", err)
	}
	r.vertexBuf = buf
	r.startTime = time.Now()
	r.onceBuilt = true
	return nil
}

func (r *triangleRenderer) DestroyOnce() {
	if !r.onceBuilt {
		return
	}
	r.vertexBuf.Destroy()
	r.onceBuilt = false
}

func (r *triangleRenderer) CreateSized(s *ash.Session, extent vk.Extent2D) error {
	var err error
	r.rasterPass, err = ash.NewRasterPass(s.Manager.Device, s.Swapchain.DisplayFormat)
	if err != nil {
		return fmt.Errorf("NewRasterPass: %w", err)
	}
	if err := s.Swapchain.CreateFramebuffers(r.rasterPass.GetRenderPass(), vk.NullImageView); err != nil {
		return fmt.Errorf("CreateFramebuffers: %w", err)
	}
	r.pipeline, err = ash.NewPipelineRasterization(s.Manager.Device, extent, r.rasterPass.GetRenderPass(), ash.PipelineOptions{
		VertShaderData: vertShaderCode,
		FragShaderData: fragShaderCode,
		PushConstantRanges: []vk.PushConstantRange{{
			StageFlags: vk.ShaderStageFlags(vk.ShaderStageVertexBit),
			Offset:     0,
			Size:       4,
		}},
	})
	if err != nil {
		return fmt.Errorf("NewPipelineRasterization: %w", err)
	}
	r.sizedBuilt = true
	return nil
}

func (r *triangleRenderer) DestroySized() {
	if !r.sizedBuilt {
		return
	}
	r.pipeline.Destroy()
	r.rasterPass.Destroy()
	r.sizedBuilt = false
}

func (r *triangleRenderer) Draw(s *ash.Session, f *ash.Frame) error {
	clearValues := []vk.ClearValue{vk.NewClearValue([]float32{0, 0, 0, 1})}
	vk.CmdBeginRenderPass(f.Cmd, &vk.RenderPassBeginInfo{
		SType:           vk.StructureTypeRenderPassBeginInfo,
		RenderPass:      r.rasterPass.GetRenderPass(),
		Framebuffer:     f.Swapchain.Framebuffers[f.ImageIndex],
		RenderArea:      vk.Rect2D{Extent: f.Extent},
		ClearValueCount: uint32(len(clearValues)),
		PClearValues:    clearValues,
	}, vk.SubpassContentsInline)

	angle := float32(time.Since(r.startTime).Seconds())
	s.CmdCtx.BindRasterPipeline(f.Cmd, r.pipeline)
	vk.CmdPushConstants(f.Cmd, r.pipeline.GetLayout(), vk.ShaderStageFlags(vk.ShaderStageVertexBit), 0, 4, unsafe.Pointer(&angle))
	s.CmdCtx.BindVertexBuffers(f.Cmd, 0, []vk.Buffer{r.vertexBuf.Buffer}, []vk.DeviceSize{0})
	s.CmdCtx.Draw(f.Cmd, 3, 1, 0, 0)

	vk.CmdEndRenderPass(f.Cmd)
	return nil
}
