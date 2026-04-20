package main

import (
	"fmt"

	vk "github.com/tomas-mraz/vulkan"
	"github.com/tomas-mraz/vulkan-ash"
)

// drawRenderer implements ash.Renderer for the static triangle demo.
//
// Lifetime buckets:
//   - "once": vertex buffer — the three NDC vertices never change.
//   - "sized": raster pass, framebuffers, graphics pipeline. The pipeline bakes
//     in viewport/scissor so a resize must rebuild it.
type drawRenderer struct {
	// once
	vertexBuf ash.BufferResource

	// sized
	rasterPass ash.RasterizationPass
	pipeline   ash.PipelineRasterization

	onceBuilt  bool
	sizedBuilt bool
}

func (r *drawRenderer) CreateOnce(s *ash.Session) error {
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
	r.onceBuilt = true
	return nil
}

func (r *drawRenderer) DestroyOnce() {
	if !r.onceBuilt {
		return
	}
	r.vertexBuf.Destroy()
	r.onceBuilt = false
}

func (r *drawRenderer) CreateSized(s *ash.Session, extent vk.Extent2D) error {
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
	})
	if err != nil {
		return fmt.Errorf("NewPipelineRasterization: %w", err)
	}
	r.sizedBuilt = true
	return nil
}

func (r *drawRenderer) DestroySized() {
	if !r.sizedBuilt {
		return
	}
	r.pipeline.Destroy()
	r.rasterPass.Destroy()
	r.sizedBuilt = false
}

func (r *drawRenderer) Draw(s *ash.Session, f *ash.Frame) error {
	clearValues := []vk.ClearValue{vk.NewClearValue([]float32{0.098, 0.71, 0.996, 1})}
	vk.CmdBeginRenderPass(f.Cmd, &vk.RenderPassBeginInfo{
		SType:           vk.StructureTypeRenderPassBeginInfo,
		RenderPass:      r.rasterPass.GetRenderPass(),
		Framebuffer:     f.Swapchain.Framebuffers[f.ImageIndex],
		RenderArea:      vk.Rect2D{Extent: f.Extent},
		ClearValueCount: uint32(len(clearValues)),
		PClearValues:    clearValues,
	}, vk.SubpassContentsInline)

	s.CmdCtx.BindRasterPipeline(f.Cmd, r.pipeline)
	s.CmdCtx.BindVertexBuffers(f.Cmd, 0, []vk.Buffer{r.vertexBuf.Buffer}, []vk.DeviceSize{0})
	s.CmdCtx.Draw(f.Cmd, 3, 1, 0, 0)

	vk.CmdEndRenderPass(f.Cmd)
	return nil
}
