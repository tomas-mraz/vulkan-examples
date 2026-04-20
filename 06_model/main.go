package main

import (
	_ "embed"
	"log/slog"
	"unsafe"

	"github.com/tomas-mraz/vulkan-ash"
)

//go:embed shaders/model.vert.spv
var vertShaderCode []byte

//go:embed shaders/model.frag.spv
var fragShaderCode []byte

const (
	appName   = "glTF Model Viewer"
	modelPath = "teapot.gltf"
)

// uboData matches the vertex shader's UBO layout.
type uboData struct {
	MVP   ash.Mat4x4
	Model ash.Mat4x4
}

const uboSize = int(unsafe.Sizeof(uboData{}))

func (u *uboData) Bytes() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(u)), uboSize)
}

func main() {
	slog.SetLogLoggerLevel(slog.LevelDebug)
	ash.SetDebug(false)
	start()
}
