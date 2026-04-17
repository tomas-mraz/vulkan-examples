package main

import (
	_ "embed"
	"unsafe"

	ash "github.com/tomas-mraz/vulkan-ash"
)

//go:embed shaders/cube.vert.spv
var vertShaderCode []byte

//go:embed shaders/cube.frag.spv
var fragShaderCode []byte

//go:embed textures/gopher.png
var gopherPng []byte

const appName = "VulkanCube"

// uniformData matches the shader's uniform buffer layout.
type uniformData struct {
	MVP      ash.Mat4x4
	Position [36][4]float32
	Attr     [36][4]float32
}

const uniformDataSize = int(unsafe.Sizeof(uniformData{}))

func (u *uniformData) Bytes() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(u)), uniformDataSize)
}

func main() {
	ash.SetDebug(false)
	ash.SetValidations(false)

	start()
}

// Cube vertex data (36 vertices = 12 triangles = 6 faces)
var gVertexBufferData = []float32{
	-1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1,
	-1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1,
	-1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1,
	-1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1,
	1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1,
	-1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1,
}

var gUVBufferData = []float32{
	0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1,
	1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
	1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
	1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
	1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0,
	0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0,
}
