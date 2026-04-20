package main

import (
	_ "embed"

	"github.com/tomas-mraz/vulkan-ash"
)

//go:embed shaders/tri-vert.spv
var vertShaderCode []byte

//go:embed shaders/tri-frag.spv
var fragShaderCode []byte

const appName = "VulkanDraw"

func main() {
	ash.SetDebug(false)
	start()
}

var triangleVertices = []float32{
	-1, -1, 0,
	1, -1, 0,
	0, 1, 0,
}
