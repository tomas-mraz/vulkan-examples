package main

import (
	_ "embed"
	"log/slog"

	"github.com/tomas-mraz/vulkan-ash"
)

//go:embed shaders/tri-vert.spv
var vertShaderCode []byte

//go:embed shaders/tri-frag.spv
var fragShaderCode []byte

const appName = "VulkanDraw"

func main() {
	slog.SetLogLoggerLevel(slog.LevelDebug)
	ash.SetDebug(false)
	start()
}

var triangleVertices = []float32{
	-1, -1, 0,
	1, -1, 0,
	0, 1, 0,
}
