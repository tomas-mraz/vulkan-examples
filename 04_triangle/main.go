package main

import (
	_ "embed"
	"math"

	"github.com/tomas-mraz/vulkan-ash"
)

//go:embed shaders/triangle.vert.spv
var vertShaderCode []byte

//go:embed shaders/triangle.frag.spv
var fragShaderCode []byte

const appName = "Rotating Triangle"

func main() {
	ash.SetLogDebug()
	ash.SetDebug(false)
	start()
}

var triangleVertices = func() []float32 {
	r := float32(0.5)
	return []float32{
		0, -r, 0,
		r * float32(math.Sin(2*math.Pi/3)), -r * float32(math.Cos(2*math.Pi/3)), 0,
		r * float32(math.Sin(4*math.Pi/3)), -r * float32(math.Cos(4*math.Pi/3)), 0,
	}
}()
