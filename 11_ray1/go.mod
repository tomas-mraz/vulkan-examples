module vulkan-examples/10_ray1

go 1.25.7

require (
	github.com/go-gl/glfw/v3.3/glfw v0.0.0-20260406072232-3ac4aa2bb164
	github.com/tomas-mraz/vulkan v0.0.0-20260407131029-2ba7ceb4ab82
	github.com/tomas-mraz/vulkan-ash v0.0.0-20260407135407-79ba9690eaa8
)

require github.com/qmuntal/gltf v0.28.0 // indirect

replace github.com/tomas-mraz/vulkan-ash => /home/tomas/git-osobni-github/vulkan-ash

replace github.com/tomas-mraz/vulkan => /home/tomas/git-osobni-github/vulkan-goki_fork
