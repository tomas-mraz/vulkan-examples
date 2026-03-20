# vulkan-examples

Examples of using [tomas-mraz/vulkan](https://github.com/tomas-mraz/vulkan) Go binding and [tomas-mraz/vulkan-ash](https://github.com/tomas-mraz/vulkan-ash) framework.

## Examples

| # | Directory | Description |
|---|-----------|-------------|
| 01 | `01_version` | Print Vulkan API and loader version |
| 02 | `02_info` | Print GPU properties, extensions, layers |
| 03 | `03_draw` | Static red triangle (GLFW, pre-recorded command buffers) |
| 04 | `04_triangle` | Rotating blue triangle (GLFW, push constants) |
| 05 | `05_cube` | Textured rotating cube with gopher texture (GLFW, uniform buffer, depth buffer, descriptor sets) |
| 06 | `06_model` | Rotating 3D teapot model from glTF file (GLFW, indexed drawing, depth buffer, directional light) |
| 07 | `07_model_textured` | Rotating textured teacup from GLB file (GLFW, texture sampling, indexed drawing, depth buffer) |

## Requirements

- Go 1.25+
- Vulkan SDK (runtime + validation layers)
- GLFW 3.3+ (Wayland) — for graphical examples
- glslang-tools — for shader compilation

## Build

Each example has its own `go.mod` and `Makefile`. To build and run:

```bash
cd 04_triangle
make run
```

To see available targets:

```bash
make help
```

# Assets and licences

DiffuseTransmissionTeacup model using Diffuse Transmission extension.
Credit:
© 2023, Public domain. CC0 1.0 Universal
- Polyhaven.com, and Eric Chadwick for Models and Textures


