# Rotating Triangle

Vulkan aplikace vykreslující rotující modrý trojúhelník na černém pozadí.

Využívá [vulkan-ash](https://github.com/tomas-mraz/vulkan-ash) framework pro inicializaci Vulkan zařízení a swapchain, [vulkan](https://github.com/tomas-mraz/vulkan) Go binding a GLFW pro správu okna.

## Požadavky

- Go 1.25+
- Vulkan SDK (runtime + validační vrstvy)
- GLFW 3.3+ (Wayland)
- glslang-tools (kompilace shaderů)

### Instalace nástrojů

```bash
make install-tools
```

## Sestavení a spuštění

```bash
make build
./rotating-triangle
```

nebo přímo:

```bash
make run
```

## Struktura

```
04_triangles/
├── main.go                      # hlavní aplikace
├── shaders/
│   ├── triangle.vert            # vertex shader (rotace přes push constant)
│   ├── triangle.frag            # fragment shader (modrá barva)
│   ├── triangle.vert.spv        # zkompilovaný SPIR-V
│   └── triangle.frag.spv        # zkompilovaný SPIR-V
├── Makefile
├── go.mod
└── README.md
```

## Jak to funguje

- Trojúhelník je definován jako vertex buffer v Go kódu (3 vrcholy rovnostranného trojúhelníku)
- Rotace probíhá v GPU přes push constant předávaný vertex shaderu každý snímek
- Fragment shader vrací konstantní modrou barvu `vec4(0, 0, 1, 1)`
- Černé pozadí je nastaveno jako clear color render passu
