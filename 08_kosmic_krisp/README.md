# Kosmic Krisp

Kopie `04_triangle` upravená tak, aby stejná aplikace fungovala i na Apple macOS přes MoltenVK portability vrstvu.

Vykresluje rotující modrý trojúhelník na černém pozadí pomocí GLFW, `github.com/tomas-mraz/vulkan` a `github.com/tomas-mraz/vulkan-ash`.

## Co je jiné proti `04_triangle`

- na macOS přidává `VK_KHR_portability_enumeration` a `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR`
- pokud GPU nabízí `VK_KHR_portability_subset`, zařízení ho při vytváření zapne
- validační layer se zapne jen tehdy, když je skutečně dostupná
- `Makefile` má oddělené targety pro Linux a macOS; Linux build používá `wayland` tag, macOS ne

## Požadavky

- Go 1.25+
- Vulkan runtime
- GLFW 3.3+
- `glslangValidator` pro kompilaci shaderů

Na macOS je potřeba mít funkční Vulkan loader + MoltenVK.

## Sestavení a spuštění

```bash
make run
```

Nebo zvlášť:

```bash
make shaders
make build
./08_kosmic_krisp
```

macOS binárku sestaví:

```bash
make build-apple
```
