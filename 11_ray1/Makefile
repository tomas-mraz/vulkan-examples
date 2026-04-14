APP = 11_ray1
SHADER_DIR = shaders
LINUX_TAGS = -tags wayland
GLSLC = glslangValidator

SHADERS_SRC = $(wildcard $(SHADER_DIR)/*.rgen $(SHADER_DIR)/*.rmiss $(SHADER_DIR)/*.rchit $(SHADER_DIR)/*.rahit)
SHADERS_SPV = $(addsuffix .spv,$(SHADERS_SRC))

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

build: shaders ## Build the Linux application binary
	go build $(LINUX_TAGS) -o $(APP) .

build-macos: shaders ## Build for macOS ARM64 (must run on macOS)
	@test "$$(go env GOOS)" = "darwin" || (echo "Error: build-macos must run on macOS"; exit 1)
	@test "$$(go env GOARCH)" = "arm64" || (echo "Error: build-macos requires arm64"; exit 1)
	go build -o $(APP)_macos .

run: shaders ## Run application on Linux
	go run $(LINUX_TAGS) .

shaders: $(SHADERS_SPV) ## Compile GLSL shaders to SPIR-V

$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%
	$(GLSLC) -I$(SHADER_DIR) -V --target-env vulkan1.2 $< -o $@

clean: ## Remove binary and compiled shaders
	rm -f $(APP) $(SHADER_DIR)/*.spv
