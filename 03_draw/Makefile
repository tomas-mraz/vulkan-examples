TAGS = wayland
GLSLC = glslangValidator
SHADER_DIR = shaders

SHADERS_SRC = $(wildcard $(SHADER_DIR)/*.vert $(SHADER_DIR)/*.frag)
SHADERS_SPV = $(SHADER_DIR)/tri-vert.spv $(SHADER_DIR)/tri-frag.spv

.PHONY: build run shaders install-tools clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

build: shaders ## Build the application
	go build -tags $(TAGS) -o 03_draw .

run: shaders ## Build and run the application
	go run -tags $(TAGS) .

shaders: $(SHADERS_SPV) ## Compile GLSL shaders to SPIR-V

$(SHADER_DIR)/tri-vert.spv: $(SHADER_DIR)/tri.vert
	$(GLSLC) -V $< -o $@

$(SHADER_DIR)/tri-frag.spv: $(SHADER_DIR)/tri.frag
	$(GLSLC) -V $< -o $@

install-tools: ## Install glslang-tools for shader compilation
	sudo apt-get install -y glslang-tools

clean: ## Remove binary and compiled shaders
	rm -f 03_draw $(SHADER_DIR)/*.spv
