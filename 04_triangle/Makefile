TAGS = wayland
GLSLC = glslangValidator
SHADER_DIR = shaders

SHADERS_SRC = $(wildcard $(SHADER_DIR)/*.vert $(SHADER_DIR)/*.frag)
SHADERS_SPV = $(addsuffix .spv,$(SHADERS_SRC))

.PHONY: build run shaders install-tools clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

build: shaders ## Build the application
	go build -tags $(TAGS) -o rotating-triangle .

run: shaders ## Build and run the application
	go run -tags $(TAGS) .

shaders: $(SHADERS_SPV) ## Compile GLSL shaders to SPIR-V

$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%
	$(GLSLC) -V $< -o $@

install-tools: ## Install glslang-tools for shader compilation
	sudo apt-get install -y glslang-tools

clean: ## Remove binary and compiled shaders
	rm -f rotating-triangle $(SHADER_DIR)/*.spv
