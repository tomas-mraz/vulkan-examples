SHADER_DIR = shaders
APP = 08_kosmic_krisp
LINUX_TAGS = -tags wayland
GLSLC = glslangValidator

SHADERS_SRC = $(wildcard $(SHADER_DIR)/*.vert $(SHADER_DIR)/*.frag)
SHADERS_SPV = $(addsuffix .spv,$(SHADERS_SRC))

.PHONY: build build-linux build-apple run run-linux shaders install-tools install-tools-linux install-tools-apple clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

build-linux: shaders ## Build the Linux application
	go build $(LINUX_TAGS) -o $(APP) .

build-apple: ## Build the macOS application binary
	GOOS=darwin GOARCH=arm64 go build -o $(APP)-macos .

run-linux: ## run application on Linux
	go run $(LINUX_TAGS) .

shaders: $(SHADERS_SPV) ## Compile GLSL shaders to SPIR-V

$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%
	$(GLSLC) -V $< -o $@

install-tools-linux: ## Install Linux build tools
	sudo apt-get install -y glslang-tools

install-tools-apple: ## Install macOS build tools
	brew install glslang

install-mesa: ## Install mesa package with KosmicKrips driver
	brew install mesa

clean: ## Remove binary and compiled shaders
	rm -f $(SHADER_DIR)/*.spv

update-deps:
	go get github.com/tomas-mraz/vulkan-ash@latest
