TAGS = wayland
GLSLC = glslangValidator
SHADER_DIR = shaders

SHADERS_SRC = $(wildcard $(SHADER_DIR)/*.rgen $(SHADER_DIR)/*.rmiss $(SHADER_DIR)/*.rchit $(SHADER_DIR)/*.rahit)
SHADERS_SPV = $(addsuffix .spv,$(SHADERS_SRC))

build: shaders
	go build -tags $(TAGS) -o 12_ray2 .

run: shaders
	go run -tags $(TAGS) .

shaders: $(SHADERS_SPV)

$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%
	$(GLSLC) -I$(SHADER_DIR) -V --target-env vulkan1.2 $< -o $@

clean:
	rm -f 12_ray2 $(SHADERS_SPV)
