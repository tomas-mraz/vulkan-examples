TAGS = wayland
GLSLC = glslangValidator
SHADER_DIR = shaders

SHADERS_SRC = $(wildcard $(SHADER_DIR)/*.vert $(SHADER_DIR)/*.frag)
SHADERS_SPV = $(addsuffix .spv,$(SHADERS_SRC))

build: shaders
	go build -tags $(TAGS) -o 00_test .

run: shaders
	go run -tags $(TAGS) .

shaders: $(SHADERS_SPV)

$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%
	$(GLSLC) -V $< -o $@

clean:
	rm -f 00_test $(SHADERS_SPV)
