APP = 05_cube
SHADER_DIR = shaders
LINUX_TAGS = -tags wayland
GLSLC = glslangValidator

SHADERS_SRC = $(wildcard $(SHADER_DIR)/*.vert $(SHADER_DIR)/*.frag)
SHADERS_SPV = $(addsuffix .spv,$(SHADERS_SRC))

# Android NDK toolchain (set NDK to your NDK root, or use standalone toolchain)
ANDROID_API ?= 34
NDK_TOOLCHAIN ?= $(ANDROID_HOME)/ndk/$(shell ls $(ANDROID_HOME)/ndk 2>/dev/null | tail -1)/toolchains/llvm/prebuilt/linux-x86_64/bin

JNILIBS = android/app/src/main/jniLibs

.PHONY: build build-macos build-android build-android-arm64 build-android-amd64 apk bundle install run shaders clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

build: shaders ## Build the Linux application binary
	go build $(LINUX_TAGS) -o $(APP) .

build-macos: shaders ## Build for macOS ARM64 (must run on macOS)
	@test "$$(go env GOOS)" = "darwin" || (echo "Error: build-macos must run on macOS"; exit 1)
	@test "$$(go env GOARCH)" = "arm64" || (echo "Error: build-macos requires arm64"; exit 1)
	go build -o $(APP)_macos .

build-android-arm64: shaders ## Build Android ARM64 shared library
	mkdir -p $(JNILIBS)/arm64-v8a
	CC="$(NDK_TOOLCHAIN)/aarch64-linux-android$(ANDROID_API)-clang" \
	CXX="$(NDK_TOOLCHAIN)/aarch64-linux-android$(ANDROID_API)-clang++" \
	GOOS=android \
	GOARCH=arm64 \
	CGO_ENABLED=1 \
	go build -buildmode=c-shared -o $(JNILIBS)/arm64-v8a/lib$(APP).so .

build-android-amd64: shaders ## Build Android x86_64 shared library
	mkdir -p $(JNILIBS)/x86_64
	CC="$(NDK_TOOLCHAIN)/x86_64-linux-android$(ANDROID_API)-clang" \
	CXX="$(NDK_TOOLCHAIN)/x86_64-linux-android$(ANDROID_API)-clang++" \
	GOOS=android \
	GOARCH=amd64 \
	CGO_ENABLED=1 \
	go build -buildmode=c-shared -o $(JNILIBS)/x86_64/lib$(APP).so .

build-android: build-android-arm64 build-android-amd64 ## Build Android shared libraries (ARM64 + x86_64)

apk: build-android ## Build Android debug APK via Gradle
	cd android && ./gradlew assembleDebug

bundle: build-android ## Build Android AAB for Google Play
	cd android && ./gradlew bundleRelease

install: apk ## Install debug APK on connected device
	cd android && ./gradlew installDebug

run: shaders ## Run application on Linux
	go run $(LINUX_TAGS) .

shaders: $(SHADERS_SPV) ## Compile GLSL shaders to SPIR-V

$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%
	$(GLSLC) -V $< -o $@

clean: ## Remove binary, compiled shaders, and Android build artifacts
	rm -f $(APP) $(SHADER_DIR)/*.spv
	rm -rf $(JNILIBS)/arm64-v8a $(JNILIBS)/x86_64
	cd android && ./gradlew clean 2>/dev/null || true

android-install:
	cd ./android/app/build/outputs/apk/debug && adb install app-x86_64-debug.apk

android-remove:
	adb uninstall com.vulkanexamples.cube || true

android-run:
	adb shell monkey -p com.example.myapp -c android.intent.category.LAUNCHER 1
