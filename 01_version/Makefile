APP = 01_version

.PHONY: build build-darwin run clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

build: ## Build the application
	go build -o $(APP) .

build-macos: ## Cross-compile for macOS (Apple Darwin)
	GOOS=darwin GOARCH=arm64 go build -o $(APP)_macos .

run: ## Build and run the application
	go run .

clean: ## Remove binary
	rm -f $(APP) $(APP)_macos
