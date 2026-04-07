.PHONY: build run clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

build: ## Build the application
	go build -tags wayland -o 02_info .

run: ## Build and run the application
	go run -tags wayland .

clean: ## Remove binary
	rm -f 02_info
