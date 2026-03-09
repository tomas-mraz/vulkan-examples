.PHONY: build run clean help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

build: ## Build the application
	go build -o 01_version .

run: ## Build and run the application
	go run .

clean: ## Remove binary
	rm -f 01_version
