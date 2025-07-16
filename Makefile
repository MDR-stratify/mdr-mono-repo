# MDR Stratify Makefile
.PHONY: help install dev build test docker-build docker-up docker-down clean

help: ## Show this help message
	@echo "MDR Stratify - AI-driven MDR Prediction System"
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}\' $(MAKEFILE_LIST)

install: ## Install dependencies
	@echo "Installing dependencies..."
	npm install

dev: ## Start development server
	@echo "Starting development server..."
	npm run dev

build: ## Build the application
	@echo "Building application..."
	npm run build

test: ## Run tests
	@echo "Running tests..."
	npm run test

docker-build: ## Build Docker images
	@echo "Building Docker images..."
	docker compose build

docker-up: ## Start Docker containers with live logs
	@echo "Starting Docker containers..."
	docker compose up --build

docker-down: ## Stop Docker containers
	@echo "Stopping Docker containers..."
	docker compose down

docker-dev: ## Start Docker containers in development mode with rebuild and live logs
	@echo "Starting Docker containers in development mode..."
	docker compose up --build

docker-rebuild: ## Force rebuild and start containers
	@echo "Rebuilding and starting containers..."
	docker compose down
	docker compose build --no-cache
	docker compose up

docker-logs: ## View Docker logs
	@echo "Viewing Docker logs..."
	docker compose logs -f
	@echo "Cleaning build artifacts..."
	rm -rf .next
	rm -rf node_modules
	rm -rf dist

setup: install ## Complete setup for development
	@echo "Setting up development environment..."
	@echo "MDR Stratify setup complete!"