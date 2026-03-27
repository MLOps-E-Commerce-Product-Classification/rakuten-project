# ============================================================
# Variables
# ============================================================

GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

export GIT_COMMIT
export GIT_BRANCH

# ============================================================
# Text Training
# ============================================================
.PHONY: train-text-build
train-text-build:
	docker compose build \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		train-text

.PHONY: train-text-run
train-text-run:
	docker compose up train-text

.PHONY: train-text
train-text: train-text-build train-text-run

.PHONY: train-text-stop
train-text-stop:
	docker compose stop train-text

.PHONY: train-text-down
train-text-down:
	docker compose down --remove-orphans train-text

.PHONY: train-text-clean
train-text-clean:
	docker compose down --rmi local --volumes --remove-orphans
	docker image rm train-text 2>/dev/null || true

# ============================================================
# Logs
# ============================================================
.PHONY: train-text-logs
train-text-logs:
	docker compose logs -f train-text

# ============================================================
# Help
# ============================================================
.PHONY: help
help:
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "  train-text-build   Build the training image"
	@echo "  train-text-run     Run the training container"
	@echo "  train-text         Build + Run (shortcut)"
	@echo "  train-text-stop    Stop the training container"
	@echo "  train-text-down    Stop + remove container"
	@echo "  train-text-clean   Stop + remove container + image + volumes"
	@echo "  train-text-logs    Follow training logs"
	@echo ""
