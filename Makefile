# ============================================================
# Variables
# ============================================================

DEVICE ?= cpu  # override via: make train-text-build DEVICE=cu121

GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

export GIT_COMMIT
export GIT_BRANCH
export DEVICE

# ============================================================
# Text Training
# ============================================================

.PHONY: train-text-build
train-text-build:
	docker compose build \
		--build-arg DEVICE=$(DEVICE) \
		--build-arg GIT_COMMIT=$$(git rev-parse HEAD) \
		--build-arg GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) \
		train-text

.PHONY: train-text-run
train-text-run:
	# Commit config changes only if there are any
	git diff --quiet configs/ || \
	(git add configs/ && git commit -m "exp: config update - $(shell date '+%Y-%m-%d %H:%M')")

	# Re-evaluate commit AFTER potential commit
	GIT_COMMIT=$$(git rev-parse HEAD) \
	GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) \
	DEVICE=$(DEVICE) \
	docker compose run --rm train-text

.PHONY: train-text-rebuild
train-text-rebuild:
	$(MAKE) train-text-build
	$(MAKE) train-text-run

.PHONY: train-text-stop
train-text-stop:
	docker compose stop train-text

.PHONY: train-text-down
train-text-down:
	docker compose down --remove-orphans

.PHONY: train-text-clean
train-text-clean:
	docker compose down --remove-orphans
	docker image rm train-text 2>/dev/null || true

# ============================================================
# Inference
# ============================================================

# Standard text for fast testing
TEXT ?= "Jeu vidéo action PS4"

.PHONY: inference-build
inference-build:
	docker compose build inference

.PHONY: inference-run
inference-run:
	docker compose run --rm inference --text $(TEXT)

.PHONY: inference-batch
inference-batch:
	docker compose run --rm inference --texts "T-shirt" "Console" "Livre"

.PHONY: inference-rebuild
inference-rebuild:
	$(MAKE) inference-build
	$(MAKE) inference-run

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
	@echo "  train-text-build    Build the training image"
	@echo "  train-text-run      Commit (if needed) + run training (ephemeral)"
	@echo "  train-text-rebuild  Build + run training"
	@echo "  train-text-stop     Stop the container"
	@echo "  train-text-down     Stop + remove container"
	@echo "  train-text-clean    Remove container + image (safe)"
	@echo "  train-text-logs     Follow logs"
	@echo ""
