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
# DVC
# ============================================================

.PHONY: dvc-init
dvc-init:
	dvc init
	git add .dvc .dvcignore
	git commit -m "chore: initialize DVC"

.PHONY: dvc-add-data
dvc-add-data:
	dvc add data/raw/X_train_update.csv data/raw/Y_train_CVw08PX.csv
	git add data/raw/*.dvc data/raw/.gitignore
	git commit -m "chore: track raw data with DVC"

.PHONY: dvc-repro
dvc-repro:
	# Commit config changes only if there are any (wie in train-text-run)
	git diff --quiet configs/ || \
	(git add configs/ && git commit -m "exp: config update - $(shell date '+%Y-%m-%d %H:%M')")

	# DVC prüft ob sich deps geändert haben und führt ggf. docker compose run aus
	GIT_COMMIT=$$(git rev-parse HEAD) \
	GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) \
	DEVICE=$(DEVICE) \
	dvc repro train-text

	# Ergebnisse committen
	git add dvc.lock
	git commit -m "exp: $(GIT_BRANCH) - $$(date '+%Y-%m-%d %H:%M') [$(DEVICE)]" || true

.PHONY: dvc-push
dvc-push:
	dvc push
	git push

.PHONY: dvc-pull
dvc-pull:
	git pull
	dvc pull

.PHONY: dvc-metrics
dvc-metrics:
	dvc metrics show
	dvc metrics diff HEAD~1

.PHONY: dvc-run
dvc-run: dvc-repro dvc-push

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

.PHONY: inference-clean
inference-clean:
	docker image rm mlops-rakuten-inference 2>/dev/null || true

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
	@echo "  inference-build     Build the inference image"
	@echo "  inference-run       Run inference (TEXT='...' optional)"
	@echo "  inference-batch     Run inference for multiple texts"
	@echo "  inference-rebuild   Build + run inference"
	@echo "  inference-clean     Remove inference image"
	@echo ""
