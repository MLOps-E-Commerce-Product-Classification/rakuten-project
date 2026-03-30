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
	uv run dvc init
	git add .dvc .dvcignore
	git commit -m "chore: initialize DVC"

.PHONY: dvc-add-data
dvc-add-data:
	uv run dvc add data/raw/
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
	uv run dvc repro train-text

	# Ergebnisse committen
	git add dvc.lock
	git commit -m "exp: $(GIT_BRANCH) - $$(date '+%Y-%m-%d %H:%M') [$(DEVICE)]" || true

.PHONY: dvc-push
dvc-push:
	uv run dvc push
	git push

.PHONY: dvc-pull
dvc-pull:
	git pull
	uv run dvc pull

.PHONY: dvc-metrics
dvc-metrics:
	dvc metrics show
	dvc metrics diff HEAD~1

.PHONY: dvc-run
dvc-run: dvc-repro dvc-push


# ============================================================
# Evaluation
# ============================================================
#
# Check if a variable is defined; otherwise, exit with an error
# Usage: $(call check_defined, VARNAME)
check_defined = \
    $(strip $(foreach 1,$1, \
        $(if $(value $1),, $(error Variable $(1) is required. Example: make evaluate-run $(1)=your_run_id))))

X_DATA    ?= data/processed/val.csv
Y_DATA    ?= data/processed/val.csv
WEIGHTS   ?= models/best_text_model.pt
ENCODING  ?= configs/label_encoding.json

.PHONY: evaluate-build
evaluate-build:
	docker compose build \
		--build-arg DEVICE=$(DEVICE) \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
		evaluate-text

.PHONY: evaluate-run
evaluate-run:
	$(call check_defined, MLFLOW_ID)
	# Inject environment variables for docker-compose interpolation
	DEVICE=$(DEVICE) \
	GIT_COMMIT=$(GIT_COMMIT) \
	GIT_BRANCH=$(GIT_BRANCH) \
	docker compose run --rm evaluate-text \
		--mlflow_run_id $(MLFLOW_ID) \
		--x_data_csv_path $(X_DATA) \
		--y_data_csv_path $(Y_DATA) \
		--model_weights_path $(WEIGHTS) \
		--label_encoding_path $(ENCODING)

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
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║              MLOps Rakuten — Makefile Reference              ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Usage:  make <target> [DEVICE=cpu|cu121]"
	@echo ""
	@echo "────────────────────────── Training ───────────────────────────"
	@echo "  train-text-build    Build the training Docker image"
	@echo "  train-text-run      Auto-commit configs (if changed) + run training"
	@echo "  train-text-rebuild  Build + run training in one step"
	@echo "  train-text-stop     Stop the running training container"
	@echo "  train-text-down     Stop + remove container (keep image)"
	@echo "  train-text-clean    Remove container + image"
	@echo "  train-text-logs     Follow live training logs"
	@echo ""
	@echo "──────────────────────────── DVC ──────────────────────────────"
	@echo "  dvc-init            Initialize DVC in the repository"
	@echo "  dvc-add-data        Track raw data directory with DVC"
	@echo "  dvc-repro           Reproduce pipeline + auto-commit results"
	@echo "  dvc-push            Push data & models to remote + git push"
	@echo "  dvc-pull            git pull + dvc pull"
	@echo "  dvc-metrics         Show metrics + diff against HEAD~1"
	@echo "  dvc-run             dvc-repro + dvc-push (full experiment run)"
	@echo ""
	@echo "─────────────────────────── Evaluation ────────────────────────"
	@echo "  evaluate-build      Build the evaluation Docker image"
	@echo "  evaluate-run        Run evaluation  (requires MLFLOW_ID=<id>)"
	@echo "                        Optional: X_DATA= Y_DATA= WEIGHTS= ENCODING="
	@echo ""
	@echo "─────────────────────────── Inference ─────────────────────────"
	@echo "  inference-build     Build the inference Docker image"
	@echo "  inference-run       Run inference  (TEXT='...' optional)"
	@echo "  inference-batch     Run inference for multiple hardcoded texts"
	@echo "  inference-rebuild   Build + run inference in one step"
	@echo "  inference-clean     Remove the inference Docker image"
	@echo ""
	@echo "────────────────────────── Variables ──────────────────────────"
	@echo "  DEVICE              Target device: cpu (default) | cu121"
	@echo "  TEXT                Input text for inference (default: sample)"
	@echo "  MLFLOW_ID           MLflow run ID (required for evaluate-run)"
	@echo ""
	@echo "  Example:  make train-text-build DEVICE=cu121"
	@echo "            make evaluate-run MLFLOW_ID=abc123"
	@echo ""
