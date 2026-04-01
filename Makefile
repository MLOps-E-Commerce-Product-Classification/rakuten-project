# ============================================================
# Variables
# ============================================================

DEVICE ?= cpu  # override via: make train-text-build DEVICE=cu121

GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

PORT ?= 3000
BENTO_SERVICE ?= src.serving.bento_service:TextBentoService
BASE_URL ?= http://127.0.0.1:$(PORT)

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
	git add configs/
	git commit -m "exp: start training run - $(shell date '+%Y-%m-%d %H:%M')" || true
	GIT_COMMIT=$(GIT_COMMIT) GIT_BRANCH=$(GIT_BRANCH) docker compose up train-text

.PHONY: train-text-rebuild
train-text-rebuild:
	git add configs/
	git commit -m "exp: start training run - $(shell date '+%Y-%m-%d %H:%M')" || true
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
<<<<<<< HEAD
# Bento serving / packaging
=======
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
>>>>>>> main
# ============================================================
.PHONY: prepare-bento-text-assets
prepare-bento-text-assets:
	uv run python -m src.serving.prepare_bento_assets

.PHONY: register-bento-text-model
register-bento-text-model: prepare-bento-text-assets
	uv run python -m src.serving.register_model

.PHONY: check-bento-text-model
check-bento-text-model:
	@uv run bentoml models get rakuten_text_classifier:latest >/dev/null 2>&1 || (echo "BentoML model rakuten_text_classifier:latest is not registered. Run 'make register-bento-text-model' first."; exit 1)

.PHONY: serve-bento-text
serve-bento-text: check-bento-text-model
	TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 uv run bentoml serve $(BENTO_SERVICE) --port $(PORT)

.PHONY: token-bento-text
token-bento-text:
	@curl -s -X POST "$(BASE_URL)/login" \
		-H "Content-Type: application/json" \
		-d '{"credentials":{"username":"user123","password":"password123"}}'

.PHONY: predict-bento-text
predict-bento-text:
	@token=$$(curl -s -X POST "$(BASE_URL)/login" \
		-H "Content-Type: application/json" \
		-d '{"credentials":{"username":"user123","password":"password123"}}' | python -c 'import sys,json; print(json.load(sys.stdin)["token"])'); \
	curl -s -X POST "$(BASE_URL)/predict" \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer $$token" \
		-d '{"input_data":{"designation":"robe femme","description":"bleu","top_k":3}}'; \
	echo

.PHONY: build-bento-text
build-bento-text: register-bento-text-model
	uv run bentoml build

.PHONY: containerize-bento-text
containerize-bento-text: build-bento-text
	uv run bentoml containerize rakuten_text_service:latest

.PHONY: docker-bento-up
docker-bento-up: containerize-bento-text
	docker compose up bento-text-service

.PHONY: docker-bento-down
docker-bento-down:
	docker compose stop bento-text-service

# ============================================================
# Logs
# ============================================================

.PHONY: train-text-logs
train-text-logs:
	docker compose logs -f train-text


.PHONY: bento-text-logs
bento-text-logs:
	docker compose logs -f bento-text-service

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
<<<<<<< HEAD
	@echo "  prepare-bento-text-assets     Download tokenizer/config assets for Bento registration"
	@echo "  register-bento-text-model     Save the trained text model in the BentoML Model Store"
	@echo "  serve-bento-text              Run the BentoML text service locally"
	@echo "  token-bento-text              Get a JWT token from /login"
	@echo "  predict-bento-text            Call the protected /predict endpoint"
	@echo "  build-bento-text              Register model + build a Bento artifact"
	@echo "  containerize-bento-text       Build and containerize the latest Bento"
	@echo "  docker-bento-up               Start Bento service in Docker Compose"
	@echo "  docker-bento-down             Stop Bento service in Docker Compose"
	@echo "  bento-text-logs               Follow Bento service logs"
=======
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
>>>>>>> main
	@echo ""
