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
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		--build-arg GIT_BRANCH=$(GIT_BRANCH) \
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
# DVC
# ============================================================

.PHONY: dvc-init
dvc-init:
	uv run dvc init
	git add .dvc .dvcignore
	git commit -m "chore: initialize DVC"

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
# Bento serving / packaging
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
		-d '{"credentials":{"username":"user123","password":"password123"}}' \
		| python -c 'import sys,json; print(json.load(sys.stdin)["token"])'); \
	curl -s -X POST "$(BASE_URL)/predict" \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer $$token" \
		-d '{"input_data":{"designation":"robe femme","description":"bleu","top_k":3}}'

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
	@echo "║           MLOps Rakuten — Makefile Reference              ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Usage:  make <target> [DEVICE=cpu|cu121] [MLFLOW_ID=...]"
	@echo ""
	@echo "────────────────────────── Data & DVC ──────────────────────────"
	@echo "  dvc-init             Initialize DVC & Git tracking"
	@echo "  dvc-add-data         Track raw data directory with DVC"
	@echo "  dvc-repro            Reproduce pipeline & auto-commit lockfile"
	@echo "  dvc-run              Full flow: dvc-repro + dvc-push"
	@echo "  dvc-push/pull        Sync data/models with remote storage"
	@echo "  dvc-metrics          Show performance metrics & diff vs HEAD~1"
	@echo ""
	@echo "─────────────────────────── Training ───────────────────────────"
	@echo "  train-text-build     Build the training Docker image"
	@echo "  train-text-run       Auto-commit configs + start training run"
	@echo "  train-text-rebuild   Rebuild image and start training"
	@echo "  train-text-stop      Stop the training container"
	@echo "  train-text-down      Stop & remove containers + networks"
	@echo "  train-text-clean     Remove containers AND Docker images"
	@echo "  train-text-logs      Follow live training logs"
	@echo ""
	@echo "────────────────────────── Evaluation ──────────────────────────"
	@echo "  evaluate-build       Build the evaluation Docker image"
	@echo "  evaluate-run         Run evaluation (requires MLFLOW_ID=<id>)"
	@echo "                       Options: X_DATA=, Y_DATA=, WEIGHTS="
	@echo ""
	@echo "─────────────────────────── Inference ──────────────────────────"
	@echo "  inference-build      Build the inference Docker image"
	@echo "  inference-run        Single test run (default or TEXT='...')"
	@echo "  inference-batch      Test with multiple hardcoded samples"
	@echo "  inference-rebuild    Rebuild and run inference immediately"
	@echo "  inference-clean      Remove the inference Docker image"
	@echo ""
	@echo "────────────────────── BentoML Serving/Deployment ───────────────"
	@echo "  prepare-bento-assets Download tokenizer/configs for registration"
	@echo "  register-bento-model Save the trained model to BentoML store"
	@echo "  serve-bento-text     Run BentoML service locally (Port 3000)"
	@echo "  predict-bento-text   Test request against running service"
	@echo "  token-bento-text     Generate JWT token via /login"
	@echo "  build-bento-text     Build the Bento artifact"
	@echo "  containerize-bento   Package the Bento into a Docker image"
	@echo "  docker-bento-up/down Start/Stop Bento service via Docker Compose"
	@echo "  bento-text-logs      Follow Bento service logs"
	@echo ""
	@echo "─────────────────────────── Variables ──────────────────────────"
	@echo "  DEVICE     Target device: cpu (default) | cu121"
	@echo "  PORT       Local port for Bento service (default: 3000)"
	@echo "  MLFLOW_ID  Required ID for 'evaluate-run'"
	@echo "  TEXT       Custom input string for 'inference-run'"
	@echo ""
	@echo "Example: make dvc-run DEVICE=cu121"
	@echo "         make evaluate-run MLFLOW_ID=abc-123"
	@echo ""
