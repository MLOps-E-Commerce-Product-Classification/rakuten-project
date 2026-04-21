# ============================================================
# Variables
# ============================================================

DEVICE ?= cpu
PORT ?= 3000
TEXT ?= "Jeu vidéo action PS4"

GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo "unknown")

BASE_URL ?= http://127.0.0.1:$(PORT)

MLFLOW_MODEL_NAME ?= text-classifier
MLFLOW_ALIAS ?= production

PROMOTION_METRIC ?= eval_macro_f1
PROMOTION_MARGIN ?= 0.0

BENTO_MODEL_NAME ?= text-classifier

export DEVICE
export GIT_BRANCH
export GIT_COMMIT

# ============================================================
# Compose files (SIMPLIFIED)
# ============================================================

COMPOSE_DEV_FULL = docker compose -f docker/compose/docker-compose.dev.yaml
COMPOSE_PROD_FULL = docker compose -f docker/compose/docker-compose.prod.yaml


# ============================================================
# Development Environment
# ============================================================

.PHONY: dev-up
dev-up:
	$(COMPOSE_DEV_FULL) up -d --build

.PHONY: dev-down
dev-down:
	$(COMPOSE_DEV_FULL) down

.PHONY: dev-restart
dev-restart:
	$(COMPOSE_DEV_FULL) down
	$(COMPOSE_DEV_FULL) up -d --build

.PHONY: dev-logs
dev-logs:
	$(COMPOSE_DEV_FULL) logs -f


# ============================================================
# Infrastructure (now part of dev compose)
# ============================================================

.PHONY: infra-up
infra-up:
	$(COMPOSE_DEV_FULL) up -d postgres redis prometheus grafana

.PHONY: infra-down
infra-down:
	$(COMPOSE_DEV_FULL) stop postgres redis prometheus grafana

.PHONY: infra-logs
infra-logs:
	$(COMPOSE_DEV_FULL) logs -f postgres redis prometheus grafana


# ============================================================
# Training
# ============================================================

.PHONY: train
train:
	git add src/ configs/ || true
	git commit -m "exp: training run $(shell date '+%Y-%m-%d %H:%M')" || true
	$(COMPOSE_DEV_FULL) run --rm train-text

.PHONY: finetune
finetune:
	git add src/ configs/ || true
	git commit -m "exp: finetuning run $(shell date '+%Y-%m-%d %H:%M')" || true
	$(COMPOSE_DEV_FULL) run --rm finetune-text

.PHONY: train-logs
train-logs:
	$(COMPOSE_DEV_FULL) logs -f train-text


# ============================================================
# Evaluation
# ============================================================

check_defined = \
$(strip $(foreach 1,$1,$(if $(value $1),,$(error Variable $(1) required))))

X_DATA ?= data/processed/val.csv
Y_DATA ?= data/processed/val.csv
WEIGHTS ?= models/best_text_model.pt
ENCODING ?= configs/label_encoding.json

.PHONY: evaluate
evaluate:
	$(call check_defined,MLFLOW_ID)
	$(COMPOSE_DEV_FULL) run --rm evaluate-text \
	--mlflow_run_id $(MLFLOW_ID) \
	--x_data_csv_path $(X_DATA) \
	--y_data_csv_path $(Y_DATA) \
	--model_weights_path $(WEIGHTS) \
	--label_encoding_path $(ENCODING)


# ============================================================
# Inference (DEV only if needed)
# ============================================================

.PHONY: inference
inference:
	$(COMPOSE_DEV_FULL) run --rm bentoml --text $(TEXT)

.PHONY: inference-batch
inference-batch:
	$(COMPOSE_DEV_FULL) run --rm bentoml --texts "T-shirt" "Console" "Livre"


# ============================================================
# BentoML
# ============================================================

.PHONY: prepare-bento
prepare-bento:
	uv run python -m src.serving.prepare_bento_assets

.PHONY: promote-model
promote-model:
	uv run python -m src.serving.promote_mlflow_model \
	--model-name $(MLFLOW_MODEL_NAME) \
	--alias $(MLFLOW_ALIAS) \
	--metric-name $(PROMOTION_METRIC) \
	--min-improvement $(PROMOTION_MARGIN)

.PHONY: sync-bento
sync-bento:
	uv run python -m src.serving.sync_mlflow_to_bento \
	--model-name $(MLFLOW_MODEL_NAME) \
	--alias $(MLFLOW_ALIAS) \
	--bento-model-name $(BENTO_MODEL_NAME)

.PHONY: build-bento
build-bento: sync-bento
	uv run bentoml build

.PHONY: containerize-bento
containerize-bento: build-bento
	uv run bentoml containerize rakuten_text_service:latest


# ============================================================
# Serving
# ============================================================

.PHONY: serve
serve:
	$(COMPOSE_DEV_FULL) up -d bentoml

.PHONY: serve-stop
serve-stop:
	$(COMPOSE_DEV_FULL) stop bentoml

.PHONY: serve-logs
serve-logs:
	$(COMPOSE_DEV_FULL) logs -f bentoml


# ============================================================
# Production
# ============================================================

.PHONY: prod-up
prod-up:
	$(COMPOSE_PROD_FULL) pull
	$(COMPOSE_PROD_FULL) up -d

.PHONY: prod-down
prod-down:
	$(COMPOSE_PROD_FULL) down

.PHONY: prod-logs
prod-logs:
	$(COMPOSE_PROD_FULL) logs -f

.PHONY: prod-restart
prod-restart:
	$(COMPOSE_PROD_FULL) down
	$(COMPOSE_PROD_FULL) up -d


# ============================================================
# Monitoring
# ============================================================

.PHONY: monitoring
monitoring:
	@echo "Grafana: http://localhost:3001"
	@echo "Prometheus: http://localhost:9090"


# ============================================================
# Utilities
# ============================================================

.PHONY: status
status:
	docker ps

.PHONY: clean
clean:
	docker system prune -f


# ============================================================
# Help
# ============================================================

.PHONY: help
help:
	@echo ""
	@echo "Rakuten MLOps Makefile"
	@echo ""
	@echo "Development:"
	@echo "  make dev-up"
	@echo "  make dev-down"
	@echo ""
	@echo "Training:"
	@echo "  make train"
	@echo "  make finetune"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate MLFLOW_ID=<id>"
	@echo ""
	@echo "Serving:"
	@echo "  make serve"
	@echo ""
	@echo "Production:"
	@echo "  make prod-up"
	@echo "  make prod-down"
	@echo ""