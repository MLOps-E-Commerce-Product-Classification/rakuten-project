# ============================================================
# Variables
# ============================================================

DEVICE ?= cpu
PORT ?= 3000
TEXT ?= "Jeu vidéo action PS4"

GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo "unknown")

MLFLOW_MODEL_NAME ?= text-classifier
MLFLOW_ALIAS ?= production

PROMOTION_METRIC ?= eval_macro_f1
PROMOTION_MARGIN ?= 0.0

BENTO_MODEL_NAME ?= text-classifier

export DEVICE
export GIT_BRANCH
export GIT_COMMIT


# ============================================================
# Compose files (3-LAYER ARCHITECTURE)
# ============================================================

COMPOSE_BASE = docker compose
INFRA = -f docker-compose.infrastructure.yaml
DEV = -f docker-compose.dev.yaml
PROD = -f docker-compose.prod.yaml

# Combined stacks
DEV_STACK = $(COMPOSE_BASE) $(INFRA) $(DEV)
PROD_STACK = $(COMPOSE_BASE) $(INFRA) $(PROD)


# ============================================================
# INFRASTRUCTURE LAYER
# ============================================================

.PHONY: infra-up
infra-up:
	$(COMPOSE_BASE) $(INFRA) up -d

.PHONY: infra-down
infra-down:
	$(COMPOSE_BASE) $(INFRA) down

.PHONY: infra-logs
infra-logs:
	$(COMPOSE_BASE) $(INFRA) logs -f

.PHONY: infra-restart
infra-restart:
	$(COMPOSE_BASE) $(INFRA) down
	$(COMPOSE_BASE) $(INFRA) up -d


# ============================================================
# DEV ENVIRONMENT (infra + dev overrides + builds)
# ============================================================

.PHONY: dev-build
dev-build:
	$(DEV_STACK) build \
		--build-arg DEVICE=$(DEVICE) \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		--build-arg GIT_BRANCH=$(GIT_BRANCH)
	$(MAKE) containerize-bento

.PHONY: dev-up
dev-up:
	$(DEV_STACK) up

.PHONY: dev-down
dev-down:
	$(DEV_STACK) down

.PHONY: dev-restart
dev-restart:
	$(DEV_STACK) down
	$(DEV_STACK) up -d --build

.PHONY: dev-logs
dev-logs:
	$(DEV_STACK) logs -f

.PHONY: train-text-run
train-text-run:
	git add src/ configs/ || true
	git commit -m "exp: start training run - $(shell date '+%Y-%m-%d %H:%M')" || true
	$(DEV_STACK) --profile train run --rm train-text

# ============================================================
# TRAINING (DEV only)
# ============================================================

.PHONY: train
train:
	git add src/ configs/ || true
	git commit -m "exp: training run $(shell date '+%Y-%m-%d %H:%M')" || true
	$(DEV_STACK) run --rm train-text

.PHONY: finetune
finetune:
	git add src/ configs/ || true
	git commit -m "exp: finetuning run $(shell date '+%Y-%m-%d %H:%M')" || true
	DEVICE=$(DEVICE) \
	GIT_COMMIT=`git rev-parse HEAD` \
	GIT_BRANCH=`git rev-parse --abbrev-ref HEAD` \
	$(DEV_STACK) run --profile finetune --rm finetune-text

.PHONY: train-logs
train-logs:
	$(DEV_STACK) logs -f train-text


# ============================================================
# EVALUATION
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
	$(DEV_STACK) --profile evaluate run --rm evaluate-text \
	--mlflow_run_id $(MLFLOW_ID) \
	--x_data_csv_path $(X_DATA) \
	--y_data_csv_path $(Y_DATA) \
	--model_weights_path $(WEIGHTS) \
	--label_encoding_path $(ENCODING)

# ============================================================
# SERVING (DEV)
# ============================================================

.PHONY: serve-internal
serve-internal:
	$(DEV_STACK) up -d bentoml

.PHONY: serve-external
serve-external:
	$(DEV_STACK) up -d bentoml streamlit

.PHONY: serve-down
serve-down:
	$(DEV_STACK) stop bentoml streamlit

.PHONY: serve-logs
serve-logs:
	$(DEV_STACK) logs -f bentoml streamlit

.PHONY: inference
inference:
	$(DEV_STACK) run --rm bentoml --text $(TEXT)

.PHONY: inference-batch
inference-batch:
	$(DEV_STACK) run --rm bentoml --texts "T-shirt" "Console" "Livre"


# ============================================================
# BENTO
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
	uv run bentoml containerize rakuten-text-service:latest -t rakuten-ml/rakuten-text-service:latest


# ============================================================
# PRODUCTION (infra + prod images)
# ============================================================

.PHONY: prod-up
prod-up:
	$(PROD_STACK) pull
	$(PROD_STACK) up -d

.PHONY: prod-down
prod-down:
	$(PROD_STACK) down

.PHONY: prod-restart
prod-restart:
	$(PROD_STACK) down
	$(PROD_STACK) up -d

.PHONY: prod-logs
prod-logs:
	$(PROD_STACK) logs -f


# ============================================================
# MONITORING
# ============================================================

.PHONY: monitoring
monitoring:
	@echo "Grafana: http://localhost:3001"
	@echo "Prometheus: http://localhost:9090"


# ============================================================
# UTILITIES
# ============================================================

.PHONY: status
status:
	docker ps

.PHONY: clean
clean:
	docker system prune -f


# ============================================================
# HELP
# ============================================================

.PHONY: help
help:
	@echo ""
	@echo "Rakuten MLOps Makefile"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make infra-up"
	@echo "  make infra-down"
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
	@echo "  make serve-internal (BentoML)"
	@echo "  make serve-external (BentoML & Streamlit)"
	@echo ""
	@echo "Production:"
	@echo "  make prod-up"
	@echo "  make prod-down"
