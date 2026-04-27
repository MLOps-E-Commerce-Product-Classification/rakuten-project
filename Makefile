# ============================================================
# Variables
# ============================================================

-include .env

GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null || echo "unknown")

MLFLOW_MODEL_NAME ?= text-classifier
MLFLOW_ALIAS      ?= production
BENTO_MODEL_NAME  ?= text-classifier
BENTO_TAG         ?= latest

COMPOSE_BASE := docker compose
PROD_FILES   := -f docker-compose.prod.yaml
PROD_STACK   := $(COMPOSE_BASE) $(PROD_FILES)

export DEVICE
export GIT_BRANCH
export GIT_COMMIT

# ============================================================
# INITIAL SETUP & VALIDATION
# ============================================================

.PHONY: check-env
check-env: ## Check if .env exists and is configured
	@if [ ! -f .env ]; then \
		echo "========================================================================"; \
		echo "❌ ERROR: .env file missing!"; \
		echo "1. Copy .env.example to .env"; \
		echo "2. Fill in all required variables, especially those in < >"; \
		echo "========================================================================"; \
		exit 1; \
	fi
	@if grep -q "<.*>" .env; then \
		echo "========================================================================"; \
		echo "⚠️  WARNING: Some variables in your .env still contain placeholders < >!"; \
		echo "Please check your .env file and replace all <placeholder> values."; \
		echo "========================================================================"; \
		exit 1; \
	fi

.PHONY: setup
setup: check-env ## Initial project setup (runs only if .env is valid)
	@echo "✅ .env validation passed. Running setup.sh..."
	@chmod +x setup.sh
	@./setup.sh
	@echo "Setup complete."

# ============================================================
# DEV ENVIRONMENT
# ============================================================

dev-build: ## Build the development environment and containerize Bento
	$(COMPOSE_BASE) build \
		--build-arg DEVICE=$(DEVICE) \
		--build-arg GIT_COMMIT=$(GIT_COMMIT) \
		--build-arg GIT_BRANCH=$(GIT_BRANCH)
	$(MAKE) containerize-bento

dev-up: ## Start dev services
	$(COMPOSE_BASE) up -d

dev-down: ## Stop dev services
	$(COMPOSE_BASE) down

dev-restart: ## Restart dev services with a fresh build
	$(MAKE) dev-down
	$(MAKE) dev-up

dev-logs: ## Follow dev logs
	$(COMPOSE_BASE) logs -f

# ============================================================
# TRAINING & EVALUATION
# ============================================================

train-text-run: ## Run full training (commits changes to track experiment state)
	@git add src/ configs/ || true
	@git commit -m "exp: start training run - $$(date '+%Y-%m-%d %H:%M')" || echo "No changes to commit"
	$(COMPOSE_BASE) --profile train run --rm train-text

finetune: ## Run finetuning
	@git add src/ configs/ || true
	@git commit -m "exp: finetuning run $$(date '+%Y-%m-%d %H:%M')" || echo "No changes to commit"
	$(COMPOSE_BASE) --profile finetune run --rm finetune-text

X_DATA   ?= data/processed/val.csv
Y_DATA   ?= data/processed/val.csv
WEIGHTS  ?= models/best_text_model.pt
ENCODING ?= configs/label_encoding.json

evaluate: ## Evaluate model. Usage: make evaluate MLFLOW_ID=xxx
	@if [ -z "$(MLFLOW_ID)" ]; then echo "Error: MLFLOW_ID is required. Usage: make evaluate MLFLOW_ID=xxx"; exit 1; fi
	$(COMPOSE_BASE) run --rm evaluate-text \
		--mlflow_run_id $(MLFLOW_ID) \
		--x_data_csv_path $(X_DATA) \
		--y_data_csv_path $(Y_DATA) \
		--model_weights_path $(WEIGHTS) \
		--label_encoding_path $(ENCODING)

# ============================================================
# BENTOML WORKFLOW
# ============================================================

sync-bento: ## Sync latest MLFlow model to BentoLocal store
	uv run python -m src.serving.sync_mlflow_to_bento \
		--model-name $(MLFLOW_MODEL_NAME) \
		--alias $(MLFLOW_ALIAS) \
		--bento-model-name $(BENTO_MODEL_NAME)

build-bento: ## Build Bento bundle
	$(MAKE) sync-bento
	uv run bentoml build

containerize-bento: ## Containerize the Bento bundle into a Docker Image
	$(MAKE) build-bento
	uv run bentoml containerize $(BENTO_MODEL_NAME):latest -t rakuten-ml/rakuten-text-service:$(BENTO_TAG)

# ============================================================
# PRODUCTION
# ============================================================

prod-up: ## Pull and start production stack
	$(PROD_STACK) pull
	$(PROD_STACK) up -d

prod-down: ## Stop production stack
	$(PROD_STACK) down

prod-logs: ## Follow production logs
	$(PROD_STACK) logs -f

# ============================================================
# HELP
# ============================================================

.PHONY: help
help: ## Show this help message
	@echo "========================================================================"
	@echo "RAKUTEN MLOPS FRAMEWORK"
	@echo "========================================================================"
	@echo "Usage: make <target> [VARIABLE=value]"
	@echo ""
	@echo "Targets:"
	@grep -E -h '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make dev-build DEVICE=cuda          Build for GPU"
	@echo "  make evaluate MLFLOW_ID=abc-123     Run evaluation for a specific run"
	@echo "  make prod-up                        Deploy to production"
	@echo "========================================================================"
