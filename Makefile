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
	@echo ""
