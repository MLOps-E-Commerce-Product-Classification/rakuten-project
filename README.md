# Rakuten Multimodal Product Classifier

Multimodal e-commerce classification pipeline for predicting the
**Rakuten product category (`prdtypecode`)** using product **text**.
Product **images** are available but not used, as they provided limited
additional predictive power while significantly increasing computational
requirements.

The repository includes:

-   Text classification with Hugging Face Transformers
-   Experiment tracking with MLflow / DagsHub
-   Reproducible pipelines with DVC
-   Model serving via BentoML
-   UI with Streamlit
-   Containerized workflows using Docker Compose
-   Workflow automation with Airflow
-   Monitoring with Prometheus, Grafana, and Evidently

------------------------------------------------------------------------

# Project Goal

The objective is to classify Rakuten products into **27 product
categories**.

Input features:

-   product **designation**
-   product **description**

Output:

-   predicted **prdtypecode**
-   probability scores
-   top-k predictions

------------------------------------------------------------------------

# Repository Structure

    rakuten-project/

    artifacts/            # DVC artifacts (splits, models)
    configs/              # experiment configurations
    dags/                 # Airflow DAGs
    docker/               # Docker & Compose setup
    monitoring/           # Prometheus / Grafana / Evidently

    src/
      data/
      evaluation/
      inference/
      models/
      pipeline/
      serving/
      training/

    streamlit/            # UI
    tests/

    Makefile
    pyproject.toml
    README.md

------------------------------------------------------------------------

# Environment Setup

Install dependencies:

``` bash
uv lock
uv sync --all-extras
```

Activate environment:

``` bash
source .venv/bin/activate
```

------------------------------------------------------------------------

# Development Environment

Start development stack:

``` bash
make dev-up
```

Stop development services:

``` bash
make dev-down
```

Restart development environment:

``` bash
make dev-restart
```

View logs:

``` bash
make dev-logs
```

------------------------------------------------------------------------

# Infrastructure

Start infrastructure services:

``` bash
make infra-up
```

Stop infrastructure services:

``` bash
make infra-down
```

View infrastructure logs:

``` bash
make infra-logs
```

------------------------------------------------------------------------

# Training

Run training experiment:

``` bash
make train
```

Run finetuning:

``` bash
make finetune
```

View training logs:

``` bash
make train-logs
```

------------------------------------------------------------------------

# Evaluation

Evaluate a trained model:

``` bash
make evaluate MLFLOW_ID=<run_id>
```

Optional parameters:

-   X_DATA: data/processed/val.csv
-   Y_DATA: data/processed/val.csv
-   WEIGHTS: models/best_text_model.pt
-   ENCODING: configs/label_encoding.json

------------------------------------------------------------------------

# Inference

Single prediction:

``` bash
make inference TEXT="Jeu vidéo action PS4"
```

Batch prediction:

``` bash
make inference-batch
```

------------------------------------------------------------------------

# Model Promotion & BentoML

Prepare Bento assets:

``` bash
make prepare-bento
```

Promote best MLflow model:

``` bash
make promote-model
```

Sync MLflow model to BentoML:

``` bash
make sync-bento
```

Build Bento:

``` bash
make build-bento
```

Containerize Bento service:

``` bash
make containerize-bento
```

------------------------------------------------------------------------

# Serving

Start BentoML service:

``` bash
make serve-internal
```

Start BentoML and streamlit service:

``` bash
make serve-external
```

Stop all services:

``` bash
make serve-down
```

View logs:

``` bash
make serve-logs
```

------------------------------------------------------------------------

# Production

Start production stack:

``` bash
make prod-up
```

Stop production services:

``` bash
make prod-down
```

View logs:

``` bash
make prod-logs
```

Restart production:

``` bash
make prod-restart
```

------------------------------------------------------------------------

# Monitoring

Show monitoring endpoints:

``` bash
make monitoring
```

Services:

-   Grafana → http://localhost:3001
-   Prometheus → http://localhost:9090

------------------------------------------------------------------------

# Utilities

Check running containers:

``` bash
make status
```

Clean Docker resources:

``` bash
make clean
```

------------------------------------------------------------------------

# Help

Show all commands:

``` bash
make help
```
