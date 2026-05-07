# Rakuten Text-First Product Classification MLOps Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/) [![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-orange)](https://mlflow.org/) [![DVC](https://img.shields.io/badge/DVC-Pipelines-945DD6)](https://dvc.org/) [![BentoML](https://img.shields.io/badge/BentoML-Serving-00A3A3)](https://www.bentoml.com/) [![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)](https://www.docker.com/) [![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)](https://streamlit.io/) [![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF)](https://github.com/features/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Production-oriented MLOps project for classifying Rakuten e-commerce products into 27 `prdtypecode` categories from product title and description text.

This repository demonstrates an end-to-end machine learning workflow for e-commerce product classification. The current production path is **text-first**: it uses the product `designation` and `description` fields for inference.

The project focuses on MLOps, combining data versioning, preprocessing, training, experiment tracking, evaluation, BentoML serving, a Streamlit UI, Airflow orchestration, monitoring components, Docker-based execution, and CI validation.

------------------------------------------------------------------------

## Table of Contents

-   [Project Overview](#project-overview)
-   [Architecture?](#architecture)
-   [Requirements](#requirements)
-   [Quickstart](#quickstart)
-   [Additional Information](#additional-information)
    -   [1. Components](#components)
    -   [2. Local Services and Ports](#local-services-and-ports)
    -   [3. Separate Workflow Commands](#separate-workflow-commands)
    -   [4. Airflow Orchestration](#airflow-orchestration)
    -   [5. Testing and CI](#testing-and-ci)
    -   [6. Monitoring and Observability](#monitoring-and-observability)
    -   [7. Streamlit UI](#streamlit-ui)
    -   [8. API Usage](#api-usage)
    -   [9. Repository Structure](#repository-structure)
-   [License](#license)

------------------------------------------------------------------------

## Project Overview

The business use case is automatic product category classification for e-commerce listings. Given product text, the system predicts the corresponding Rakuten product category code (`prdtypecode`).

The project goals are to:

-   predict the correct category code for each product;
-   return probabilities for the category code predictions;
-   provide reproducible and automated training and evaluation workflows;
-   implement authentication and APIs in our MLOps system;
-   provide a user interface for inspecting predictions;
-   support local development and production-style deployment;
-   ensure full version control across the model lifecycle.

------------------------------------------------------------------------

## Architecture

![](images/MLOps%20-%20Liora.png)

The development stack is coordinated through Docker Compose. The core service path is:

1.  DVC pulls or tracks data. 2.  Text preprocessing prepares the model input fields. 3.  Training runs a transformer-based text model. 4.  Metrics are written to `results/` and tracked through MLflow/DagsHub. 5.  The selected MLflow model alias is synced into the local BentoML model store. 6.  BentoML packages and serves the text classifier. 7.  Streamlit provides a UI on top of the prediction API, with NGINX acting as reverse proxy. 8.  Prometheus, Grafana, and Evidently provide monitoring-related components.

------------------------------------------------------------------------

## Requirements

-   Python `>=3.11,<3.13`
-   `uv`
-   Git
-   Make
-   Docker
-   Docker Compose `>=2.20.0`
-   Bash-compatible Unix environment
-   Access credentials and environment variables configured in `.env`

Strongly recommended:

-   NVIDIA GPU support, including compatible NVIDIA drivers/container runtime, for CUDA-based Docker builds, GPU training, or monitoring through DCGM exporter.
-   Sufficient Docker memory and disk space for ML images, model artifacts, DVC data, and dependency caches.

------------------------------------------------------------------------

### Quickstart

> Notes: - Use `DEVICE=cu121` only if you have a compatible NVIDIA GPU and configured CUDA. - To run on a machine without GPU support, omit the `DEVICE` environment variable. - If a command fails due to permissions, check whether sudo privileges are required and consult your system administrator.

#### 1. Clone the repository

``` bash
git clone https://github.com/MLOps-E-Commerce-Product-Classification/rakuten-project.git
cd rakuten-project
```

#### 2. Run project setup

Run `make help` to get further make commands.

To set up the project, run:

``` bash
make setup
```

> **Note:** This step may require sudo (sudoers) privileges.

#### 3. Configure environment variables

**IMPORTANT:** Edit `.env` and fill in the required credentials (for details, see `.env.example`).

#### 4. Build the development stack

> **Note:** `DEVICE=cu121` requires an NVIDIA GPU. Remove the `DEVICE` argument if you want to run on CPU.

``` bash
make dev-build DEVICE=cu121
```

#### 5. Start and stop the development stack

Start the development stack:

``` bash
make dev-up DEVICE=cu121
```

Stop the development stack:

``` bash
make dev-down DEVICE=cu121
```

View logs:

``` bash
make dev-logs
```

#### 6. Start the production stack

The production Compose file uses prebuilt images and an Nginx reverse proxy configuration.

Start the production stack:

``` bash
make prod-up DEVICE=cu121
```

Stop it:

``` bash
make prod-down DEVICE=cu121
```

View logs:

``` bash
make prod-logs
```

#### 7. Run a text training or fine-tune experiment manually

``` bash
make train-text-run DEVICE=cu121
```

``` bash
make finetune DEVICE=cu121
```

#### 8. Evaluate a model by MLflow run ID

``` bash
make evaluate MLFLOW_ID=<run_id>
```

------------------------------------------------------------------------

## Additional information:

### 1. Components

+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Area                | Implemented component                                                           | Where to look                              |
+=====================+=================================================================================+============================================+
| Data versioning     | DVC-tracked raw data directory and pipeline definition                          | `data/raw.dvc` · `.dvc/config`\            |
|                     |                                                                                 | `dvc.yaml` · `dvc.lock`                    |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Text preprocessing  | Cleaning, field combination, tokenization configuration                         | `configs/text_preprocessing_config.yaml`\  |
|                     |                                                                                 | `src/pipeline/preprocess_text_pipeline.py` |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Text model training | Transformer-based text classifier training                                      | `configs/text_train_config.yaml`\          |
|                     |                                                                                 | `src/training/`\                           |
|                     |                                                                                 | `src/pipeline/text_pipeline.py`            |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Fine-tuning         | Fine-tuning workflow for new data                                               | `configs/text_finetune_config.yaml`\       |
|                     |                                                                                 | `src/training/run_text_finetuning.py`\     |
|                     |                                                                                 | `dags/finetune_new_data.py`                |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Inference           | Single and batch inference scripts                                              | `src/inference/`\                          |
|                     |                                                                                 | `dags/infere_new_data.py`                  |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Experiment tracking | MLflow / DagsHub tracking and model registry                                    | `.env.example`\                            |
|                     |                                                                                 | `src/training/mlflow_text_registry.py`\    |
|                     |                                                                                 | `src/training/mlflow_text_pyfunc.py`       |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Evaluation          | DVC metrics and evaluation scripts                                              | `results/dvc_metrics.json`\                |
|                     |                                                                                 | `src/evaluation/`                          |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Model definitions   | Text and image classifier modules                                               | `src/models/`                              |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Model registry sync | MLflow model alias to local BentoML model store                                 | `src/serving/sync_mlflow_to_bento.py`      |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Serving             | JWT-protected BentoML text service                                              | `bentofile.yaml`\                          |
|                     |                                                                                 | `src/serving/bento_service.py`             |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| API schemas         | Request and response models for prediction                                      | `src/serving/schemas.py`                   |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| UI                  | Streamlit application for interacting with the service                          | `streamlit/`                               |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Orchestration       | Airflow DAGs for training, inference, simulation, and fine-tuning workflows     | `dags/`                                    |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Monitoring          | Prometheus, Grafana, Evidently, node exporter, DCGM exporter                    | `monitoring/`\                             |
|                     |                                                                                 | `docker-compose.base.yaml`                 |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Containers          | Development and production Docker Compose setups                                | `docker-compose.yaml`\                     |
|                     |                                                                                 | `docker-compose.base.yaml`\                |
|                     |                                                                                 | `docker-compose.prod.yaml`\                |
|                     |                                                                                 | `docker/`                                  |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Reverse proxy       | Nginx configuration for production routing                                      | `nginx/`\                                  |
|                     |                                                                                 | `docker-compose.prod.yaml`                 |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Build automation    | Unified command interface for all workflows                                     | `Makefile`                                 |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| Testing             | Unit/integration tests for API, BentoML, data, training, pipeline, and registry | `tests/`                                   |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+
| CI                  | Pre-commit, pytest, Docker image builds, Bento build/containerization           | `.github/workflows/integration.yml`        |
+---------------------+---------------------------------------------------------------------------------+--------------------------------------------+

Note. Raw data is managed through DVC. The repository contains a DVC pointer file:

``` text
data/raw.dvc
```

------------------------------------------------------------------------

### 2. Local Services and Ports

| Service                 | Local URL / port        |
|-------------------------|-------------------------|
| BentoML API             | `http://localhost:3000` |
| Streamlit UI            | `http://localhost:8502` |
| Airflow webserver       | `http://localhost:8080` |
| Flower                  | `http://localhost:5555` |
| Prometheus              | `http://localhost:9090` |
| Grafana                 | `http://localhost:3001` |
| Evidently drift service | `http://localhost:8899` |
| Redis                   | `localhost:6379`        |
| DCGM exporter           | `localhost:9400`        |
| Nginx                   | `http://localhost:80`   |

------------------------------------------------------------------------

### 3. Separate workflow commands

#### DVC

Pull DVC-tracked data:

``` bash
uv run dvc pull
```

Run text preprocessing:

``` bash
uv run dvc repro preprocess-text
```

Run text training:

``` bash
uv run dvc repro train-text
```

Run the full DVC graph as defined in `dvc.yaml`:

``` bash
uv run dvc repro
```

Show DVC-tracked metrics:

``` bash
uv run dvc metrics show
```

#### Training

Run a text training experiment through Docker Compose:

``` bash
make train-text-run
```

This target stages `src/` and `configs/`, attempts to create an experiment-state commit, and continues if there are no changes to commit.

Run fine-tuning:

``` bash
make finetune
```

#### Evaluation

Evaluate a trained model by MLflow run ID:

``` bash
make evaluate MLFLOW_ID=<run_id>
```

Default evaluation parameters:

| Variable   | Default                       |
|------------|-------------------------------|
| `X_DATA`   | `data/processed/val.csv`      |
| `Y_DATA`   | `data/processed/val.csv`      |
| `WEIGHTS`  | `models/best_text_model.pt`   |
| `ENCODING` | `configs/label_encoding.json` |

By default, both `X_DATA` and `Y_DATA` point to the same validation CSV. Override these paths if your evaluation inputs are split across separate files, e.g.:

``` bash
make evaluate \
  MLFLOW_ID=<run_id> \
  X_DATA=data/processed/val.csv \
  Y_DATA=data/processed/val.csv
```

#### Model Registry and BentoML

Sync the selected MLflow model alias to the local BentoML model store:

``` bash
make sync-bento
```

Default values from the Makefile:

| Variable            | Default           |
|---------------------|-------------------|
| `MLFLOW_MODEL_NAME` | `text-classifier` |
| `MLFLOW_ALIAS`      | `production`      |
| `BENTO_MODEL_NAME`  | `text-classifier` |

Build the Bento bundle:

``` bash
make build-bento
```

This target runs `make sync-bento` before building the Bento bundle.

Containerize the Bento bundle:

``` bash
make containerize-bento
```

This target builds the Bento bundle first, then containerizes `rakuten-text-service:latest` as `rakuten-ml/rakuten-text-service:latest`.

The BentoML service target is:

``` text
src.serving.bento_service:TextBentoService
```

The Bento build configuration is defined in:

``` text
bentofile.yaml
```

------------------------------------------------------------------------

### 4. Airflow Orchestration

Airflow is included in the Docker Compose stack. The repository contains DAGs for:

| DAG file                   | Purpose inferred from filename    |
|----------------------------|-----------------------------------|
| `manual_train_text_dag.py` | Manual text training workflow     |
| `simulate_new_data.py`     | New-data simulation workflow      |
| `infere_new_data.py`       | New-data inference workflow       |
| `finetune_new_data.py`     | Fine-tuning workflow for new data |

New-data is being retrieved from the file `data/raw/X_test_update.csv`.

------------------------------------------------------------------------

### 5. Testing and CI

Run the test suite locally:

``` bash
pytest
```

The `tests/` directory contains automated tests covering API behavior, BentoML authentication, BentoML HTTP behavior, BentoML packaging, data handling, the MLflow registry flow, model code, text dataset handling, text training, text preprocessing, and text pipeline behavior.

The GitHub Actions CI workflow is triggered for pushes and pull requests targeting `main`. It runs:

-   pre-commit checks,
-   the pytest test suite,
-   Docker image builds and pushes to Docker Hub on pushes to `main`,
-   BentoML service containerization and Docker Hub publishing on pushes to `main`.

------------------------------------------------------------------------

### 6. Monitoring and Observability

| Component | Purpose | Location / port |
|----|----|----|
| Prometheus | Scrapes metrics from BentoML, Evidently, node exporter, DCGM exporter | `monitoring/prometheus.yaml`, `localhost:9090` |
| Grafana | Dashboarding and visualization | `localhost:3001` |
| Evidently drift service | Drift/data-quality monitoring service | `monitoring/`, `localhost:8899` |
| Node exporter | Host/system metrics | Docker Compose service |
| DCGM exporter | NVIDIA GPU metrics where supported | `localhost:9400` |

------------------------------------------------------------------------

### 7. Streamlit UI

The user interface is exposed on:

``` text
http://localhost:8502
```

It allows for single and batch prediction and offers the possibility to select the correct product code from the top-k.

![](images/streamlit.png)

The Streamlit container receives BentoML connection and authentication settings through environment variables in `.env`.

------------------------------------------------------------------------

### 8. API Usage

The BentoML service exposes:

-   `/health`
-   `/login`
-   `/predict`
-   `/predict_batch`

Prediction endpoints are protected with JWT authentication. First call `/login` with the BentoML credentials configured in `.env`.

### Health check

``` bash
curl -X POST http://localhost:3000/health
```

Example response shape:

``` json
{
  "status": "ok",
  "model_ready": true,
  "model_tag": "text-classifier:latest"
}
```

If the model is not available in the BentoML model store, the service may report degraded readiness.

### Login

``` bash
curl -X POST http://localhost:3000/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user123",
    "password": "password123"
  }'
```

Example response:

``` json
{
  "token": "<jwt-token>"
}
```

Store the token locally:

``` bash
TOKEN=$(curl -s -X POST http://localhost:3000/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user123","password":"password123"}' \
  | python -c "import sys, json; print(json.load(sys.stdin)['token'])")
```

### Single prediction

Request schema:

``` json
{
  "designation": "Jeu vidéo action PS4",
  "description": "Jeu vidéo pour console PlayStation 4",
  "top_k": 5
}
```

-   `designation` is required.
-   `description` is optional and defaults to an empty string.
-   `top_k` is optional and must be between 1 and 27.

cURL example:

``` bash
curl -X POST http://localhost:3000/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "designation": "Jeu vidéo action PS4",
    "description": "Jeu vidéo pour console PlayStation 4",
    "top_k": 5
  }'
```

Response shape:

``` jsonc
{
  "predicted_rakuten_code": 2462,
  "top_k_predictions": [
    {
      "rakuten_code": 2462,
      "probability": 0.5865910053253174
    },
    {
      "rakuten_code": 40,
      "probability": 0.14944210648536682
    }
  ],
  "probabilities": {
    "10": 0.0014983717119321227,
    "40": 0.14944210648536682,
    "2462": 0.5865910053253174
    // truncated: response contains probabilities for all classes
  }
}
```

### Batch prediction

Request schema:

``` json
{
  "items": [
    {
      "designation": "Jeu vidéo action PS4",
      "description": "Jeu vidéo pour console PlayStation 4",
      "top_k": 5
    }
  ]
}
```

cURL example:

``` bash
curl -X POST http://localhost:3000/predict_batch \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "designation": "Jeu vidéo action PS4",
        "description": "Jeu vidéo pour console PlayStation 4",
        "top_k": 5
      }
    ]
  }'
```

------------------------------------------------------------------------

### 9. Repository Structure

``` text
rakuten-project/
├── .dvc/                     # DVC metadata
├── .github/workflows/        # GitHub Actions CI workflows
├── configs/                  # Experiment and model configurations
├── data/                     # Raw and processed data
├── dags/                     # Airflow DAGs
├── docker/                   # Docker and service-specific container setup
├── models/                   # Saved model artifacts
├── monitoring/               # Prometheus, Grafana, and Evidently setup
├── nginx/                    # Reverse proxy configuration
├── results/                  # Reports, metrics, and experiment outputs
├── src/
│   ├── data/                 # Data ingestion and cleaning
│   ├── evaluation/           # Evaluation and reporting
│   ├── inference/            # Single and batch inference
│   ├── models/               # Model definitions and helpers
│   ├── pipeline/             # End-to-end pipeline orchestration
│   ├── serving/              # BentoML service implementation
│   └── training/             # Training and MLflow logic
├── streamlit/                # Streamlit application
├── tests/                    # Automated tests
├── Makefile                  # Unified command interface
├── bentofile.yaml            # BentoML build configuration
├── dvc.yaml                  # DVC pipeline stages
├── dvc.lock                  # DVC lock file
├── params.yaml               # DVC parameters
├── docker-compose.yaml       # Main compose file
├── docker-compose.base.yaml  # Base compose configuration
├── docker-compose.prod.yaml  # Production compose configuration
├── pyproject.toml            # Project metadata and dependencies
├── requirements.txt          # Optional pip dependency list
├── setup.sh                  # Environment bootstrap script
├── uv.lock                   # Locked dependency state for uv
└── README.md
```

Generated local folders such as `artifacts/`, `logs/`, and `plugins/` may be created by setup, DVC, Airflow, or runtime workflows. They are not necessarily present or fully populated in a fresh clone.

------------------------------------------------------------------------

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

For information on third-party packages and their licenses, see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

