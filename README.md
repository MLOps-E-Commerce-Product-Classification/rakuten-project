# Rakuten Multimodal Classifier

This project addresses a multimodal e-commerce classification task: predicting the Rakuten product category (`prdtypecode`) for catalog items from product **text** and product **images**.

The repository currently implements **two unimodal baseline pipelines**:

-   **Text classification** using Hugging Face Transformer models
-   **Image classification** using `torchvision` backbones

Text inputs are derived from:

-   `designation`
-   `description`

Image inputs are linked via:

-   `imageid`
-   `productid`

> **Note**\
> This repository is currently under active development. Some production-oriented components (e.g. Docker, serving, monitoring, CI/tests) are not yet fully implemented.

------------------------------------------------------------------------

## Key Features

-   Config-driven experiments managed via YAML files in `configs/`
-   Text baseline built on Transformer encoders such as:
    -   `bert-base-multilingual-cased`
    -   `xlm-roberta-base`
    -   `camembert-base`
-   Image baseline built on `torchvision` backbones such as:
    -   `efficientnet_b0`
    -   `resnet18`
-   Multiclass classification target: Rakuten `prdtypecode` across **27 classes**
-   Reproducible train/validation/test splits persisted as row IDs in `artifacts/splits/`
-   Random search modules for text and image hyperparameter tuning

------------------------------------------------------------------------

## Problem Statement

Given a product record, the goal is to assign the correct Rakuten category code. Specifically, the project aims to answer the following questions:

-   Based on a product title and description, which category should this item belong to?
-   Based on a product image, which Rakuten class is most likely?
-   Which alternative classes remain plausible when the model is uncertain?

------------------------------------------------------------------------

## Data

The project follows the original Rakuten challenge structure, where feature data and target labels are provided separately.

### 1. Text Inputs

The text pipeline uses the following columns:

-   `designation`
-   `description`
-   `prdtypecode`

According to the preprocessing configuration, the text pipeline:

-   optionally removes HTML
-   preserves original casing (`lowercase: false`)
-   combines `designation` and `description` into a single model input
-   inserts `[SEP]` as a field separator
-   tokenizes using the configured Hugging Face tokenizer
-   truncates and pads to `max_length = 128`

### 2. Image Inputs

The image pipeline uses the following columns:

-   `imageid`
-   `productid`
-   `prdtypecode`

Image filenames follow this pattern:

``` text
image_<imageid>_product<productid>.jpg
```

According to the preprocessing configuration, images are:

-   converted to RGB
-   resized to `224 x 224`
-   optionally normalized (e.g. `zero_one`)
-   passed to the selected `torchvision` backbone

### 3. Target

Both pipelines predict the same target:

-   `prdtypecode` = Rakuten product category code

The repository currently supports **27 target classes**, defined in `configs/label_encoding.json`.

------------------------------------------------------------------------

## Output

For each item, the inference pipeline is expected to produce:

-   predicted class index
-   predicted Rakuten category code
-   top-k ranked candidate classes
-   probability score for each candidate class

------------------------------------------------------------------------

## Evaluation Metrics

The project reports the following metrics:

-   **Macro F1** (primary classification metric)
-   **Accuracy**
-   **Loss**
-   **Per-class F1**
-   **Confusion matrix**
-   **Latency**
-   **Throughput**

------------------------------------------------------------------------

## Repository Structure

``` text
rakuten-project-text/
├── artifacts/
│   └── splits/                  # Persisted train/val/test row IDs
├── configs/                     # Training, preprocessing, evaluation, and search configs
├── docker/                      # Docker scaffolding (currently empty)
├── monitoring/                  # Monitoring scaffolding
├── pipelines/
│   └── dags/                    # Orchestration scaffolding
├── src/
│   ├── data/                    # Datasets and preprocessing
│   ├── evaluation/              # Evaluation logic
│   ├── inference/               # Inference utilities
│   ├── models/                  # Model builders
│   ├── pipeline/                # CLI entry points
│   ├── serving/                 # API scaffolding
│   └── training/                # Training and hyperparameter search
├── tests/                       # Test scaffolding (currently empty)
├── pyproject.toml               # Project metadata and dependencies
├── requirements_image.txt
└── README.md
```

------------------------------------------------------------------------

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/MLOps-E-Commerce-Product-Classification/rakuten-project.git
cd rakuten-project
```

### 2. Create the environment

``` bash
uv lock

```

API only (no PyTorch)
``` bash
uv sync --extra api

```

Text pipeline
``` bash
uv sync --extra text

```

Image pipeline
``` bash
uv sync --extra image

```

Training (Image + Text)
``` bash
uv sync --extra training

```

Everything (Image + Text + API)
``` bash
uv sync --all-extras

```

### 3. Activate the environment

``` bash
source .venv/bin/activate
```

------------------------------------------------------------------------

## Run Image Pipeline

### 1. Requirements

Install the image pipeline dependencies from:

``` text
requirements_image.txt
```

### 2. Training

``` bash
python -m src.pipeline.image_pipeline --mode train
```

The number of samples used for training can be configured in:

``` text
configs/image_train_config.yaml
```

### 3. Hyperparameter Search

``` bash
python -m src.pipeline.image_pipeline --mode random_search
```

### 4. Evaluation

``` bash
python -m src.pipeline.image_pipeline --mode evaluate
```

### 5. Inference

``` bash
python -m src.pipeline.image_pipeline --mode inference --image_path data/raw/images/image_train/image_1263597046_product_3804725264.jpg
```

------------------------------------------------------------------------

## Text Training Pipeline

### 1. Running Training

Start text model training with the following command:

``` bash
uv run python -m src.pipeline.text_pipeline --mode train
```

### 2. Configuration

Training parameters and paths are configured in:

-   `configs/text_train_config.yaml` - training hyperparameters such as epochs, batch size, and learning rate
-   `configs/label_encoding.json` - label-to-index mapping
-   `artifacts/splits/` - directory for persisted train/val/test split ID files (auto-created or loaded)

Adjust these files to customize training behavior. After training, the best model is saved to:

``` text
models/best_text_model.pt
```

Logs are saved in:

``` text
logs/text_training.log
```

------------------------------------------------------------------------

## To Do

-   [ ] train the final image model via random search and save `best_model.pt`, `best_train_config.yaml`, and `random_search_results.csv` to Google Drive
-   [ ] dockerize the project
-   [ ] add unit tests

------------------------------------------------------------------------

## Limitations

-   tbd
## MLOps Rakuten — Training Infrastructure

## Prerequisites

#### 1. Docker & Docker Compose

The project runs inside Docker containers. Make sure Docker and Docker Compose are installed on your host system.

Check installation:

```bash
docker --version
docker compose version
```

#### 2. NVIDIA Driver (only for GPU training)

If you want to run with GPU support, the NVIDIA driver must be installed on the **host machine**. CUDA itself is now provided through the Docker image when `DEVICE=cu121` is used.

Check if your driver is available:

```bash
nvidia-smi
```

If not installed:

##### Ubuntu/Debian

```bash
sudo apt-get install -y nvidia-driver-535
sudo reboot
```

#### 3. NVIDIA Container Toolkit (only for GPU training)

Allows Docker to access the GPU on the host.

##### Add NVIDIA package repository

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |   sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

##### Install

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

##### Restart Docker

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU is accessible inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Device Selection

The Makefile supports different execution targets via `DEVICE`:

```bash
make train-text-build DEVICE=cpu
make train-text-build DEVICE=cu121
```

- `DEVICE=cpu`: builds and runs the CPU image
- `DEVICE=cu121`: builds and runs the CUDA 12.1 image inside Docker

> CUDA is no longer expected to be installed manually on the host. For GPU runs, only the NVIDIA driver and NVIDIA Container Toolkit are required on the host system.

### Environment Setup

Create a `.env` file in the project root:

```env
DAGSHUB_USER_TOKEN=your_token_here
MLFLOW_TRACKING_URI=https://dagshub.com/<USERNAME>/<REPO>.mlflow
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

> ⚠️ Never commit `.env` to Git. It is already listed in `.gitignore`.

### Makefile Commands

#### Text Training

- `make train-text-build [DEVICE=cpu|cu121]`: Build the training image
- `make train-text-run [DEVICE=cpu|cu121]`: Commit config changes if needed and run training
- `make train-text-rebuild [DEVICE=cpu|cu121]`: Build image and run training
- `make train-text-stop`: Stop the training container
- `make train-text-down`: Stop and remove containers
- `make train-text-clean`: Stop containers and remove the training image
- `make train-text-logs`: Follow live training logs

#### When to use which command?

- **Config changed** (e.g. learning rate, batch size):

```bash
make train-text-run DEVICE=cpu
```

or for GPU:

```bash
make train-text-run DEVICE=cu121
```

→ No rebuild needed. Config changes are committed automatically before the run.

- **Code changed** (e.g. training pipeline, Docker dependencies, source files):

```bash
make train-text-rebuild DEVICE=cpu
```

or:

```bash
make train-text-rebuild DEVICE=cu121
```

→ Rebuilds the image and then starts training.

> `make train-text-run` automatically commits changes in `configs/` before starting the run if there are uncommitted config changes.

> `GIT_COMMIT`, `GIT_BRANCH`, and `DEVICE` are injected into the container runtime so experiment metadata stays aligned with the executed run.

## DVC — Data & Model Versioning

Data, models, and pipeline outputs are tracked with DVC.

#### Initialize DVC

```bash
make dvc-init
```

#### Track raw data

```bash
make dvc-add-data
```

#### Pull data and artifacts

```bash
make dvc-pull
```

This will:
- run `git pull`
- run `uv run dvc pull`

#### Reproduce the training pipeline with DVC

```bash
make dvc-repro DEVICE=cpu
```

or:

```bash
make dvc-repro DEVICE=cu121
```

What happens during `dvc-repro`:
- config changes in `configs/` are committed automatically if needed
- DVC checks whether dependencies changed
- DVC executes the `train-text` stage if required
- `dvc.lock` is updated and committed

#### Push DVC artifacts

```bash
make dvc-push
```

This will:
- push DVC artifacts to remote storage
- push Git commits

#### Full DVC run

```bash
make dvc-run DEVICE=cu121
```

This is equivalent to:
- `make dvc-repro`
- `make dvc-push`

#### Compare metrics

```bash
make dvc-metrics
```

Shows current metrics and the difference to `HEAD~1`.

### MLflow — Experiment Tracking

Experiments are tracked automatically during training and pushed to DagsHub / MLflow.

View experiments at:

```text
https://dagshub.com/<USERNAME>/<REPO>/experiments
```

Each run can log:
- parameters (learning rate, batch size, epochs, ...)
- metrics (accuracy, macro-f1, loss, ...)
- Git commit hash and branch
- config files and artifacts
- runtime device information

## Evaluation

Evaluation is now supported via dedicated Make targets.

Default paths:
- `X_DATA=data/processed/val.csv`
- `Y_DATA=data/processed/val.csv`
- `WEIGHTS=models/best_text_model.pt`
- `ENCODING=configs/label_encoding.json`

#### Build evaluation image

```bash
make evaluate-build DEVICE=cpu
```

or:

```bash
make evaluate-build DEVICE=cu121
```

#### Run evaluation

```bash
make evaluate-run MLFLOW_ID=<your_run_id> DEVICE=cpu
```

Example with custom paths:

```bash
make evaluate-run   MLFLOW_ID=<your_run_id>   DEVICE=cu121   X_DATA=data/processed/val_features.csv   Y_DATA=data/processed/val_labels.csv   WEIGHTS=models/best_text_model.pt   ENCODING=configs/label_encoding.json
```

`MLFLOW_ID` is required. The Makefile will stop with an error if it is missing.

## Inference

Inference is available through separate targets.

#### Build inference image

```bash
make inference-build
```

#### Run single-text inference

```bash
make inference-run TEXT="Jeu vidéo action PS4"
```

#### Run batch inference

```bash
make inference-batch
```

#### Rebuild and run inference

```bash
make inference-rebuild
```

#### Clean inference image

```bash
make inference-clean
```

## Reproducibility

Every training run is reproducible via the following anchors:

- **Code**: Git commit hash (`GIT_COMMIT`)
- **Branch**: Git branch (`GIT_BRANCH`)
- **Config**: committed `configs/` state
- **Device**: exported via `DEVICE`
- **Data & Artifacts**: tracked through DVC

To reproduce a run:

```bash
git checkout <commit-hash>
make dvc-pull
make train-text-run DEVICE=cpu
```

or for GPU:

```bash
git checkout <commit-hash>
make dvc-pull
make train-text-run DEVICE=cu121
```

