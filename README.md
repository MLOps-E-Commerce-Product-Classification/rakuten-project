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
> This repository is currently under active development. Monitoring and CI coverage are still evolving, while the text serving layer is now implemented with BentoML.

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

For BentoML packaging, vendor the Hugging Face text backbone locally before building the Bento:

``` bash
make prepare-bento-text-assets
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

#### 1. NVIDIA Driver (Host)
CUDA runs on the **host machine**, not inside Docker. Docker only needs the NVIDIA Container Toolkit.

Check if your driver is installed:

nvidia-smi

If not installed:

## Ubuntu/Debian
sudo apt-get install -y nvidia-driver-535
sudo reboot

#### 2. NVIDIA Container Toolkit
Allows Docker to access the GPU on the host.

## Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

## Install
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

## Restart Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

Verify GPU is accessible inside Docker:

docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

### Environment Setup

Create a `.env` file in the project root:

DAGSHUB_USER_TOKEN=your_token_here
MLFLOW_TRACKING_URI=https://dagshub.com/<USERNAME>/<REPO>.mlflow
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token

> ⚠️ Never commit `.env` to Git. It is already listed in `.gitignore`.

### Makefile Commands

#### Text Model Training

- `make train-text-run`: Commit configs + run training (no rebuild)
- `make train-text-rebuild`: Commit configs + build image + run training
- `make train-text-build`: Build Docker image only
- `make train-text-stop`: Stop the running container
- `make train-text-down`: Stop + remove the container
- `make train-text-clean`: Stop + remove container + image + volumes
- `make train-text-logs`: Follow live training logs

#### When to use which command?

- **Config changed** (e.g. learning rate, batch size):

  make train-text-run

  → No rebuild needed. Config is mounted as a volume.

- **Code changed** (e.g. `src/pipeline/text_pipeline.py`):

  make train-text-rebuild

  → Rebuilds the Docker image with the latest code.

> `make train-text-run` automatically commits changes in `configs/` before starting
> the container, ensuring the Git commit logged in MLflow matches the actual config used.

## DVC — Data & Model Versioning

Data and models are **not stored in Git**. They are tracked via DVC and stored in DagsHub Storage.

#### Pull data and models

uv run dvc pull

#### After training: version a new model

## 1. Register the new model with DVC
uv run dvc add models/best_text_model.pt

## 2. Push model to DagsHub Storage
uv run dvc push

## 3. Commit the DVC pointer + config to Git
git add best_text_model.pt.dvc configs/
git commit -m "feat: text classifier v2 - macro-f1=0.88"
git push

### MLflow — Experiment Tracking

Experiments are tracked automatically during training and pushed to DagsHub.

View experiments at:

https://dagshub.com/<USERNAME>/<REPO>/experiments

Each run logs:
- Parameters (learning rate, batch size, epochs, …)
- Metrics (accuracy, macro-f1, loss, …)
- Git commit hash + branch
- Config file as artifact

### Reproducibility

Every training run is fully reproducible via three anchors:

- **Code**: Git commit hash (logged in MLflow)
- **Config**: `configs/` committed before each run + logged as MLflow artifact
- **Data & Model**: DVC pointer files (`data.dvc`, `best_text_model.pt.dvc`)

To reproduce a specific run:

git checkout <commit-hash>
uv run dvc pull
make train-text-run


------------------------------------------------------------------------

## BentoML Serving

FastAPI is no longer the primary serving layer in this repository. The supported serving path follows the BentoML workflow from the course material more closely:

1. prepare lightweight tokenizer/config assets
2. register the trained PyTorch model in the BentoML Model Store
3. serve a JWT-protected BentoML API
4. build and containerize the Bento artifact

Prepare assets and register the model in the BentoML Model Store:

``` bash
make prepare-bento-text-assets
make register-bento-text-model
```

This step requires a local `models/best_text_model.pt`. If the weight file is tracked via DVC, pull it before registration.

Run the local BentoML service:

``` bash
make serve-bento-text
```

`make serve-bento-text` now performs a preflight check and stops early with a clear error if `rakuten_text_classifier:latest` has not been registered in the local BentoML Model Store yet.

Available endpoints:

- `GET /health`
- `POST /login`
- `POST /predict`
- `POST /predict_batch`
- `GET /metrics` (enabled by BentoML by default)

Get a JWT and call the protected prediction endpoint:

``` bash
make token-bento-text
make predict-bento-text
```

Example request shapes:

``` json
{
  "credentials": {
    "username": "user123",
    "password": "password123"
  }
}
```

``` json
{
  "input_data": {
    "designation": "robe femme",
    "description": "bleu",
    "top_k": 3
  }
}
```

Build and package the Bento through the supported Makefile path:

``` bash
make build-bento-text
make containerize-bento-text
```

Start the packaged Bento container through Docker Compose:

``` bash
make docker-bento-up
```

`make build-bento-text` first prepares the local Hugging Face tokenizer assets, then registers `rakuten_text_classifier:latest` in the BentoML Model Store, and finally builds the Bento. The Bento itself references the registered model via the `models:` section in `bentofile.yaml` instead of bundling the raw `.pt` file directly.

Dependency note: `pyproject.toml` and `bentofile.yaml` are the authoritative sources for BentoML serving and packaging. `requirements.txt` is a broader exported development snapshot and is not used by the Bento build itself.
