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
uv sync
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
python -m src.pipeline.image_pipeline --mode inference --image_path data/images/image_train/image_1263597046_product_3804725264.jpg
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
