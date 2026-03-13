## Run Image Pipeline

## Requirements

requirements_image.txt

### Training

python -m src.pipeline.image_pipeline --mode train

Number of samples used for training can be set in configs/image_train_config.yaml

### Hyperparameter Search

python -m src.pipeline.image_pipeline --mode random_search

### Evaluation

python -m src.pipeline.image_pipeline --mode evaluate

### Inference 

python -m src.pipeline.image_pipeline --mode inference --image_path data/images/image_train/image_1263597046_product_3804725264.jpg


## To Dos:

* train "final" model
 python -m src.pipeline.image_pipeline --mode random_search
 save best_model.pt, best_train_config.yaml & random_search_results.csv to google drive

 * dockerize

 * unit tests 


 # Text Training Pipeline

## Environment Setup

This project uses **uv** for environment and dependency management.

To install dependencies and set up the environment, run:

```bash
uv sync
```

## Running Training

Start training with the following command:

```bash
uv run python -m src.pipeline.text_pipeline --mode train
```

## Configuration

Training parameters and paths are configured in:

- `configs/text_train_config.yaml` 	6 training hyperparameters (epochs, batch size, learning rate, etc.)
- `configs/label_encoding.json` 	6 label to index mapping
- `artifacts/splits/` 	6 directory for train/val/test split ID files (auto-created or loaded)

Adjust these files to customize training behavior.

---

After training, the best model is saved to:

```
models/best_text_model.pt
```

Logs are saved in:

```
logs/text_training.log
```
