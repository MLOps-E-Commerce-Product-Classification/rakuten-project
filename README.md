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