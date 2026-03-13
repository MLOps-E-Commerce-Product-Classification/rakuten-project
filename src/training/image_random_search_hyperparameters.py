from pathlib import Path
import copy
import json
import logging
import random
import shutil

import pandas as pd
import yaml

from src.training.run_image_training import run_image_training


LOG_PATH = Path("logs")
LOG_PATH.mkdir(parents=True, exist_ok=True)

SEARCH_PATH = Path("hyperparameter_search")
SEARCH_PATH.mkdir(parents=True, exist_ok=True)

SEARCH_CONFIGS_PATH = SEARCH_PATH / "trial_configs"
SEARCH_CONFIGS_PATH.mkdir(parents=True, exist_ok=True)

SEARCH_RESULTS_PATH = SEARCH_PATH / "results"
SEARCH_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

TEMP_MODELS_PATH = SEARCH_PATH / "temp_models"
TEMP_MODELS_PATH.mkdir(parents=True, exist_ok=True)

CONFIGS_PATH = Path("configs")
CONFIGS_PATH.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: str | Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


SEARCH_LOGGER = setup_logger("random_search", LOG_PATH / "random_search.log")


def load_yaml_config(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml_config(config: dict, output_path: str | Path) -> None:
    with Path(output_path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_json(data: dict, output_path: str | Path) -> None:
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def set_nested_value(config: dict, key: str, value) -> None:
    keys = key.split(".")
    current = config
    for k in keys[:-1]:
        current = current.setdefault(k, {})
    current[keys[-1]] = value


def sample_random_config(base_config: dict, search_space: dict):
    sampled_config = copy.deepcopy(base_config)
    sampled_params = {}

    for key, values in search_space.items():
        sampled_value = random.choice(values)
        set_nested_value(sampled_config, key, sampled_value)
        sampled_params[key] = sampled_value

    return sampled_config, sampled_params


def get_optimization_score(history: dict, metric: str):
    values = history[metric]
    if metric == "val_loss":
        return float(min(values))
    return float(max(values))


def is_better_score(score, best, metric):
    if best is None:
        return True
    if metric == "val_loss":
        return score < best
    return score > best


def run_random_search(
    x_data_csv_path,
    y_data_csv_path,
    image_dir,
    split_ids_dir,
    force_new_split=False,
    base_train_config_path="configs/image_train_config.yaml",
    preprocessing_config_path="configs/image_preprocessing_config.yaml",
    search_space_config_path="configs/image_parameter_search_space.yaml",
    final_best_config_path="configs/image_best_train_config.yaml",
    final_best_model_path="search/best_model.pt",
    label_encoding_path="configs/label_encoding.json",
    random_seed=42,
):

    random.seed(random_seed)

    base_config = load_yaml_config(base_train_config_path)
    search_config = load_yaml_config(search_space_config_path)

    search_space = search_config["search_space"]
    rs_cfg = search_config["random_search"]

    n_trials = rs_cfg.get("n_trials", 10)
    optimization_metric = rs_cfg.get("optimization_metric", "val_macro_f1")

    trial_epochs = rs_cfg.get("trial_epochs", 3)
    final_epochs = rs_cfg.get("final_epochs", 15)

    results = []

    best_score = None
    best_config = None
    best_trial_config_path = None

    for trial in range(1, n_trials + 1):

        SEARCH_LOGGER.info(f"Trial {trial}/{n_trials}")

        sampled_config, sampled_params = sample_random_config(
            base_config,
            search_space
        )

        # SPEED TRICK
        sampled_config["training"]["epochs"] = trial_epochs

        trial_config_path = SEARCH_CONFIGS_PATH / f"trial_{trial:03d}.yaml"
        trial_model_path = TEMP_MODELS_PATH / f"trial_{trial:03d}.pt"

        save_yaml_config(sampled_config, trial_config_path)

        try:

            _, history, _ = run_image_training(
                x_data_csv_path=x_data_csv_path,
                y_data_csv_path=y_data_csv_path,
                image_dir=image_dir,
                split_ids_dir=split_ids_dir,
                force_new_split=force_new_split,
                train_config_path=trial_config_path,
                preprocessing_config_path=preprocessing_config_path,
                model_save_path=trial_model_path,
                label_encoding_path=label_encoding_path,
                use_best_config_if_available=False,
            )

            score = get_optimization_score(history, optimization_metric)

            result_row = {
                "trial": trial,
                **sampled_params,
                optimization_metric: score
            }

            results.append(result_row)

            if is_better_score(score, best_score, optimization_metric):

                best_score = score
                best_config = sampled_config
                best_trial_config_path = trial_config_path

                SEARCH_LOGGER.info(f"New best score: {score}")

        except Exception as e:

            SEARCH_LOGGER.exception(f"Trial {trial} failed: {e}")

        if trial_model_path.exists():
            trial_model_path.unlink()

    # SAVE CSV
    df = pd.DataFrame(results)
    df = df.sort_values(optimization_metric, ascending=False)

    csv_path = SEARCH_PATH / "random_search_results.csv"
    df.to_csv(csv_path, index=False)

    SEARCH_LOGGER.info(f"Saved CSV results to {csv_path}")

    # FINAL FULL TRAINING
    SEARCH_LOGGER.info("Starting final full training of best config")

    best_config["training"]["epochs"] = final_epochs

    save_yaml_config(best_config, final_best_config_path)

    run_image_training(
        x_data_csv_path=x_data_csv_path,
        y_data_csv_path=y_data_csv_path,
        image_dir=image_dir,
        split_ids_dir=split_ids_dir,
        force_new_split=False,
        train_config_path=final_best_config_path,
        preprocessing_config_path=preprocessing_config_path,
        model_save_path=final_best_model_path,
        label_encoding_path=label_encoding_path,
        use_best_config_if_available=False,
    )

    SEARCH_LOGGER.info("Random search finished")

    return {
        "best_score": best_score,
        "best_config": str(final_best_config_path),
        "best_model": str(final_best_model_path),
        "results_csv": str(csv_path)
    }