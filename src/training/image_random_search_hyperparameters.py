from pathlib import Path
import copy
import json
import logging
import random
import shutil

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
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml_config(config: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_json(data: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def set_nested_value(config: dict, key: str, value) -> None:
    """
    Set a nested config value using dot notation, e.g. 'training.learning_rate'.
    """
    keys = key.split(".")
    current = config

    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]

    current[keys[-1]] = value


def sample_random_config(base_config: dict, search_space: dict) -> tuple[dict, dict]:
    """
    Sample one random configuration from the search space.

    Returns
    -------
    tuple[dict, dict]
        sampled_config, sampled_params
    """
    sampled_config = copy.deepcopy(base_config)
    sampled_params = {}

    for key, candidate_values in search_space.items():
        if not isinstance(candidate_values, list) or len(candidate_values) == 0:
            raise ValueError(
                f"Search space for '{key}' must be a non-empty list."
            )

        sampled_value = random.choice(candidate_values)
        set_nested_value(sampled_config, key, sampled_value)
        sampled_params[key] = sampled_value

    return sampled_config, sampled_params


def get_optimization_score(history: dict, optimization_metric: str) -> float:
    """
    Extract the optimization score from training history.

    Supported metrics:
    - val_macro_f1
    - val_accuracy
    - val_loss
    """
    if optimization_metric not in history:
        raise ValueError(
            f"Optimization metric '{optimization_metric}' not found in history. "
            f"Available keys: {list(history.keys())}"
        )

    values = history[optimization_metric]
    if not values:
        raise ValueError(
            f"No values recorded for optimization metric '{optimization_metric}'."
        )

    if optimization_metric == "val_loss":
        return float(min(values))

    return float(max(values))


def is_better_score(
    current_score: float,
    best_score: float | None,
    optimization_metric: str,
) -> bool:
    if best_score is None:
        return True

    if optimization_metric == "val_loss":
        return current_score < best_score

    return current_score > best_score


def safe_delete_file(file_path: str | Path) -> None:
    file_path = Path(file_path)
    if file_path.exists():
        file_path.unlink()


def run_random_search(
    x_data_csv_path: str | Path,
    y_data_csv_path: str | Path,
    image_dir: str | Path,
    split_ids_dir: str | Path,
    force_new_split: bool = False,
    base_train_config_path: str | Path = "configs/image_train_config.yaml",
    preprocessing_config_path: str | Path = "configs/image_preprocessing_config.yaml",
    search_space_config_path: str | Path = "configs/image_parameter_search_space.yaml",
    final_best_config_path: str | Path = "configs/image_best_train_config.yaml",
    final_best_model_path: str | Path = "search/best_model.pt",
    label_encoding_path: str | Path = "configs/label_encoding.json",
    random_seed: int = 42,
) -> dict:
    """
    Run random search over hyperparameters for image training.

    Parameters
    ----------
    x_data_csv_path : str | Path
        Path to X data CSV.
    y_data_csv_path : str | Path
        Path to Y data CSV.
    image_dir : str | Path
        Directory containing images.
    split_ids_dir : str | Path
        Directory with stored train/val/test split ids.
    force_new_split : bool
        Whether to force a new split. Usually keep this False during random search.
    base_train_config_path : str | Path
        Path to base training config.
    preprocessing_config_path : str | Path
        Path to preprocessing config.
    search_space_config_path : str | Path
        Path to random search config.
    final_best_config_path : str | Path
        Path where the best config should be exported.
    final_best_model_path : str | Path
        Path where the best model should be exported.
    label_encoding_path : str | Path
        Path to predefined label encoding JSON.
    random_seed : int
        Random seed for hyperparameter sampling.

    Returns
    -------
    dict
        Summary of all trials and best trial.
    """
    random.seed(random_seed)

    base_config = load_yaml_config(base_train_config_path)
    search_config = load_yaml_config(search_space_config_path)

    search_space = search_config.get("search_space", {})
    random_search_config = search_config.get("random_search", {})

    n_trials = int(random_search_config.get("n_trials", 10))
    optimization_metric = random_search_config.get(
        "optimization_metric", "val_macro_f1"
    )

    if not search_space:
        raise ValueError("Search space is empty.")

    final_best_config_path = Path(final_best_config_path)
    final_best_model_path = Path(final_best_model_path)

    final_best_config_path.parent.mkdir(parents=True, exist_ok=True)
    final_best_model_path.parent.mkdir(parents=True, exist_ok=True)

    all_trials = []
    best_score = None
    best_trial = None

    SEARCH_LOGGER.info(
        f"Starting random search with n_trials={n_trials}, "
        f"optimization_metric={optimization_metric}, "
        f"force_new_split={force_new_split}"
    )

    for trial_idx in range(1, n_trials + 1):
        SEARCH_LOGGER.info(f"Starting trial {trial_idx}/{n_trials}")

        sampled_config, sampled_params = sample_random_config(base_config, search_space)

        trial_config_path = SEARCH_CONFIGS_PATH / f"trial_{trial_idx:03d}.yaml"
        trial_model_path = TEMP_MODELS_PATH / f"trial_{trial_idx:03d}.pt"

        save_yaml_config(sampled_config, trial_config_path)

        try:
            _, history, label_encoding = run_image_training(
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

            trial_result = {
                "trial": trial_idx,
                "status": "success",
                "sampled_params": sampled_params,
                "config_path": str(trial_config_path),
                "optimization_metric": optimization_metric,
                "score": score,
                "num_classes": len(label_encoding["classes"]),
                "history": history,
            }

            SEARCH_LOGGER.info(
                f"Trial {trial_idx} finished successfully with "
                f"{optimization_metric}={score:.4f}"
            )

            if is_better_score(score, best_score, optimization_metric):
                best_score = score

                shutil.copy2(trial_model_path, final_best_model_path)
                shutil.copy2(trial_config_path, final_best_config_path)

                best_trial = {
                    **trial_result,
                    "best_model_path": str(final_best_model_path),
                    "best_config_path": str(final_best_config_path),
                }

                SEARCH_LOGGER.info(
                    f"New best trial: {trial_idx} with "
                    f"{optimization_metric}={score:.4f}"
                )
                SEARCH_LOGGER.info(
                    f"Exported best config to {final_best_config_path}, "
                    f"best model to {final_best_model_path}"
                )

            safe_delete_file(trial_model_path)

        except Exception as e:
            trial_result = {
                "trial": trial_idx,
                "status": "failed",
                "sampled_params": sampled_params,
                "config_path": str(trial_config_path),
                "optimization_metric": optimization_metric,
                "error": str(e),
            }

            SEARCH_LOGGER.exception(
                f"Trial {trial_idx} failed with error: {e}"
            )

            safe_delete_file(trial_model_path)

        all_trials.append(trial_result)

        trial_result_path = SEARCH_RESULTS_PATH / f"trial_{trial_idx:03d}.json"
        save_json(trial_result, trial_result_path)

    summary = {
        "n_trials": n_trials,
        "optimization_metric": optimization_metric,
        "best_score": best_score,
        "best_trial": best_trial,
        "all_trials": all_trials,
    }

    summary_path = SEARCH_RESULTS_PATH / "random_search_summary.json"
    save_json(summary, summary_path)

    SEARCH_LOGGER.info(
        f"Random search finished. Best score for {optimization_metric}: {best_score}"
    )
    SEARCH_LOGGER.info(f"Summary saved to {summary_path}")

    return summary


if __name__ == "__main__":
    summary = run_random_search(
        x_data_csv_path="data/X_train_update.csv",
        y_data_csv_path="data/Y_train_CVw08PX.csv",
        image_dir="data/images/image_train",
        split_ids_dir="artifacts/splits",
        force_new_split=False,
        base_train_config_path="configs/image_train_config.yaml",
        preprocessing_config_path="configs/image_preprocessing_config.yaml",
        search_space_config_path="configs/image_parameter_search_space.yaml",
        final_best_config_path="configs/image_best_train_config.yaml",
        final_best_model_path="search/best_model.pt",
        label_encoding_path="configs/label_encoding.json",
        random_seed=42,
    )

    print("Random search finished.")
    print(f"Best score: {summary['best_score']}")
    if summary["best_trial"] is not None:
        print(f"Best trial: {summary['best_trial']['trial']}")
        print(f"Best config: {summary['best_trial']['best_config_path']}")
        print(f"Best model: {summary['best_trial']['best_model_path']}")