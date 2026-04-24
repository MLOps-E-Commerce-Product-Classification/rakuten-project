from pathlib import Path
import json

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.image_dataset import RakutenImageDataset
from src.evaluation.image_evaluate import evaluate_model, save_evaluation_results
from src.models.image_classifier import build_image_model


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_encoding(label_encoding_path: str | Path) -> dict:
    label_encoding_path = Path(label_encoding_path)

    if not label_encoding_path.exists():
        raise FileNotFoundError(f"Label encoding file not found: {label_encoding_path}")

    with label_encoding_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split_ids(split_path: str | Path) -> pd.Index:
    split_path = Path(split_path)

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split_df = pd.read_csv(split_path)

    if "row_id" not in split_df.columns:
        raise ValueError(f"Split file {split_path} must contain a 'row_id' column.")

    return pd.Index(split_df["row_id"].astype(int).tolist())


def validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    df_name: str,
) -> None:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{df_name} is missing required columns: {sorted(missing_columns)}"
        )


def validate_labels_in_encoding(
    df: pd.DataFrame,
    label_col: str,
    code_to_idx: dict,
    df_name: str,
) -> None:
    labels_as_str = df[label_col].astype(str)
    unknown_labels = sorted(set(labels_as_str) - set(code_to_idx.keys()))

    if unknown_labels:
        raise ValueError(
            f"{df_name} contains labels not present in the predefined label encoding: "
            f"{unknown_labels[:10]}" + (" ..." if len(unknown_labels) > 10 else "")
        )


def run_image_evaluation(
    x_data_csv_path: str | Path,
    y_data_csv_path: str | Path,
    image_dir: str | Path,
    split_ids_dir: str | Path,
    train_config_path: str | Path = "configs/image_train_config.yaml",
    eval_config_path: str | Path = "configs/image_evaluate_config.yaml",
    preprocessing_config_path: str | Path = "configs/image_preprocessing_config.yaml",
    model_weights_path: str | Path = "models/best_image_model.pt",
    label_encoding_path: str | Path = "configs/label_encoding.json",
    results_output_path: str | Path = "results/evaluation_results.json",
) -> dict:
    """
    Run end-to-end image evaluation on the saved test split.

    Parameters
    ----------
    x_data_csv_path : str | Path
        Path to X data CSV.
    y_data_csv_path : str | Path
        Path to Y data CSV.
    image_dir : str | Path
        Directory containing images.
    split_ids_dir : str | Path
        Directory containing saved train/val/test split ids.
    train_config_path : str | Path
        Path to training YAML config.
    eval_config_path : str | Path
        Path to evaluation YAML config.
    preprocessing_config_path : str | Path
        Path to preprocessing YAML config.
    model_weights_path : str | Path
        Path to saved model weights.
    label_encoding_path : str | Path
        Path to predefined label encoding JSON.
    results_output_path : str | Path
        Path to save evaluation results JSON.

    Returns
    -------
    dict
        Evaluation results dictionary.
    """
    train_config = load_config(train_config_path)
    eval_config = load_config(eval_config_path)
    label_encoding = load_label_encoding(label_encoding_path)

    training_config = train_config.get("training", {})
    model_config = train_config.get("model", {})
    data_config = train_config.get("data", {})
    metrics_config = eval_config.get("metrics", {})

    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))

    model_name = model_config.get("name", "efficientnet_b0")
    freeze_backbone = bool(model_config.get("freeze_backbone", False))

    image_id_col = data_config.get("image_id_col", "imageid")
    product_id_col = data_config.get("product_id_col", "productid")
    label_col = data_config.get("label_col", "prdtypecode")
    return_quality_report = bool(data_config.get("return_quality_report", False))

    compute_per_class_f1 = bool(metrics_config.get("compute_per_class_f1", True))
    main_metric = metrics_config.get("main_metric", "macro_f1")

    x_data_csv_path = Path(x_data_csv_path)
    y_data_csv_path = Path(y_data_csv_path)
    image_dir = Path(image_dir)
    split_ids_dir = Path(split_ids_dir)
    model_weights_path = Path(model_weights_path)
    results_output_path = Path(results_output_path)
    label_encoding_path = Path(label_encoding_path)

    if not x_data_csv_path.exists():
        raise FileNotFoundError(f"X data CSV not found: {x_data_csv_path}")
    if not y_data_csv_path.exists():
        raise FileNotFoundError(f"Y data CSV not found: {y_data_csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")

    test_ids_path = split_ids_dir / "test_ids.csv"
    test_ids = load_split_ids(test_ids_path)

    x_df = pd.read_csv(x_data_csv_path)
    y_df = pd.read_csv(y_data_csv_path)

    if len(x_df) != len(y_df):
        raise ValueError(
            f"X and Y CSVs must have the same number of rows, got {len(x_df)} and {len(y_df)}."
        )

    df = pd.concat([x_df, y_df], axis=1)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    required_columns = {image_id_col, product_id_col, label_col}
    validate_dataframe_columns(df, required_columns, "Merged evaluation data")

    code_to_idx = label_encoding["code_to_idx"]
    validate_labels_in_encoding(
        df,
        label_col=label_col,
        code_to_idx=code_to_idx,
        df_name="Merged evaluation data",
    )

    missing_test_ids = sorted(set(test_ids.tolist()) - set(df.index.tolist()))
    if missing_test_ids:
        raise ValueError(
            f"Some saved test ids are not present in the merged dataframe: "
            f"{missing_test_ids[:10]}" + (" ..." if len(missing_test_ids) > 10 else "")
        )

    test_df = df.loc[test_ids].copy()

    if len(test_df) == 0:
        raise ValueError("Test split is empty. Cannot run evaluation.")

    eval_dataset = RakutenImageDataset(
        dataframe=test_df,
        image_dir=image_dir,
        config_path=preprocessing_config_path,
        image_id_col=image_id_col,
        product_id_col=product_id_col,
        label_col=label_col,
        return_quality_report=return_quality_report,
        label_encoding_path=label_encoding_path,
    )

    pin_memory = torch.cuda.is_available()

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    num_classes = len(label_encoding["classes"])

    model = build_image_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=freeze_backbone,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state_dict)

    criterion = torch.nn.CrossEntropyLoss()

    results = evaluate_model(
        model=model,
        dataloader=eval_dataloader,
        criterion=criterion,
        device=device,
        config_path=eval_config_path,
        num_classes=num_classes,
    )

    results["metadata"] = {
        "model_name": model_name,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "compute_per_class_f1": compute_per_class_f1,
        "main_metric": main_metric,
        "model_weights_path": str(model_weights_path),
        "label_encoding_path": str(label_encoding_path),
        "x_data_csv_path": str(x_data_csv_path),
        "y_data_csv_path": str(y_data_csv_path),
        "split_ids_dir": str(split_ids_dir),
        "test_ids_path": str(test_ids_path),
        "image_dir": str(image_dir),
        "test_size": int(len(test_df)),
    }

    save_evaluation_results(results, results_output_path)

    return results


if __name__ == "__main__":
    results = run_image_evaluation(
        x_data_csv_path="data/X_train_update.csv",
        y_data_csv_path="data/Y_train_CVw08PX.csv",
        image_dir="data/images/image_train",
        split_ids_dir="artifacts/splits",
        train_config_path="configs/image_train_config.yaml",
        eval_config_path="configs/image_evaluate_config.yaml",
        preprocessing_config_path="configs/image_preprocessing_config.yaml",
        model_weights_path="models/best_image_model.pt",
        label_encoding_path="configs/label_encoding.json",
        results_output_path="results/evaluation_results.json",
    )

    print("Evaluation finished.")
    print(f"Main metric: {results['main_metric']} = {results['main_metric_value']:.4f}")
    print("Results saved to: results/evaluation_results.json")
