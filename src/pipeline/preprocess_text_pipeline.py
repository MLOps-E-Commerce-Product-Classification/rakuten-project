from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.text_preprocessing import (
    clean_text,
    load_text_preprocessing_config,
)
from src.training.run_text_training import (
    load_config,
    load_label_encoding,
    load_or_create_splits,
    set_seed,
    validate_dataframe_columns,
    validate_labels_in_mapping,
)


def preprocess_and_save(
    x_data_csv_path: str | Path,
    y_data_csv_path: str | Path,
    output_dir: str | Path,
    split_ids_dir: str | Path,
    train_config_path: str | Path,
    preprocessing_config_path: str | Path,
    label_encoding_path: str | Path,
    force_new_split: bool = False,
    pseudo_label_csv_path: str | Path = Path("data/raw/pseudo_labeled_samples.csv"),
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure pseudo_label_csv_path is a Path object
    if pseudo_label_csv_path:
        pseudo_label_csv_path = Path(pseudo_label_csv_path)

    # Load configuration files
    config = load_config(train_config_path)
    label_encoding = load_label_encoding(label_encoding_path)
    preprocessing_config = load_text_preprocessing_config(preprocessing_config_path)

    training_config = config.get("training", {})
    split_config = config.get("split", {})
    data_config = config.get("data", {})
    preprocessing_settings = preprocessing_config.get("preprocessing", {})

    seed = int(training_config.get("seed", 42))
    val_size = float(split_config.get("val_size", 0.1))
    test_size = float(split_config.get("test_size", 0.1))

    label_col = data_config.get("label_col", "prdtypecode")
    designation_col = data_config.get("designation_col", "designation")
    description_col = data_config.get("description_col", "description")

    remove_html = bool(preprocessing_settings.get("remove_html", True))
    lowercase = bool(preprocessing_settings.get("lowercase", False))
    combine_fields = bool(preprocessing_settings.get("combine_fields", True))
    separator = str(preprocessing_settings.get("separator", " "))

    set_seed(seed)

    # Load and merge base raw datasets
    print(f"Loading base data from {x_data_csv_path} and {y_data_csv_path}...")
    x_df = pd.read_csv(x_data_csv_path, index_col=0)
    y_df = pd.read_csv(y_data_csv_path, index_col=0)
    base_df = pd.concat([x_df, y_df], axis=1).reset_index(drop=True)

    dfs = [base_df]

    # OPTIONAL: load pseudo-labeled samples if the file exists
    # Internal hardcoded re-assignment removed to allow flexible path usage
    if pseudo_label_csv_path and pseudo_label_csv_path.exists():
        try:
            pseudo_df = pd.read_csv(pseudo_label_csv_path)
            # Validate required columns in pseudo-labeled data
            required_cols = {designation_col, label_col}
            missing = required_cols - set(pseudo_df.columns)
            if not missing:
                if description_col not in pseudo_df.columns:
                    pseudo_df[description_col] = ""
                print(
                    f"Loaded {len(pseudo_df)} pseudo-labeled samples from {pseudo_label_csv_path}"
                )
                dfs.append(pseudo_df)
            else:
                print(
                    f"Warning: Pseudo-labeled CSV missing columns {missing}. Skipping optional data."
                )
        except Exception as e:
            print(f"Warning: Could not read pseudo-labels file: {e}")
    else:
        print(
            f"Info: Optional pseudo-labeled file not found at {pseudo_label_csv_path}. Proceeding with base data only."
        )

    # Create final merged dataset
    df = pd.concat(dfs, ignore_index=True)
    if description_col not in df.columns:
        df[description_col] = ""

    validate_dataframe_columns(df, {designation_col, label_col}, "Merged data")
    validate_labels_in_mapping(
        df,
        label_col=label_col,
        code_to_idx=label_encoding["code_to_idx"],
        df_name="Merged data",
    )

    # Apply text cleaning and preprocessing
    print("Applying text preprocessing...")
    df["designation_clean"] = df[designation_col].apply(
        lambda x: clean_text(x, remove_html=remove_html, lowercase=lowercase)
    )
    df["description_clean"] = df[description_col].apply(
        lambda x: clean_text(x, remove_html=remove_html, lowercase=lowercase)
    )

    def combine(row):
        parts = [p for p in [row["designation_clean"], row["description_clean"]] if p]
        return (
            separator.join(parts).strip()
            if combine_fields
            else row["designation_clean"]
        )

    df["text"] = df.apply(combine, axis=1).replace("", "[EMPTY_TEXT]")

    # Map target labels to numerical indices
    df["label"] = df[label_col].astype(str).map(label_encoding["code_to_idx"])

    # Split data into train, validation, and test sets
    train_df, val_df, test_df = load_or_create_splits(
        df=df,
        label_col=label_col,
        split_ids_dir=split_ids_dir,
        seed=seed,
        force_new_split=force_new_split,
        val_size=val_size,
        test_size=test_size,
    )

    # Save processed splits to disk
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Saved processed splits to {output_dir}")
    print(f"  train: {len(train_df)} rows")
    print(f"  val:   {len(val_df)} rows")
    print(f"  test:  {len(test_df)} rows")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess text data and save splits."
    )
    parser.add_argument(
        "--x_data_csv_path", type=str, default="data/raw/X_train_update.csv"
    )
    parser.add_argument(
        "--y_data_csv_path", type=str, default="data/raw/Y_train_CVw08PX.csv"
    )
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--split_ids_dir", type=str, default="artifacts/splits")
    parser.add_argument(
        "--train_config_path", type=str, default="configs/text_train_config.yaml"
    )
    parser.add_argument(
        "--preprocessing_config_path",
        type=str,
        default="configs/text_preprocessing_config.yaml",
    )
    parser.add_argument(
        "--label_encoding_path", type=str, default="configs/label_encoding.json"
    )
    parser.add_argument(
        "--pseudo_label_csv_path",
        type=str,
        default="data/raw/pseudo_labeled_samples.csv",
    )
    parser.add_argument("--force_new_split", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    preprocess_and_save(
        x_data_csv_path=args.x_data_csv_path,
        y_data_csv_path=args.y_data_csv_path,
        output_dir=args.output_dir,
        split_ids_dir=args.split_ids_dir,
        train_config_path=args.train_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        label_encoding_path=args.label_encoding_path,
        force_new_split=args.force_new_split,
        pseudo_label_csv_path=args.pseudo_label_csv_path,
    )


if __name__ == "__main__":
    main()
