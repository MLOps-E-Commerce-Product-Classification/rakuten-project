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
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
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

    # Load and merge raw data
    x_df = pd.read_csv(x_data_csv_path, index_col=0)
    y_df = pd.read_csv(y_data_csv_path, index_col=0)
    df = pd.concat([x_df, y_df], axis=1).reset_index(drop=True)

    if description_col not in df.columns:
        df[description_col] = ""

    validate_dataframe_columns(df, {designation_col, label_col}, "Merged data")
    validate_labels_in_mapping(
        df,
        label_col=label_col,
        code_to_idx=label_encoding["code_to_idx"],
        df_name="Merged data",
    )

    # Apply preprocessing
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

    # Map label to index
    df["label"] = df[label_col].astype(str).map(label_encoding["code_to_idx"])

    # Create splits
    train_df, val_df, test_df = load_or_create_splits(
        df=df,
        label_col=label_col,
        split_ids_dir=split_ids_dir,
        seed=seed,
        force_new_split=force_new_split,
        val_size=val_size,
        test_size=test_size,
    )

    # Save processed splits
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
    )


if __name__ == "__main__":
    main()
