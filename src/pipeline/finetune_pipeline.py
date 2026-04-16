from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finetuning pipeline entry point.")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["finetune"],
        help="Pipeline mode.",
    )
    parser.add_argument("--processed_data_dir", type=str, default="data/processed")
    parser.add_argument(
        "--new_processed_data_dir", type=str, default="data/processed_new"
    )
    parser.add_argument(
        "--finetune_config_path", type=str, default="configs/text_finetune_config.yaml"
    )
    parser.add_argument(
        "--preprocessing_config_path",
        type=str,
        default="configs/text_preprocessing_config.yaml",
    )
    parser.add_argument(
        "--model_save_path", type=str, default="models/best_text_model.pt"
    )
    parser.add_argument(
        "--label_encoding_path", type=str, default="configs/label_encoding.json"
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "finetune":
        from src.training.run_text_finetuning import run_text_finetuning

        history, label_encoding = run_text_finetuning(
            processed_data_dir=args.processed_data_dir,
            new_processed_data_dir=args.new_processed_data_dir,
            finetune_config_path=args.finetune_config_path,
            preprocessing_config_path=args.preprocessing_config_path,
            model_save_path=args.model_save_path,
            label_encoding_path=args.label_encoding_path,
        )

        print("Finetuning finished.")
        print(f"Best val macro-F1: {max(history['val_macro_f1']):.4f}")
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
