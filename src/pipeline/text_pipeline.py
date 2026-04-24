import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified pipeline entry point for text training, evaluation, "
            "inference, and random search."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate", "inference", "random_search"],
        help="Which pipeline step to run.",
    )

    # ----------------------------------------------------
    # Data inputs
    # ----------------------------------------------------
    parser.add_argument(
        "--x_data_csv_path",
        type=str,
        default="data/raw/X_train_update.csv",
        help="Path to X data CSV.",
    )
    parser.add_argument(
        "--y_data_csv_path",
        type=str,
        default="data/raw/Y_train_CVw08PX.csv",
        help="Path to Y data CSV.",
    )

    # ----------------------------------------------------
    # Data preprocessing
    # ----------------------------------------------------
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/processed",
        help="Path to preprocessed data directory.",
    )

    # ----------------------------------------------------
    # Split handling
    # ----------------------------------------------------
    parser.add_argument(
        "--split_ids_dir",
        type=str,
        default="artifacts/splits",
        help="Directory where train/val/test split ids are stored.",
    )
    parser.add_argument(
        "--force_new_split",
        action="store_true",
        help="Create a new train/val/test split even if saved split ids already exist.",
    )

    # ----------------------------------------------------
    # Config paths
    # ----------------------------------------------------
    parser.add_argument(
        "--train_config_path",
        type=str,
        default="configs/text_train_config.yaml",
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--best_train_config_path",
        type=str,
        default="configs/text_best_train_config.yaml",
        help="Path to best training config YAML.",
    )
    parser.add_argument(
        "--preprocessing_config_path",
        type=str,
        default="configs/text_preprocessing_config.yaml",
        help="Path to preprocessing config YAML.",
    )
    parser.add_argument(
        "--eval_config_path",
        type=str,
        default="configs/text_evaluate_config.yaml",
        help="Path to evaluation config YAML.",
    )
    parser.add_argument(
        "--search_space_config_path",
        type=str,
        default="configs/text_parameter_search_space.yaml",
        help="Path to random search config YAML.",
    )
    parser.add_argument(
        "--label_encoding_path",
        type=str,
        default="configs/label_encoding.json",
        help="Path to predefined label encoding JSON.",
    )

    # ----------------------------------------------------
    # Model / output paths
    # ----------------------------------------------------
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="models/best_text_model.pt",
        help="Path to save trained model.",
    )
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="models/best_text_model.pt",
        help="Path to model weights for evaluation or inference.",
    )
    parser.add_argument(
        "--results_output_path",
        type=str,
        default="results/text_evaluation_results.json",
        help="Path to save evaluation results.",
    )
    parser.add_argument(
        "--inference_output_path",
        type=str,
        default="results/text_inference_results.json",
        help="Path to save inference results.",
    )

    # ----------------------------------------------------
    # Random search outputs
    # ----------------------------------------------------
    parser.add_argument(
        "--final_best_config_path",
        type=str,
        default="configs/text_best_train_config.yaml",
        help="Path to export best config during random search.",
    )
    parser.add_argument(
        "--final_best_model_path",
        type=str,
        default="search/best_text_model.pt",
        help="Path to export best model during random search.",
    )

    # ----------------------------------------------------
    # Inference inputs
    # ----------------------------------------------------
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text input for inference.",
    )
    parser.add_argument(
        "--texts",
        type=str,
        nargs="*",
        default=None,
        help="Multiple text inputs for inference.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k predictions for inference.",
    )
    parser.add_argument(
        "--mlflow_run_id",
        type=str,
        default=None,
        help="If provided, log evaluation results to this existing MLflow run ID.",
    )

    # ----------------------------------------------------
    # Misc
    # ----------------------------------------------------
    parser.add_argument(
        "--random_seed",
        type=int,
        default=17,
        help="Random seed for random search.",
    )

    return parser


def run_train_mode(args: argparse.Namespace) -> None:
    from src.training.run_text_training import run_text_training

    history, label_encoding = run_text_training(
        processed_data_dir=args.processed_data_dir,
        train_config_path=args.train_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        model_save_path=args.model_save_path,
        label_encoding_path=args.label_encoding_path,
    )

    print("Training finished.")
    print(f"Model saved to: {args.model_save_path}")
    print(f"Label encoding loaded from: {args.label_encoding_path}")
    print(f"Number of classes: {len(label_encoding['classes'])}")
    print(f"Best validation macro-F1: {max(history['val_macro_f1']):.4f}")


def run_evaluate_mode(args: argparse.Namespace) -> None:
    from src.evaluation.run_text_evaluation import run_text_evaluation

    results = run_text_evaluation(
        x_data_csv_path=args.x_data_csv_path,
        y_data_csv_path=args.y_data_csv_path,
        split_ids_dir=args.split_ids_dir,
        train_config_path=args.best_train_config_path,
        eval_config_path=args.eval_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        model_weights_path=args.model_weights_path,
        label_encoding_path=args.label_encoding_path,
        results_output_path=args.results_output_path,
        mlflow_run_id=args.mlflow_run_id,  # ← NEU
    )

    print("Evaluation finished.")
    print(f"Main metric: {results['main_metric']} = {results['main_metric_value']:.4f}")
    print(f"Results saved to: {args.results_output_path}")


def run_inference_mode(args: argparse.Namespace) -> None:
    from src.inference.run_text_inference import run_text_inference

    if args.text is None and not args.texts:
        raise ValueError("For inference mode, provide either --text or --texts.")

    text_input = args.text if args.text is not None else args.texts

    results = run_text_inference(
        text_input=text_input,
        train_config_path=args.best_train_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        model_weights_path=args.model_weights_path,
        label_encoding_path=args.label_encoding_path,
        output_path=args.inference_output_path,
        top_k=args.top_k,
    )

    print("Inference finished.")
    print(f"Results saved to: {args.inference_output_path}")

    if isinstance(results, dict):
        print(f"Predicted Rakuten code: {results['predicted_rakuten_code']}")


def run_random_search_mode(args: argparse.Namespace) -> None:
    from src.training.text_random_search_hyperparameters import run_random_search

    summary = run_random_search(
        x_data_csv_path=args.x_data_csv_path,
        y_data_csv_path=args.y_data_csv_path,
        split_ids_dir=args.split_ids_dir,
        force_new_split=args.force_new_split,
        base_train_config_path=args.train_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        search_space_config_path=args.search_space_config_path,
        final_best_config_path=args.final_best_config_path,
        final_best_model_path=args.final_best_model_path,
        label_encoding_path=args.label_encoding_path,
        random_seed=args.random_seed,
    )

    print("Random search finished.")
    print(f"Best score: {summary['best_score']}")
    if summary["best_trial"] is not None:
        print(f"Best trial: {summary['best_trial']['trial']}")
        print(f"Best config: {summary['best_trial']['best_config_path']}")
        print(f"Best model: {summary['best_trial']['best_model_path']}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        run_train_mode(args)
    elif args.mode == "evaluate":
        run_evaluate_mode(args)
    elif args.mode == "inference":
        run_inference_mode(args)
    elif args.mode == "random_search":
        run_random_search_mode(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
