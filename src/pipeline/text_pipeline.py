import argparse

from src.training.run_text_training import run_text_training
from src.evaluation.run_text_evaluation import run_text_evaluation
from src.inference.run_text_inference import run_text_inference
from src.inference.text_random_search_hyperparameters import run_random_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified pipeline entry point for text training, evaluation, inference, and random search."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate", "inference", "random_search"],
        help="Which pipeline step to run.",
    )

    # Data Paths
    parser.add_argument(
        "--train_csv_path",
        type=str,
        default="data/train_split.csv",
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--val_csv_path",
        type=str,
        default="data/val_split.csv",
        help="Path to validation CSV.",
    )
    parser.add_argument(
        "--eval_csv_path",
        type=str,
        default="data/val_split.csv",
        help="Path to evaluation CSV.",
    )

    # Config Paths
    parser.add_argument(
        "--train_config_path",
        type=str,
        default="configs/text_train_config.yaml",
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--best_train_config_path",
        type=str,
        default="configs/text_train_config.yaml",
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

    # Model & Output Paths
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
        "--label_mapping_path",
        type=str,
        default="artifacts/label_mapping.json",
        help="Path to label mapping JSON.",
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

    # Random Search Paths
    parser.add_argument(
        "--final_best_config_path",
        type=str,
        default="configs/best_text_train_config.yaml",
        help="Path to export best config during random search.",
    )
    parser.add_argument(
        "--final_best_model_path",
        type=str,
        default="search/best_text_model.pt",
        help="Path to export best model during random search.",
    )
    parser.add_argument(
        "--final_best_label_mapping_path",
        type=str,
        default="artifacts/best_text_label_mapping.json",
        help="Path to export best label mapping during random search.",
    )

    # Inference specific
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

    # Misc
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for random search.",
    )

    return parser


def run_train_mode(args: argparse.Namespace) -> None:
    trained_model, history, label_encoder = run_text_training(
        train_csv_path=args.train_csv_path,
        val_csv_path=args.val_csv_path,
        train_config_path=args.train_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        model_save_path=args.model_save_path,
        label_mapping_path=args.label_mapping_path,
    )

    print("Training finished.")
    print(f"Model saved to: {args.model_save_path}")
    print(f"Label mapping saved to: {args.label_mapping_path}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Best validation macro-F1: {max(history['val_macro_f1']):.4f}")


def run_evaluate_mode(args: argparse.Namespace) -> None:
    results = run_text_evaluation(
        eval_csv_path=args.eval_csv_path,
        train_config_path=args.best_train_config_path,
        eval_config_path=args.eval_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        model_weights_path=args.model_weights_path,
        label_mapping_path=args.label_mapping_path,
        results_output_path=args.results_output_path,
    )

    print("Evaluation finished.")
    print(f"Main metric: {results['main_metric']} = {results['main_metric_value']:.4f}")
    print(f"Results saved to: {args.results_output_path}")


def run_inference_mode(args: argparse.Namespace) -> None:
    if args.text is None and not args.texts:
        raise ValueError(
            "For inference mode, provide either --text or --texts."
        )

    text_input = args.text if args.text is not None else args.texts

    results = run_text_inference(
        text_input=text_input,
        train_config_path=args.best_train_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        model_weights_path=args.model_weights_path,
        label_mapping_path=args.label_mapping_path,
        output_path=args.inference_output_path,
        top_k=args.top_k,
    )

    print("Inference finished.")
    print(f"Results saved to: {args.inference_output_path}")

    if isinstance(results, dict):
        print(f"Predicted label: {results['predicted_rakuten_code']}")


def run_random_search_mode(args: argparse.Namespace) -> None:
    summary = run_random_search(
        train_csv_path=args.train_csv_path,
        val_csv_path=args.val_csv_path,
        base_train_config_path=args.train_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        search_space_config_path=args.search_space_config_path,
        final_best_config_path=args.final_best_config_path,
        final_best_model_path=args.final_best_model_path,
        final_best_label_mapping_path=args.final_best_label_mapping_path,
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
