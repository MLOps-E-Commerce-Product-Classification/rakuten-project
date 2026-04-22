## unit tests for functions from text_pipeline.py

import argparse
import pytest
from src.pipeline import text_pipeline as pipeline


@pytest.fixture
def base_args() -> argparse.Namespace:
    return argparse.Namespace(
        mode="train",
        x_data_csv_path="data/x.csv",
        y_data_csv_path="data/y.csv",
        processed_data_dir="data/processed",
        split_ids_dir="artifacts/splits",
        force_new_split=False,
        train_config_path="configs/text_train_config.yaml",
        best_train_config_path="configs/text_best_train_config.yaml",
        preprocessing_config_path="configs/text_preprocessing_config.yaml",
        eval_config_path="configs/text_evaluate_config.yaml",
        search_space_config_path="configs/text_parameter_search_space.yaml",
        label_encoding_path="configs/label_encoding.json",
        model_save_path="models/best_text_model.pt",
        model_weights_path="models/best_text_model.pt",
        results_output_path="results/text_evaluation_results.json",
        inference_output_path="results/text_inference_results.json",
        final_best_config_path="configs/text_best_train_config.yaml",
        final_best_model_path="search/best_text_model.pt",
        text=None,
        texts=None,
        top_k=5,
        mlflow_run_id=None,
        random_seed=17,
    )


def test_build_parser_exposes_expected_defaults():
    parser = pipeline.build_parser()
    args = parser.parse_args(["--mode", "train"])

    assert args.mode == "train"
    assert args.x_data_csv_path == "data/raw/X_train_update.csv"
    assert args.y_data_csv_path == "data/raw/Y_train_CVw08PX.csv"
    assert args.split_ids_dir == "artifacts/splits"
    assert args.force_new_split is False
    assert args.train_config_path == "configs/text_train_config.yaml"
    assert args.best_train_config_path == "configs/text_best_train_config.yaml"
    assert args.preprocessing_config_path == "configs/text_preprocessing_config.yaml"
    assert args.eval_config_path == "configs/text_evaluate_config.yaml"
    assert args.search_space_config_path == "configs/text_parameter_search_space.yaml"
    assert args.label_encoding_path == "configs/label_encoding.json"
    assert args.model_save_path == "models/best_text_model.pt"
    assert args.model_weights_path == "models/best_text_model.pt"
    assert args.results_output_path == "results/text_evaluation_results.json"
    assert args.inference_output_path == "results/text_inference_results.json"
    assert args.final_best_config_path == "configs/text_best_train_config.yaml"
    assert args.final_best_model_path == "search/best_text_model.pt"
    assert args.text is None
    assert args.texts is None
    assert args.top_k == 5
    assert args.random_seed == 17


@pytest.mark.parametrize("invalid_mode", ["fit", "predict", "serve"])
def test_build_parser_rejects_unsupported_modes(invalid_mode: str):
    parser = pipeline.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", invalid_mode])


def test_build_parser_parses_single_text_inference_args():
    parser = pipeline.build_parser()
    args = parser.parse_args(
        ["--mode", "inference", "--text", "une robe rouge", "--top_k", "3"]
    )

    assert args.mode == "inference"
    assert args.text == "une robe rouge"
    assert args.texts is None
    assert args.top_k == 3


def test_build_parser_parses_multiple_texts_inference_args():
    parser = pipeline.build_parser()
    args = parser.parse_args(
        ["--mode", "inference", "--texts", "texte 1", "texte 2", "texte 3"]
    )

    assert args.mode == "inference"
    assert args.text is None
    assert args.texts == ["texte 1", "texte 2", "texte 3"]


def test_run_train_mode_forwards_arguments_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base_args: argparse.Namespace,
):
    captured: dict[str, object] = {}

    def fake_run_text_training(**kwargs):
        captured.update(kwargs)
        history = {"val_macro_f1": [0.42, 0.57]}
        label_encoding = {"classes": [10, 20]}
        return history, label_encoding

    monkeypatch.setattr(
        "src.training.run_text_training.run_text_training", fake_run_text_training
    )

    pipeline.run_train_mode(base_args)
    output = capsys.readouterr().out

    assert captured == {
        "processed_data_dir": "data/processed",
        "train_config_path": "configs/text_train_config.yaml",
        "preprocessing_config_path": "configs/text_preprocessing_config.yaml",
        "model_save_path": "models/best_text_model.pt",
        "label_encoding_path": "configs/label_encoding.json",
    }
    assert "Training finished." in output
    assert "Model saved to: models/best_text_model.pt" in output
    assert "Label encoding loaded from: configs/label_encoding.json" in output
    assert "Number of classes: 2" in output
    assert "Best validation macro-F1: 0.5700" in output


def test_run_evaluate_mode_forwards_arguments_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base_args: argparse.Namespace,
):
    captured: dict[str, object] = {}

    def fake_run_text_evaluation(**kwargs):
        captured.update(kwargs)
        return {"main_metric": "macro_f1", "main_metric_value": 0.61}

    monkeypatch.setattr(
        "src.evaluation.run_text_evaluation.run_text_evaluation",
        fake_run_text_evaluation,
    )

    pipeline.run_evaluate_mode(base_args)
    output = capsys.readouterr().out

    assert captured == {
        "x_data_csv_path": "data/x.csv",
        "y_data_csv_path": "data/y.csv",
        "split_ids_dir": "artifacts/splits",
        "train_config_path": "configs/text_best_train_config.yaml",
        "eval_config_path": "configs/text_evaluate_config.yaml",
        "preprocessing_config_path": "configs/text_preprocessing_config.yaml",
        "model_weights_path": "models/best_text_model.pt",
        "label_encoding_path": "configs/label_encoding.json",
        "results_output_path": "results/text_evaluation_results.json",
        "mlflow_run_id": None,
    }
    assert "Evaluation finished." in output
    assert "Main metric: macro_f1 = 0.6100" in output
    assert "Results saved to: results/text_evaluation_results.json" in output


def test_run_inference_mode_requires_text_or_texts(
    base_args: argparse.Namespace,
):
    with pytest.raises(ValueError, match="provide either --text or --texts"):
        pipeline.run_inference_mode(base_args)


def test_run_inference_mode_forwards_single_text_and_prints_prediction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base_args: argparse.Namespace,
):
    captured: dict[str, object] = {}
    base_args.text = "une robe rouge"

    def fake_run_text_inference(**kwargs):
        captured.update(kwargs)
        return {"predicted_rakuten_code": "100"}

    monkeypatch.setattr(
        "src.inference.run_text_inference.run_text_inference", fake_run_text_inference
    )

    pipeline.run_inference_mode(base_args)
    output = capsys.readouterr().out

    assert captured == {
        "text_input": "une robe rouge",
        "train_config_path": "configs/text_best_train_config.yaml",
        "preprocessing_config_path": "configs/text_preprocessing_config.yaml",
        "model_weights_path": "models/best_text_model.pt",
        "label_encoding_path": "configs/label_encoding.json",
        "output_path": "results/text_inference_results.json",
        "top_k": 5,
    }
    assert "Inference finished." in output
    assert "Results saved to: results/text_inference_results.json" in output
    assert "Predicted Rakuten code: 100" in output


def test_run_inference_mode_prefers_text_over_texts(
    monkeypatch: pytest.MonkeyPatch,
    base_args: argparse.Namespace,
):
    captured: dict[str, object] = {}
    base_args.text = "une robe rouge"
    base_args.texts = ["unused", "values"]

    def fake_run_text_inference(**kwargs):
        captured.update(kwargs)
        return {"predicted_rakuten_code": "100"}

    monkeypatch.setattr(
        "src.inference.run_text_inference.run_text_inference", fake_run_text_inference
    )

    pipeline.run_inference_mode(base_args)

    assert captured["text_input"] == "une robe rouge"


def test_run_inference_mode_forwards_multiple_texts_without_single_prediction_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base_args: argparse.Namespace,
):
    captured: dict[str, object] = {}
    base_args.texts = ["texte 1", "texte 2"]

    def fake_run_text_inference(**kwargs):
        captured.update(kwargs)
        return [
            {"predicted_rakuten_code": "100"},
            {"predicted_rakuten_code": "200"},
        ]

    monkeypatch.setattr(
        "src.inference.run_text_inference.run_text_inference", fake_run_text_inference
    )

    pipeline.run_inference_mode(base_args)
    output = capsys.readouterr().out

    assert captured == {
        "text_input": ["texte 1", "texte 2"],
        "train_config_path": "configs/text_best_train_config.yaml",
        "preprocessing_config_path": "configs/text_preprocessing_config.yaml",
        "model_weights_path": "models/best_text_model.pt",
        "label_encoding_path": "configs/label_encoding.json",
        "output_path": "results/text_inference_results.json",
        "top_k": 5,
    }
    assert "Inference finished." in output
    assert "Results saved to: results/text_inference_results.json" in output
    assert "Predicted Rakuten code:" not in output


def test_run_random_search_mode_forwards_arguments_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base_args: argparse.Namespace,
):
    captured: dict[str, object] = {}

    def fake_run_random_search(**kwargs):
        captured.update(kwargs)
        return {
            "best_score": 0.72,
            "best_trial": {
                "trial": 3,
                "best_config_path": "configs/best.yaml",
                "best_model_path": "search/best.pt",
            },
        }

    monkeypatch.setattr(
        "src.training.text_random_search_hyperparameters.run_random_search",
        fake_run_random_search,
    )

    pipeline.run_random_search_mode(base_args)
    output = capsys.readouterr().out

    assert captured == {
        "x_data_csv_path": "data/x.csv",
        "y_data_csv_path": "data/y.csv",
        "split_ids_dir": "artifacts/splits",
        "force_new_split": False,
        "base_train_config_path": "configs/text_train_config.yaml",
        "preprocessing_config_path": "configs/text_preprocessing_config.yaml",
        "search_space_config_path": "configs/text_parameter_search_space.yaml",
        "final_best_config_path": "configs/text_best_train_config.yaml",
        "final_best_model_path": "search/best_text_model.pt",
        "label_encoding_path": "configs/label_encoding.json",
        "random_seed": 17,
    }
    assert "Random search finished." in output
    assert "Best score: 0.72" in output
    assert "Best trial: 3" in output
    assert "Best config: configs/best.yaml" in output
    assert "Best model: search/best.pt" in output


def test_run_random_search_mode_handles_missing_best_trial(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base_args: argparse.Namespace,
):
    def fake_run_random_search(**kwargs):
        return {"best_score": 0.55, "best_trial": None}

    monkeypatch.setattr(
        "src.training.text_random_search_hyperparameters.run_random_search",
        fake_run_random_search,
    )

    pipeline.run_random_search_mode(base_args)
    output = capsys.readouterr().out

    assert "Random search finished." in output
    assert "Best score: 0.55" in output
    assert "Best trial:" not in output
    assert "Best config:" not in output
    assert "Best model:" not in output


@pytest.mark.parametrize(
    ("mode", "runner_name"),
    [
        ("train", "run_train_mode"),
        ("evaluate", "run_evaluate_mode"),
        ("inference", "run_inference_mode"),
        ("random_search", "run_random_search_mode"),
    ],
)
def test_main_dispatches_to_the_correct_runner(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    runner_name: str,
):
    calls: list[str] = []

    class DummyParser:
        def parse_args(self):
            return argparse.Namespace(mode=mode)

    monkeypatch.setattr(pipeline, "build_parser", lambda: DummyParser())

    for name in [
        "run_train_mode",
        "run_evaluate_mode",
        "run_inference_mode",
        "run_random_search_mode",
    ]:
        monkeypatch.setattr(
            pipeline,
            name,
            lambda args, name=name: calls.append(name),
        )

    pipeline.main()

    assert calls == [runner_name]
