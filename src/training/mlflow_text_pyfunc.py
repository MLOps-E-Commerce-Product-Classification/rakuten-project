from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from src.data.text_preprocessing import build_tokenizer
from src.inference.run_text_inference import invert_label_mapping, predict_single_text
from src.models.text_classifier import build_text_model

try:  # pragma: no cover - exercised in environments where mlflow is installed
    import mlflow

    _PythonModelBase = mlflow.pyfunc.PythonModel
except ImportError:  # pragma: no cover - unit tests import this module without mlflow
    _PythonModelBase = object


class TextClassifierPyFuncModel(_PythonModelBase):
    """MLflow pyfunc wrapper around the existing PyTorch + tokenizer inference stack."""

    def load_context(
        self, context
    ) -> None:  # pragma: no cover - covered via integration once mlflow exists
        weights_path = Path(context.artifacts["weights"])
        train_config_path = Path(context.artifacts["train_config"])
        preprocessing_config_path = Path(context.artifacts["preprocessing_config"])
        label_encoding_path = Path(context.artifacts["label_encoding"])
        backbone_path = Path(context.artifacts["backbone"])

        with train_config_path.open("r", encoding="utf-8") as handle:
            train_config = yaml.safe_load(handle)
        with label_encoding_path.open("r", encoding="utf-8") as handle:
            label_encoding = json.load(handle)

        self.preprocessing_config_path = preprocessing_config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.idx_to_label = invert_label_mapping(label_encoding["code_to_idx"])
        self.tokenizer = build_tokenizer(
            preprocessing_config_path,
            local_model_dir=backbone_path if backbone_path.exists() else None,
        )

        model_name = train_config.get("model", {}).get(
            "name", "bert-base-multilingual-cased"
        )
        model_name_or_path = (
            str(backbone_path) if backbone_path.exists() else model_name
        )
        self.model = build_text_model(
            model_name=model_name_or_path,
            num_classes=len(label_encoding["classes"]),
            pretrained=False,
            freeze_backbone=False,
        )
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(
        self, context, model_input, params=None
    ):  # pragma: no cover - covered in integration once mlflow exists
        if isinstance(model_input, pd.DataFrame):
            records = model_input.to_dict(orient="records")
        elif isinstance(model_input, dict):
            records = [model_input]
        else:
            records = list(model_input)

        predictions: list[dict[str, Any]] = []
        for row in records:
            top_k_value = row.get("top_k", 5)
            try:
                top_k = int(top_k_value)
            except (TypeError, ValueError):
                top_k = 5
            result = predict_single_text(
                model=self.model,
                text_input={
                    "designation": row.get("designation", ""),
                    "description": row.get("description", ""),
                },
                tokenizer=self.tokenizer,
                preprocessing_config_path=self.preprocessing_config_path,
                device=self.device,
                idx_to_label=self.idx_to_label,
                top_k=top_k,
            )
            predictions.append(result)

        return pd.DataFrame(predictions)
