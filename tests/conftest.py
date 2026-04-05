from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "transformers" not in sys.modules:
    import types

    transformers_stub = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str):
            return DummyTokenizer()

    class _DummyModel:
        def __init__(self):
            self.logits = torch.zeros((1, 2))

        def named_parameters(self):
            return []

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state_dict):
            return None

        def __call__(self, *args, **kwargs):
            return self

    class _AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _DummyModel()

        @staticmethod
        def from_config(config):
            return _DummyModel()

    class _AutoConfig:
        @staticmethod
        def from_pretrained(model_name: str, num_labels: int):
            return {"model_name": model_name, "num_labels": num_labels}

    transformers_stub.AutoTokenizer = _AutoTokenizer
    transformers_stub.AutoModelForSequenceClassification = (
        _AutoModelForSequenceClassification
    )
    transformers_stub.AutoConfig = _AutoConfig
    sys.modules["transformers"] = transformers_stub


class DummyTokenizer:
    """Small tokenizer stub that mimics the Hugging Face tokenizer interface."""

    def __call__(
        self,
        text: str,
        padding: str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"

        token_count = min(max(len(text.split()), 1), max_length)
        input_ids = torch.zeros((1, max_length), dtype=torch.long)
        attention_mask = torch.zeros((1, max_length), dtype=torch.long)
        input_ids[0, :token_count] = torch.arange(1, token_count + 1)
        attention_mask[0, :token_count] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


@pytest.fixture
def dummy_tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


@pytest.fixture
def label_encoding() -> dict:
    return {
        "classes": [100, 200],
        "code_to_idx": {"100": 0, "200": 1},
    }


@pytest.fixture
def text_preprocessing_config_path(tmp_path: Path) -> Path:
    config_path = tmp_path / "text_preprocessing_config.yaml"
    config_path.write_text(
        """
preprocessing:
  tokenizer_model: dummy-tokenizer
  max_length: 8
  remove_html: true
  lowercase: false
  combine_fields: true
  separator: " [SEP] "
quality:
  compute_quality_report: false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def quality_text_preprocessing_config_path(tmp_path: Path) -> Path:
    config_path = tmp_path / "text_preprocessing_quality_config.yaml"
    config_path.write_text(
        """
preprocessing:
  tokenizer_model: dummy-tokenizer
  max_length: 8
  remove_html: true
  lowercase: true
  combine_fields: true
  separator: " [SEP] "
quality:
  compute_quality_report: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path
