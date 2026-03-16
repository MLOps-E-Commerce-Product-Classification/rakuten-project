## unit tests for functions from text_preprocessing.py

from pathlib import Path

import pytest
import yaml

from src.data import text_preprocessing as tp


def _write_config(tmp_path: Path, config: dict) -> Path:
    config_path = tmp_path / "text_preprocessing_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


@pytest.fixture
def text_preprocessing_config_path(tmp_path: Path) -> Path:
    return _write_config(
        tmp_path,
        {
            "preprocessing": {
                "remove_html": True,
                "lowercase": False,
                "combine_fields": True,
                "separator": " [SEP] ",
                "tokenizer_model": "dummy-tokenizer",
            },
            "quality": {
                "compute_quality_report": False,
            },
        },
    )


@pytest.fixture
def quality_text_preprocessing_config_path(tmp_path: Path) -> Path:
    return _write_config(
        tmp_path,
        {
            "preprocessing": {
                "remove_html": True,
                "lowercase": True,
                "combine_fields": True,
                "separator": " ",
            },
            "quality": {
                "compute_quality_report": True,
            },
        },
    )


def test_load_text_preprocessing_config_reads_yaml(tmp_path: Path):
    config = {
        "preprocessing": {
            "lowercase": True,
            "separator": " [SEP] ",
        }
    }
    config_path = _write_config(tmp_path, config)

    loaded = tp.load_text_preprocessing_config(config_path)

    assert loaded == config


def test_load_text_preprocessing_config_raises_on_missing_file(tmp_path: Path):
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        tp.load_text_preprocessing_config(missing_path)


def test_clean_text_removes_html_and_normalizes_whitespace():
    raw_text = "  Bonjour   <b>Monde</b> &amp;\n tout\t le monde  "

    cleaned = tp.clean_text(raw_text, remove_html=True, lowercase=False)

    assert cleaned == "Bonjour Monde & tout le monde"


def test_clean_text_respects_remove_html_flag():
    raw_text = "Bonjour <b>Monde</b>"

    cleaned = tp.clean_text(raw_text, remove_html=False, lowercase=False)

    assert cleaned == "Bonjour <b>Monde</b>"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (None, ""),
        ("nan", ""),
        (123, "123"),
    ],
)
def test_clean_text_handles_common_non_standard_inputs(raw_value, expected):
    assert tp.clean_text(raw_value) == expected


def test_preprocess_text_combines_designation_and_description(
    text_preprocessing_config_path: Path,
):
    processed = tp.preprocess_text(
        designation="<b>Robe</b>",
        description="Très   élégante",
        config_path=text_preprocessing_config_path,
    )

    assert processed == "Robe [SEP] Très élégante"


@pytest.mark.parametrize(
    ("designation", "description"),
    [
        (None, None),
        ("", ""),
        ("nan", "nan"),
        (float("nan"), None),
    ],
)
def test_preprocess_text_returns_placeholder_for_empty_inputs(
    designation,
    description,
    text_preprocessing_config_path: Path,
):
    processed = tp.preprocess_text(
        designation=designation,
        description=description,
        config_path=text_preprocessing_config_path,
    )

    assert processed == "[EMPTY_TEXT]"


def test_preprocess_text_uses_only_designation_when_combine_fields_disabled(
    tmp_path: Path,
):
    config_path = _write_config(
        tmp_path,
        {
            "preprocessing": {
                "remove_html": True,
                "lowercase": False,
                "combine_fields": False,
                "separator": " [SEP] ",
            },
            "quality": {
                "compute_quality_report": False,
            },
        },
    )

    processed = tp.preprocess_text(
        designation="<b>Robe</b>",
        description="Très élégante",
        config_path=config_path,
    )

    assert processed == "Robe"


def test_preprocess_text_returns_quality_report_when_enabled(
    quality_text_preprocessing_config_path: Path,
):
    processed_text, quality_report = tp.preprocess_text(
        designation="  ",
        description="Petit texte",
        config_path=quality_text_preprocessing_config_path,
    )

    assert processed_text == "petit texte"
    assert quality_report == {
        "designation_empty": True,
        "description_empty": False,
        "combined_char_length": 11,
        "combined_word_count": 2,
        "very_short_text": False,
    }


def test_build_tokenizer_uses_model_from_config(
    monkeypatch: pytest.MonkeyPatch,
    text_preprocessing_config_path: Path,
):
    captured = {}
    sentinel = object()

    def fake_from_pretrained(model_name: str):
        captured["model_name"] = model_name
        return sentinel

    monkeypatch.setattr(tp.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    tokenizer = tp.build_tokenizer(text_preprocessing_config_path)

    assert tokenizer is sentinel
    assert captured["model_name"] == "dummy-tokenizer"


def test_build_tokenizer_falls_back_to_default_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    config_path = _write_config(
        tmp_path,
        {
            "preprocessing": {},
            "quality": {},
        },
    )

    captured = {}
    sentinel = object()

    def fake_from_pretrained(model_name: str):
        captured["model_name"] = model_name
        return sentinel

    monkeypatch.setattr(tp.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    tokenizer = tp.build_tokenizer(config_path)

    assert tokenizer is sentinel
    assert captured["model_name"] == "bert-base-multilingual-cased"


def test_build_tokenizer_reraises_loading_errors(
    monkeypatch: pytest.MonkeyPatch,
    text_preprocessing_config_path: Path,
):
    def fake_from_pretrained(_: str):
        raise OSError("tokenizer load failed")

    monkeypatch.setattr(tp.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    with pytest.raises(OSError, match="tokenizer load failed"):
        tp.build_tokenizer(text_preprocessing_config_path)