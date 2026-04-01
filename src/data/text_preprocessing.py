from __future__ import annotations

from pathlib import Path
import html
import logging
import re
from typing import Any

from transformers import AutoTokenizer
import yaml


LOG_PATH = Path("logs")
LOG_PATH.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str, log_file: str | Path) -> logging.Logger:
    """
    Create and return a logger writing to a file.
    Prevent duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger


QUALITY_LOGGER = setup_logger(
    "text_quality", LOG_PATH / "text_quality.log"
)

PREPROCESSING_LOGGER = setup_logger(
    "text_preprocessing", LOG_PATH / "text_preprocessing.log"
)


HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def load_text_preprocessing_config(config_path: str | Path) -> dict:
    """
    Load preprocessing configuration from YAML.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        PREPROCESSING_LOGGER.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        PREPROCESSING_LOGGER.exception(
            f"Failed to load config file {config_path}"
        )
        raise


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        if value.lower() == "nan":
            return ""
        return value

    try:
        import pandas as pd  # local import to avoid hard dependency at module load
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value)


def clean_text(
    text: Any,
    remove_html: bool = True,
    lowercase: bool = False,
) -> str:
    """
    Normalize a raw text field.
    """
    text = _coerce_text(text)

    if remove_html:
        text = html.unescape(text)
        text = HTML_TAG_PATTERN.sub(" ", text)

    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def compute_quality_report(
    designation: Any,
    description: Any,
    combined_text: str,
) -> dict:
    """
    Compute simple text quality indicators and log potential issues.
    """
    designation_clean = _coerce_text(designation).strip()
    description_clean = _coerce_text(description).strip()

    report = {
        "designation_empty": len(designation_clean) == 0,
        "description_empty": len(description_clean) == 0,
        "combined_char_length": int(len(combined_text)),
        "combined_word_count": int(len(combined_text.split())),
        "very_short_text": bool(len(combined_text) < 10),
    }

    if report["designation_empty"]:
        QUALITY_LOGGER.warning("Encountered sample with empty designation")

    if report["description_empty"]:
        QUALITY_LOGGER.info("Encountered sample with empty description")

    if report["very_short_text"]:
        QUALITY_LOGGER.warning(
            f"Very short combined text detected: '{combined_text[:80]}'"
        )

    return report


def preprocess_text(
    designation: Any,
    description: Any = None,
    config_path: str | Path = "configs/text_preprocessing_config.yaml",
) -> str | tuple[str, dict]:
    """
    Deterministic preprocessing of one text sample.

    Parameters
    ----------
    designation : Any
        Main title / designation field.
    description : Any
        Longer product description field.
    config_path : str | Path
        Path to YAML preprocessing config.

    Returns
    -------
    str | tuple[str, dict]
        Preprocessed text, and optionally a quality report.
    """
    config = load_text_preprocessing_config(config_path)

    preprocessing_config = config.get("preprocessing", {})
    quality_config = config.get("quality", {})

    remove_html = bool(preprocessing_config.get("remove_html", True))
    lowercase = bool(preprocessing_config.get("lowercase", False))
    combine_fields = bool(preprocessing_config.get("combine_fields", True))
    separator = str(preprocessing_config.get("separator", " "))
    compute_quality = bool(quality_config.get("compute_quality_report", False))

    designation_text = clean_text(
        designation,
        remove_html=remove_html,
        lowercase=lowercase,
    )
    description_text = clean_text(
        description,
        remove_html=remove_html,
        lowercase=lowercase,
    )

    if combine_fields:
        text_parts = [part for part in [designation_text, description_text] if part]
        combined_text = separator.join(text_parts).strip()
    else:
        combined_text = designation_text

    if not combined_text:
        combined_text = "[EMPTY_TEXT]"

    quality_report = None
    if compute_quality:
        quality_report = compute_quality_report(
            designation=designation,
            description=description,
            combined_text=combined_text,
        )

    if quality_report is not None:
        return combined_text, quality_report

    return combined_text


def build_tokenizer(
    config_path: str | Path = "configs/text_preprocessing_config.yaml",
    local_model_dir: str | Path | None = None,
):
    """
    Build tokenizer defined in preprocessing config.
    """
    config = load_text_preprocessing_config(config_path)
    preprocessing_config = config.get("preprocessing", {})

    tokenizer_model = preprocessing_config.get(
        "tokenizer_model", "bert-base-multilingual-cased"
    )

    try:
        if local_model_dir is not None:
            local_model_dir = Path(local_model_dir)
            if local_model_dir.exists():
                return AutoTokenizer.from_pretrained(
                    str(local_model_dir),
                    local_files_only=True,
                )
        return AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception:
        PREPROCESSING_LOGGER.exception(
            f"Failed to load tokenizer '{tokenizer_model}'"
        )
        raise
