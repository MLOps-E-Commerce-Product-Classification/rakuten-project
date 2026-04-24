from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_CONFIG_PATH = BASE_DIR / "configs/text_train_config.yaml"
PREPROCESSING_CONFIG_PATH = BASE_DIR / "configs/text_preprocessing_config.yaml"
BACKBONE_DIR = BASE_DIR / "models/text_backbone"
REQUIRED_BACKBONE_FILES = (
    "config.json",
    "tokenizer_config.json",
)
TOKENIZER_ASSET_CANDIDATES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.txt",
    "vocab.json",
)
ALLOWED_BACKBONE_PATTERNS = [
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "sentencepiece.bpe.model",
]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _model_name_from_configs(
    train_config_path: str | Path = TRAIN_CONFIG_PATH,
    preprocessing_config_path: str | Path = PREPROCESSING_CONFIG_PATH,
) -> str:
    train_config = _load_yaml(Path(train_config_path))
    preprocessing_config = _load_yaml(Path(preprocessing_config_path))
    model_name = train_config.get("model", {}).get("name")
    tokenizer_name = preprocessing_config.get("preprocessing", {}).get(
        "tokenizer_model"
    )
    if tokenizer_name and model_name and tokenizer_name != model_name:
        raise ValueError(
            f"Tokenizer/model mismatch for Bento packaging: {tokenizer_name!r} != {model_name!r}"
        )
    return tokenizer_name or model_name or "bert-base-multilingual-cased"


def _has_minimum_file_set(path: Path) -> bool:
    return (
        path.exists()
        and all((path / filename).exists() for filename in REQUIRED_BACKBONE_FILES)
        and any((path / filename).exists() for filename in TOKENIZER_ASSET_CANDIDATES)
    )


def _validate_local_backbone_assets(path: Path) -> bool:
    if not _has_minimum_file_set(path):
        return False
    try:
        AutoConfig.from_pretrained(path, local_files_only=True)
        AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception:
        return False
    return True


def ensure_local_text_backbone(
    *,
    train_config_path: str | Path = TRAIN_CONFIG_PATH,
    preprocessing_config_path: str | Path = PREPROCESSING_CONFIG_PATH,
    output_dir: str | Path = BACKBONE_DIR,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _validate_local_backbone_assets(output_dir):
        return output_dir

    model_name = _model_name_from_configs(
        train_config_path=train_config_path,
        preprocessing_config_path=preprocessing_config_path,
    )
    snapshot_download(
        repo_id=model_name,
        local_dir=str(output_dir),
        allow_patterns=ALLOWED_BACKBONE_PATTERNS,
        ignore_patterns=[
            "*.bin",
            "*.safetensors",
            "*.h5",
            "*.msgpack",
            "*.ckpt",
            "onnx/*",
            "tf_model.h5",
            "flax_model.msgpack",
            "rust_model.ot",
        ],
    )

    if not _validate_local_backbone_assets(output_dir):
        missing_required = [
            name for name in REQUIRED_BACKBONE_FILES if not (output_dir / name).exists()
        ]
        has_tokenizer_assets = any(
            (output_dir / name).exists() for name in TOKENIZER_ASSET_CANDIDATES
        )
        detail_parts = []
        if missing_required:
            detail_parts.append(
                "missing required files: " + ", ".join(missing_required)
            )
        if not has_tokenizer_assets:
            detail_parts.append(
                "missing tokenizer asset; expected one of: "
                + ", ".join(TOKENIZER_ASSET_CANDIDATES)
            )
        if not detail_parts:
            detail_parts.append(
                "downloaded assets could not be loaded via AutoConfig/AutoTokenizer"
            )
        raise FileNotFoundError(
            "Backbone asset preparation failed: " + "; ".join(detail_parts)
        )

    return output_dir


if __name__ == "__main__":
    path = ensure_local_text_backbone()
    print(f"Prepared local text backbone assets in: {path}")
