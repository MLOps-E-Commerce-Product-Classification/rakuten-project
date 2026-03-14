import json
from pathlib import Path


def load_label_encoding(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Label encoding not found: {path}")

    with path.open() as f:
        encoding = json.load(f)

    return encoding