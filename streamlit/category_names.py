from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_category_names() -> dict[str, dict[str, Any]]:
    path = Path(__file__).resolve().parent.parent / "configs" / "label_encoding.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "idx_to_name": {str(k): str(v) for k, v in data.get("idx_to_name", {}).items()},
        "idx_to_code": {str(k): v for k, v in data.get("idx_to_code", {}).items()},
        "code_to_name": {
            str(k): str(v) for k, v in data.get("code_to_name", {}).items()
        },
    }


def resolve_to_code(value: int | str, maps: dict[str, dict[str, Any]]) -> int | None:
    s = str(value)

    if s in maps["code_to_name"]:
        try:
            return int(s)
        except ValueError:
            return None

    if s in maps["idx_to_code"]:
        code = maps["idx_to_code"].get(s)
        try:
            return int(code)
        except Exception:
            return None

    return None


def format_category(value: int | str, maps: dict[str, dict[str, Any]]) -> str:
    s = str(value)

    if s in maps["idx_to_name"]:
        name = maps["idx_to_name"].get(s)
        code = maps["idx_to_code"].get(s, "?")
        return f"{name} ({code})"

    # Else treat as real code
    name = maps["code_to_name"].get(s)
    if name:
        return f"{name} ({s})"

    return f"Code {s}"
