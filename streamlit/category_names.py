"""Utilities for loading and formatting category names for Streamlit."""

from __future__ import annotations

import json
from pathlib import Path


def load_category_names() -> dict[str, str]:
    """Load category names from streamlit/data/category_names.json."""
    path = Path(__file__).resolve().parent / "data" / "category_names.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def format_category(code: int | str, category_names: dict[str, str]) -> str:
    """Format a category code as 'Name (Code)' if known, else 'Code <code>'."""
    code_str = str(code)
    name = category_names.get(code_str)
    print(name)
    print(name)
    print(name)
    print(name)
    print(name)
    print(name)
    if name:
        return f"{name} ({code_str})"
    return f"Code {code_str}"
