"""Settings manager for loading and saving config.yaml."""

import os
import yaml
from filelock import FileLock

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
CONFIG_LOCK = CONFIG_PATH + ".lock"

_cache = {}
_cache_mtime = 0.0


def _resolve_path() -> str:
    return CONFIG_PATH


def load_config(force: bool = False) -> dict:
    """Load config from YAML file with simple mtime cache."""
    global _cache, _cache_mtime
    path = _resolve_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0

    if not force and _cache and mtime == _cache_mtime:
        return _cache

    lock = FileLock(CONFIG_LOCK, timeout=5)
    with lock:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    _cache = data
    _cache_mtime = mtime
    return data


def save_config(data: dict) -> None:
    """Save config dict back to YAML file."""
    global _cache, _cache_mtime
    path = _resolve_path()
    lock = FileLock(CONFIG_LOCK, timeout=5)
    with lock:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    _cache = data
    try:
        _cache_mtime = os.path.getmtime(path)
    except OSError:
        _cache_mtime = 0.0


def get(section: str, key: str, default=None):
    """Get a specific config value."""
    cfg = load_config()
    return cfg.get(section, {}).get(key, default)
