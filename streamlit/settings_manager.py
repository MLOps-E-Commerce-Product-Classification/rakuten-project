"""Settings manager for loading and saving config.yaml.

Credentials are loaded from the .env file in the project root (parent of
the streamlit/ directory).  The following environment variables are supported:

    STREAMLIT_BASE_URL        API base URL
    STREAMLIT_API_KEY         Nginx API key (X-API-Key header)
    STREAMLIT_NGINX_USER      Nginx basic-auth user
    STREAMLIT_NGINX_PASS      Nginx basic-auth password
    STREAMLIT_BENTO_USER      BentoML login user
    STREAMLIT_BENTO_PASS      BentoML login password
    STREAMLIT_ADMIN_PASSWORD  Default password for the admin user (first start)
    STREAMLIT_DEMO_PASSWORD   Default password for the demo_user (first start)

If a variable is set it always takes precedence over the value stored in
config.yaml so that credentials never need to be committed to git.
"""

from __future__ import annotations

import os
import yaml
from filelock import FileLock

# ── Locate project root (.env lives next to docker-compose.yml) ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))          # …/streamlit/
_PROJECT_ROOT = os.path.dirname(_HERE)                       # …/rakuten-project/

# Load .env from project root (silent if missing)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=False)
except ImportError:
    pass  # python-dotenv not installed — fall back to real env vars only

# ── Config file ──────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(_HERE, "config.yaml")
CONFIG_LOCK = CONFIG_PATH + ".lock"

_cache: dict = {}
_cache_mtime: float = 0.0


def _resolve_path() -> str:
    return CONFIG_PATH


def load_config(force: bool = False) -> dict:
    """Load config from YAML, then overlay credentials from environment."""
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
            data: dict = yaml.safe_load(f) or {}

    # ── Overlay credentials from environment ─────────────────────────────────
    api = data.setdefault("api", {})
    _env_override(api, "base_url",        "STREAMLIT_BASE_URL")
    _env_override(api, "api_key",         "STREAMLIT_API_KEY")
    _env_override(api, "nginx_user",      "STREAMLIT_NGINX_USER")
    _env_override(api, "nginx_pass",      "STREAMLIT_NGINX_PASS")
    _env_override(api, "bento_user",      "STREAMLIT_BENTO_USER")
    _env_override(api, "bento_pass",      "STREAMLIT_BENTO_PASS")

    _cache = data
    _cache_mtime = mtime
    return data


def _env_override(section: dict, key: str, env_var: str) -> None:
    """If env_var is set and non-empty, write its value into section[key]."""
    value = os.environ.get(env_var, "").strip()
    if value:
        section[key] = value


def save_config(data: dict) -> None:
    """Persist config dict to YAML (credentials from env are NOT written back)."""
    global _cache, _cache_mtime
    path = _resolve_path()

    # Strip env-sourced credentials before writing so they never land in git
    safe = _strip_env_credentials(data)

    lock = FileLock(CONFIG_LOCK, timeout=5)
    with lock:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(safe, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    _cache = data          # keep full (incl. env values) in memory cache
    try:
        _cache_mtime = os.path.getmtime(path)
    except OSError:
        _cache_mtime = 0.0


def _strip_env_credentials(data: dict) -> dict:
    """Return a copy of data with env-controlled credential fields cleared."""
    import copy
    safe = copy.deepcopy(data)
    api = safe.get("api", {})
    env_keys = {
        "base_url":   "STREAMLIT_BASE_URL",
        "api_key":    "STREAMLIT_API_KEY",
        "nginx_user": "STREAMLIT_NGINX_USER",
        "nginx_pass": "STREAMLIT_NGINX_PASS",
        "bento_user": "STREAMLIT_BENTO_USER",
        "bento_pass": "STREAMLIT_BENTO_PASS",
    }
    for key, env_var in env_keys.items():
        if os.environ.get(env_var, "").strip():
            api[key] = ""   # clear in YAML — value comes from env
    return safe


def get(section: str, key: str, default=None):
    """Get a specific config value."""
    cfg = load_config()
    return cfg.get(section, {}).get(key, default)
