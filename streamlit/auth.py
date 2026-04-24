"""Authentication module with bcrypt password hashing and session management."""

import os
import time
import bcrypt
import streamlit as st
from settings_manager import load_config, save_config

# ── Project root (.env lives there) ──────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=False)
except ImportError:
    pass


def _default_passwords() -> dict[str, str]:
    """
    Read default passwords from environment variables.
    Falls back to empty string — user must set a password via admin UI.
    """
    return {
        "admin": os.environ.get("STREAMLIT_ADMIN_PASSWORD", "").strip(),
        "demo_user": os.environ.get("STREAMLIT_DEMO_PASSWORD", "").strip(),
    }


def _hash_password(password: str) -> str:
    """Hash a password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _check_password(password: str, password_hash: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def ensure_password_hashes() -> None:
    """Generate and store bcrypt hashes for users without a hash on first start.

    Passwords are sourced from STREAMLIT_ADMIN_PASSWORD / STREAMLIT_DEMO_PASSWORD.
    If neither env var is set the hash stays empty and the account cannot be used
    until a password is set via the admin UI.
    """
    cfg = load_config(force=True)
    users = cfg.get("users", {})
    defaults = _default_passwords()
    changed = False
    for username, user_data in users.items():
        if not user_data.get("password_hash"):
            default_pw = defaults.get(username, "")
            if default_pw:
                user_data["password_hash"] = _hash_password(default_pw)
                changed = True
    if changed:
        cfg["users"] = users
        save_config(cfg)


def check_login(username: str, password: str) -> bool:
    """Check login credentials against config."""
    cfg = load_config(force=True)
    users = cfg.get("users", {})
    user_data = users.get(username)
    if not user_data:
        return False
    if user_data.get("disabled", False):
        return False
    pw_hash = user_data.get("password_hash", "")
    if not pw_hash:
        return False
    return _check_password(password, pw_hash)


def get_current_user() -> dict | None:
    """Get current logged-in user info from session state, or None if not logged in."""
    if "username" not in st.session_state:
        return None

    cfg = load_config()
    timeout_minutes = cfg.get("app", {}).get("session_timeout_minutes", 60)
    last_activity = st.session_state.get("last_activity", 0)
    if time.time() - last_activity > timeout_minutes * 60:
        logout()
        return None

    st.session_state["last_activity"] = time.time()

    cfg_users = cfg.get("users", {})
    user_data = cfg_users.get(st.session_state["username"], {})
    return {
        "username": st.session_state["username"],
        "role": user_data.get("role", "user"),
    }


def require_role(role: str) -> bool:
    """Check if current user has the required role. Shows error if not."""
    user = get_current_user()
    if not user:
        st.error("Not logged in.")
        return False
    if role == "admin" and user["role"] != "admin":
        st.error("Access denied. Admin privileges required.")
        return False
    return True


def logout() -> None:
    """Clear session state to log out."""
    for key in ["username", "last_activity", "jwt_token"]:
        st.session_state.pop(key, None)


def login_user(username: str) -> None:
    """Set session state for logged-in user."""
    st.session_state["username"] = username
    st.session_state["last_activity"] = time.time()


def render_login_page() -> bool:
    """Render the login form. Returns True if user is now logged in."""
    cfg = load_config()
    title = cfg.get("app", {}).get("title", "Rakuten Product Classification")
    st.title(title)
    st.subheader("Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if check_login(username, password):
            login_user(username)
            st.rerun()
        else:
            st.error("Invalid username or password.")

    return False
