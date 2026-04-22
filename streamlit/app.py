"""Main Streamlit application for Rakuten product classification."""

import sys
import os

# Ensure the streamlit directory is on the path for imports
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from auth import get_current_user, render_login_page, logout, ensure_password_hashes
from settings_manager import load_config

# Page config
st.set_page_config(
    page_title="Rakuten Product Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure password hashes exist on first start
ensure_password_hashes()

# Check authentication
user = get_current_user()

if not user:
    render_login_page()
    st.stop()

# Load config
cfg = load_config()
app_title = cfg.get("app", {}).get("title", "Rakuten Product Classification")

# Navigation
PAGES_USER = {
    "Single Prediction": "single_prediction",
    "Batch Prediction": "batch_prediction",
    "History / Corrections": "corrections",
    "Monitoring": "monitoring",
}

PAGES_ADMIN = {
    **PAGES_USER,
    "Settings": "admin_settings",
    "Unit Tests": "admin_tests",
}

pages = PAGES_ADMIN if user["role"] == "admin" else PAGES_USER

# Sidebar
with st.sidebar:
    st.title(app_title)
    st.divider()
    st.text(f"User: {user['username']}")
    st.text(f"Role: {user['role']}")
    st.divider()

    page_selection = st.radio(
        "Navigation",
        list(pages.keys()),
        key="nav_radio",
    )

    st.divider()
    if st.button("Logout"):
        logout()
        st.rerun()

# Render selected page
page_module_name = pages[page_selection]

if page_module_name == "single_prediction":
    from views.single_prediction import render

    render()
elif page_module_name == "batch_prediction":
    from views.batch_prediction import render

    render()
elif page_module_name == "corrections":
    from views.corrections import render

    render()
elif page_module_name == "monitoring":
    from views.monitoring import render

    render()
elif page_module_name == "admin_settings":
    from views.admin_settings import render

    render()
elif page_module_name == "admin_tests":
    from views.admin_tests import render

    render()
