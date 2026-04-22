"""Corrections and demo selections page."""

import os
import pandas as pd
import streamlit as st
from auth import get_current_user
from settings_manager import load_config


def _load_csv(filepath: str, username: str | None = None) -> pd.DataFrame | None:
    """Load a CSV file, optionally filtering by username."""
    if not os.path.isfile(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return None
    if df.empty:
        return None
    if username and "username" in df.columns:
        df = df[df["username"] == username]
    return df if not df.empty else None


def render():
    """Render the corrections/history page."""
    st.header("Prediction History and Corrections")
    user = get_current_user()
    if not user:
        return

    cfg = load_config()
    paths = cfg.get("paths", {})
    corrections_path = paths.get("corrections_csv", "streamlit/data/corrections.csv")
    demo_path = paths.get("demo_selections_csv", "streamlit/data/demo_selections.csv")

    is_admin = user["role"] == "admin"
    filter_user = None if is_admin else user["username"]

    tab1, tab2 = st.tabs(["Corrections", "All Selections"])

    with tab1:
        st.subheader("Corrections" + (" (all users)" if is_admin else ""))
        df_corr = _load_csv(corrections_path, filter_user)
        if df_corr is not None:
            st.dataframe(df_corr, use_container_width=True)
            st.text(f"{len(df_corr)} entries")
        else:
            st.info("No corrections available.")

    with tab2:
        st.subheader(
            "All Selections (Demo Selections)" + (" (all users)" if is_admin else "")
        )
        df_demo = _load_csv(demo_path, filter_user)
        if df_demo is not None:
            st.dataframe(df_demo, use_container_width=True)
            st.text(f"{len(df_demo)} entries")
        else:
            st.info("No selections available.")
