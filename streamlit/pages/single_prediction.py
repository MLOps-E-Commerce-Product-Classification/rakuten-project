"""Single Prediction page."""

import os
import csv
import datetime
import streamlit as st
from filelock import FileLock
from auth import get_current_user
from api_client import get_client
from settings_manager import load_config


def _csv_path(key: str) -> str:
    cfg = load_config()
    return cfg.get("paths", {}).get(key, f"streamlit/data/{key}.csv")


def _write_csv_row(filepath: str, row: dict) -> None:
    """Thread-safe append to CSV."""
    lock = FileLock(filepath + ".lock", timeout=5)
    file_exists = os.path.isfile(filepath)
    with lock:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writeheader()
            writer.writerow(row)


def _build_row(user: dict, designation: str, description: str,
               top1_code: int, top1_prob: float,
               selected_rank: int, selected_code: int) -> dict:
    is_correction = selected_rank > 1
    escaped_des = designation.replace("'", "''")
    escaped_desc = description.replace("'", "''")
    sql = (
        f"INSERT INTO corrected_labels (designation, description, prdtypecode) "
        f"VALUES ('{escaped_des}', '{escaped_desc}', {selected_code});"
    )
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "username": user["username"],
        "designation": designation,
        "description": description,
        "model_top1_code": top1_code,
        "model_top1_prob": round(top1_prob, 4),
        "selected_rank": selected_rank,
        "selected_code": selected_code,
        "is_correction": is_correction,
        "sql_command": sql,
    }


def render():
    """Render the single prediction page."""
    st.header("Einzelvorhersage")
    user = get_current_user()
    if not user:
        return

    cfg = load_config()
    pred_cfg = cfg.get("prediction", {})
    default_top_k = pred_cfg.get("default_top_k", 5)
    max_top_k = pred_cfg.get("max_top_k", 27)

    with st.form("single_pred_form"):
        designation = st.text_input("Designation (Pflichtfeld)")
        description = st.text_area("Description (optional)", value="")
        top_k = st.number_input("Top-K", min_value=1, max_value=max_top_k, value=default_top_k)
        submitted = st.form_submit_button("Vorhersage starten")

    if submitted:
        if not designation.strip():
            st.error("Designation ist ein Pflichtfeld.")
            return
        try:
            client = get_client()
            with st.spinner("Vorhersage wird durchgefuehrt..."):
                result = client.predict_single(designation.strip(), description.strip(), int(top_k))
            st.session_state["single_result"] = result
            st.session_state["single_designation"] = designation.strip()
            st.session_state["single_description"] = description.strip()
        except Exception as e:
            st.error(f"Fehler bei der Vorhersage: {e}")
            return

    # Show results
    if "single_result" in st.session_state:
        result = st.session_state["single_result"]
        predictions = result.get("top_k_predictions", [])
        if not predictions:
            st.warning("Keine Vorhersagen erhalten.")
            return

        # Always show top 5
        display_preds = predictions[:5]
        st.subheader("Vorhersage-Ergebnisse")

        options = []
        for i, pred in enumerate(display_preds):
            rank = i + 1
            code = pred.get("rakuten_code", "?")
            prob = pred.get("probability", 0)
            label = f"Rang {rank}: Code {code} (Wahrscheinlichkeit: {prob:.2%})"
            options.append(label)

        selected_idx = st.radio(
            "Welcher Code ist korrekt?",
            range(len(options)),
            format_func=lambda i: options[i],
            key="single_radio",
        )

        if st.button("Auswahl bestaetigen", key="single_confirm"):
            designation = st.session_state["single_designation"]
            description = st.session_state["single_description"]
            sel_pred = display_preds[selected_idx]
            top1 = display_preds[0]
            selected_rank = selected_idx + 1
            selected_code = sel_pred["rakuten_code"]

            row = _build_row(
                user, designation, description,
                top1["rakuten_code"], top1.get("probability", 0),
                selected_rank, selected_code,
            )

            # Always write to demo_selections
            demo_path = _csv_path("demo_selections_csv")
            _write_csv_row(demo_path, row)

            # Write to corrections only if rank > 1
            if selected_rank > 1:
                corr_path = _csv_path("corrections_csv")
                _write_csv_row(corr_path, row)

            st.success(
                f"Auswahl gespeichert: Code {selected_code} (Rang {selected_rank})"
            )

            # Clear result
            for k in ["single_result", "single_designation", "single_description"]:
                st.session_state.pop(k, None)
