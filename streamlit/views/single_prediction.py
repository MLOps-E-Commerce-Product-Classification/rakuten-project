import os
import csv
import datetime
import streamlit as st
from filelock import FileLock
from auth import get_current_user
from api_client import get_client
from settings_manager import load_config
from category_names import load_category_names, format_category


def _csv_path(key: str) -> str:
    cfg = load_config()
    return cfg.get("paths", {}).get(key, f"streamlit/data/{key}.csv")


def _write_csv_row(filepath: str, row: dict) -> None:
    """Thread-safe append to CSV."""
    lock = FileLock(filepath + ".lock", timeout=5)
    with lock:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def _build_row(
    user: dict,
    designation: str,
    description: str,
    top1_code: int,
    top1_prob: float,
    selected_rank: int,
    selected_code: int,
) -> dict:
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
    st.header("Single Prediction")
    user = get_current_user()
    if not user:
        return

    cfg = load_config()
    category_names = load_category_names()
    pred_cfg = cfg.get("prediction", {})
    default_top_k = pred_cfg.get("default_top_k", 5)
    max_top_k = pred_cfg.get("max_top_k", 27)

    with st.form("single_pred_form"):
        designation = st.text_input("Designation (required)")
        description = st.text_area("Description (optional)", value="")
        top_k = st.number_input(
            "Top-K",
            min_value=1,
            max_value=max_top_k,
            value=default_top_k,
        )
        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        if not designation.strip():
            st.error("Designation is a required field.")
            return
        try:
            client = get_client()
            with st.spinner("Running prediction..."):
                result = client.predict_single(
                    designation.strip(),
                    description.strip(),
                    int(top_k),
                )
            st.session_state["single_result"] = result
            st.session_state["single_designation"] = designation.strip()
            st.session_state["single_description"] = description.strip()
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

    if "single_result" in st.session_state:
        result = st.session_state["single_result"]
        predictions = result.get("top_k_predictions", [])
        if not predictions:
            st.warning("No predictions received.")
            return

        display_preds = predictions[:5]
        st.subheader("Prediction Results")

        options = []
        for i, pred in enumerate(display_preds):
            rank = i + 1
            code = pred.get("rakuten_code", "?")
            prob = pred.get("probability", 0)
            label = (
                f"Rank {rank}: "
                f"{format_category(code, category_names)} "
                f"(Probability: {prob:.2%})"
            )
            options.append(label)

        selected_idx = st.radio(
            "Which code is correct?",
            range(len(options)),
            format_func=lambda i: options[i],
            key="single_radio",
        )

        if st.button("Confirm Selection", key="single_confirm"):
            designation = st.session_state["single_designation"]
            description = st.session_state["single_description"]
            sel_pred = display_preds[selected_idx]
            top1 = display_preds[0]
            selected_rank = selected_idx + 1
            selected_code = sel_pred["rakuten_code"]

            row = _build_row(
                user,
                designation,
                description,
                top1["rakuten_code"],
                top1.get("probability", 0),
                selected_rank,
                selected_code,
            )

            demo_path = _csv_path("demo_selections_csv")
            _write_csv_row(demo_path, row)

            if selected_rank > 1:
                corr_path = _csv_path("corrections_csv")
                _write_csv_row(corr_path, row)

            st.success(
                f"Selection saved: "
                f"{format_category(selected_code, category_names)} "
                f"(Rank {selected_rank})"
            )

            for k in ["single_result", "single_designation", "single_description"]:
                st.session_state.pop(k, None)
