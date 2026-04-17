"""Batch Prediction page."""

import os
import csv
import datetime
import io
import pandas as pd
import streamlit as st
from filelock import FileLock
from auth import get_current_user
from api_client import get_client
from settings_manager import load_config


def _csv_path(key: str) -> str:
    cfg = load_config()
    return cfg.get("paths", {}).get(key, f"streamlit/data/{key}.csv")


def _write_csv_row(filepath: str, row: dict) -> None:
    lock = FileLock(filepath + ".lock", timeout=5)
    with lock:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
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
    """Render the batch prediction page."""
    st.header("Batch Prediction")
    user = get_current_user()
    if not user:
        return

    cfg = load_config()
    pred_cfg = cfg.get("prediction", {})
    default_top_k = pred_cfg.get("default_top_k", 5)
    batch_limit = pred_cfg.get("batch_limit", 100)
    max_upload_mb = cfg.get("app", {}).get("max_csv_upload_mb", 10)

    uploaded_file = st.file_uploader(
        f"Upload CSV file (max. {max_upload_mb} MB, columns: designation, description)",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_file is not None:
        if uploaded_file.size > max_upload_mb * 1024 * 1024:
            st.error(f"File too large. Maximum: {max_upload_mb} MB.")
            return

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return

        if "designation" not in df.columns:
            st.error("Column 'designation' is missing from the CSV file.")
            return

        if "description" not in df.columns:
            df["description"] = ""

        df = df.head(batch_limit)
        st.info(f"{len(df)} entries loaded (limit: {batch_limit}).")

        if st.button("Start Batch Prediction", key="batch_start"):
            items = []
            for _, row in df.iterrows():
                items.append({
                    "designation": str(row["designation"]),
                    "description": str(row.get("description", "")),
                    "top_k": default_top_k,
                })

            try:
                client = get_client()
                progress = st.progress(0)
                status_text = st.empty()

                chunk_size = max(1, len(items) // 10)
                all_results = []

                for i in range(0, len(items), chunk_size):
                    chunk = items[i:i + chunk_size]
                    status_text.text(f"Processing {i + 1} to {min(i + chunk_size, len(items))} of {len(items)}...")
                    results = client.predict_batch(chunk)
                    if isinstance(results, list):
                        all_results.extend(results)
                    else:
                        all_results.append(results)
                    progress.progress(min((i + chunk_size) / len(items), 1.0))

                progress.progress(1.0)
                status_text.text("Done.")
                st.session_state["batch_results"] = all_results
                st.session_state["batch_items"] = items
            except Exception as e:
                st.error(f"Batch prediction error: {e}")
                return

    if "batch_results" in st.session_state and "batch_items" in st.session_state:
        results = st.session_state["batch_results"]
        items = st.session_state["batch_items"]
        st.subheader("Results")

        table_data = []
        for i, (item, result) in enumerate(zip(items, results)):
            top_k_preds = result.get("top_k_predictions", [])[:5]
            row_data = {
                "No.": i + 1,
                "Designation": item["designation"],
                "Description": item["description"],
            }
            for j, pred in enumerate(top_k_preds):
                row_data[f"Rank {j+1}"] = f"{pred.get('rakuten_code', '?')} ({pred.get('probability', 0):.2%})"
            table_data.append(row_data)

        result_df = pd.DataFrame(table_data)
        st.dataframe(result_df, use_container_width=True)

        st.subheader("Code Selection per Row")
        selections = {}
        for i, (item, result) in enumerate(zip(items, results)):
            top_k_preds = result.get("top_k_predictions", [])[:5]
            if not top_k_preds:
                continue
            options = []
            for j, pred in enumerate(top_k_preds):
                code = pred.get("rakuten_code", "?")
                prob = pred.get("probability", 0)
                options.append(f"Rank {j+1}: Code {code} ({prob:.2%})")

            sel = st.radio(
                f"Row {i+1}: {item['designation'][:50]}",
                range(len(options)),
                format_func=lambda idx, opts=options: opts[idx],
                key=f"batch_radio_{i}",
                horizontal=True,
            )
            selections[i] = sel

        if st.button("Confirm All Selections", key="batch_confirm_all"):
            count_corrections = 0
            count_total = 0
            for i, sel_idx in selections.items():
                item = items[i]
                result = results[i]
                top_k_preds = result.get("top_k_predictions", [])[:5]
                if not top_k_preds:
                    continue

                top1 = top_k_preds[0]
                selected = top_k_preds[sel_idx]
                selected_rank = sel_idx + 1

                row = _build_row(
                    user,
                    item["designation"],
                    item["description"],
                    top1["rakuten_code"],
                    top1.get("probability", 0),
                    selected_rank,
                    selected["rakuten_code"],
                )

                demo_path = _csv_path("demo_selections_csv")
                _write_csv_row(demo_path, row)

                if selected_rank > 1:
                    corr_path = _csv_path("corrections_csv")
                    _write_csv_row(corr_path, row)
                    count_corrections += 1

                count_total += 1

            st.success(f"{count_total} selections saved, {count_corrections} of which are corrections.")

        if result_df is not None:
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv_buffer.getvalue(),
                file_name="batch_results.csv",
                mime="text/csv",
            )
