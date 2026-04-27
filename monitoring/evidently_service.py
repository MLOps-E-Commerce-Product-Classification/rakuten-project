"""
Evidently AI Drift Detection Service
=====================================
Compares data/raw/*_update.csv (reference) vs data/raw/*_new.csv (current)
and exposes Prometheus metrics on /metrics.

Metrics exposed:
  - rakuten_drift_detected          (gauge, 0/1 per feature)
  - rakuten_drift_score             (gauge, drift score per feature)
  - rakuten_dataset_drift_share     (gauge, share of drifted features)
  - rakuten_dataset_drift_detected  (gauge, 0/1 overall)
  - rakuten_reference_row_count     (gauge)
  - rakuten_current_row_count       (gauge)
  - rakuten_label_distribution      (gauge, per label)
  - rakuten_label_drift_detected    (gauge, 0/1)
  - rakuten_last_run_timestamp      (gauge, unix timestamp)
  - rakuten_service_ready           (gauge, 1 if data available)
"""

from __future__ import annotations

import logging
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread, Lock

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import (
    DatasetDriftMetric,
    ColumnDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from prometheus_client import (
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data/raw"))
REFERENCE_X = DATA_DIR / "X_train_update.csv"
REFERENCE_Y = DATA_DIR / "Y_train_CVw08PX.csv"
CURRENT_X = DATA_DIR / "X_train_new.csv"
CURRENT_Y = DATA_DIR / "Y_train_new.csv"

RUN_INTERVAL = int(os.getenv("DRIFT_RUN_INTERVAL", "300"))   # seconds between runs
PORT = int(os.getenv("DRIFT_PORT", "8080"))

TEXT_FEATURES = ["designation", "description"]

# ---------------------------------------------------------------------------
# Prometheus Registry & Metrics
# ---------------------------------------------------------------------------
registry = CollectorRegistry()

g_drift_detected = Gauge(
    "rakuten_drift_detected",
    "1 if drift detected for this feature, 0 otherwise",
    ["feature"],
    registry=registry,
)
g_drift_score = Gauge(
    "rakuten_drift_score",
    "Drift score (p-value or statistic) for this feature",
    ["feature"],
    registry=registry,
)
g_dataset_drift_share = Gauge(
    "rakuten_dataset_drift_share",
    "Share of drifted features (0.0 – 1.0)",
    registry=registry,
)
g_dataset_drift_detected = Gauge(
    "rakuten_dataset_drift_detected",
    "1 if overall dataset drift detected, 0 otherwise",
    registry=registry,
)
g_reference_rows = Gauge(
    "rakuten_reference_row_count",
    "Number of rows in the reference dataset (*_update)",
    registry=registry,
)
g_current_rows = Gauge(
    "rakuten_current_row_count",
    "Number of rows in the current dataset (*_new)",
    registry=registry,
)
g_label_dist_ref = Gauge(
    "rakuten_label_distribution_reference",
    "Fraction of label in reference dataset",
    ["label"],
    registry=registry,
)
g_label_dist_cur = Gauge(
    "rakuten_label_distribution_current",
    "Fraction of label in current dataset",
    ["label"],
    registry=registry,
)
g_label_drift = Gauge(
    "rakuten_label_drift_detected",
    "1 if label distribution drift detected, 0 otherwise",
    registry=registry,
)
g_label_drift_score = Gauge(
    "rakuten_label_drift_score",
    "Drift score for target label column",
    registry=registry,
)
g_missing_ref = Gauge(
    "rakuten_missing_values_reference_share",
    "Share of missing values in reference dataset",
    registry=registry,
)
g_missing_cur = Gauge(
    "rakuten_missing_values_current_share",
    "Share of missing values in current dataset",
    registry=registry,
)
g_last_run = Gauge(
    "rakuten_drift_last_run_timestamp",
    "Unix timestamp of the last successful drift computation",
    registry=registry,
)
g_service_ready = Gauge(
    "rakuten_service_ready",
    "1 if both reference and current datasets are available",
    registry=registry,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_metrics_lock = Lock()
_last_metrics_output: bytes = b""


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------
def _build_text_features(df_x: pd.DataFrame) -> pd.DataFrame:
    """
    Create numeric features from raw text columns that Evidently can analyse:
      - <col>_length    : character count (or 0 if NaN)
      - <col>_word_count: word count
      - <col>_is_empty  : 1 if empty/NaN
    """
    frames = {}
    for col in TEXT_FEATURES:
        if col not in df_x.columns:
            continue
        s = df_x[col].fillna("")
        frames[f"{col}_length"] = s.str.len().astype(float)
        frames[f"{col}_word_count"] = s.str.split().str.len().fillna(0).astype(float)
        frames[f"{col}_is_empty"] = (s == "").astype(float)
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Core drift computation
# ---------------------------------------------------------------------------
def run_drift_analysis() -> None:
    """Load data, run Evidently, push metrics to Prometheus gauges."""
    log.info("Starting drift analysis …")

    # --- availability check ------------------------------------------------
    ref_x_ok = REFERENCE_X.exists()
    cur_x_ok = CURRENT_X.exists()
    ref_y_ok = REFERENCE_Y.exists()
    cur_y_ok = CURRENT_Y.exists()

    if not (ref_x_ok and ref_y_ok):
        log.warning("Reference data not found: %s / %s", REFERENCE_X, REFERENCE_Y)
        g_service_ready.set(0)
        return

    if not (cur_x_ok and cur_y_ok):
        log.info(
            "Current (*_new) data not yet available – will retry in %ss", RUN_INTERVAL
        )
        g_service_ready.set(0)
        return

    g_service_ready.set(1)

    # --- load ---------------------------------------------------------------
    log.info("Loading reference  : %s, %s", REFERENCE_X, REFERENCE_Y)
    log.info("Loading current    : %s, %s", CURRENT_X, CURRENT_Y)

    ref_x = pd.read_csv(REFERENCE_X, index_col=0)
    cur_x = pd.read_csv(CURRENT_X, index_col=0)
    ref_y = pd.read_csv(REFERENCE_Y, index_col=0)
    cur_y = pd.read_csv(CURRENT_Y, index_col=0)

    # align index
    ref_x = ref_x.reset_index(drop=True)
    cur_x = cur_x.reset_index(drop=True)
    ref_y = ref_y.reset_index(drop=True)
    cur_y = cur_y.reset_index(drop=True)

    g_reference_rows.set(len(ref_x))
    g_current_rows.set(len(cur_x))

    # --- feature engineering -----------------------------------------------
    ref_feat = _build_text_features(ref_x)
    cur_feat = _build_text_features(cur_x)

    target_col = "prdtypecode"
    ref_feat[target_col] = ref_y[target_col].astype(str)
    cur_feat[target_col] = cur_y[target_col].astype(str)

    numeric_features = [c for c in ref_feat.columns if c != target_col]

    column_mapping = ColumnMapping(
        target=target_col,
        numerical_features=numeric_features,
        categorical_features=[target_col],
    )

    # --- label distribution ------------------------------------------------
    for df_g, g_dist in [(ref_feat, g_label_dist_ref), (cur_feat, g_label_dist_cur)]:
        counts = df_g[target_col].value_counts(normalize=True)
        for label, frac in counts.items():
            g_dist.labels(label=str(label)).set(frac)

    # --- Evidently report ---------------------------------------------------
    metrics_list = [
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
    for feat in numeric_features:
        metrics_list.append(ColumnDriftMetric(column_name=feat))

    # label drift
    metrics_list.append(ColumnDriftMetric(column_name=target_col))

    report = Report(metrics=metrics_list)

    log.info("Running Evidently report (ref=%d rows, cur=%d rows) …", len(ref_feat), len(cur_feat))
    report.run(
        reference_data=ref_feat,
        current_data=cur_feat,
        column_mapping=column_mapping,
    )

    result = report.as_dict()
    metrics_results = result.get("metrics", [])

    # --- parse results ------------------------------------------------------
    for m in metrics_results:
        m_type = m.get("metric", "")
        val = m.get("result", {})

        if m_type == "DatasetDriftMetric":
            share = val.get("share_of_drifted_columns", 0.0)
            detected = int(val.get("dataset_drift", False))
            g_dataset_drift_share.set(share)
            g_dataset_drift_detected.set(detected)
            log.info(
                "Dataset drift: detected=%s, share=%.3f", bool(detected), share
            )

        elif m_type == "DatasetMissingValuesMetric":
            cur_mv = val.get("current", {})
            ref_mv = val.get("reference", {})
            ref_total = ref_mv.get("number_of_rows", 1) or 1
            cur_total = cur_mv.get("number_of_rows", 1) or 1
            g_missing_ref.set(
                ref_mv.get("number_of_missing_values", 0) / ref_total
            )
            g_missing_cur.set(
                cur_mv.get("number_of_missing_values", 0) / cur_total
            )

        elif m_type == "ColumnDriftMetric":
            col = val.get("column_name", "")
            detected = int(val.get("drift_detected", False))
            score = float(val.get("drift_score", 0.0))
            if col == target_col:
                g_label_drift.set(detected)
                g_label_drift_score.set(score)
                log.info("Label drift: detected=%s, score=%.4f", bool(detected), score)
            else:
                g_drift_detected.labels(feature=col).set(detected)
                g_drift_score.labels(feature=col).set(score)
                log.info(
                    "Feature '%s': drift=%s, score=%.4f", col, bool(detected), score
                )

    g_last_run.set(time.time())

    # --- serialise for /metrics endpoint ------------------------------------
    with _metrics_lock:
        global _last_metrics_output
        _last_metrics_output = generate_latest(registry)

    log.info("Drift analysis complete.")


# ---------------------------------------------------------------------------
# Background scheduler
# ---------------------------------------------------------------------------
def _scheduler_loop() -> None:
    while True:
        try:
            run_drift_analysis()
        except Exception as exc:  # noqa: BLE001
            log.exception("Drift analysis failed: %s", exc)
        time.sleep(RUN_INTERVAL)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default access log
        pass

    def do_GET(self):  # noqa: N802
        if self.path in ("/metrics", "/metrics/"):
            with _metrics_lock:
                data = _last_metrics_output
            if not data:
                data = generate_latest(registry)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(data)

        elif self.path in ("/health", "/healthz"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK\n")

        else:
            self.send_response(404)
            self.end_headers()


def main() -> None:
    log.info("Evidently Drift Service starting on port %d …", PORT)
    log.info("Reference : %s / %s", REFERENCE_X, REFERENCE_Y)
    log.info("Current   : %s / %s", CURRENT_X, CURRENT_Y)
    log.info("Run interval: %ds", RUN_INTERVAL)

    # initial run before accepting traffic
    try:
        run_drift_analysis()
    except Exception as exc:  # noqa: BLE001
        log.exception("Initial drift run failed: %s", exc)

    t = Thread(target=_scheduler_loop, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    log.info("Serving on http://0.0.0.0:%d/metrics", PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()
