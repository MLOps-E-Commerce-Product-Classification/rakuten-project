from __future__ import annotations

import argparse
import os
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.text_dataset import RakutenTextDataset
from src.evaluation.text_evaluate import evaluate_model, save_evaluation_results
from src.models.text_classifier import build_text_model


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_label_mapping(mapping_path: str | Path) -> dict[str, int]:
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "code_to_idx" in data:
        return data["code_to_idx"]
    return data


def load_full_label_encoding(mapping_path: str | Path) -> dict:
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_dataframe_columns(df, required_columns, df_name):
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{df_name} is missing required columns: {sorted(missing_columns)}"
        )


def apply_label_mapping(df, label_col, mapping, encoded_label_col="label"):
    df = df.copy()
    labels_as_str = df[label_col].astype(str)
    unseen_labels = sorted(set(labels_as_str) - set(mapping.keys()))
    if unseen_labels:
        raise ValueError(
            "Evaluation set contains labels not present in the training label mapping: "
            f"{unseen_labels[:10]}" + (" ..." if len(unseen_labels) > 10 else "")
        )
    df[encoded_label_col] = labels_as_str.map(mapping)
    df[encoded_label_col] = df[encoded_label_col].astype(int)
    return df


_STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.8,
    "axes.facecolor": "#fafafa",
    "figure.facecolor": "white",
    "axes.labelcolor": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "text.color": "#222222",
}

_PALETTE = {
    "excellent": "#2ecc71",
    "good": "#3498db",
    "fair": "#f39c12",
    "poor": "#e74c3c",
    "neutral": "#7f8c8d",
    "highlight": "#2c3e50",
}


def _f1_color(score: float) -> str:
    if score >= 0.90:
        return _PALETTE["excellent"]
    if score >= 0.80:
        return _PALETTE["good"]
    if score >= 0.70:
        return _PALETTE["fair"]
    return _PALETTE["poor"]


def generate_evaluation_plots(
    results: dict,
    output_dir: Path,
    idx_to_name: dict[str, str] | None = None,
    predictions: dict | None = None,
) -> None:
    """
    Generate and save a comprehensive set of professional evaluation plots.
    Outputs:
        1.  confusion_matrix.png          – row-normalised heatmap
        2.  per_class_f1.png              – sorted horizontal bar chart
        3.  classification_report.png     – colour-coded metrics table
        4.  metrics_summary.png           – headline KPI card
        5.  f1_distribution.png           – histogram + KDE of per-class F1
        6.  top_confusions.png            – top-N most confused class pairs
        7.  precision_recall_f1.png       – grouped bar chart (P / R / F1)
        8.  confidence_distribution.png   – correct vs incorrect confidence histogram
        9.  misclassified_samples.csv     – table of misclassified examples with confidence
        10. evaluation_summary.txt        – human-readable text report
    """
    import datetime

    output_dir.mkdir(parents=True, exist_ok=True)

    cm = results.get("confusion_matrix")
    per_class_f1 = results.get("per_class_f1", {})
    per_class_prec = results.get("per_class_precision", {})
    per_class_rec = results.get("per_class_recall", {})
    macro_f1 = results.get(
        "macro_f1", float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0
    )
    weighted_f1 = results.get("weighted_f1", None)
    accuracy = results.get("accuracy", None)
    loss = results.get("loss", None)

    def _label(idx) -> str:
        if idx_to_name:
            return idx_to_name.get(str(idx), str(idx))
        return str(idx)

    # ------------------------------------------------------------------ #
    # 1. Confusion Matrix (row-normalised)                                #
    # ------------------------------------------------------------------ #
    if cm:
        with plt.rc_context(_STYLE):
            cm_array = np.array(cm)
            cm_norm = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True).clip(
                min=1
            )
            n = cm_norm.shape[0]
            cell_sz = max(0.42, min(0.6, 14 / n))
            fig, ax = plt.subplots(figsize=(n * cell_sz + 2, n * cell_sz * 0.9 + 1.5))
            im = ax.imshow(
                cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1
            )
            cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("Recall", fontsize=9)
            ax.set_title(
                "Confusion Matrix  (row-normalised recall)",
                fontsize=13,
                fontweight="bold",
                pad=14,
            )
            ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
            ax.set_ylabel("True Label", fontsize=10, labelpad=8)
            tick_labels = [_label(i) for i in range(n)]
            ax.set_xticks(range(n))
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=max(5, 8 - n // 10))
            ax.set_yticks(range(n))
            ax.set_yticklabels(tick_labels, fontsize=max(5, 8 - n // 10))
            if n <= 20:
                for i in range(n):
                    for j in range(n):
                        val = cm_norm[i, j]
                        if val > 0.01:
                            ax.text(
                                j,
                                i,
                                f"{val:.2f}",
                                ha="center",
                                va="center",
                                fontsize=6,
                                color="white" if val > 0.6 else "#333",
                            )
            fig.tight_layout()
            fig.savefig(
                output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
        print("[Plots] Saved confusion_matrix.png")

    # ------------------------------------------------------------------ #
    # 2. Per-Class F1 – sorted horizontal bar chart                       #
    # ------------------------------------------------------------------ #
    if per_class_f1:
        with plt.rc_context(_STYLE):
            sorted_items = sorted(per_class_f1.items(), key=lambda x: x[1])
            classes = [_label(k) for k, _ in sorted_items]
            scores = [v for _, v in sorted_items]
            colors = [_f1_color(s) for s in scores]
            mean_f1 = float(np.mean(scores))

            fig, ax = plt.subplots(figsize=(9, max(5, len(classes) * 0.32 + 1)))
            bars = ax.barh(
                classes,
                scores,
                color=colors,
                edgecolor="white",
                linewidth=0.4,
                height=0.7,
            )
            ax.axvline(
                mean_f1,
                color=_PALETTE["highlight"],
                linestyle="--",
                linewidth=1.4,
                label=f"Mean F1: {mean_f1:.3f}",
                zorder=3,
            )
            ax.set_xlim(0, 1.08)
            ax.set_xlabel("F1 Score", fontsize=10)
            ax.set_title(
                "Per-Class F1 Score  (sorted)", fontsize=13, fontweight="bold", pad=12
            )
            for bar, score in zip(bars, scores):
                ax.text(
                    score + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}",
                    va="center",
                    fontsize=7.5,
                    color="#333",
                )
            from matplotlib.patches import Patch

            legend_els = [
                Patch(facecolor=_PALETTE["excellent"], label="≥ 0.90  Excellent"),
                Patch(facecolor=_PALETTE["good"], label="≥ 0.80  Good"),
                Patch(facecolor=_PALETTE["fair"], label="≥ 0.70  Fair"),
                Patch(facecolor=_PALETTE["poor"], label="< 0.70  Poor"),
            ]
            ax.legend(handles=legend_els, fontsize=8, loc="lower right", framealpha=0.9)
            fig.tight_layout()
            fig.savefig(output_dir / "per_class_f1.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        print("[Plots] Saved per_class_f1.png")

    # ------------------------------------------------------------------ #
    # 3. Classification Report – colour-coded table                       #
    # ------------------------------------------------------------------ #
    if per_class_f1:
        with plt.rc_context({**_STYLE, "axes.grid": False}):
            rows = []
            for cls, f1 in per_class_f1.items():
                prec = per_class_prec.get(cls, float("nan"))
                rec = per_class_rec.get(cls, float("nan"))
                rows.append(
                    [
                        _label(cls),
                        f"{prec:.4f}" if not np.isnan(prec) else "—",
                        f"{rec:.4f}" if not np.isnan(rec) else "—",
                        f"{f1:.4f}",
                    ]
                )
            rows.append(["", "", "", ""])
            rows.append(["macro avg", "—", "—", f"{macro_f1:.4f}"])
            if weighted_f1 is not None:
                rows.append(["weighted avg", "—", "—", f"{weighted_f1:.4f}"])
            if accuracy is not None:
                rows.append(["accuracy", "—", "—", f"{accuracy:.4f}"])

            col_labels = ["Class", "Precision", "Recall", "F1 Score"]
            n_rows = len(rows)
            fig, ax = plt.subplots(figsize=(6, max(4, n_rows * 0.26 + 1.2)))
            ax.axis("off")
            table = ax.table(
                cellText=rows, colLabels=col_labels, cellLoc="center", loc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.15, 1.18)
            for j in range(len(col_labels)):
                table[0, j].set_facecolor(_PALETTE["highlight"])
                table[0, j].set_text_props(color="white", fontweight="bold")
            for i, (cls, *_, f1_str) in enumerate(rows, start=1):
                try:
                    f1_val = float(f1_str)
                    c = _f1_color(f1_val)
                    table[i, 3].set_facecolor(c + "33")
                except ValueError:
                    pass
                if i % 2 == 0:
                    for j in range(len(col_labels)):
                        if table[i, j].get_facecolor()[3] < 0.1:
                            table[i, j].set_facecolor("#f5f5f5")
            ax.set_title(
                "Classification Report", fontsize=13, fontweight="bold", pad=14
            )
            fig.tight_layout()
            fig.savefig(
                output_dir / "classification_report.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
        print("[Plots] Saved classification_report.png")

    # ------------------------------------------------------------------ #
    # 4. Metrics Summary Card                                             #
    # ------------------------------------------------------------------ #
    with plt.rc_context({**_STYLE, "axes.grid": False, "axes.facecolor": "white"}):
        kpis = []
        if accuracy is not None:
            kpis.append(("Accuracy", f"{accuracy:.4f}", "#2c3e50"))
        if macro_f1 is not None:
            kpis.append(("Macro F1", f"{macro_f1:.4f}", _f1_color(macro_f1)))
        if weighted_f1 is not None:
            kpis.append(("Weighted F1", f"{weighted_f1:.4f}", _f1_color(weighted_f1)))
        if loss is not None:
            kpis.append(("Loss", f"{loss:.4f}", "#7f8c8d"))
        if per_class_f1:
            best_cls, best_f1 = max(per_class_f1.items(), key=lambda x: x[1])
            worst_cls, worst_f1 = min(per_class_f1.items(), key=lambda x: x[1])
            kpis.append(
                (
                    "Best Class",
                    f"{_label(best_cls)}\nF1={best_f1:.3f}",
                    _PALETTE["excellent"],
                )
            )
            kpis.append(
                (
                    "Worst Class",
                    f"{_label(worst_cls)}\nF1={worst_f1:.3f}",
                    _PALETTE["poor"],
                )
            )

        n_kpis = len(kpis)
        cols = min(n_kpis, 4)
        rows_n = (n_kpis + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 2.8, rows_n * 1.8))
        axes = np.array(axes).flatten()
        for ax in axes:
            ax.axis("off")
        for ax, (label, value, color) in zip(axes, kpis):
            ax.set_facecolor(color + "18")
            ax.patch.set_visible(True)
            ax.text(
                0.5,
                0.62,
                value,
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=color,
                transform=ax.transAxes,
                wrap=True,
            )
            ax.text(
                0.5,
                0.18,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="#555",
                transform=ax.transAxes,
            )
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color + "55")
                spine.set_linewidth(1.5)
        fig.suptitle("Evaluation Summary", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("[Plots] Saved metrics_summary.png")

    # ------------------------------------------------------------------ #
    # 5. F1 Distribution – histogram + KDE                               #
    # ------------------------------------------------------------------ #
    if per_class_f1 and len(per_class_f1) >= 4:
        with plt.rc_context(_STYLE):
            scores = np.array(list(per_class_f1.values()))
            fig, ax = plt.subplots(figsize=(8, 4))
            n_bins = min(15, max(5, len(scores) // 3))
            ax.hist(
                scores,
                bins=n_bins,
                color=_PALETTE["good"],
                edgecolor="white",
                linewidth=0.6,
                alpha=0.75,
                label="Class F1 distribution",
            )
            try:
                from scipy.stats import gaussian_kde

                kde_x = np.linspace(0, 1, 300)
                kde = gaussian_kde(scores, bw_method=0.3)
                ax2 = ax.twinx()
                ax2.plot(
                    kde_x,
                    kde(kde_x),
                    color=_PALETTE["highlight"],
                    linewidth=2,
                    label="KDE",
                )
                ax2.set_ylabel("Density", fontsize=9, color=_PALETTE["neutral"])
                ax2.tick_params(axis="y", labelcolor=_PALETTE["neutral"], labelsize=8)
                ax2.set_ylim(bottom=0)
                ax2.spines["top"].set_visible(False)
            except ImportError:
                pass
            ax.axvline(
                float(np.mean(scores)),
                color=_PALETTE["highlight"],
                linestyle="--",
                linewidth=1.4,
                label=f"Mean: {np.mean(scores):.3f}",
            )
            ax.axvline(
                float(np.median(scores)),
                color=_PALETTE["fair"],
                linestyle=":",
                linewidth=1.4,
                label=f"Median: {np.median(scores):.3f}",
            )
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("F1 Score", fontsize=10)
            ax.set_ylabel("Number of Classes", fontsize=10)
            ax.set_title(
                "Per-Class F1 Distribution", fontsize=13, fontweight="bold", pad=12
            )
            ax.legend(fontsize=9, framealpha=0.9)
            fig.tight_layout()
            fig.savefig(
                output_dir / "f1_distribution.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
        print("[Plots] Saved f1_distribution.png")

    # ------------------------------------------------------------------ #
    # 6. Top Confusions – most confused class pairs                       #
    # ------------------------------------------------------------------ #
    if cm:
        with plt.rc_context(_STYLE):
            cm_array = np.array(cm)
            n = cm_array.shape[0]
            pairs = []
            for i in range(n):
                row_total = cm_array[i].sum()
                for j in range(n):
                    if i != j and cm_array[i, j] > 0:
                        pairs.append((cm_array[i, j], row_total, i, j))
            pairs.sort(reverse=True)
            top_k = pairs[: min(20, len(pairs))]

            if top_k:
                labels_conf = [f"{_label(i)} → {_label(j)}" for _, _, i, j in top_k]
                counts = [c for c, _, _, _ in top_k]
                rates = [c / max(rt, 1) for c, rt, _, _ in top_k]

                fig, (ax1, ax2) = plt.subplots(
                    1, 2, figsize=(14, max(5, len(top_k) * 0.38 + 1.5))
                )
                colors_conf = [
                    _PALETTE["poor"]
                    if r > 0.2
                    else _PALETTE["fair"]
                    if r > 0.1
                    else _PALETTE["good"]
                    for r in rates
                ]
                ax1.barh(
                    labels_conf[::-1],
                    counts[::-1],
                    color=colors_conf[::-1],
                    edgecolor="white",
                    linewidth=0.4,
                    height=0.7,
                )
                ax1.set_xlabel("Misclassification Count", fontsize=10)
                ax1.set_title(
                    "Top Confused Pairs  (absolute)",
                    fontsize=12,
                    fontweight="bold",
                    pad=10,
                )
                for bar, cnt in zip(ax1.patches, counts[::-1]):
                    ax1.text(
                        bar.get_width() + 0.3,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(cnt)),
                        va="center",
                        fontsize=7.5,
                    )

                ax2.barh(
                    labels_conf[::-1],
                    [r * 100 for r in rates[::-1]],
                    color=colors_conf[::-1],
                    edgecolor="white",
                    linewidth=0.4,
                    height=0.7,
                )
                ax2.set_xlabel("Confusion Rate  (%)", fontsize=10)
                ax2.set_title(
                    "Top Confused Pairs  (% of true class)",
                    fontsize=12,
                    fontweight="bold",
                    pad=10,
                )
                for bar, rate in zip(ax2.patches, rates[::-1]):
                    ax2.text(
                        bar.get_width() + 0.3,
                        bar.get_y() + bar.get_height() / 2,
                        f"{rate * 100:.1f}%",
                        va="center",
                        fontsize=7.5,
                    )

                fig.suptitle(
                    "Most Confused Class Pairs", fontsize=14, fontweight="bold", y=1.01
                )
                fig.tight_layout()
                fig.savefig(
                    output_dir / "top_confusions.png", dpi=150, bbox_inches="tight"
                )
                plt.close(fig)
            print("[Plots] Saved top_confusions.png")

    # ------------------------------------------------------------------ #
    # 7. Precision / Recall / F1 grouped bar chart                        #
    # ------------------------------------------------------------------ #
    if per_class_f1 and (per_class_prec or per_class_rec):
        with plt.rc_context(_STYLE):
            sorted_keys = sorted(
                per_class_f1, key=lambda k: per_class_f1[k], reverse=True
            )
            cls_labels = [_label(k) for k in sorted_keys]
            f1s = [per_class_f1.get(k, 0.0) for k in sorted_keys]
            precs = [per_class_prec.get(k, 0.0) for k in sorted_keys]
            recs = [per_class_rec.get(k, 0.0) for k in sorted_keys]

            x = np.arange(len(cls_labels))
            width = 0.26
            fig, ax = plt.subplots(figsize=(max(12, len(cls_labels) * 0.55), 5))
            ax.bar(
                x - width,
                precs,
                width,
                label="Precision",
                color=_PALETTE["good"],
                alpha=0.85,
                edgecolor="white",
            )
            ax.bar(
                x,
                recs,
                width,
                label="Recall",
                color=_PALETTE["fair"],
                alpha=0.85,
                edgecolor="white",
            )
            ax.bar(
                x + width,
                f1s,
                width,
                label="F1 Score",
                color=_PALETTE["highlight"],
                alpha=0.85,
                edgecolor="white",
            )
            ax.set_ylim(0, 1.12)
            ax.set_xticks(x)
            ax.set_xticklabels(cls_labels, rotation=90, fontsize=8)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title(
                "Precision / Recall / F1  per Class",
                fontsize=13,
                fontweight="bold",
                pad=12,
            )
            ax.legend(fontsize=9, framealpha=0.9)
            ax.axhline(
                float(np.mean(f1s)),
                color=_PALETTE["poor"],
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
            )
            fig.tight_layout()
            fig.savefig(
                output_dir / "precision_recall_f1.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
        print("[Plots] Saved precision_recall_f1.png")

    # ------------------------------------------------------------------ #
    # 8. Confidence Distribution – correct vs incorrect                   #
    # ------------------------------------------------------------------ #
    if predictions and predictions.get("confidences"):
        with plt.rc_context(_STYLE):
            y_true = predictions["y_true"]
            y_pred = predictions["y_pred"]
            confs = predictions["confidences"]
            correct = [c for yt, yp, c in zip(y_true, y_pred, confs) if yt == yp]
            wrong = [c for yt, yp, c in zip(y_true, y_pred, confs) if yt != yp]

            fig, ax = plt.subplots(figsize=(8, 4))
            bins = np.linspace(0, 1, 30)
            ax.hist(
                correct,
                bins=bins,
                alpha=0.65,
                color=_PALETTE["excellent"],
                edgecolor="white",
                linewidth=0.5,
                label=f"Correct  (n={len(correct)})",
            )
            ax.hist(
                wrong,
                bins=bins,
                alpha=0.65,
                color=_PALETTE["poor"],
                edgecolor="white",
                linewidth=0.5,
                label=f"Incorrect (n={len(wrong)})",
            )
            ax.axvline(
                float(np.mean(correct)) if correct else 0,
                color=_PALETTE["excellent"],
                linestyle="--",
                linewidth=1.2,
                label=f"Mean correct: {np.mean(correct):.3f}" if correct else "",
            )
            ax.axvline(
                float(np.mean(wrong)) if wrong else 0,
                color=_PALETTE["poor"],
                linestyle="--",
                linewidth=1.2,
                label=f"Mean incorrect: {np.mean(wrong):.3f}" if wrong else "",
            )
            ax.set_xlabel("Confidence (max softmax probability)", fontsize=10)
            ax.set_ylabel("Number of Samples", fontsize=10)
            ax.set_title(
                "Prediction Confidence Distribution",
                fontsize=13,
                fontweight="bold",
                pad=12,
            )
            ax.legend(fontsize=9, framealpha=0.9)
            fig.tight_layout()
            fig.savefig(
                output_dir / "confidence_distribution.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
        print("[Plots] Saved confidence_distribution.png")

    # ------------------------------------------------------------------ #
    # 9. Misclassified Samples CSV                                        #
    # ------------------------------------------------------------------ #
    if predictions and predictions.get("y_true"):
        y_true = predictions["y_true"]
        y_pred = predictions["y_pred"]
        confs = predictions.get("confidences", [None] * len(y_true))
        texts = predictions.get("texts", [None] * len(y_true))
        sids = predictions.get("sample_ids", [None] * len(y_true))
        pids = predictions.get("product_ids", [None] * len(y_true))

        rows = []
        for i, (yt, yp, cf, tx, sid, pid) in enumerate(
            zip(y_true, y_pred, confs, texts, sids, pids)
        ):
            if yt != yp:
                rows.append(
                    {
                        "index": i,
                        "sample_id": sid,
                        "product_id": pid,
                        "true_label": _label(yt),
                        "pred_label": _label(yp),
                        "confidence": round(cf, 4) if cf is not None else None,
                        "text_snippet": str(tx)[:200] if tx else None,
                    }
                )
        if rows:
            df_err = pd.DataFrame(rows).sort_values("confidence", ascending=False)
            df_err.to_csv(output_dir / "misclassified_samples.csv", index=False)
            print(f"[Plots] Saved misclassified_samples.csv  ({len(rows)} errors)")

    # ------------------------------------------------------------------ #
    # 10. Human-readable text summary                                     #
    # ------------------------------------------------------------------ #
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        "  TEXT MODEL EVALUATION SUMMARY",
        f"  Generated: {ts}",
        "=" * 60,
        "",
    ]
    if accuracy is not None:
        lines.append(f"  Accuracy       : {accuracy:.4f}")
    if macro_f1 is not None:
        lines.append(f"  Macro F1       : {macro_f1:.4f}")
    if weighted_f1 is not None:
        lines.append(f"  Weighted F1    : {weighted_f1:.4f}")
    if loss is not None:
        lines.append(f"  Loss           : {loss:.4f}")
    lines.append("")
    if per_class_f1:
        scores_arr = np.array(list(per_class_f1.values()))
        lines += [
            f"  # Classes      : {len(per_class_f1)}",
            f"  F1 std-dev     : {float(np.std(scores_arr)):.4f}",
            f"  F1 median      : {float(np.median(scores_arr)):.4f}",
            f"  F1 min         : {float(np.min(scores_arr)):.4f}",
            f"  F1 max         : {float(np.max(scores_arr)):.4f}",
            "",
            "  Classes below 0.80 F1:",
        ]
        for cls, score in sorted(per_class_f1.items(), key=lambda x: x[1]):
            if score < 0.80:
                lines.append(f"    {_label(cls):<30s} {score:.4f}")
        lines += [
            "",
            "  Per-Class Breakdown (sorted by F1 desc):",
            f"  {'Class':<30s} {'Precision':>10} {'Recall':>10} {'F1':>10}",
            "  " + "-" * 62,
        ]
        for cls in sorted(per_class_f1, key=lambda k: per_class_f1[k], reverse=True):
            p = per_class_prec.get(cls, float("nan"))
            r = per_class_rec.get(cls, float("nan"))
            f = per_class_f1[cls]
            p_str = f"{p:.4f}" if not np.isnan(p) else "   —  "
            r_str = f"{r:.4f}" if not np.isnan(r) else "   —  "
            lines.append(f"  {_label(cls):<30s} {p_str:>10} {r_str:>10} {f:>10.4f}")
    lines += ["", "=" * 60]
    summary_path = output_dir / "evaluation_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("[Plots] Saved evaluation_summary.txt")


_PLOT_FILES = [
    "confusion_matrix.png",
    "per_class_f1.png",
    "classification_report.png",
    "metrics_summary.png",
    "f1_distribution.png",
    "top_confusions.png",
    "precision_recall_f1.png",
    "confidence_distribution.png",
    "misclassified_samples.csv",
    "evaluation_summary.txt",
]


def log_evaluation_to_mlflow(results, results_output_path, mlflow_run_id):
    plots_dir = results_output_path.parent
    with mlflow.start_run(run_id=mlflow_run_id):
        metrics = results.get("metrics", {})
        for key in [
            "macro_f1",
            "weighted_f1",
            "accuracy",
            "loss",
            "macro_precision",
            "macro_recall",
        ]:
            if key in metrics:
                mlflow.log_metric(f"eval_{key}", float(metrics[key]))
        for cls, score in metrics.get("per_class_f1", {}).items():
            mlflow.log_metric(f"eval_f1_class_{cls}", float(score))
        for plot_file in _PLOT_FILES:
            plot_path = plots_dir / plot_file
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path), artifact_path="evaluation_plots")
            else:
                print(f"[MLflow] Artifact not found, skipping: {plot_path}")
        if results_output_path.exists():
            mlflow.log_artifact(
                str(results_output_path), artifact_path="evaluation_plots"
            )
    print(f"[MLflow] Evaluation results logged to run: {mlflow_run_id}")


def run_text_evaluation(
    x_data_csv_path: str | Path,
    y_data_csv_path: str | Path,
    split_ids_dir: str | Path | None = None,
    train_config_path: str | Path = "configs/text_train_config.yaml",
    eval_config_path: str | Path = "configs/text_evaluate_config.yaml",
    preprocessing_config_path: str | Path = "configs/text_preprocessing_config.yaml",
    model_weights_path: str | Path = "models/best_text_model.pt",
    label_encoding_path: str | Path = "artifacts/label_mapping.json",
    results_output_path: str | Path = "results/text_evaluation_results.json",
    mlflow_run_id: str | None = None,
) -> dict:
    x_data_csv_path = Path(x_data_csv_path)
    y_data_csv_path = Path(y_data_csv_path)
    model_weights_path = Path(model_weights_path)
    label_encoding_path = Path(label_encoding_path)
    results_output_path = Path(results_output_path)

    if not x_data_csv_path.exists():
        raise FileNotFoundError(f"X data CSV not found: {x_data_csv_path}")
    if not y_data_csv_path.exists():
        raise FileNotFoundError(f"Y data CSV not found: {y_data_csv_path}")
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")

    train_config = load_config(train_config_path)
    label_mapping = load_label_mapping(label_encoding_path)

    full_encoding = load_full_label_encoding(label_encoding_path)
    idx_to_name = {str(k): v for k, v in full_encoding.get("idx_to_name", {}).items()}

    training_config = train_config.get("training", {})
    model_config = train_config.get("model", {})
    data_config = train_config.get("data", {})

    if x_data_csv_path == y_data_csv_path:
        eval_df = pd.read_csv(x_data_csv_path)
    else:
        x_df = pd.read_csv(x_data_csv_path)
        y_df = pd.read_csv(y_data_csv_path)
        eval_df = pd.concat([x_df, y_df], axis=1)

    designation_col = data_config.get("designation_col", "designation")
    description_col = data_config.get("description_col", "description")
    label_col = data_config.get("label_col", "prdtypecode")
    encoded_label_col = data_config.get("encoded_label_col", "label")
    return_quality_report = bool(data_config.get("return_quality_report", False))

    required_columns = {designation_col}
    if encoded_label_col not in eval_df.columns:
        required_columns.add(label_col)
    validate_dataframe_columns(eval_df, required_columns, "Merged Evaluation Data")
    if description_col not in eval_df.columns:
        eval_df[description_col] = ""

    if encoded_label_col in eval_df.columns and pd.api.types.is_integer_dtype(
        eval_df[encoded_label_col]
    ):
        print(
            f"Column '{encoded_label_col}' already exists and is numeric. Using existing labels."
        )
    else:
        eval_df = apply_label_mapping(
            eval_df,
            label_col=label_col,
            mapping=label_mapping,
            encoded_label_col=encoded_label_col,
        )

    model_name = model_config.get("name", "bert-base-multilingual-cased")
    num_classes = len(label_mapping)
    identity_encoding = {"code_to_idx": {str(i): i for i in range(num_classes)}}

    eval_dataset = RakutenTextDataset(
        dataframe=eval_df,
        config_path=preprocessing_config_path,
        label_encoding=identity_encoding,
        designation_col=designation_col,
        description_col=description_col,
        label_col=encoded_label_col,
        return_quality_report=return_quality_report,
    )

    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_text_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    results = evaluate_model(
        model=model,
        dataloader=eval_dataloader,
        criterion=criterion,
        device=device,
        config_path=eval_config_path,
        num_classes=num_classes,
    )

    results["metadata"] = {
        "model_name": model_name,
        "num_classes": num_classes,
        "x_data_csv_path": str(x_data_csv_path),
        "y_data_csv_path": str(y_data_csv_path),
        "split_ids_dir": str(split_ids_dir) if split_ids_dir else None,
        "model_weights_path": str(model_weights_path),
    }
    save_evaluation_results(results, results_output_path)

    plots_dir = results_output_path.parent
    generate_evaluation_plots(
        results["metrics"],
        plots_dir,
        idx_to_name=idx_to_name,
        predictions=results.get("predictions"),
    )

    if mlflow_run_id:
        log_evaluation_to_mlflow(
            results=results,
            results_output_path=results_output_path,
            mlflow_run_id=mlflow_run_id,
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run text model evaluation.")
    parser.add_argument("--x_data_csv_path", default="data/raw/X_val.csv")
    parser.add_argument("--y_data_csv_path", default="data/raw/Y_val.csv")
    parser.add_argument("--split_ids_dir", default="artifacts/splits")
    parser.add_argument("--train_config_path", default="configs/text_train_config.yaml")
    parser.add_argument(
        "--eval_config_path", default="configs/text_evaluate_config.yaml"
    )
    parser.add_argument(
        "--preprocessing_config_path", default="configs/text_preprocessing_config.yaml"
    )
    parser.add_argument("--model_weights_path", default="models/best_text_model.pt")
    parser.add_argument("--label_encoding_path", default="artifacts/label_mapping.json")
    parser.add_argument(
        "--results_output_path", default="results/text_evaluation_results.json"
    )
    parser.add_argument("--mlflow_run_id", default=None)
    args = parser.parse_args()

    mlflow_run_id = args.mlflow_run_id or os.environ.get("MLFLOW_RUN_ID")

    results = run_text_evaluation(
        x_data_csv_path=args.x_data_csv_path,
        y_data_csv_path=args.y_data_csv_path,
        split_ids_dir=args.split_ids_dir,
        train_config_path=args.train_config_path,
        eval_config_path=args.eval_config_path,
        preprocessing_config_path=args.preprocessing_config_path,
        model_weights_path=args.model_weights_path,
        label_encoding_path=args.label_encoding_path,
        results_output_path=args.results_output_path,
        mlflow_run_id=mlflow_run_id,
    )

    print("Evaluation finished.")
    print(f"Main metric: {results['main_metric']} = {results['main_metric_value']:.4f}")
    print("Results saved to: results/text_evaluation_results.json")
