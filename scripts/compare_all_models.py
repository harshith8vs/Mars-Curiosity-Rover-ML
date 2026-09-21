#!/usr/bin/env python3
"""
Cross-model final TEST comparison for all 9 models.

Evaluation Protocol (Phase B — Final Comparison):
  Computes canonical Macro-F1 over ALL 24 ACTIVE classes, including NASA IDs
  5 and 23 which have zero samples in the TEST split (F1=0.0, counted in average).
  Class 22 ("sun") is GLOBALLY INACTIVE and is excluded from all metrics.

Models compared (5 classical + 4 deep learning):
  01 KNN               02 Naive Bayes        03 Decision Tree
  04 Random Forest     05 SVM
  06 MRSCAtt           07 ViT-B/16           08 ResNet-50
  09 EfficientNet-B3

Outputs to results/10_Overall_Comparison/:
  final_test_comparison.csv           — numeric table
  final_test_comparison.json          — structured JSON
  final_test_comparison.md            — GitHub markdown table
  final_test_comparison_bar.png       — Macro-F1 + Accuracy bar chart
  final_test_comparison_heatmap.png   — Metric heatmap across all models

Existing artifacts (model_comparison.csv / model_comparison.png) are
PRESERVED; only new "final_test_*" prefixed files are generated.
"""

import json
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from evaluation.mapping import ACTIVE_NASA_CLASS_IDS, ACTIVE_INDICES, NUM_ACTIVE_CLASSES
from evaluation.metrics import compute_canonical_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Model specification: prediction file paths and label space
# ──────────────────────────────────────────────────────────────────────────────

MODEL_SPECS = [
    {
        "rank": 1,
        "key": "knn",
        "display_name": "KNN (Scaled+PCA)",
        "category": "Classical",
        "preds": PROJECT_ROOT / "results" / "01_KNN" / "test_predictions.npy",
        "labels": PROJECT_ROOT / "results" / "01_KNN" / "test_labels.npy",
        "label_space": "nasa",
    },
    {
        "rank": 2,
        "key": "naive_bayes",
        "display_name": "Naive Bayes (Scaled)",
        "category": "Classical",
        "preds": PROJECT_ROOT / "results" / "02_Naive_Bayes" / "test_predictions.npy",
        "labels": PROJECT_ROOT / "results" / "02_Naive_Bayes" / "test_labels.npy",
        "label_space": "nasa",
    },
    {
        "rank": 3,
        "key": "decision_tree",
        "display_name": "Decision Tree (Balanced)",
        "category": "Classical",
        "preds": PROJECT_ROOT / "results" / "03_Decision_Tree" / "test_predictions.npy",
        "labels": PROJECT_ROOT / "results" / "03_Decision_Tree" / "test_labels.npy",
        "label_space": "nasa",
    },
    {
        "rank": 4,
        "key": "random_forest",
        "display_name": "Random Forest (Balanced)",
        "category": "Classical",
        "preds": PROJECT_ROOT / "results" / "04_Random_Forest" / "test_predictions.npy",
        "labels": PROJECT_ROOT / "results" / "04_Random_Forest" / "test_labels.npy",
        "label_space": "nasa",
    },
    {
        "rank": 5,
        "key": "svm",
        "display_name": "SVM (Scaled+PCA, Balanced)",
        "category": "Classical",
        "preds": PROJECT_ROOT / "results" / "05_SVM" / "test_predictions.npy",
        "labels": PROJECT_ROOT / "results" / "05_SVM" / "test_labels.npy",
        "label_space": "nasa",
    },
    {
        "rank": 6,
        "key": "mrscatt",
        "display_name": "MRSCAtt (Attention)",
        "category": "Deep Learning",
        "preds": PROJECT_ROOT / "results" / "06_MRSCAtt" / "predictions" / "test_predictions.npy",
        "labels": PROJECT_ROOT / "results" / "06_MRSCAtt" / "predictions" / "test_labels.npy",
        "label_space": "active_index",
    },
    {
        "rank": 7,
        "key": "vit",
        "display_name": "ViT-B/16 (Transformer)",
        "category": "Deep Learning",
        "preds": PROJECT_ROOT / "results" / "07_ViT_B16" / "predictions" / "test_predictions.npy",
        "labels": PROJECT_ROOT / "results" / "07_ViT_B16" / "predictions" / "test_labels.npy",
        "label_space": "active_index",
    },
    {
        "rank": 8,
        "key": "resnet50",
        "display_name": "ResNet-50 (Baseline)",
        "category": "Deep Learning",
        "preds": PROJECT_ROOT / "results" / "08_ResNet50" / "predictions" / "test_predictions.npy",
        "labels": PROJECT_ROOT / "results" / "08_ResNet50" / "predictions" / "test_labels.npy",
        "label_space": "active_index",
    },
    {
        "rank": 9,
        "key": "efficientnet_b3",
        "display_name": "EfficientNet-B3 (Champion)",
        "category": "Deep Learning",
        "preds": PROJECT_ROOT / "results" / "09_EfficientNet_B3" / "predictions" / "test_predictions.npy",
        "labels": PROJECT_ROOT / "results" / "09_EfficientNet_B3" / "predictions" / "test_labels.npy",
        "label_space": "active_index",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(verbose: bool = True) -> list[dict]:
    """Load stored predictions and compute canonical test metrics for all 9 models."""
    results = []
    missing = []

    for spec in MODEL_SPECS:
        if not spec["preds"].is_file() or not spec["labels"].is_file():
            missing.append(spec["display_name"])
            continue

        y_true = np.load(spec["labels"])
        y_pred = np.load(spec["preds"])
        n_samples = len(y_true)
        m = compute_canonical_metrics(y_true, y_pred, label_space=spec["label_space"])

        row = {
            "rank": spec["rank"],
            "key": spec["key"],
            "model": spec["display_name"],
            "category": spec["category"],
            "n_test_samples": n_samples,
            "accuracy_pct": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 6),
            "macro_precision": round(m["macro_precision"], 6),
            "macro_recall": round(m["macro_recall"], 6),
            "weighted_f1": round(m["weighted_f1"], 6),
            "label_space": spec["label_space"],
        }
        results.append(row)
        if verbose:
            print(
                f"  {spec['display_name']:<32}  "
                f"Acc={m['accuracy']:6.2f}%  "
                f"Macro-F1={m['macro_f1']:.4f}  "
                f"Weighted-F1={m['weighted_f1']:.4f}"
            )

    if missing:
        print(f"\n  [WARNING] Prediction files not found for: {missing}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Output generation
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(results: list[dict], out_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df = df.sort_values("rank").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def save_json(results: list[dict], out_path: Path) -> None:
    payload = {
        "evaluation_split": "TEST",
        "metric_protocol": (
            "Canonical 24-class Macro-F1: labels=ACTIVE_NASA_CLASS_IDS "
            "(excl. class 22 'sun'). IDs 5 and 23 are active but have 0 "
            "test samples — counted in denominator with F1=0.0."
        ),
        "num_active_classes": NUM_ACTIVE_CLASSES,
        "num_test_samples": results[0]["n_test_samples"] if results else 0,
        "models": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def save_markdown(results: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(results).sort_values("rank")
    lines = [
        "# Final TEST Set Comparison — All 9 Models",
        "",
        "> **Evaluation Protocol (Phase B)**  ",
        "> Canonical 24-class Macro-F1: `labels=ACTIVE_NASA_CLASS_IDS` (IDs 0–21, 23, 24).  ",
        "> Class 22 (\"sun\") is globally inactive (excluded).  ",
        "> NASA IDs 5 and 23 are active but have 0 test samples;  ",
        "> F1=0.0 is assigned and counted in the 24-class denominator.",
        "",
        "| # | Model | Category | Test Samples | Accuracy (%) | Macro-F1 | Macro-P | Macro-R | Weighted-F1 |",
        "|---|-------|----------|:------------:|:------------:|:--------:|:-------:|:-------:|:-----------:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {int(row['rank'])} "
            f"| {row['model']} "
            f"| {row['category']} "
            f"| {int(row['n_test_samples'])} "
            f"| {row['accuracy_pct']:.2f} "
            f"| **{row['macro_f1']:.4f}** "
            f"| {row['macro_precision']:.4f} "
            f"| {row['macro_recall']:.4f} "
            f"| {row['weighted_f1']:.4f} |"
        )
    lines += ["", "---", "_Generated by scripts/compare_all_models.py_"]
    out_path.write_text("\n".join(lines))


def save_bar_chart(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: Macro-F1 + Accuracy for all 9 models."""
    df = df.sort_values("rank")
    n = len(df)
    x = np.arange(n)
    width = 0.38

    colors_f1 = ["#4C8CBF" if c == "Classical" else "#E88D27" for c in df["category"]]
    colors_acc = ["#A8C8E8" if c == "Classical" else "#F5CA87" for c in df["category"]]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, df["macro_f1"], width, label="Macro-F1",
                   color=colors_f1, edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width / 2, df["accuracy_pct"] / 100, width, label="Accuracy",
                   color=colors_acc, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Final TEST Set Comparison — All 9 Models\n"
        "(Canonical 24-class Macro-F1 | Class 22 excluded | IDs 5 & 23 counted at F1=0.0)",
        fontsize=12, weight="bold", pad=14,
    )

    # Value annotations
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h*100:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    # Custom legend combining category color and metric
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#4C8CBF", label="Classical — Macro-F1"),
        Patch(facecolor="#A8C8E8", label="Classical — Accuracy"),
        Patch(facecolor="#E88D27", label="Deep Learning — Macro-F1"),
        Patch(facecolor="#F5CA87", label="Deep Learning — Accuracy"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left", framealpha=0.85)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metric_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Metric heatmap: one row per model, one column per metric."""
    df = df.sort_values("rank")
    metric_cols = ["macro_f1", "accuracy_pct", "macro_precision", "macro_recall", "weighted_f1"]
    display_cols = ["Macro-F1", "Accuracy (%)", "Macro-Precision", "Macro-Recall", "Weighted-F1"]

    heatmap_data = df[metric_cols].copy()
    # Normalize accuracy to 0-1 for uniform colour scale
    heatmap_data["accuracy_pct"] = heatmap_data["accuracy_pct"] / 100.0
    heatmap_data.columns = display_cols
    heatmap_data.index = df["model"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".3f", cmap="YlOrRd",
        vmin=0.0, vmax=1.0,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Score (0–1)"},
        ax=ax,
    )
    ax.set_title(
        "Test Set Metric Heatmap — All 9 Models\n"
        "(Canonical 24-class | Accuracy displayed as fraction)",
        fontsize=12, weight="bold", pad=12,
    )
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    out_dir = PROJECT_ROOT / "results" / "10_Overall_Comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" PHASE B — FINAL TEST SET COMPARISON (ALL 9 MODELS)")
    print(" Canonical 24-class Macro-F1 | Using stored prediction arrays")
    print(" (No model checkpoint files required)")
    print("=" * 72)

    results = compute_all_metrics(verbose=True)
    if not results:
        print("\n[ERROR] No prediction artifacts found. Cannot generate comparison.")
        sys.exit(1)

    df = save_csv(results, out_dir / "final_test_comparison.csv")
    save_json(results, out_dir / "final_test_comparison.json")
    save_markdown(results, out_dir / "final_test_comparison.md")
    save_bar_chart(df, out_dir / "final_test_comparison_bar.png")
    save_metric_heatmap(df, out_dir / "final_test_comparison_heatmap.png")

    print("\n" + "=" * 72)
    print(f" Outputs saved to: {out_dir}")
    print(" Files generated:")
    for fname in [
        "final_test_comparison.csv",
        "final_test_comparison.json",
        "final_test_comparison.md",
        "final_test_comparison_bar.png",
        "final_test_comparison_heatmap.png",
    ]:
        print(f"   ✓ {fname}")
    print()

    # Print summary table
    df_sorted = df.sort_values("rank")
    print(f"{'#':<4} {'Model':<33} {'Category':<14} {'Acc%':>7} {'Macro-F1':>10}")
    print("-" * 72)
    for _, row in df_sorted.iterrows():
        flag = " ← CHAMPION" if row["macro_f1"] == df["macro_f1"].max() else ""
        print(
            f"{int(row['rank']):<4} {row['model']:<33} {row['category']:<14} "
            f"{row['accuracy_pct']:>7.2f} {row['macro_f1']:>10.4f}{flag}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
