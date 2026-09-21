#!/usr/bin/env python3
"""
Automated Classification of Martian Surface Images Captured by NASA's Curiosity Rover Using Machine Learning
Unified Project CLI Orchestrator """

import argparse
from collections import Counter
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Central configuration
import config


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def status_pass(msg: str) -> str:
    return f"{Colors.GREEN}[PASS]{Colors.RESET} {msg}"


def status_warn(msg: str) -> str:
    return f"{Colors.YELLOW}[WARNING]{Colors.RESET} {msg}"


def status_fail(msg: str) -> str:
    return f"{Colors.RED}[FAIL]{Colors.RESET} {msg}"


# =====================================================================
# Official Benchmark Table & Display
# =====================================================================

# NOTE on evaluation protocol:
# Phase A (Model Selection): Classical ML config tuned on VALIDATION split.
#   Validation metrics shown in Phase A only; these are NOT the final comparison metrics.
# Phase B (Final Cross-Model Comparison): ALL 9 models evaluated on TEST split.
#   Use --compare or scripts/compare_all_models.py to display Phase B results.
# Canonical 24-class Macro-F1: class 22 ('sun') excluded, IDs 5 and 23 are active
# but have 0 test samples; they count in the denominator with F1=0.0.
OFFICIAL_BENCHMARK = [
    {"family": "Classical ML", "model": "KNN",            "config": "KNN-4",       "metric_split": "Test",  "accuracy": 47.20, "macro_f1": 0.5182, "note": "Canonical 24-class | Phase B TEST"},
    {"family": "Classical ML", "model": "Naive Bayes",    "config": "NB-2",        "metric_split": "Test",  "accuracy": 18.01, "macro_f1": 0.2093, "note": "Canonical 24-class | Phase B TEST"},
    {"family": "Classical ML", "model": "Decision Tree",  "config": "DT-2",        "metric_split": "Test",  "accuracy": 25.44, "macro_f1": 0.2551, "note": "Canonical 24-class | Phase B TEST"},
    {"family": "Classical ML", "model": "Random Forest",  "config": "RF-2",        "metric_split": "Test",  "accuracy": 50.42, "macro_f1": 0.4305, "note": "Canonical 24-class | Phase B TEST"},
    {"family": "Classical ML", "model": "SVM",            "config": "SVM-4",       "metric_split": "Test",  "accuracy": 48.89, "macro_f1": 0.4920, "note": "Canonical 24-class | Phase B TEST"},
    {"family": "Deep Learning", "model": "MRSCAtt",       "config": "Attention-CNN","metric_split": "Test",  "accuracy": 64.29, "macro_f1": 0.5851, "note": "Spatial & Channel Attention"},
    {"family": "Deep Learning", "model": "ViT-B/16",      "config": "196 Patches", "metric_split": "Test",  "accuracy": 73.26, "macro_f1": 0.6536, "note": "Vision Transformer"},
    {"family": "Deep Learning", "model": "ResNet-50",     "config": "224x224 RGB", "metric_split": "Test",  "accuracy": 77.78, "macro_f1": 0.6631, "note": "Deep Learning Baseline"},
    {"family": "Deep Learning", "model": "EfficientNet-B3","config": "300x300 RGB","metric_split": "Test",  "accuracy": 80.61, "macro_f1": 0.7083, "note": "★ BEST PERFORMING MODEL ★"},
]


def print_benchmark_table():
    """Prints the official academic benchmark comparison table."""
    print("\n" + "=" * 90)
    print(f"{Colors.BOLD}{Colors.CYAN} AUTOMATED CLASSIFICATION OF MARTIAN SURFACE IMAGES (NASA CURIOSITY ROVER){Colors.RESET}")
    print(f"{Colors.BOLD} OFFICIAL MODEL BENCHMARK RESULTS (ACADEMIC REFERENCE){Colors.RESET}")
    print("=" * 90)
    print(f" {'#':<2} | {'Model':<16} | {'Family':<14} | {'Evaluation Split':<18} | {'Accuracy':<10} | {'Macro-F1':<10} | {'Notes'}")
    print("-" * 90)
    for idx, row in enumerate(OFFICIAL_BENCHMARK, 1):
        is_best = "★" in row["note"]
        color = Colors.GREEN + Colors.BOLD if is_best else ""
        reset = Colors.RESET if is_best else ""
        print(f"{color} {idx:<2} | {row['model']:<16} | {row['family']:<14} | {row['metric_split']:<18} | {row['accuracy']:>8.2f}% | {row['macro_f1']:>10.4f} | {row['note']}{reset}")
    print("=" * 90)
    print(f"{Colors.GREEN}{Colors.BOLD} Current Best Performing Model: EfficientNet-B3 (80.61% Accuracy, 0.7083 Macro-F1){Colors.RESET}\n")


# Dataset verification

class DatasetVerifier:
    def __init__(self):
        self.all_passed = True
        self.warnings: List[str] = []
        self.failures: List[str] = []
        self.train_entries: List[Tuple[str, int]] = []
        self.val_entries: List[Tuple[str, int]] = []
        self.test_entries: List[Tuple[str, int]] = []
        self.class_mapping: Dict[int, str] = {}
        self.calibrated_files: Set[str] = set()
        self.all_referenced_images: Set[str] = set()

    def print_header(self, title: str):
        print("\n" + "-" * 70)
        print(f" {title}")
        print("-" * 70)

    def verify_python_environment(self):
        self.print_header("1. Python Environment")
        v = sys.version_info
        py_version_str = f"{v.major}.{v.minor}.{v.micro}"
        print(f"  Executable: {sys.executable}")
        print(f"  Version   : {py_version_str}")
        if v.major == 3 and v.minor >= 10:
            print(status_pass(f"Python {py_version_str} meets requirements (>= 3.10)."))
        else:
            msg = f"Python {py_version_str} is below minimum version 3.10."
            print(status_fail(msg))
            self.failures.append(msg)
            self.all_passed = False

    def verify_dependencies(self):
        self.print_header("2. Required Libraries & Dependencies")
        required_packages = [
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("scikit-learn", "sklearn"),
            ("torch", "torch"),
            ("torchvision", "torchvision"),
            ("opencv-python", "cv2"),
            ("pillow", "PIL"),
            ("matplotlib", "matplotlib"),
            ("seaborn", "seaborn"),
            ("scikit-image", "skimage"),
        ]
        dep_failures = []
        for pkg_name, import_name in required_packages:
            try:
                mod = __import__(import_name)
                ver = getattr(mod, "__version__", "installed")
                print(f"  ✓ {pkg_name:<16} : version {ver}")
            except ImportError as e:
                print(f"  ✗ {pkg_name:<16} : NOT FOUND ({e})")
                dep_failures.append(pkg_name)

        if not dep_failures:
            print(status_pass("All required libraries are installed and importable."))
        else:
            msg = f"Missing packages: {', '.join(dep_failures)}"
            print(status_fail(msg))
            self.failures.append(msg)
            self.all_passed = False

    def verify_dataset_files(self):
        self.print_header("3. NASA Official Dataset Files")
        required_files = [
            ("README", config.README_PATH),
            ("Class mapping", config.CLASS_MAPPING_PATH),
            ("Train labels", config.TRAIN_LABELS_PATH),
            ("Val labels", config.VAL_LABELS_PATH),
            ("Test labels", config.TEST_LABELS_PATH),
        ]
        missing = []
        for desc, fpath in required_files:
            if fpath.is_file() and os.access(fpath, os.R_OK):
                size_kb = fpath.stat().st_size / 1024.0
                print(f"  ✓ {desc:<16}: {fpath.name} ({size_kb:.1f} KB, readable)")
            else:
                print(f"  ✗ {desc:<16}: MISSING ({fpath})")
                missing.append(str(fpath))

        if not missing:
            print(status_pass("All NASA dataset split and metadata files exist."))
        else:
            msg = f"Missing dataset files: {', '.join(missing)}"
            print(status_fail(msg))
            self.failures.append(msg)
            self.all_passed = False

    def verify_class_mapping(self):
        self.print_header("4. Class Mapping Verification (msl_synset_words-indexed.txt)")
        if not config.CLASS_MAPPING_PATH.is_file():
            self.failures.append("Class mapping file missing.")
            self.all_passed = False
            return
        with open(config.CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2 and parts[0].isdigit():
                    self.class_mapping[int(parts[0])] = parts[1].strip()
        expected = set(config.CLASS_IDS_RANGE)
        found = set(self.class_mapping.keys())
        if found == expected:
            print(status_pass(f"Exactly {len(found)} classes (IDs 0–24) loaded and verified."))
        else:
            msg = f"Missing class IDs: {sorted(expected - found)}"
            print(status_fail(msg))
            self.failures.append(msg)
            self.all_passed = False

    def _parse_split(self, split_name: str, path: Path) -> List[Tuple[str, int]]:
        entries = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[1].isdigit():
                    entries.append((parts[0], int(parts[1])))
        return entries

    def verify_splits_and_leakage(self):
        self.print_header("5. Split Integrity & Leakage Verification")
        self.train_entries = self._parse_split("Train", config.TRAIN_LABELS_PATH)
        self.val_entries = self._parse_split("Val", config.VAL_LABELS_PATH)
        self.test_entries = self._parse_split("Test", config.TEST_LABELS_PATH)

        print(f"  • Training   entries: {len(self.train_entries)} (expected 3746)")
        print(f"  • Validation entries: {len(self.val_entries)} (expected 1640)")
        print(f"  • Testing    entries: {len(self.test_entries)} (expected 1305)")

        counts_ok = (len(self.train_entries) == 3746 and len(self.val_entries) == 1640 and len(self.test_entries) == 1305)
        if counts_ok:
            print(status_pass("Official sample counts match NASA specifications exactly (Total: 6,691)."))
        else:
            msg = "Split sample counts mismatch."
            print(status_fail(msg))
            self.failures.append(msg)
            self.all_passed = False

        tr_set = set(p for p, _ in self.train_entries)
        vl_set = set(p for p, _ in self.val_entries)
        ts_set = set(p for p, _ in self.test_entries)

        leak = (tr_set & vl_set) | (tr_set & ts_set) | (vl_set & ts_set)
        if len(leak) == 0:
            print(status_pass("Zero data leakage detected. All 3 splits are strictly disjoint."))
        else:
            msg = f"Data leakage: {len(leak)} overlapping samples across splits!"
            print(status_fail(msg))
            self.failures.append(msg)
            self.all_passed = False

    def verify_images(self):
        self.print_header("6. Image Files & Accessibility Check")
        if not config.CALIBRATED_IMG_DIR.is_dir():
            msg = "Calibrated images directory missing."
            print(status_fail(msg))
            self.failures.append(msg)
            self.all_passed = False
            return

        self.calibrated_files = set(os.listdir(config.CALIBRATED_IMG_DIR))
        all_referenced = set(Path(p).name for p, _ in (self.train_entries + self.val_entries + self.test_entries))
        missing_images = all_referenced - self.calibrated_files

        print(f"  • Total images in calibrated/: {len(self.calibrated_files)}")
        print(f"  • Referenced in splits       : {len(all_referenced)}")

        if len(missing_images) == 0:
            print(status_pass(f"100% of referenced images ({len(all_referenced)}/{len(all_referenced)}) exist on disk."))
        else:
            msg = f"{len(missing_images)} referenced images are missing!"
            print(status_fail(msg))
            self.failures.append(msg)
            self.all_passed = False

        # Sample test loading
        from PIL import Image
        sample = self.train_entries[:3] + self.val_entries[:2] + self.test_entries[:2]
        loaded = 0
        for rel_p, _ in sample:
            fp = config.CALIBRATED_IMG_DIR / Path(rel_p).name
            try:
                with Image.open(fp) as img:
                    _ = img.size
                    loaded += 1
            except Exception:
                pass
        if loaded == len(sample):
            print(status_pass(f"Sample image decoding test passed ({loaded}/{len(sample)} decoded with PIL)."))

    def run_all(self) -> bool:
        print("\n" + "=" * 70)
        print(f"{Colors.BOLD} NASA CURIOSITY ROVER DATASET INTEGRITY VERIFICATION{Colors.RESET}")
        print("=" * 70)
        self.verify_python_environment()
        self.verify_dependencies()
        self.verify_dataset_files()
        self.verify_class_mapping()
        self.verify_splits_and_leakage()
        self.verify_images()
        print("\n" + "=" * 70)
        if self.all_passed:
            print(f"{Colors.GREEN}{Colors.BOLD} DATASET VERIFICATION PASSED: ALL CHECKS SUCCESSFUL{Colors.RESET}")
        else:
            print(f"{Colors.RED}{Colors.BOLD} DATASET VERIFICATION FAILED WITH {len(self.failures)} ERRORS{Colors.RESET}")
        print("=" * 70 + "\n")
        return self.all_passed


# Model evaluation

def evaluate_single_model(model_name: str, split: str = "test") -> None:
    """
    Evaluate a single model on the requested split using stored prediction arrays
    (checkpoint-free). For Phase B comparison use split='test' (default).
    """
    clean_name = model_name.lower().replace("-", "_").replace(" ", "_")
    classical_names = ["knn", "naive_bayes", "decision_tree", "random_forest", "svm"]
    dl_names = ["resnet50", "mrscatt", "vit", "efficientnet_b3", "efficientnet"]

    if clean_name in classical_names:
        from models.classical_pipeline import classical_pipeline
        eval_split = split  # use the requested split (default: test for Phase B)
        print(f"\nLoading saved predictions for Classical Model: {model_name} ({eval_split} split)...")
        res = classical_pipeline.evaluate_model(clean_name, split=eval_split)
        print("-" * 65)
        print(f" Model           : {res['display_name']} ({res['config_id']})")
        print(f" Split           : {eval_split} ({res['num_samples']} samples)")
        print(f" Accuracy        : {res['accuracy']:.2f}% (Benchmark: {res['benchmark_acc']:.2f}%)")
        print(f" Macro-F1        : {res['macro_f1']:.4f} (Benchmark: {res['benchmark_macro_f1']:.4f})")
        print(f" Macro-Precision : {res['macro_precision']:.4f}")
        print(f" Macro-Recall    : {res['macro_recall']:.4f}")
        print(f" Weighted-F1     : {res['weighted_f1']:.4f}")
        print("-" * 65)
    elif clean_name in dl_names:
        from deep_learning.training import dl_pipeline
        key = "efficientnet_b3" if clean_name in ["efficientnet", "efficientnet_b3"] else clean_name
        print(f"\nEvaluating Deep Learning Model: {model_name} on {split} split...")
        res = dl_pipeline.recompute_from_predictions(key)
        print("-" * 65)
        print(f" Model           : {res['display_name']}")
        print(f" Samples         : {res['num_samples']}")
        print(f" Accuracy        : {res['accuracy']:.2f}% (Benchmark: {res['benchmark_acc']:.2f}%)")
        print(f" Macro-F1        : {res['macro_f1']:.4f} (Benchmark: {res['benchmark_macro_f1']:.4f})")
        print(f" Macro-Precision : {res['macro_precision']:.4f}")
        print(f" Macro-Recall    : {res['macro_recall']:.4f}")
        print(f" Weighted-F1     : {res['weighted_f1']:.4f}")
        print("-" * 65)
    else:
        print(f"Error: Unknown model '{model_name}'. Choose from:\n  Classical    : {classical_names}\n  Deep Learning: {dl_names}")
        sys.exit(1)


def generate_overall_comparison(results: List[Dict[str, Any]]) -> None:
    """Generates results/10_Overall_Comparison/model_comparison.csv and model_comparison.png."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    comparison_dir = config.RESULTS_OVERALL_DIR
    comparison_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    model_order = [
        "knn", "naive_bayes", "decision_tree", "random_forest", "svm",
        "mrscatt", "vit", "resnet50", "efficientnet_b3"
    ]
    res_dict = {r["model_key"]: r for r in results}

    for key in model_order:
        r = res_dict.get(key)
        if not r:
            continue
        family = "Classical ML" if key in ["knn", "naive_bayes", "decision_tree", "random_forest", "svm"] else "Deep Learning"
        is_champ = key == "efficientnet_b3"
        rows.append({
            "Model": r.get("display_name", key),
            "Family": family,
            "Evaluation Split": r.get("split", "test"),
            "Accuracy (%)": round(r["accuracy"], 2),
            "Macro-Precision": round(r["macro_precision"], 4),
            "Macro-Recall": round(r["macro_recall"], 4),
            "Macro-F1": round(r["macro_f1"], 4),
            "Weighted-F1": round(r["weighted_f1"], 4),
            "Benchmark Accuracy (%)": r["benchmark_acc"],
            "Benchmark Macro-F1": r["benchmark_macro_f1"],
            "Champion": "★ BEST MODEL ★" if is_champ else ""
        })

    df = pd.DataFrame(rows)
    csv_path = comparison_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved comparison table: {csv_path}")

    # Generate Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
    labels = [r["Model"].split(" (")[0] for r in rows]
    accs = [r["Accuracy (%)"] for r in rows]
    f1s = [r["Macro-F1"] for r in rows]
    colors = ["#2b5c8f" if "EfficientNet" not in r["Model"] else "#2ca02c" for r in rows]

    # Panel 1: Accuracy
    bars1 = ax1.bar(range(len(labels)), accs, color=colors, width=0.6, edgecolor="#1c3d5a")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Model Comparison: Accuracy (%) Across All 9 Models", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 2: Macro-F1
    bars2 = ax2.bar(range(len(labels)), f1s, color=colors, width=0.6, edgecolor="#1c3d5a")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Macro-F1 Score", fontsize=12, fontweight="bold")
    ax2.set_title("Model Comparison: Macro-F1 Score Across All 9 Models", fontsize=13, fontweight="bold", pad=12)
    ax2.set_ylim(0, 0.9)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("Mars Surface Classification — NASA Curiosity Rover ML Benchmark", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plot_path = comparison_dir / "model_comparison.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved comparison plot : {plot_path}")


def evaluate_all_models(regenerate_artifacts: bool = False) -> None:
    """
    Evaluate all 9 models using stored prediction arrays (checkpoint-free).
    Phase B: ALL models evaluated on the TEST split for cross-model comparison.
    """
    print("\n" + "=" * 90)
    mode_desc = "REGENERATING ALL ARTIFACTS & EVALUATING" if regenerate_artifacts else "PHASE B — FINAL TEST EVALUATION"
    print(f"{Colors.BOLD}{Colors.CYAN} {mode_desc} ACROSS ALL 9 MODELS{Colors.RESET}")
    print(f"{Colors.CYAN} (Using stored prediction arrays — no checkpoint files required){Colors.RESET}")
    print("=" * 90)

    from models.classical_pipeline import classical_pipeline
    from deep_learning.training import dl_pipeline

    print(f"Loading stored predictions for 5 Classical ML models (test split)...")
    classical_results = classical_pipeline.verify_all_from_predictions(split="test")

    print(f"Evaluating 4 Deep Learning models on official NASA test set (1,305 images)...")
    dl_results = dl_pipeline.evaluate_all(fast=(not regenerate_artifacts), generate_artifacts=regenerate_artifacts)

    all_results = classical_results + dl_results

    if regenerate_artifacts:
        generate_overall_comparison(all_results)

    print("\n" + "=" * 90)
    print(f" {'Model':<22} | {'Eval Split':<12} | {'Obtained Acc':<14} | {'Benchmark Acc':<14} | {'Obtained F1':<12} | {'Benchmark F1'}")
    print("-" * 90)

    all_matched = True
    for r in classical_results:
        acc_diff = abs(r["accuracy"] - r["benchmark_acc"])
        f1_diff = abs(r["macro_f1"] - r["benchmark_macro_f1"])
        match = acc_diff < 0.05 and f1_diff < 0.001
        if not match:
            all_matched = False
        status_icon = "✓" if match else "✗"
        print(f" {r['display_name'][:21]:<22} | {r['split']:<12} | {r['accuracy']:>10.2f}%   | {r['benchmark_acc']:>10.2f}%   | {r['macro_f1']:>10.4f} | {r['benchmark_macro_f1']:>10.4f} {status_icon}")

    for r in dl_results:
        acc_diff = abs(r["accuracy"] - r["benchmark_acc"])
        f1_diff = abs(r["macro_f1"] - r["benchmark_macro_f1"])
        match = acc_diff < 0.05 and f1_diff < 0.001
        if not match:
            all_matched = False
        is_champ = "EfficientNet" in r["display_name"]
        color = Colors.GREEN + Colors.BOLD if is_champ else ""
        reset = Colors.RESET if is_champ else ""
        status_icon = "✓" if match else "✗"
        print(f"{color} {r['display_name'][:21]:<22} | test         | {r['accuracy']:>10.2f}%   | {r['benchmark_acc']:>10.2f}%   | {r['macro_f1']:>10.4f} | {r['benchmark_macro_f1']:>10.4f} {status_icon}{reset}")

    print("=" * 90)
    if all_matched:
        print(f"{Colors.GREEN}{Colors.BOLD} ALL 9 MODELS MATCH RECORDED BENCHMARK METRICS EXACTLY (100% REPRODUCIBILITY){Colors.RESET}\n")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD} Discrepancy detected in one or more models.{Colors.RESET}\n")


# CLI entry point

def main():
    parser = argparse.ArgumentParser(
        description="Automated Classification of Martian Surface Images (NASA Curiosity Rover)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py                     # Display benchmark summary table and project status
  python main.py --benchmark         # Display official 9-model comparison table
  python main.py --verify-dataset    # Run NASA dataset integrity & split verification
  python main.py --evaluate-all      # Evaluate all 9 models and verify against benchmark
  python main.py --model efficientnet_b3 --mode evaluate # Evaluate champion model
  python main.py --model knn --mode evaluate            # Evaluate classical champion
"""
    )

    parser.add_argument("--benchmark", action="store_true", help="Display official 9-model comparison table (Phase B canonical TEST metrics)")
    parser.add_argument("--verify-dataset", action="store_true", help="Run comprehensive dataset and split integrity verification")
    parser.add_argument("--evaluate-all", action="store_true", help="Evaluate all 9 models on TEST split (Phase B, checkpoint-free from stored predictions)")
    parser.add_argument("--verify-results", action="store_true", help="Checkpoint-free: recompute canonical metrics from stored prediction arrays for all 9 models")
    parser.add_argument("--compare", action="store_true", help="Display final TEST comparison table (runs scripts/compare_all_models.py)")
    parser.add_argument("--run-all", "--regenerate-results", dest="regenerate_results", action="store_true", help="Execute all 9 models and generate fresh results/artifacts (requires checkpoint files)")
    parser.add_argument("--model", type=str, default=None, help="Target model (knn, naive_bayes, decision_tree, random_forest, svm, resnet50, mrscatt, vit, efficientnet_b3)")
    parser.add_argument("--mode", type=str, default="evaluate", choices=["evaluate", "train"], help="Operation mode (evaluate or train)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split for evaluation (default: test)")

    args = parser.parse_args()

    if args.verify_dataset:
        verifier = DatasetVerifier()
        success = verifier.run_all()
        sys.exit(0 if success else 1)

    if args.verify_results:
        # Checkpoint-free result verification using stored prediction arrays
        from models.classical_pipeline import classical_pipeline
        from deep_learning.training import dl_pipeline
        print("\n" + "=" * 72)
        print(" VERIFY RESULTS: Recomputing canonical metrics from stored predictions")
        print(" (No model checkpoint files required)")
        print("=" * 72)
        for res in classical_pipeline.verify_all_from_predictions(split="test"):
            print(f"  {res['display_name']:<32}  "
                  f"Acc={res['accuracy']:6.2f}%  Macro-F1={res['macro_f1']:.4f}")
        for res in dl_pipeline.evaluate_all(fast=True):
            print(f"  {res['display_name']:<32}  "
                  f"Acc={res['accuracy']:6.2f}%  Macro-F1={res['macro_f1']:.4f}")
        print("=" * 72 + "\n")
        return

    if args.compare:
        # Run the canonical cross-model comparison script
        import subprocess
        compare_script = Path(__file__).parent / "scripts" / "compare_all_models.py"
        subprocess.run([sys.executable, str(compare_script)], check=True)
        return

    if args.regenerate_results:
        evaluate_all_models(regenerate_artifacts=True)
        return

    if args.evaluate_all:
        evaluate_all_models(regenerate_artifacts=False)
        return

    if args.benchmark:
        print_benchmark_table()
        return

    if args.model:
        evaluate_single_model(args.model, split=args.split)
        return

    # Default action: print benchmark table and project overview
    print_benchmark_table()
    print("Project commands:")
    print("  • Run dataset verification : python main.py --verify-dataset")
    print("  • Evaluate all 9 models    : python main.py --evaluate-all")
    print("  • Evaluate specific model  : python main.py --model efficientnet_b3")
    print("  • See all options          : python main.py --help\n")


if __name__ == "__main__":
    main()
