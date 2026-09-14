#!/usr/bin/env python3
"""Batch execution script to evaluate all 9 models and regenerate results artifacts."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import evaluate_all_models


def main():
    print("=" * 80)
    print(" Running Full Project Pipeline: All 9 Models & Fresh Results Generation")
    print("=" * 80)
    evaluate_all_models(regenerate_artifacts=True)


if __name__ == "__main__":
    main()
