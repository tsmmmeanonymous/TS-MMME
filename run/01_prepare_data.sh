#!/bin/zsh
# Build TS-MMME train / val / test files from the MMM-Trend release (same rows, labels and split as the baselines).
# Requires MMMTREND_DATA = folder with complete_dataset_v1.2.xlsx, article texts and images (see README).
set -e; cd "$(dirname "$0")/.."
python data_prep/build_full_master.py
