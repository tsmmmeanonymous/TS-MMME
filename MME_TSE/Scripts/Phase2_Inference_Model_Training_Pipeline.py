#!/usr/bin/env python3
"""
=====================================================================
MME-TSE PHASE 2
Inference Model Training Pipeline
=====================================================================

Purpose
-------
Train the inference modules using the graph-enhanced memory
embeddings (ZMM) generated in Phase 1.

Prerequisites
-------------
✓ Phase1_Offline_Graph_Construction_Pipeline.py completed successfully

Required Inputs
---------------
Graph/Experiment/Training/
    ├── Models/
    │      best_graph_transformer.pth
    │
    ├── ZMM/
    │      memory_zmm.pt
    │
    ├── Hetero/
    │      heterodata.pt
    │
    └── Dataset/
           train_dataset.json
           test_dataset.json

Outputs
-------
Graph/Experiment/InferenceModels/
    best_inference_model.pt
    last_inference_model.pt

Graph/Experiment/InferenceResults/
    training_history.json
    training_statistics.json

=====================================================================
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "Logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"phase2_{datetime.now():%Y%m%d_%H%M%S}.log"

# ==========================================================
# Pipeline
# ==========================================================

PIPELINE = [

    (
        "Step13_3 Train Inference Modules",
        "Scripts/step13_3_train_inference_modules.py"
    ),

]

# ==========================================================
# Logger
# ==========================================================

def log(message=""):

    print(message)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# ==========================================================
# Execute Script
# ==========================================================

def run_script(title, script):

    log()
    log("=" * 80)
    log(title)
    log("=" * 80)

    start = time.time()

    process = subprocess.Popen(
        [sys.executable, script],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    for line in process.stdout:
        log(line.rstrip())

    process.wait()

    elapsed = time.time() - start

    if process.returncode != 0:

        log()
        log(f"FAILED : {title}")
        log(f"Elapsed : {elapsed:.2f} sec")

        sys.exit(process.returncode)

    log(f"Completed in {elapsed:.2f} sec")

# ==========================================================
# Main
# ==========================================================

def main():

    overall_start = time.time()

    log("=" * 80)
    log("MME-TSE PHASE 2")
    log("Inference Model Training Pipeline")
    log("=" * 80)

    log(f"Started : {datetime.now()}")

    for title, script in PIPELINE:

        run_script(title, script)

    total = time.time() - overall_start

    log()
    log("=" * 80)
    log("PHASE 2 COMPLETED SUCCESSFULLY")
    log("=" * 80)

    log(f"Finished : {datetime.now()}")
    log(f"Total Time : {total:.2f} sec")
    log(f"Log File : {LOG_FILE}")

if __name__ == "__main__":
    main()