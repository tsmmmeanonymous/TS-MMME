#!/usr/bin/env python3
"""
==============================================================
MME-TSE OFFLINE PIPELINE
==============================================================

Offline execution from preprocessing to ZMM generation.

Stages
-------
A. Preprocessing
B. Graph Construction
C. Feature Generation
D. Heterogeneous Graph
E. Graph Transformer Training

==============================================================
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# ==============================================================
# Paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "Logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"offline_pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"

# ==============================================================
# Pipeline
# ==============================================================

PIPELINE = [

# ==============================================================
# STAGE A : PREPROCESSING
# ==============================================================

("P0  Generate Caption Cache",
 "Scripts/step00_generate_caption_cache.py"),

("P1  Generate Text Embeddings",
 "Scripts/step01_generate_text_embeddings.py"),

("P2  Generate Caption Embeddings",
 "Scripts/step02_generate_caption_embeddings.py"),

("Step14.1 Prepare Topic Embeddings",
 "Scripts/step14_1_prepare_topic_embeddings.py"),

("P4  Validate Cached Resources",
 "Scripts/step04_load_cached_resources.py"),

# ==============================================================
# STAGE B : GRAPH
# ==============================================================

("Step1  Create Graph Nodes",
 "Scripts/step1_create_graph_nodes.py"),

("Step2  Create Graph Edges",
 "Scripts/step2_create_graph_edges.py"),

("Step2.1 Validate Graph",
 "Scripts/step2.1_validate_graph.py"),

("Step3  Create Temporal Groups",
 "Scripts/step3_create_temporal_groups.py"),

("Step4  Create Memory Nodes",
 "Scripts/step4_create_memory_nodes.py"),

("Step5  Create Memory Edges",
 "Scripts/step5_create_memory_edges.py"),

("Step5.1 Validate Memory Graph",
 "Scripts/step5.1_validate_memory_graph.py"),

# ==============================================================
# STAGE C : FEATURES
# ==============================================================

("Step6  Prepare Article Text Features",
 "Scripts/step6_generate_article_text_features.py"),

("Step7  Generate Article Image Features",
 "Scripts/step7_generate_article_image_features.py"),

("Step8  Generate Memory Text Features",
 "Scripts/step8_generate_memory_text_features.py"),

("Step9  Generate Memory Image Features",
 "Scripts/step9_generate_memory_image_features.py"),

("Step10 Multimodal Fusion",
 "Scripts/Step10_Memory_multimodal_fusion.py"),

# ==============================================================
# STAGE D : HETEROGRAPH
# ==============================================================

("Step11 Create Temporal Memory Edges",
 "Scripts/step11_create_temporal_memory_edges.py"),

("Step12 Build HeteroData",
 "Scripts/step12_build_heterodata.py"),

("Step12.1 Validate HeteroData",
 "Scripts/step12.1_validate_heterodata.py"),

# Step12.2 (graph visualisation) is optional and slow on the full graph (~4 h); run it separately if needed.

# ==============================================================
# STAGE E : TRAINING
# ==============================================================

("Step13.1 Train Graph Transformer",
 "Scripts/step13.1_train_graph_transformer.py"),

("Step13.2 Generate ZMM",
 "Scripts/step13.2_generate_zmm.py")

]

# ==============================================================
# Logger
# ==============================================================

def log(text=""):
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# ==============================================================
# Runner
# ==============================================================

def run_script(title, script):

    log("")
    log("=" * 80)
    log(title)
    log("=" * 80)

    start = time.time()

    process = subprocess.Popen(
        [sys.executable, script],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    for line in process.stdout:
        log(line.rstrip())

    process.wait()

    elapsed = time.time() - start

    if process.returncode != 0:
        log("")
        log(f"FAILED : {title}")
        log(f"Elapsed : {elapsed:.2f} sec")
        sys.exit(process.returncode)

    log(f"Completed in {elapsed:.2f} sec")

# ==============================================================
# Main
# ==============================================================

def main():

    overall = time.time()

    log("=" * 80)
    log("MME-TSE OFFLINE PIPELINE")
    log("=" * 80)
    log(f"Started : {datetime.now()}")
    log("")

    for title, script in PIPELINE:
        run_script(title, script)

    total = time.time() - overall

    log("")
    log("=" * 80)
    log("OFFLINE PIPELINE COMPLETED SUCCESSFULLY")
    log("=" * 80)
    log(f"Finished : {datetime.now()}")
    log(f"Total Time : {total:.2f} sec")
    log(f"Log File : {LOG_FILE}")

if __name__ == "__main__":
    main()