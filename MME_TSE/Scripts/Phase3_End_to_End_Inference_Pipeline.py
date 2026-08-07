#!/usr/bin/env python3
"""
====================================================================
MME-TSE
PHASE 3 : END-TO-END INFERENCE PIPELINE
====================================================================

Purpose
-------
Run complete MME-TSE inference using pretrained models.

No training is performed.

Inputs
------
Article Text
Image (Optional)
Month
Location

Loads
------
✓ Cached Resources
✓ Graph Transformer
✓ ZMM
✓ Cross Attention
✓ Veracity Classifier

Outputs
-------
Prediction
Confidence
Retrieved Memories
Explanation

====================================================================
"""

import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import json
import csv

from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from Phase2.step14_7_temporal_spatial_veracity_evolution import (
                export_evolution_analysis
            )

# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "Logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"phase3_{datetime.now():%Y%m%d_%H%M%S}.log"

DATASET_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "Experiment"
    / "test_dataset.json"
)

# ============================================================
# Ablation Experiment
# ============================================================

from Phase2.step15_mme_tse_pipeline import (
    MMETSEPipeline,
    ABLATION_MODE
)

EXPERIMENT_NAME = ABLATION_MODE

RESULT_DIR = PROJECT_ROOT / "Results"
RESULT_DIR.mkdir(exist_ok=True)

PREDICTION_FILE = RESULT_DIR / f"{EXPERIMENT_NAME}_predictions.csv"
METRIC_FILE = RESULT_DIR / f"{EXPERIMENT_NAME}_metrics.json"

def log(message=""):
    message = str(message)

    print(message)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# ============================================================
# Import Complete Inference Pipeline
# ============================================================

from Phase2.step15_mme_tse_pipeline import MMETSEPipeline

# ============================================================
# Logger
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("MME-TSE")

# ============================================================
# Phase 3
# ============================================================

class MMETSEInferencePipeline:

    def __init__(self):

        log("=" * 70)
        log("MME-TSE PHASE 3")
        log("END-TO-END INFERENCE")
        log(f"Experiment : {EXPERIMENT_NAME}")
        log("=" * 70)

        self.pipeline = MMETSEPipeline()

        log("Inference Pipeline Initialized.")

    # =========================================================

    def evaluate_test_dataset(self):

        log("")
        log("=" * 70)
        log("TEST DATASET EVALUATION")
        log("=" * 70)

        dataset = []

        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                dataset.append(json.loads(line))

        y_true = []
        y_pred = []
        rows = []

        LABEL_MAP = {
            "real": "truth",
            "true": "truth",
            "truth": "truth",

            "fake": "rumor",
            "false": "rumor",
            "rumor": "rumor",
        }

        total = len(dataset)

        for idx, sample in enumerate(dataset, start=1):

            start = time.time()

            image_path = sample.get("image_path", "")

            try:
                result = self.predict(
                    article_text=sample["content"],
                    image_path=image_path,
                    month=sample["year_month"],
                    location=sample["location"],
                    verbose=False
                )

            except Exception:

                import traceback

                print("\n" + "=" * 70)
                print(f"FAILED ARTICLE : {sample['id']}")
                print("=" * 70)

                traceback.print_exc()

                raise

            elapsed = time.time() - start

            if not result["success"]:
                log(f"[{idx}/{total}] ID={sample['id']} Prediction Failed")
                continue

            gt = LABEL_MAP.get(
                sample["label"].strip().lower(),
                sample["label"].strip().lower()
            )

            pred = LABEL_MAP.get(
                result["veracity_result"]["prediction"].strip().lower(),
                result["veracity_result"]["prediction"].strip().lower()
            )

            confidence = result["veracity_result"]["confidence"]

            topic = result["topic_result"]["predicted_topic"]

            topic_name = result["topic_result"]["predicted_topic_name"]

            y_true.append(gt)
            y_pred.append(pred)


            # ==========================================================
            # Evolution Information
            # ==========================================================

            if result["evolution_result"] is not None:

                evolution_score = result["evolution_result"].evolution_score

                evolution_confidence = (
                    result["evolution_result"].evolution_confidence
                )

                dominant_state = (
                    result["evolution_result"].dominant_state
                )

                evolution_direction = (
                    result["evolution_result"]
                    .statistics["evolution_direction"]
                )

                historical_stability = (
                    result["evolution_result"]
                    .statistics["historical_stability"]
                )

                export_evolution_analysis(

                    result["evolution_result"],

                    RESULT_DIR
                    / EXPERIMENT_NAME
                    / "Evolution"
                    / f"Article_{sample['id']}"

                )

            else:

                evolution_score = ""

                evolution_confidence = ""

                dominant_state = ""

                evolution_direction = ""

                historical_stability = ""

            rows.append([

                sample["id"],

                gt,

                pred,

                confidence,

                topic,

                topic_name,

                sample["year_month"],

                sample["location"],

                evolution_score,

                evolution_confidence,

                dominant_state,

                evolution_direction,

                historical_stability,

                elapsed

            ])

            correct = "✓" if gt == pred else "✗"

            log(
                f"[{idx}/{total}] "
                f"ID={sample['id']} "
                f"GT={gt} "
                f"Pred={pred} "
                f"{correct}"
            )

        self.save_predictions(rows)

        print("\nUnique Ground Truth Labels:")
        print(sorted(set(y_true)))

        print("\nUnique Predicted Labels:")
        print(sorted(set(y_pred)))

        self.save_metrics(y_true, y_pred)

    def save_predictions(self, rows):

        with open(
            PREDICTION_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([

                "Article_ID",

                "GroundTruth",

                "Prediction",

                "Confidence",

                "Topic_ID",

                "Topic_Name",

                "Month",

                "Location",

                "Evolution_Score",

                "Evolution_Confidence",

                "Dominant_State",

                "Evolution_Direction",

                "Historical_Stability",

                "Execution_Time"

            ])

            writer.writerows(rows)

        log("")
        log(f"Predictions saved : {PREDICTION_FILE}")

    def save_metrics(
        self,
        y_true,
        y_pred
    ):

        print("\n" + "=" * 70)
        print("LABEL CHECK")
        print("=" * 70)

        print("Unique Ground Truth Labels:")
        print(sorted(set(y_true)))

        print("\nUnique Predicted Labels:")
        print(sorted(set(y_pred)))

        print("\nGround Truth Counts:")
        print(Counter(y_true))

        print("\nPrediction Counts:")
        print(Counter(y_pred))

        print("=" * 70)

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            average="macro"
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="macro"
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="macro"
        )

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        report = classification_report(
            y_true,
            y_pred,
            digits=4
        )

        metrics = {

            "experiment": EXPERIMENT_NAME,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1_score": f1,

            "confusion_matrix": cm.tolist(),

            "classification_report": report

        }

        with open(
            METRIC_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4
            )

        log("")
        log("=" * 70)
        log("FINAL RESULTS")
        log("=" * 70)

        log(f"Accuracy  : {accuracy:.4f}")
        log(f"Precision : {precision:.4f}")
        log(f"Recall    : {recall:.4f}")
        log(f"F1 Score  : {f1:.4f}")

        log("")
        log("Confusion Matrix")
        log(str(cm))

        log("")
        log(report)

        log("")
        log(f"Metrics saved : {METRIC_FILE}")

    def predict(
        self,
        article_text,
        month,
        location,
        image_path=None,
        verbose=True
    ):

        start = time.time()

        result = self.pipeline.predict(

            article_text=article_text,
            image_path=image_path,
            month=month,
            location=location

        )

        elapsed = time.time() - start
        if verbose:
            log("")
            log("=" * 70)
            log("PREDICTION COMPLETED")
            log("=" * 70)
            log(f"Execution Time : {elapsed:.2f} sec")

        if verbose:
            self.pipeline.display_result(
                result,
                log_fn=log
            )

        return result

# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MME-TSE PHASE 3")
    print("END-TO-END INFERENCE")
    print("=" * 70)

    print("\nChoose Mode")
    print("1. Manual Prediction")
    print("2. Evaluate Test Dataset")
    print("3. Exit")

    choice = input("\nEnter Choice : ").strip()

    pipeline = MMETSEInferencePipeline()

    # =========================================================
    # Manual Prediction
    # =========================================================
    if choice == "1":

        article = input("\nEnter Article Text :\n")

        image = input("\nEnter Image Path (Optional) : ").strip()

        if image == "":
            image = None

        month = input("\nEnter Month (YYYY-MM) : ").strip()

        location = input("\nEnter Location : ").strip()

        pipeline.predict(
            article_text=article,
            image_path=image,
            month=month,
            location=location,
            verbose=True
        )

    # =========================================================
    # Test Dataset Evaluation
    # =========================================================
    elif choice == "2":

        pipeline.evaluate_test_dataset()

    # =========================================================
    # Exit
    # =========================================================
    elif choice == "3":

        print("\nExiting Phase 3.")
        return

    # =========================================================
    # Invalid Choice
    # =========================================================
    else:

        print("\nInvalid choice.")
        return

if __name__ == "__main__":
    main()