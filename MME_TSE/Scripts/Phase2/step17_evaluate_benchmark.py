import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from Phase2.step15_mme_tse_pipeline import MMETSEPipeline

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "Dataset"
    / "Experiment"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "Results"
    / "Experiment"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_DATASET = DATASET_DIR / "test_dataset.json"

PREDICTION_FILE = RESULT_DIR / "predictions.csv"

METRIC_FILE = RESULT_DIR / "evaluation_metrics.json"

CLASSIFICATION_REPORT_FILE = (
    RESULT_DIR
    / "classification_report.txt"
)

MISCLASSIFIED_FILE = (
    RESULT_DIR
    / "misclassified_samples.csv"
)

CONFUSION_MATRIX_FIGURE = (
    RESULT_DIR
    / "confusion_matrix.png"
)

ROC_CURVE_FIGURE = (
    RESULT_DIR
    / "roc_curve.png"
)

PR_CURVE_FIGURE = (
    RESULT_DIR
    / "precision_recall_curve.png"
)
# ==========================================================
# Load Test Dataset
# ==========================================================

print("\n==============================================")
print("MME-TSE EVALUATION")
print("==============================================")

print("\nLoading Test Dataset...")

with open(TEST_DATASET, "r", encoding="utf-8") as f:
    content = f.read().strip()

if content.startswith("["):
    test_dataset = json.loads(content)
else:
    test_dataset = [
        json.loads(line)
        for line in content.splitlines()
        if line.strip()
    ]
print(f"Test Samples : {len(test_dataset)}")

# ==========================================================
# Initialize Pipeline
# ==========================================================

print("\nInitializing MME-TSE Pipeline...")

pipeline = MMETSEPipeline()

# ==========================================================
# Prediction Storage
# ==========================================================

y_true = []
y_pred = []
y_score = []

prediction_rows = []
misclassified_rows = []
failed_rows = []
# ==========================================================
# Run Inference
# ==========================================================

print("\n==============================================")
print("RUNNING TEST SET INFERENCE")
print("==============================================")


for index, sample in enumerate(
    tqdm(
        test_dataset,
        total=len(test_dataset),
        desc="Evaluating",
        unit="article",
        dynamic_ncols=True
    ),
    start=1
):


    article_text = sample["content"]

    image_path = sample.get("image_path", None)

    month = sample["month"]

    location = sample["location"]

    label = sample["label"].strip().lower()

    LABEL_MAP = {
        "truth":0,
        "real":0,
        "rumor":1,
        "fake":1,
        "false":1
    }

    ground_truth = LABEL_MAP[
        sample["label"].strip().lower()
    ]

    try:

        result = pipeline.predict(
            article_text=article_text,
            image_path=image_path,
            month=month,
            location=location
        )

        if not result["success"]:
            continue

        veracity = result["veracity_result"]

        prediction = (
                    veracity["prediction"]
                    .strip()
                    .lower()
                )
        
        LABEL_MAP = {
            "truth":0,
            "real":0,
            "rumor":1,
            "fake":1,
            "false":1
        }
        
        predicted_label = LABEL_MAP[prediction]

        confidence = float(veracity["confidence"])

        probabilities = veracity["probabilities"]

        real_probability = float(probabilities[0])

        rumor_probability = float(probabilities[1])

    except Exception as e:

        print(f"\nSample {index} failed : {e}")

        failed_rows.append({

            "article_id": sample.get("id", index),

            "error": str(e),

            "image_path": image_path

        })

        continue
    

        

    # ----------------------------------------------
    # Store metrics
    # ----------------------------------------------

    y_true.append(ground_truth)

    y_pred.append(predicted_label)

    y_score.append(rumor_probability)

    prediction_rows.append({

        "article_id": sample.get("id", index),

        "ground_truth": ground_truth,

        "prediction": predicted_label,

        "confidence": confidence,

        "real_probability": real_probability,

        "rumor_probability": rumor_probability

    })

    # ----------------------------------------------
    # Misclassified samples
    # ----------------------------------------------

    if predicted_label != ground_truth:

        misclassified_rows.append({

            "article_id": sample.get("id", index),

            "ground_truth": ground_truth,

            "prediction": predicted_label,

            "confidence": confidence,

            "month": month,

            "location": location

        })

print("\nInference Completed.")
print(f"Successful Predictions : {len(y_true)}")
print(f"Skipped Samples        : {len(test_dataset) - len(y_true)}")

# ==========================================================
# Evaluation Metrics
# ==========================================================

print("\n==============================================")
print("COMPUTING EVALUATION METRICS")
print("==============================================")

if len(y_true) == 0:
    raise RuntimeError(
        "No successful predictions were generated. "
        "Evaluation cannot continue."
    )

# ----------------------------------------------------------
# Basic Metrics
# ----------------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

# Binary metrics (Rumor as positive class)
precision = precision_score(
    y_true,
    y_pred,
    average="binary",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="binary",
    zero_division=0
)

# Macro metrics
macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

# Weighted metrics
weighted_precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred
)

mcc = matthews_corrcoef(
    y_true,
    y_pred
)

kappa = cohen_kappa_score(
    y_true,
    y_pred
)

# ----------------------------------------------------------
# Confusion Matrix
# ----------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

# ----------------------------------------------------------
# ROC-AUC
# ----------------------------------------------------------

fpr, tpr, roc_thresholds = roc_curve(
    y_true,
    y_score
)

roc_auc = roc_auc_score(
    y_true,
    y_score
)

# ----------------------------------------------------------
# Precision-Recall Curve
# ----------------------------------------------------------

precision_curve, recall_curve, pr_thresholds = (
    precision_recall_curve(
        y_true,
        y_score
    )
)

average_precision = average_precision_score(
    y_true,
    y_score
)

# ----------------------------------------------------------
# Classification Report
# ----------------------------------------------------------

report_text = classification_report(
    y_true,
    y_pred,
    target_names=["Truth","Rumor"],
    digits=4,
    zero_division=0
)

metrics = {

    "Split": "Testing",

    "Total Test Samples": len(test_dataset),

    "Successful Predictions": len(y_true),

    "Skipped Samples": len(test_dataset) - len(y_true),

    "Accuracy": float(accuracy),

    "Binary Precision": float(precision),
    "Binary Recall": float(recall),
    "Binary F1": float(f1),

    "Macro Precision": float(macro_precision),
    "Macro Recall": float(macro_recall),
    "Macro F1": float(macro_f1),

    "Weighted Precision": float(weighted_precision),
    "Weighted Recall": float(weighted_recall),
    "Weighted F1": float(weighted_f1),

    "Balanced Accuracy": float(balanced_accuracy),

    "Matthews Correlation Coefficient": float(mcc),

    "Cohen Kappa": float(kappa),

    "ROC AUC": float(roc_auc),

    "Average Precision": float(average_precision)
}

# ==========================================================
# Save Results
# ==========================================================

# Save predictions
pd.DataFrame(prediction_rows).to_csv(
    PREDICTION_FILE,
    index=False
)

# Save misclassified samples
pd.DataFrame(misclassified_rows).to_csv(
    MISCLASSIFIED_FILE,
    index=False
)

# Save failed samples
pd.DataFrame(failed_rows).to_csv(
    RESULT_DIR / "failed_samples.csv",
    index=False
)

# Save classification report
with open(CLASSIFICATION_REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(report_text)

# Save evaluation metrics
with open(METRIC_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print(f"\nResults saved to: {RESULT_DIR}")

print("\nEvaluation Completed.")

print("\n==============================================")
print("FINAL RESULTS")
print("==============================================")

print(f"Accuracy            : {accuracy:.4f}")

print(f"Macro Precision     : {macro_precision:.4f}")
print(f"Macro Recall        : {macro_recall:.4f}")
print(f"Macro F1            : {macro_f1:.4f}")

print(f"Weighted Precision  : {weighted_precision:.4f}")
print(f"Weighted Recall     : {weighted_recall:.4f}")
print(f"Weighted F1         : {weighted_f1:.4f}")

print(f"Balanced Accuracy   : {balanced_accuracy:.4f}")
print(f"MCC                 : {mcc:.4f}")
print(f"Cohen Kappa         : {kappa:.4f}")
print(f"ROC-AUC             : {roc_auc:.4f}")
print(f"Average Precision   : {average_precision:.4f}")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(report_text)