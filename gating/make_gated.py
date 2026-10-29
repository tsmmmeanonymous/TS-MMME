#!/usr/bin/env python3
"""Shared-encoder TS-MMME_TI (Table 4): image-gated combination of two TS-MMME runs that share the fine-tuned text encoder.

For each 2024 test article, the prediction of the text+image run (TSMMME_TI_ftclip, projected CLIP image feature) is
used when the images selected for the article by BLIP and by CLIP agree (ids listed in gate_ids.json); otherwise the
prediction of the text-only run (TSMMME_T_ft) is used.

Usage:  python gating/make_gated.py
Writes: results/runs/TSMMME_TI_gated/FULL_Q{1..4}_predictions.csv and prints the metrics averaged over the quarters.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"
GATE = json.loads((Path(__file__).parent / "gate_ids.json").read_text())


def load(run, quarter):
    with open(RUNS / run / f"FULL_Q{quarter}_predictions.csv", encoding="utf-8") as f:
        return {r["Article_ID"]: r for r in csv.DictReader(f)}


def main():
    out_dir = RUNS / "TSMMME_TI_gated"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    for k in range(1, 5):
        image_run, text_run = load("TSMMME_TI_ftclip", k), load("TSMMME_T_ft", k)
        gate = set(GATE[f"Q{k}"])
        rows = []
        for aid, t in text_run.items():
            use_image = aid in gate and aid in image_run
            r = image_run[aid] if use_image else t
            rows.append({"Article_ID": aid, "GroundTruth": r["GroundTruth"], "Prediction": r["Prediction"],
                         "Confidence": r["Confidence"], "Source": "TI" if use_image else "T"})
        with open(out_dir / f"FULL_Q{k}_predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        y = [r["GroundTruth"] == "rumor" for r in rows]
        h = [r["Prediction"] == "rumor" for r in rows]
        scores.append([f1_score(y, h, average="macro"), accuracy_score(y, h), f1_score(y, h), f1_score(y, h, pos_label=False)])
    m = np.mean(scores, axis=0)
    print(f"TS-MMME_TI (gated)  Macro-F1 {m[0]:.4f}  Acc {m[1]:.4f}  F1_fake {m[2]:.4f}  F1_real {m[3]:.4f}")


if __name__ == "__main__":
    main()
