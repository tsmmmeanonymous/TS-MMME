#!/usr/bin/env python3
"""Score the LLM baselines exactly like the TS-MMME results: metrics per 2024 quarter, averaged over the four quarters.
Unparseable answers count as 'real' and are reported. Writes results/llm/scores.json.

Usage:  python llm_baselines/score.py
"""
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import common


def main():
    scores = {}
    for path in sorted(common.OUT_DIR.glob("*.jsonl")):
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        per_q = []
        for k in range(1, 5):
            sub = [r for r in rows if r["quarter"] == k]
            if not sub:
                continue
            y = [r["gold"] == "fake" for r in sub]
            h = [(r["pred"] or "real") == "fake" for r in sub]
            per_q.append([f1_score(y, h, average="macro"), accuracy_score(y, h), f1_score(y, h), f1_score(y, h, pos_label=False)])
        m = np.mean(per_q, axis=0)
        lat = [r["latency_s"] for r in rows if "latency_s" in r]
        name = path.stem
        scores[name] = {"n": len(rows), "unparsed": sum(r["pred"] is None for r in rows), "macro_f1": m[0], "acc": m[1],
                        "f1_fake": m[2], "f1_real": m[3]}
        if lat:
            scores[name]["latency_median_s"] = float(np.median(lat))
        print(f"{name:32s} n={len(rows):4d}  Macro-F1 {m[0]:.4f}  Acc {m[1]:.4f}  F1_fake {m[2]:.4f}  F1_real {m[3]:.4f}"
              + (f"  latency {np.median(lat):.2f}s" if lat else ""))
    (common.OUT_DIR / "scores.json").write_text(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
