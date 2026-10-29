#!/usr/bin/env python3
"""Prepare one public benchmark (Weibo, FakeNewsNet, PHEME, Fakeddit) for TS-MMME (Table 2).

Input: the dataset's train and test files in the TS-MMME JSON-lines format (one article per line with the fields
content, image_path, cluster_id, label, year, month, year_month, location, id; see ts_mmme/Scripts/converters/).
10% of the training articles are held out for validation (stratified, seed 2378, as for MMM-TREND), and a working copy
of the TS-MMME code is created under work/<dataset>/ with the train / val / test files in place.

Usage:  python other_datasets/prepare.py --name pheme --train path/train.json --test path/test.json [--images dir]
"""
import argparse
import json
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--images", help="folder with the article images (optional)")
    a = ap.parse_args()

    train, test = read(a.train), read(a.test)
    train, val = train_test_split(train, test_size=0.10, random_state=2378, stratify=[x["label"] for x in train])

    work = REPO / "other_datasets" / "work" / a.name
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(REPO / "ts_mmme" / "Scripts", work / "Scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "ts_mmme" / "Resources" / "Topic", work / "Resources" / "Topic")
    for d in ["Dataset/Experiment", "Resources/Cache", "Graph", "Results", "Logs"]:
        (work / d).mkdir(parents=True, exist_ok=True)
    write(train, work / "Dataset/Experiment/train_dataset.json")
    write(val, work / "Dataset/Experiment/val_dataset.json")
    write(test, work / "Dataset/Experiment/test_dataset.json")
    if a.images:
        (work / "Dataset/article_images").symlink_to(Path(a.images).resolve())
    print(f"{a.name}: train {len(train)}, val {len(val)}, test {len(test)} -> {work}")


if __name__ == "__main__":
    main()
