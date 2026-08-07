#!/usr/bin/env python3
"""
===========================================================
MME-TSE
Step 2 : Create Train/Test Split
===========================================================

Input
-----
Dataset/Experiment/topic_dataset.json

Output
------
train_dataset.json
test_dataset.json
split_statistics.json

===========================================================
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# PATHS
# ==========================================================

CURRENT_DIR = Path(__file__).resolve().parent

BASELINE_DIR = CURRENT_DIR.parents[2]

DATASET_DIR = (
    BASELINE_DIR
    / "Dataset"
    / "Experiment"
)

INPUT_FILE = DATASET_DIR / "topic_dataset.json"

TRAIN_FILE = DATASET_DIR / "train_dataset.json"

TEST_FILE = DATASET_DIR / "test_dataset.json"

STAT_FILE = DATASET_DIR / "split_statistics.json"

TEST_SIZE = 0.20

RANDOM_STATE = 42



# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if line:

            records.append(json.loads(line))

df = pd.DataFrame(records)

print(f"Loaded Events : {len(df)}")




# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

train_df, test_df = train_test_split(

    df,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=df["label"],

    shuffle=True

)

print(f"Training Events : {len(train_df)}")
print(f"Testing Events  : {len(test_df)}")



# ==========================================================
# SAVE JSONL
# ==========================================================

def save_jsonl(df, output_file):

    with open(output_file, "w", encoding="utf-8") as f:

        for _, row in df.iterrows():

            f.write(
                json.dumps(
                    row.to_dict(),
                    ensure_ascii=False
                ) + "\n"
            )

save_jsonl(train_df, TRAIN_FILE)

save_jsonl(test_df, TEST_FILE)




# ==========================================================
# SAVE JSONL
# ==========================================================

def save_jsonl(df, output_file):

    with open(output_file, "w", encoding="utf-8") as f:

        for _, row in df.iterrows():

            f.write(
                json.dumps(
                    row.to_dict(),
                    ensure_ascii=False
                ) + "\n"
            )

save_jsonl(train_df, TRAIN_FILE)

save_jsonl(test_df, TEST_FILE)



# ==========================================================
# STATISTICS
# ==========================================================

statistics = {

    "Total Events": len(df),

    "Training Events": len(train_df),

    "Testing Events": len(test_df),

    "Training Rumor": int((train_df["label"] == "rumor").sum()),

    "Training Truth": int((train_df["label"] == "truth").sum()),

    "Testing Rumor": int((test_df["label"] == "rumor").sum()),

    "Testing Truth": int((test_df["label"] == "truth").sum())

}

with open(STAT_FILE, "w", encoding="utf-8") as f:

    json.dump(
        statistics,
        f,
        indent=4
    )

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT COMPLETED")
print("=" * 60)

print(f"Training : {len(train_df)}")
print(f"Testing  : {len(test_df)}")

print("\nSaved Files")

print(TRAIN_FILE)

print(TEST_FILE)

print(STAT_FILE)
