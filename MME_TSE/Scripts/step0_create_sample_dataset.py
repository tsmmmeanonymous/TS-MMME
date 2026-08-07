import os
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config


# =====================================================
# Configuration
# =====================================================

# Dataset Mode
# --------------------------
# FULL   -> Uses the complete dataset
# SAMPLE -> Creates a representative sample first
# --------------------------

DATASET_MODE = "FULL"          # FULL / SAMPLE

# Used only if DATASET_MODE == "SAMPLE"
SAMPLE_SIZE = 300

# Train/Test Split
TRAIN_RATIO = config.TRAIN_RATIO

# Random seed
RANDOM_SEED = config.RANDOM_SEED

random.seed(RANDOM_SEED)

# =====================================================
# Paths
# =====================================================

INPUT_FILE = config.MASTER_FILE

OUTPUT_DIR = config.EXPERIMENT_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_JSON = OUTPUT_DIR / "train_dataset.json"
TEST_JSON = OUTPUT_DIR / "test_dataset.json"

TRAIN_CSV = OUTPUT_DIR / "train_summary.csv"
TEST_CSV = OUTPUT_DIR / "test_summary.csv"

STATISTICS_JSON = OUTPUT_DIR / "dataset_statistics.json"

# =====================================================
# Helper Functions
# =====================================================


def load_dataset(path):
    """
    Load JSONL dataset.
    """

    print("\nLoading dataset...")

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    print(f"Loaded {len(df):,} articles.")

    return df


def representative_sampling(df, sample_size):
    """
    Topic-balanced sampling while maximizing
    (year, location) diversity.
    """

    print("\nCreating representative sample...")

    topics = sorted(df["cluster_id"].unique())

    num_topics = len(topics)

    base = sample_size // num_topics

    extra = sample_size % num_topics

    topic_quota = {}

    for i, topic in enumerate(topics):

        topic_quota[topic] = base + (1 if i < extra else 0)

    sampled_parts = []

    for topic in topics:

        topic_df = df[df["cluster_id"] == topic].copy()

        quota = topic_quota[topic]

        shuffled = topic_df.sample(
            frac=1,
            random_state=RANDOM_SEED
        )

        selected = []

        used_pairs = set()

        for idx, row in shuffled.iterrows():

            pair = (
                row["year"],
                row["location"]
            )

            if pair not in used_pairs:

                selected.append(idx)

                used_pairs.add(pair)

            if len(selected) >= quota:
                break

        if len(selected) < quota:

            remaining = topic_df.drop(index=selected)

            need = quota - len(selected)

            if len(remaining) > 0:

                extra_rows = remaining.sample(
                    min(need, len(remaining)),
                    random_state=RANDOM_SEED
                )

                selected.extend(extra_rows.index.tolist())

        sampled_parts.append(
            topic_df.loc[selected]
        )

    sample_df = pd.concat(
        sampled_parts,
        ignore_index=True
    )

    sample_df = sample_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(f"Representative sample created: {len(sample_df)} articles")

    return sample_df


def prepare_dataset():

    df = load_dataset(INPUT_FILE)

    if DATASET_MODE.upper() == "FULL":

        print("\nUsing FULL dataset.")

        working_df = df.copy()

    elif DATASET_MODE.upper() == "SAMPLE":

        print(f"\nUsing SAMPLE dataset ({SAMPLE_SIZE} articles).")

        working_df = representative_sampling(
            df,
            SAMPLE_SIZE
        )

    else:

        raise ValueError(
            "DATASET_MODE must be FULL or SAMPLE."
        )

    return working_df

# =====================================================
# Save Utilities
# =====================================================

def save_jsonl(df, output_file):
    """
    Save dataframe as JSONL.
    """

    with open(output_file, "w", encoding="utf-8") as f:

        for _, row in df.iterrows():

            f.write(
                json.dumps(
                    row.to_dict(),
                    ensure_ascii=False
                )
                + "\n"
            )


def create_summary(df):

    return pd.DataFrame(
        [
            {
                "Metric": "Articles",
                "Value": len(df)
            },
            {
                "Metric": "Truth",
                "Value": int((df["label"] == "truth").sum())
            },
            {
                "Metric": "Rumor",
                "Value": int((df["label"] == "rumor").sum())
            },
            {
                "Metric": "Topics",
                "Value": int(df["cluster_id"].nunique())
            },
            {
                "Metric": "Years",
                "Value": int(df["year"].nunique())
            },
            {
                "Metric": "Locations",
                "Value": int(df["location"].nunique())
            }
        ]
    )


def print_statistics(title, df):

    print("\n--------------------------------------------")
    print(title)
    print("--------------------------------------------")

    print(f"Articles   : {len(df)}")
    print(f"Truth      : {(df['label']=='truth').sum()}")
    print(f"Rumor      : {(df['label']=='rumor').sum()}")
    print(f"Topics     : {df['cluster_id'].nunique()}")
    print(f"Years      : {df['year'].nunique()}")
    print(f"Locations  : {df['location'].nunique()}")

    print("\nTopic Distribution")

    print(
        df["cluster_id"]
        .value_counts()
        .sort_index()
    )


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MME-TSE DATASET PREPARATION")
    print("=" * 60)

    dataset = prepare_dataset()

    # --------------------------------------------
    # Stratified Split
    # --------------------------------------------

    dataset = dataset.copy()

    dataset["strata"] = (
        dataset["cluster_id"].astype(str)
        + "_"
        + dataset["label"].astype(str)
    )

    # Stratify over topic and label together where possible. A topic-label
    # combination holding a single record cannot be split, which scikit-learn
    # rejects outright, so fall back to stratifying on the label alone.
    smallest = dataset["strata"].value_counts().min()

    if smallest < 2:
        print(
            f"\nSmallest topic-label group has {smallest} record(s). "
            "Stratifying on label only."
        )
        strata = dataset["label"]
    else:
        strata = dataset["strata"]

    train_df, test_df = train_test_split(
        dataset,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=strata
    )

    train_df = train_df.drop(columns=["strata"])
    test_df = test_df.drop(columns=["strata"])

    train_df = train_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    test_df = test_df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    # --------------------------------------------
    # Save JSON
    # --------------------------------------------

    save_jsonl(train_df, TRAIN_JSON)
    save_jsonl(test_df, TEST_JSON)

    # --------------------------------------------
    # Save CSV summaries
    # --------------------------------------------

    create_summary(train_df).to_csv(
        TRAIN_CSV,
        index=False
    )

    create_summary(test_df).to_csv(
        TEST_CSV,
        index=False
    )

    # --------------------------------------------
    # Statistics JSON
    # --------------------------------------------

    statistics = {

        "dataset_mode": DATASET_MODE,

        "sample_size": (
            SAMPLE_SIZE
            if DATASET_MODE.upper() == "SAMPLE"
            else len(dataset)
        ),

        "train_ratio": TRAIN_RATIO,

        "test_ratio": round(
            1 - TRAIN_RATIO,
            2
        ),

        "total_articles": int(len(dataset)),

        "train_articles": int(len(train_df)),

        "test_articles": int(len(test_df)),

        "train_truth": int(
            (train_df["label"] == "truth").sum()
        ),

        "train_rumor": int(
            (train_df["label"] == "rumor").sum()
        ),

        "test_truth": int(
            (test_df["label"] == "truth").sum()
        ),

        "test_rumor": int(
            (test_df["label"] == "rumor").sum()
        ),

        "topics": int(
            dataset["cluster_id"].nunique()
        ),

        "years": int(
            dataset["year"].nunique()
        ),

        "locations": int(
            dataset["location"].nunique()
        )
    }

    with open(
        STATISTICS_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            statistics,
            f,
            indent=4
        )

    # --------------------------------------------
    # Display Report
    # --------------------------------------------

    print("\n")
    print("=" * 60)
    print("DATASET CREATED SUCCESSFULLY")
    print("=" * 60)

    print_statistics("TRAIN DATASET", train_df)

    print_statistics("TEST DATASET", test_df)

    print("\nSaved Files")
    print("--------------------------------------------")

    print(TRAIN_JSON)
    print(TEST_JSON)
    print(TRAIN_CSV)
    print(TEST_CSV)
    print(STATISTICS_JSON)

    print("\nDone.")