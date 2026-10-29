#!/usr/bin/env python3


import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm



MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHT = 1

MAX_SAMPLES = 10000

RANDOM_SEED = 42

LABEL_MAP = {
    1: "truth",
    0: "rumor"
}




CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = CURRENT_DIR.parents[2]

BASELINE_ROOT = PROJECT_ROOT / "Baseline"

DATASET_DIR = (
    BASELINE_ROOT
    / "Dataset"
    / "Fakeddit"
)

TOPIC_FILE = (
    BASELINE_ROOT
    / "dataset_converters"
    / "reference"
    / "topic_keywords.json"
)

OUTPUT_FILE = (
    BASELINE_ROOT
    / "MME_TSE"
    / "Dataset"
    / "fakeddit_master.json"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)



def load_dataset():

    print("\nLoading Fakeddit dataset...")

    files = [
        "all_train.tsv",
        "all_validate.tsv",
        "all_test_public.tsv"
    ]

    frames = []

    for file in files:

        path = DATASET_DIR / file

        print(f"Reading {file}")

        frames.append(
            pd.read_csv(path, sep="\t")
        )


    
    df = pd.concat(
        frames,
        ignore_index=True
    )

    print(f"\nOriginal Articles : {len(df):,}")

   

    if len(df) > MAX_SAMPLES:

        df, _ = train_test_split(

            df,

            train_size=MAX_SAMPLES,

            random_state=RANDOM_SEED,

            stratify=df["2_way_label"]

        )

    print(f"Selected Articles : {len(df):,}")

    records = df.to_dict("records")

    return records




def load_topics():

    print("\nLoading topic keywords...")

    with open(TOPIC_FILE, "r", encoding="utf-8") as f:

        topics = json.load(f)

    topic_ids = sorted(
        topics.keys(),
        key=int
    )

    topic_sentences = [

        " ".join(
            topics[t]["keywords"]
        )

        for t in topic_ids

    ]

    return topic_ids, topic_sentences




def load_model():

    print(f"\nLoading SentenceTransformer ({MODEL_NAME})")
    print(f"Device : {DEVICE}")

    return SentenceTransformer(
        MODEL_NAME,
        device=DEVICE
    )




def assign_topics(records, model):

    topic_ids, topic_sentences = load_topics()

    print("\nEncoding topic keywords...")

    topic_embeddings = model.encode(

        topic_sentences,

        convert_to_tensor=True,

        normalize_embeddings=True

    )

    print("Encoding article titles...")

    article_embeddings = model.encode(

        [
            str(r["clean_title"])
            for r in records
        ],

        batch_size=64,

        convert_to_tensor=True,

        normalize_embeddings=True,

        show_progress_bar=True

    )

    similarity = article_embeddings @ topic_embeddings.T

    assignment = similarity.argmax(dim=1).tolist()

    for record, idx in zip(records, assignment):

        record["cluster_id"] = int(topic_ids[idx])

    return records




def build_record(record):

    
    timestamp = int(float(record["created_utc"]))

    dt = datetime.fromtimestamp(
        timestamp,
        tz=timezone.utc
    )

    midnight = datetime(
        dt.year,
        dt.month,
        dt.day,
        tzinfo=timezone.utc
    )

    timestamp = int(midnight.timestamp() * 1000)

    season = (dt.month - 1) // 3 + 1

    return {

        "content": str(record["clean_title"]),

        "image_path": "",

        "cluster_id": int(record["cluster_id"]),

        "time": timestamp,

        "label": LABEL_MAP[
            int(record["2_way_label"])
        ],

        "year": dt.year,

        "season": season,

        "month": dt.month,

        "id": str(record["id"]),

        "weight": WEIGHT,

        "day": dt.day,

        "year_month": f"{dt.year}-{dt.month:02d}",

        "year_season": f"{dt.year}-{season:02d}",

        "location": "USA",

        "location_bengali": "USA",

        "original_location": "USA"

    }




def save_dataset(records):

    print("\nWriting master dataset...")

    truth = 0
    rumor = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as writer:

        for record in tqdm(records):

            sample = build_record(record)

            if sample["label"] == "truth":
                truth += 1
            else:
                rumor += 1

            writer.write(
                json.dumps(
                    sample,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\n========================================")
    print("MASTER DATASET CREATED")
    print("========================================")
    print(f"Articles : {len(records):,}")
    print(f"Truth    : {truth:,}")
    print(f"Rumor    : {rumor:,}")
    print(f"Saved To : {OUTPUT_FILE}")




def main():

    records = load_dataset()

    model = load_model()

    records = assign_topics(
        records,
        model
    )

    save_dataset(records)


if __name__ == "__main__":

    main()
