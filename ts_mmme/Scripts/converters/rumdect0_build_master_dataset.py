#!/usr/bin/env python3


import json
from pathlib import Path
from datetime import datetime, timezone

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm



MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHT = 1


CURRENT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = CURRENT_DIR.parents[2]

BASELINE_ROOT = PROJECT_ROOT / "Baseline"

INPUT_FILE = (
    BASELINE_ROOT
    / "Dataset"
    / "Experiment"
    / "full_dataset.json"
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
    / "rumdect_master.json"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)





def load_dataset():

    print("\nLoading RumDect dataset...")

    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:

                records.append(json.loads(line))

    print(f"Loaded {len(records):,} articles.")

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
            topics[topic]["keywords"]
        )

        for topic in topic_ids

    ]

    return topic_ids, topic_sentences



def load_model():

    print(f"\nLoading SentenceTransformer ({MODEL_NAME})")
    print(f"Device : {DEVICE}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE
    )

    return model




def assign_topics(records, model):

    topic_ids, topic_sentences = load_topics()

    print("\nEncoding topic keywords...")

    topic_embeddings = model.encode(

        topic_sentences,

        convert_to_tensor=True,

        normalize_embeddings=True

    )

    print("Encoding article content...")

    article_embeddings = model.encode(

        [r["content"] for r in records],

        batch_size=64,

        show_progress_bar=True,

        convert_to_tensor=True,

        normalize_embeddings=True

    )

    similarity = article_embeddings @ topic_embeddings.T

    assignments = similarity.argmax(dim=1).tolist()

    for record, index in zip(records, assignments):

        record["cluster_id"] = int(topic_ids[index])

    return records







def build_record(record):

    year = int(record["year"])

    month = int(record["month"])

    day = 1

    season = (month - 1) // 3 + 1

    dt = datetime(
        year,
        month,
        day,
        tzinfo=timezone.utc
    )

    epoch_ms = int(dt.timestamp() * 1000)

    location = str(
        record.get("location", "")
    ).strip()

    return {

        "content": record["content"],

        "image_path": record.get(
            "image_path",
            ""
        ),

        "cluster_id": int(
            record["cluster_id"]
        ),

        "time": epoch_ms,

        "label": record["label"],

        "year": year,

        "season": season,

        "month": month,

        "id": record["id"],

        "weight": WEIGHT,

        "day": day,

        "year_month": f"{year}-{month:02d}",

        "year_season": f"{year}-{season:02d}",

        "location": location,

        "location_bengali": location,

        "original_location": location

    }




def save_dataset(records):

    print("\nWriting master dataset...")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as writer:

        for record in tqdm(records):

            sample = build_record(record)

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
