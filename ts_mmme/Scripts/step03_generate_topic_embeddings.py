"""
===========================================================
MME-TSE PREPROCESSING
Step 03 : Generate Topic Embeddings
===========================================================

Project : MME-TSE

Description
-----------
Reads topics from the training dataset, extracts unique
topics, and converts them into SentenceTransformer
embeddings.

Input
-----
dataset/train_dataset.json

Output
------
Resources/Cache/topic_embeddings.pt

This script is executed ONLY ONCE before training.

Training will later read topic embeddings directly from
the cache, eliminating repeated SentenceTransformer
inference.
===========================================================
"""

import json
from pathlib import Path

import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ===========================================================
# PATH CONFIGURATION
# ===========================================================

CURRENT_DIR = Path(__file__).resolve().parent

# Project Root
PROJECT_ROOT = CURRENT_DIR.parent

# Dataset
DATASET_FILE = PROJECT_ROOT / "dataset" / "train_dataset.json"

# Cache Folder
CACHE_FOLDER = PROJECT_ROOT.parent / "Resources" / "Cache"

CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

# Output File
OUTPUT_FILE = CACHE_FOLDER / "topic_embeddings.pt"


# ===========================================================
# MODEL CONFIGURATION
# ===========================================================

MODEL_NAME = "all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Loading SentenceTransformer ({MODEL_NAME})...")
print(f"Device : {DEVICE}")

model = SentenceTransformer(MODEL_NAME)

model = model.to(DEVICE)

print("SentenceTransformer loaded successfully.\n")


# ===========================================================
# DATASET LOADER
# ===========================================================

def load_dataset(dataset_path):
    """
    Reads JSONL dataset.

    Returns
    -------
    list
        List of article dictionaries.
    """

    dataset = []

    with open(dataset_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line == "":
                continue

            try:

                dataset.append(json.loads(line))

            except Exception as e:

                print(f"Skipping malformed line : {e}")

    return dataset


# ===========================================================
# EXTRACT UNIQUE TOPICS
# ===========================================================

def extract_topics(dataset):
    """
    Extracts unique topics from the dataset.

    Returns
    -------
    list
        Sorted list of unique topics.
    """

    topics = set()

    for article in dataset:

        topic = article.get("topic", "")

        if topic is None:
            continue

        topic = str(topic).strip()

        if len(topic) == 0:
            continue

        topics.add(topic)

    return sorted(topics)


# ===========================================================
# STATISTICS
# ===========================================================

class Statistics:

    def __init__(self):

        self.total_topics = 0

        self.success = 0

        self.failed = 0

        self.empty_topic = 0
# ===========================================================
# GENERATE TOPIC EMBEDDINGS
# ===========================================================

BATCH_SIZE = 64


def generate_embeddings(topics):
    """
    Generates SentenceTransformer embeddings for topics.

    Returns
    -------
    dict
        {
            topic_name : embedding_tensor
        }

    Statistics
    """

    stats = Statistics()

    stats.total_topics = len(topics)

    embeddings = {}

    print("=" * 60)
    print("Generating Topic Embeddings")
    print("=" * 60)
    print(f"Total Topics : {stats.total_topics}")
    print(f"Batch Size   : {BATCH_SIZE}")
    print()

    progress_bar = tqdm(
        range(0, len(topics), BATCH_SIZE),
        desc="Encoding Topics",
        unit="batch",
        ncols=100
    )

    for start in progress_bar:

        end = min(start + BATCH_SIZE, len(topics))

        batch_topics = topics[start:end]

        valid_topics = []

        # ----------------------------------------
        # Collect valid topics
        # ----------------------------------------

        for topic in batch_topics:

            topic = str(topic).strip()

            if len(topic) == 0:

                stats.empty_topic += 1

                continue

            valid_topics.append(topic)

        # ----------------------------------------
        # Skip empty batch
        # ----------------------------------------

        if len(valid_topics) == 0:

            continue

        # ----------------------------------------
        # Generate embeddings
        # ----------------------------------------

        try:

            batch_embeddings = model.encode(
                valid_topics,
                batch_size=BATCH_SIZE,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=DEVICE
            )

            for topic, embedding in zip(valid_topics, batch_embeddings):

                embeddings[topic] = embedding.cpu()

                stats.success += 1

        except Exception as e:

            print(f"\nBatch failed : {e}")

            stats.failed += len(valid_topics)

        progress_bar.set_postfix({
            "Success": stats.success,
            "Failed": stats.failed,
            "Empty": stats.empty_topic
        })

    progress_bar.close()

    return embeddings, stats
# ===========================================================
# SAVE EMBEDDINGS
# ===========================================================

def save_embeddings(embeddings, output_file):
    """
    Saves topic embeddings as a PyTorch file.
    """

    torch.save(
        {
            "model": MODEL_NAME,
            "embedding_dim": model.get_sentence_embedding_dimension(),
            "num_topics": len(embeddings),
            "embeddings": embeddings
        },
        output_file
    )


# ===========================================================
# SUMMARY
# ===========================================================

def print_summary(stats):

    print("\n")
    print("=" * 60)
    print("Topic Embedding Generation Completed")
    print("=" * 60)

    print(f"Total Topics       : {stats.total_topics}")
    print(f"Successful         : {stats.success}")
    print(f"Failed             : {stats.failed}")
    print(f"Empty Topic        : {stats.empty_topic}")

    print()
    print("Embedding Cache Saved :")
    print(OUTPUT_FILE)

    print("=" * 60)


# ===========================================================
# MAIN
# ===========================================================

def main():

    print("=" * 60)
    print("MME-TSE PREPROCESSING")
    print("Step 03 : Topic Embedding Generation")
    print("=" * 60)

    # ------------------------------------------
    # Check dataset
    # ------------------------------------------

    if not DATASET_FILE.exists():

        print(f"\nDataset not found:\n{DATASET_FILE}")
        return

    # ------------------------------------------
    # Load dataset
    # ------------------------------------------

    print("\nLoading dataset...")

    dataset = load_dataset(DATASET_FILE)

    print(f"Loaded {len(dataset)} articles.")

    # ------------------------------------------
    # Extract Topics
    # ------------------------------------------

    print("\nExtracting unique topics...")

    topics = extract_topics(dataset)

    print(f"Found {len(topics)} unique topics.\n")

    # ------------------------------------------
    # Generate embeddings
    # ------------------------------------------

    embeddings, stats = generate_embeddings(topics)

    # ------------------------------------------
    # Save cache
    # ------------------------------------------

    print("\nSaving embedding cache...")

    save_embeddings(
        embeddings=embeddings,
        output_file=OUTPUT_FILE
    )

    # ------------------------------------------
    # Print report
    # ------------------------------------------

    print_summary(stats)


# ===========================================================
# ENTRY POINT
# ===========================================================

if __name__ == "__main__":

    main()