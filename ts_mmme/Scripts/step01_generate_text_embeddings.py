"""
===========================================================
MME-TSE PREPROCESSING
Step 01 : Generate Text Embeddings
===========================================================

Project : MME-TSE

Description
-----------
Generates SentenceTransformer embeddings for every article
text and stores them in

Resources/Cache/text_embeddings.pt

This script is executed ONLY ONCE before training.

Training will later read embeddings directly from the cache,
eliminating repeated SentenceTransformer inference.
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
DATASET_FILE = PROJECT_ROOT / "Dataset" / "Experiment" / "train_dataset.json"

# Cache Folder
CACHE_FOLDER = PROJECT_ROOT / "Resources" / "Cache"

CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

# Output File
OUTPUT_FILE = CACHE_FOLDER / "text_embeddings.pt"


# ===========================================================
# MODEL CONFIGURATION
# ===========================================================

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

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
# STATISTICS
# ===========================================================

class Statistics:

    def __init__(self):

        self.total_articles = 0

        self.success = 0

        self.failed = 0

        self.empty_text = 0
# ===========================================================
# GENERATE TEXT EMBEDDINGS
# ===========================================================

BATCH_SIZE = 64


def generate_embeddings(dataset):

    """
    Generates SentenceTransformer embeddings in batches.

    Returns
    -------
    dict
        {
            article_id : embedding_tensor
        }

    Statistics
    """

    stats = Statistics()

    stats.total_articles = len(dataset)

    embeddings = {}

    print("=" * 60)
    print("Generating Text Embeddings")
    print("=" * 60)
    print(f"Total Articles : {stats.total_articles}")
    print(f"Batch Size     : {BATCH_SIZE}")
    print()

    progress_bar = tqdm(
        range(0, len(dataset), BATCH_SIZE),
        desc="Encoding Text",
        unit="batch",
        ncols=100
    )

    for start in progress_bar:

        end = min(start + BATCH_SIZE, len(dataset))

        batch = dataset[start:end]

        batch_ids = []
        batch_texts = []

        # ----------------------------------------
        # Collect valid text
        # ----------------------------------------

        for article in batch:
            article_id = article.get("id", None)
            if article_id is None:
                stats.failed += 1
                continue

            text = article.get("content", "")

            if text is None:
                text = ""

            text = str(text).strip()

            if len(text) == 0:

                stats.empty_text += 1

                continue

            batch_ids.append(article_id)

            batch_texts.append(text)

        # ----------------------------------------
        # Skip empty batch
        # ----------------------------------------

        if len(batch_texts) == 0:

            continue

        # ----------------------------------------
        # Generate embeddings
        # ----------------------------------------

        try:

            batch_embeddings = model.encode(
                batch_texts,
                batch_size=BATCH_SIZE,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=DEVICE
            )

            for article_id, embedding in zip(batch_ids, batch_embeddings):

                embeddings[article_id] = embedding.cpu()

                stats.success += 1

        except Exception as e:

            print(f"\nBatch failed : {e}")

            stats.failed += len(batch_texts)

        progress_bar.set_postfix({
            "Success": stats.success,
            "Failed": stats.failed,
            "Empty": stats.empty_text
        })

    progress_bar.close()

    return embeddings, stats
# ===========================================================
# SAVE EMBEDDINGS
# ===========================================================

def save_embeddings(embeddings, output_file):
    """
    Saves embeddings dictionary as a PyTorch file.
    """

    torch.save(
        {
            "model": MODEL_NAME,
            "embedding_dim": model.get_sentence_embedding_dimension(),
            "num_articles": len(embeddings),
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
    print("Text Embedding Generation Completed")
    print("=" * 60)

    print(f"Total Articles     : {stats.total_articles}")
    print(f"Successful         : {stats.success}")
    print(f"Failed             : {stats.failed}")
    print(f"Empty Text         : {stats.empty_text}")

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
    print("Step 01 : Text Embedding Generation")
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

    print(f"Loaded {len(dataset)} articles.\n")

    # ------------------------------------------
    # Generate embeddings
    # ------------------------------------------

    embeddings, stats = generate_embeddings(dataset)

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