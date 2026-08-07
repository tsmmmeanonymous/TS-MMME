"""
===========================================================
MME-TSE PREPROCESSING
Step 02 : Generate Caption Embeddings
===========================================================

Project : MME-TSE

Description
-----------
Reads image captions generated in Step 00 and converts
them into SentenceTransformer embeddings.

Input
-----
Resources/Cache/captions.csv

Output
------
Resources/Cache/caption_embeddings.pt

This script is executed ONLY ONCE before training.

Training will later read caption embeddings directly from
the cache, eliminating repeated SentenceTransformer
inference.
===========================================================
"""

import csv
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

# Cache Folder
CACHE_FOLDER = PROJECT_ROOT / "Resources" / "Cache"

CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

# Input CSV
CAPTION_FILE = CACHE_FOLDER / "captions.csv"

# Output File
OUTPUT_FILE = CACHE_FOLDER / "caption_embeddings.pt"


# ===========================================================
# MODEL CONFIGURATION
# ===========================================================

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading SentenceTransformer ({MODEL_NAME})...")
print(f"Device : {DEVICE}")

model = SentenceTransformer(MODEL_NAME)

model = model.to(DEVICE)

print("SentenceTransformer loaded successfully.\n")


# ===========================================================
# CAPTION LOADER
# ===========================================================

def load_captions(caption_file):
    """
    Reads captions.csv.

    Returns
    -------
    list
        List of caption dictionaries.
    """

    captions = []

    with open(caption_file, "r", encoding="utf-8", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            captions.append(row)

    return captions


# ===========================================================
# STATISTICS
# ===========================================================

class Statistics:

    def __init__(self):

        self.total_captions = 0

        self.success = 0

        self.failed = 0

        self.empty_caption = 0
# ===========================================================
# GENERATE CAPTION EMBEDDINGS
# ===========================================================

BATCH_SIZE = 64


def generate_embeddings(captions):
    """
    Generates SentenceTransformer embeddings for image captions.

    Returns
    -------
    dict
        {
            image_id : embedding_tensor
        }

    Statistics
    """

    stats = Statistics()

    stats.total_captions = len(captions)

    embeddings = {}

    print("=" * 60)
    print("Generating Caption Embeddings")
    print("=" * 60)
    print(f"Total Captions : {stats.total_captions}")
    print(f"Batch Size     : {BATCH_SIZE}")
    print()

    progress_bar = tqdm(
        range(0, len(captions), BATCH_SIZE),
        desc="Encoding Captions",
        unit="batch",
        ncols=100
    )

    for start in progress_bar:

        end = min(start + BATCH_SIZE, len(captions))

        batch = captions[start:end]

        batch_ids = []
        batch_captions = []

        # ----------------------------------------
        # Collect valid captions
        # ----------------------------------------

        for row in batch:

            image_id = row.get("id", "").strip()

            if image_id == "":

                stats.failed += 1

                continue

            try:
                image_id = int(image_id)
            except ValueError:

                stats.failed += 1

                continue

            caption = row.get("caption", "")

            if caption is None:
                caption = ""

            caption = str(caption).strip()

            if len(caption) == 0:

                stats.empty_caption += 1

                continue

            batch_ids.append(image_id)

            batch_captions.append(caption)

        # ----------------------------------------
        # Skip empty batch
        # ----------------------------------------

        if len(batch_captions) == 0:

            continue

        # ----------------------------------------
        # Generate embeddings
        # ----------------------------------------

        try:

            batch_embeddings = model.encode(
                batch_captions,
                batch_size=BATCH_SIZE,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=DEVICE
            )

            for image_id, embedding in zip(batch_ids, batch_embeddings):

                embeddings[image_id] = embedding.cpu()

                stats.success += 1

        except Exception as e:

            print(f"\nBatch failed : {e}")

            stats.failed += len(batch_captions)

        progress_bar.set_postfix({
            "Success": stats.success,
            "Failed": stats.failed,
            "Empty": stats.empty_caption
        })

    progress_bar.close()

    return embeddings, stats
# ===========================================================
# SAVE EMBEDDINGS
# ===========================================================

def save_embeddings(embeddings, output_file):
    """
    Saves caption embeddings as a PyTorch file.
    """

    torch.save(
        {
            "model": MODEL_NAME,
            "embedding_dim": model.get_embedding_dimension(),
            "num_captions": len(embeddings),
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
    print("Caption Embedding Generation Completed")
    print("=" * 60)

    print(f"Total Captions     : {stats.total_captions}")
    print(f"Successful         : {stats.success}")
    print(f"Failed             : {stats.failed}")
    print(f"Empty Caption      : {stats.empty_caption}")

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
    print("Step 02 : Caption Embedding Generation")
    print("=" * 60)

    # ------------------------------------------
    # Check caption file
    # ------------------------------------------

    if not CAPTION_FILE.exists():

        print(f"\nCaption file not found:\n{CAPTION_FILE}")
        return

    # ------------------------------------------
    # Load captions
    # ------------------------------------------

    print("\nLoading captions...")

    captions = load_captions(CAPTION_FILE)

    print(f"Loaded {len(captions)} captions.\n")

    # ------------------------------------------
    # Generate embeddings
    # ------------------------------------------

    embeddings, stats = generate_embeddings(captions)

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