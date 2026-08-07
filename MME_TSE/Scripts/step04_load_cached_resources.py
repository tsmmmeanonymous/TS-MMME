"""
===========================================================
MME-TSE PREPROCESSING
Step 04 : Load Cached Resources
===========================================================

Project : MME-TSE

Description
-----------
Loads all cached preprocessing resources required for
training and inference.

Input
-----
Resources/Cache/
    captions.csv
    text_embeddings.pt
    caption_embeddings.pt
    topic_embeddings.pt

Output
------
Dictionary containing all cached resources.

This script eliminates repeated file loading throughout
the training pipeline.
===========================================================
"""

from pathlib import Path

import pandas as pd
import torch


# ===========================================================
# PATH CONFIGURATION
# ===========================================================

CURRENT_DIR = Path(__file__).resolve().parent

# Project Root
PROJECT_ROOT = CURRENT_DIR.parent

# Cache Folder
CACHE_FOLDER = PROJECT_ROOT / "Resources" / "Cache"

# Cache Files
CAPTION_FILE = CACHE_FOLDER / "captions.csv"

TEXT_EMBEDDINGS_FILE = CACHE_FOLDER / "text_embeddings.pt"

CAPTION_EMBEDDINGS_FILE = CACHE_FOLDER / "caption_embeddings.pt"

TOPIC_FOLDER = PROJECT_ROOT / "Resources" / "Topic"

TOPIC_EMBEDDINGS_FILE = TOPIC_FOLDER / "topic_embeddings.pt"


# ===========================================================
# VERIFY CACHE FILES
# ===========================================================

def verify_cache():

    cache_files = {
        "Captions": CAPTION_FILE,
        "Text Embeddings": TEXT_EMBEDDINGS_FILE,
        "Caption Embeddings": CAPTION_EMBEDDINGS_FILE,
        "Topic Embeddings": TOPIC_EMBEDDINGS_FILE,
    }

    missing = []

    print("=" * 60)
    print("Checking Cache Files")
    print("=" * 60)

    for name, path in cache_files.items():

        if path.exists():

            print(f"✓ {name}")

        else:

            print(f"✗ {name}")

            missing.append(path)

    if missing:

        print("\nMissing cache files:\n")

        for path in missing:

            print(path)

        raise FileNotFoundError(
            "One or more cache files are missing."
        )

    print()


# ===========================================================
# LOAD CAPTIONS
# ===========================================================

def load_captions():

    print("Loading captions...")

    captions = pd.read_csv(CAPTION_FILE)

    print(f"Loaded {len(captions)} captions.\n")

    return captions


# ===========================================================
# LOAD PYTORCH CACHE
# ===========================================================

def load_torch_cache(cache_file):

    print(f"Loading {cache_file.name}...")

    cache = torch.load(
        cache_file,
        map_location="cpu"
    )

    print("Loaded successfully.\n")

    return cache
# ===========================================================
# VALIDATE CACHE
# ===========================================================

def validate_cache(cache, expected_key):

    if expected_key not in cache:

        raise KeyError(
            f"Missing key '{expected_key}' in cache."
        )

    if "model" not in cache:

        raise KeyError(
            "Missing 'model' metadata."
        )

    if "embedding_dim" not in cache:

        raise KeyError(
            "Missing 'embedding_dim' metadata."
        )

    print(f"Model          : {cache['model']}")
    print(f"Embedding Dim  : {cache['embedding_dim']}")
    print(f"Entries        : {len(cache[expected_key])}")
    print()


# ===========================================================
# LOAD ALL CACHED RESOURCES
# ===========================================================

def load_cached_resources():

    verify_cache()

    # ------------------------------------------
    # Captions
    # ------------------------------------------

    captions = load_captions()

    # ------------------------------------------
    # Text Embeddings
    # ------------------------------------------

    text_cache = load_torch_cache(
        TEXT_EMBEDDINGS_FILE
    )

    validate_cache(
        text_cache,
        "embeddings"
    )

    # ------------------------------------------
    # Caption Embeddings
    # ------------------------------------------

    caption_cache = load_torch_cache(
        CAPTION_EMBEDDINGS_FILE
    )

    validate_cache(
        caption_cache,
        "embeddings"
    )

    # ------------------------------------------
    # Topic Embeddings
    # ------------------------------------------

    topic_cache = load_torch_cache(
        TOPIC_EMBEDDINGS_FILE
    )

    validate_cache(
        topic_cache,
        "embeddings"
    )

    # ------------------------------------------
    # Build Resource Dictionary
    # ------------------------------------------

    resources = {

        "captions": captions,

        "text_embeddings":
            text_cache["embeddings"],

        "caption_embeddings":
            caption_cache["embeddings"],

        "topic_embeddings":
            topic_cache["embeddings"],

        "metadata": {

            "text": {

                "model":
                    text_cache["model"],

                "embedding_dim":
                    text_cache["embedding_dim"],

                "num_entries":
                    text_cache["num_articles"]
            },

            "caption": {

                "model":
                    caption_cache["model"],

                "embedding_dim":
                    caption_cache["embedding_dim"],

                "num_entries":
                    caption_cache["num_captions"]
            },

            "topic": {

                "model":
                    topic_cache["model"],

                "embedding_dim":
                    topic_cache["embedding_dim"],

                "num_entries":
                    topic_cache["num_topics"]
            }
        }
    }

    return resources
# ===========================================================
# SUMMARY
# ===========================================================

def print_summary(resources):

    print("\n")
    print("=" * 60)
    print("Cached Resources Summary")
    print("=" * 60)

    print(f"Captions               : {len(resources['captions'])}")

    print(f"Text Embeddings        : "
          f"{len(resources['text_embeddings'])}")

    print(f"Caption Embeddings     : "
          f"{len(resources['caption_embeddings'])}")

    print(f"Topic Embeddings       : "
          f"{len(resources['topic_embeddings'])}")

    print()

    print("Embedding Models")

    print("------------------------------")

    print(f"Text Model            : "
          f"{resources['metadata']['text']['model']}")

    print(f"Caption Model         : "
          f"{resources['metadata']['caption']['model']}")

    print(f"Topic Model           : "
          f"{resources['metadata']['topic']['model']}")

    print()

    print("Embedding Dimensions")

    print("------------------------------")

    print(f"Text                 : "
          f"{resources['metadata']['text']['embedding_dim']}")

    print(f"Caption              : "
          f"{resources['metadata']['caption']['embedding_dim']}")

    print(f"Topic                : "
          f"{resources['metadata']['topic']['embedding_dim']}")

    print("=" * 60)


# ===========================================================
# MAIN
# ===========================================================

def main():

    print("=" * 60)
    print("MME-TSE PREPROCESSING")
    print("Step 04 : Load Cached Resources")
    print("=" * 60)

    resources = load_cached_resources()

    print_summary(resources)


# ===========================================================
# ENTRY POINT
# ===========================================================

if __name__ == "__main__":

    main()
