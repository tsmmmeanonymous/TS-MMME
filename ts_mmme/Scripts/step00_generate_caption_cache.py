"""
===========================================================
MME-TSE PREPROCESSING
Step 00 : Generate Caption Cache
===========================================================

Project : MME-TSE

Description
-----------
Generates BLIP captions for every image in the training
dataset and stores them in

Resources/Cache/captions.csv

This script is executed ONLY ONCE before training.

Training will later read captions directly from the cache,
eliminating repeated BLIP inference.
===========================================================
"""

import os
import json
import csv
from pathlib import Path

from tqdm import tqdm
from PIL import Image

# Existing BLIP module
from Modules.image_captioning import generate_caption


# ===========================================================
# PATH CONFIGURATION
# ===========================================================

CURRENT_DIR = Path(__file__).resolve().parent

# Project Root
PROJECT_ROOT = CURRENT_DIR.parent

# Dataset
DATASET_FILE = PROJECT_ROOT / "Dataset" / "Experiment" / "train_dataset.json"

# Image Folder
IMAGE_FOLDER = PROJECT_ROOT / "Dataset" / "article_images"

# Cache Folder
CACHE_FOLDER = PROJECT_ROOT / "Resources" / "Cache"

CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

# Output CSV
OUTPUT_CSV = CACHE_FOLDER / "captions.csv"


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
# IMAGE PATH
# ===========================================================

def get_image_path(image_path):

    """
    Converts dataset path into actual filesystem path.

    Example

    dataset/article_images/img_610.jpg

    →

    Scripts/dataset/article_images/img_610.jpg
    """

    if image_path is None:
        return None

    image_path = image_path.strip()

    if image_path == "":
        return None

    if Path(image_path).is_absolute() and Path(image_path).exists():   # final master stores absolute paths
        return Path(image_path)

    filename = Path(image_path).name

    return IMAGE_FOLDER / filename


# ===========================================================
# CSV WRITER
# ===========================================================

def initialize_csv(csv_file):

    # Always recreate the cache
    if csv_file.exists():
        csv_file.unlink()

    with open(csv_file, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "id",
            "image_path",
            "image_exists",
            "caption",
            "status"
        ])
        f.flush()
        os.fsync(f.fileno())


# ===========================================================
# VALIDATE IMAGE
# ===========================================================

def image_exists(image_file):

    """
    Checks if image exists and is readable.
    """

    if image_file is None:
        return False

    if not image_file.exists():
        return False

    try:

        with Image.open(image_file) as img:
            img.verify()

        return True

    except Exception:

        return False


# ===========================================================
# SUMMARY COUNTERS
# ===========================================================

class Statistics:

    def __init__(self):

        self.total_articles = 0

        self.images_found = 0

        self.no_image = 0

        self.missing_image = 0

        self.success = 0

        self.failed = 0
# ===========================================================
# CAPTION GENERATION
# ===========================================================

def process_dataset(dataset, output_csv):

    """
    Generate captions for every article image and
    append the results to captions.csv.
    """

    stats = Statistics()

    stats.total_articles = len(dataset)

    print("\n" + "=" * 60)
    print("Generating Caption Cache")
    print("=" * 60)
    print(f"Total Articles : {stats.total_articles}")
    print()

    with open(output_csv, "a", newline="", encoding="utf-8") as csv_file:

        writer = csv.writer(csv_file)

        progress_bar = tqdm(
            dataset,
            desc="Generating Captions",
            unit="article",
            ncols=100
        )

        for article in progress_bar:

            article_id = article.get("id", "")

            image_path = article.get("image_path", "")

            image_file = get_image_path(image_path)

            # -----------------------------------------
            # No image mentioned in dataset
            # -----------------------------------------

            if image_file is None:

                stats.no_image += 1

                writer.writerow([
                    article_id,
                    "",
                    "No",
                    "",
                    "NO_IMAGE"
                ])

                progress_bar.set_postfix({
                    "Success": stats.success,
                    "Missing": stats.missing_image,
                    "NoImg": stats.no_image
                })

                continue

            # -----------------------------------------
            # Missing image file
            # -----------------------------------------

            if not image_exists(image_file):

                stats.missing_image += 1

                writer.writerow([
                    article_id,
                    image_path,
                    "No",
                    "",
                    "MISSING_FILE"
                ])

                progress_bar.set_postfix({
                    "Success": stats.success,
                    "Missing": stats.missing_image,
                    "NoImg": stats.no_image
                })

                continue

            # -----------------------------------------
            # Generate BLIP caption
            # -----------------------------------------

            try:

                stats.images_found += 1

                caption = generate_caption(str(image_file))

                if caption is None:
                    caption = ""

                caption = str(caption).replace("\n", " ").strip()

                writer.writerow([
                    article_id,
                    image_path,
                    "Yes",
                    caption,
                    "SUCCESS"
                ])

                stats.success += 1

            except Exception as e:

                writer.writerow([
                    article_id,
                    image_path,
                    "Yes",
                    "",
                    f"FAILED : {e}"
                ])

                stats.failed += 1

            progress_bar.set_postfix({
                "Success": stats.success,
                "Failed": stats.failed,
                "Missing": stats.missing_image,
                "NoImg": stats.no_image
            })

        progress_bar.close()

    return stats

# ===========================================================
# SUMMARY
# ===========================================================

def print_summary(stats):

    print("\n")
    print("=" * 60)
    print("Caption Cache Generation Completed")
    print("=" * 60)

    print(f"Total Articles     : {stats.total_articles}")
    print(f"Images Found       : {stats.images_found}")
    print(f"Successful         : {stats.success}")
    print(f"No Image           : {stats.no_image}")
    print(f"Missing Image      : {stats.missing_image}")
    print(f"Failed             : {stats.failed}")

    print()
    print(f"Caption Cache Saved :")
    print(f"{OUTPUT_CSV}")

    print("=" * 60)


# ===========================================================
# MAIN
# ===========================================================

def main():

    print("=" * 60)
    print("MME-TSE PREPROCESSING")
    print("Step 00 : Caption Cache Generation")
    print("=" * 60)

    # ------------------------------------------
    # Check dataset
    # ------------------------------------------

    if not DATASET_FILE.exists():

        print(f"\nDataset not found:\n{DATASET_FILE}")
        return

    # ------------------------------------------
    # Check image directory
    # ------------------------------------------

    if not IMAGE_FOLDER.exists():

        print(f"\nImage directory not found:\n{IMAGE_FOLDER}")
        return

    # ------------------------------------------
    # Create CSV
    # ------------------------------------------

    initialize_csv(OUTPUT_CSV)

    # ------------------------------------------
    # Load dataset
    # ------------------------------------------

    print("\nLoading dataset...")

    dataset = load_dataset(DATASET_FILE)

    print(f"Loaded {len(dataset)} articles.")

    # ------------------------------------------
    # Process dataset
    # ------------------------------------------

    stats = process_dataset(
        dataset=dataset,
        output_csv=OUTPUT_CSV
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