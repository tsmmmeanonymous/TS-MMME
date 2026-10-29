from pathlib import Path

# =====================================================
# Active Dataset
# =====================================================
# Set this to the dataset being run.
# Every path below is derived from it; no other file
# needs editing when switching datasets.
#
#   mmm_trend    -> Dataset/final_model_dataset_images_normalized_location.json
#   fakenewsnet  -> Dataset/fakenewsnet_master.json
#   pheme        -> Dataset/pheme_master.json
#   fakeddit     -> Dataset/fakeddit_master.json
#   weibo        -> Dataset/rumdect_master.json
# =====================================================

DATASET = "experimental"

MASTER_FILES = {
    "mmm_trend": "final_model_dataset_images_normalized_location.json",
    "experimental": "experimental_master.json",
    "fakenewsnet": "fakenewsnet_master.json",
    "pheme": "pheme_master.json",
    "fakeddit": "fakeddit_master.json",
    "weibo": "rumdect_master.json",
}

if DATASET not in MASTER_FILES:
    raise ValueError(
        f"Unknown DATASET '{DATASET}'. "
        f"Expected one of: {', '.join(sorted(MASTER_FILES))}"
    )

# =====================================================
# Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "Dataset"

MASTER_FILE = DATASET_DIR / MASTER_FILES[DATASET]

EXPERIMENT_DIR = DATASET_DIR / "Experiment"

IMAGE_DIR = DATASET_DIR / "article_images"

RESOURCE_DIR = PROJECT_ROOT / "Resources"

GRAPH_DIR = PROJECT_ROOT / "Graph"

RESULT_DIR = PROJECT_ROOT / "Results"

LOG_DIR = PROJECT_ROOT / "Logs"

# =====================================================
# Split
# =====================================================

TRAIN_RATIO = 0.80

RANDOM_SEED = 42
