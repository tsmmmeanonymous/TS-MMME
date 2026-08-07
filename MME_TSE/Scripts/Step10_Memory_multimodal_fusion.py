from pathlib import Path
import json
import torch

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPH_DIR = (
    PROJECT_ROOT
    / "Graph"
    / "Experiment"
    / "Training"
)

FEATURE_DIR = GRAPH_DIR / "Features"

# =====================================================
# Load Features
# =====================================================

print("Loading feature tensors...")

article_text = torch.load(
    FEATURE_DIR / "article_text.pt",
    weights_only=True
)

article_image = torch.load(
    FEATURE_DIR / "article_image.pt",
    weights_only=True
)

memory_text = torch.load(
    FEATURE_DIR / "memory_text.pt",
    weights_only=True
)

memory_image = torch.load(
    FEATURE_DIR / "memory_image.pt",
    weights_only=True
)

# =====================================================
# Move everything to the same device
# =====================================================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

article_text = article_text.to(device)
article_image = article_image.to(device)
memory_text = memory_text.to(device)
memory_image = memory_image.to(device)

print(f"Article Text   : {tuple(article_text.shape)}")
print(f"Article Image  : {tuple(article_image.shape)}")
print(f"Memory Text    : {tuple(memory_text.shape)}")
print(f"Memory Image   : {tuple(memory_image.shape)}")

# =====================================================
# Concatenation Fusion
# =====================================================

print("\nPerforming multimodal fusion...")

print("Article Text  :", article_text.device)
print("Article Image :", article_image.device)
print("Memory Text   :", memory_text.device)
print("Memory Image  :", memory_image.device)

article_multimodal = torch.cat(
    (
        article_text,
        article_image
    ),
    dim=1
)

memory_multimodal = torch.cat(
    (
        memory_text,
        memory_image
    ),
    dim=1
)

# =====================================================
# Save
# =====================================================

ARTICLE_FILE = FEATURE_DIR / "article_multimodal.pt"

MEMORY_FILE = FEATURE_DIR / "memory_multimodal.pt"

torch.save(
    article_multimodal,
    ARTICLE_FILE
)

torch.save(
    memory_multimodal,
    MEMORY_FILE
)

# =====================================================
# Metadata
# =====================================================

metadata = {

    "Split": "Training",

    "fusion_method": "Concatenation",

    "article_feature_dimension": int(article_multimodal.shape[1]),

    "memory_feature_dimension": int(memory_multimodal.shape[1]),

    "article_shape": list(article_multimodal.shape),

    "memory_shape": list(memory_multimodal.shape),

    "text_dimension": 768,

    "image_dimension": 512,

    "multimodal_dimension": 1280
}

METADATA_FILE = FEATURE_DIR / "multimodal_metadata.json"

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

# =====================================================
# Summary
# =====================================================

print("\n======================================")
print("MULTIMODAL FEATURE FUSION COMPLETED")
print("Split                     : Training")
print("======================================")

print(f"Article Feature Shape : {tuple(article_multimodal.shape)}")

print(f"Memory Feature Shape  : {tuple(memory_multimodal.shape)}")

print(f"Final Feature Size    : {article_multimodal.shape[1]}")

print("\nSaved Files")

print(ARTICLE_FILE)

print(MEMORY_FILE)

print(METADATA_FILE)