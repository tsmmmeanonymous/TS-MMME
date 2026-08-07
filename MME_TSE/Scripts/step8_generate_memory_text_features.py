from pathlib import Path
import json
import torch
import pandas as pd

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

MEMORY_DIR = GRAPH_DIR / "Memory"

# =====================================================
# Load Article Text Features
# =====================================================

print("Loading article text features...")

article_embeddings = torch.load(
    FEATURE_DIR / "article_text.pt",
    map_location="cpu"
)

print(f"Article Embeddings : {tuple(article_embeddings.shape)}")

# =====================================================
# Load Temporal Groups
# =====================================================

groups = pd.read_csv(
    MEMORY_DIR / "temporal_groups.csv"
)

print(f"Memory Groups : {len(groups)}")

# =====================================================
# Mean Pooling
# =====================================================

memory_embeddings = []

print("Generating memory text features...")

for _, row in groups.iterrows():

    article_indices = [

        int(x)

        for x in str(row["article_indices"]).split(",")

        if str(x).strip() != ""

    ]

    embeddings = article_embeddings[
        article_indices
    ]

    memory_feature = embeddings.mean(dim=0)

    memory_embeddings.append(
        memory_feature
    )

memory_embeddings = torch.stack(
    memory_embeddings
)

# =====================================================
# Save
# =====================================================

FEATURE_FILE = FEATURE_DIR / "memory_text.pt"

torch.save(
    memory_embeddings,
    FEATURE_FILE
)

# =====================================================
# Metadata
# =====================================================

metadata = {

    "Split": "Training",

    "pooling": "mean",

    "memory_nodes": len(memory_embeddings),

    "embedding_dimension": memory_embeddings.shape[1],

    "tensor_shape": list(memory_embeddings.shape)
}

with open(

    FEATURE_DIR / "memory_text_metadata.json",

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
print("MEMORY TEXT FEATURES CREATED")
print("Split          : Training")
print("======================================")

print(f"Memory Nodes : {len(memory_embeddings)}")

print(f"Embedding Shape : {tuple(memory_embeddings.shape)}")

print(f"Embedding Size  : {memory_embeddings.shape[1]}")

print("\nSaved Files")

print(FEATURE_FILE)

print(FEATURE_DIR / "memory_text_metadata.json")