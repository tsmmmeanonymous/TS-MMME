from pathlib import Path
import json
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = PROJECT_ROOT / "Dataset" / "Experiment"

DATASET_FILE = EXPERIMENT_DIR / "train_dataset.json"

GRAPH_DIR = (
    PROJECT_ROOT
    / "Graph"
    / "Experiment"
    / "Training"
)

NODE_DIR = GRAPH_DIR / "Nodes"

FEATURE_DIR = GRAPH_DIR / "Features"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# Model
# =====================================================

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

print("Loading SentenceTransformer...")

model = SentenceTransformer(MODEL_NAME)

# =====================================================
# Load Dataset
# =====================================================

import os as _os
_FT = None
if _os.environ.get("FT_TEXT") == "1":   # use fine-tuned BERT_T text features (data_prep/extract_ft_features.py)
    _FT = torch.load(PROJECT_ROOT / "Resources" / "Cache" / "ft_text_features.pt")
print("Loading Training Dataset...")

records = []

with open(DATASET_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if line.strip():

            records.append(json.loads(line))

dataset_df = pd.DataFrame(records)

# =====================================================
# Load Graph Nodes
# =====================================================

article_nodes = pd.read_csv(
    NODE_DIR / "article_nodes.csv"
)

print(f"Articles : {len(article_nodes)}")

# =====================================================
# Generate Embeddings
# =====================================================

embeddings = []

print("Generating text embeddings...")

for _, row in tqdm(
    article_nodes.iterrows(),
    total=len(article_nodes),
    desc="Encoding"
):

    article_index = int(row["article_index"])

    text = str(
        dataset_df.iloc[article_index]["content"]
    )

    if _FT is not None:
        embeddings.append(_FT[int(dataset_df.iloc[article_index]["id"])].to(model.device)); continue
    emb = model.encode(

        text,

        convert_to_tensor=True,

        normalize_embeddings=True,

        show_progress_bar=False

    )

    embeddings.append(emb)

embeddings = torch.stack(embeddings)

# =====================================================
# Save Tensor
# =====================================================

FEATURE_FILE = FEATURE_DIR / "article_text.pt"

torch.save(

    embeddings.cpu(),

    FEATURE_FILE

)

# =====================================================
# Metadata
# =====================================================

metadata = {

    "Split": "Training",

    "model": MODEL_NAME,

    "articles": len(article_nodes),

    "embedding_dimension": int(embeddings.shape[1]),

    "tensor_shape": list(embeddings.shape)
}

with open(

    FEATURE_DIR / "article_text_metadata.json",

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
print("ARTICLE TEXT FEATURES CREATED")
print("Split           : Training")
print("======================================")

print(f"Embedding Shape : {tuple(embeddings.shape)}")

print(f"Embedding Size  : {embeddings.shape[1]}")

print("\nSaved Files")

print(FEATURE_FILE)

print(FEATURE_DIR / "article_text_metadata.json")