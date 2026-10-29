import sys
from pathlib import Path
import json

import pandas as pd
import torch

from sentence_transformers import SentenceTransformer

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )

# ==========================================================
# Project Paths
# ==========================================================

RESOURCE_DIR = (
    PROJECT_ROOT
    / "Resources"
    / "Topic"
)

RESOURCE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# Files
# ==========================================================

KEYWORD_FILE = (
    RESOURCE_DIR
    / "topic_keywords.json"
)

EMBEDDING_PT_FILE = (
    RESOURCE_DIR
    / "topic_embeddings.pt"
)

EMBEDDING_CSV_FILE = (
    RESOURCE_DIR
    / "topic_embeddings.csv"
)

METADATA_FILE = (
    RESOURCE_DIR
    / "topic_metadata.json"
)

# ==========================================================
# Embedding Model
# ==========================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# ==========================================================
# Device
# ==========================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

print("\n==============================================")
print("PREPARE TOPIC EMBEDDINGS")
print("==============================================")

print(f"Device : {device}")

if device.type == "cuda":

    print(
        f"GPU : {torch.cuda.get_device_name(0)}"
    )

else:

    print("Running on CPU")



# ==========================================================
# Load Topic Keywords
# ==========================================================

print("\nLoading Topic Keywords...")

with open(
    KEYWORD_FILE,
    "r",
    encoding="utf-8"
) as f:

    topic_data = json.load(f)

print(f"Total Topics : {len(topic_data)}")

# ==========================================================
# Build Topic Sentences
# ==========================================================

topic_ids = []

topic_names = []

topic_keywords = []

topic_sentences = []

for topic_id, info in topic_data.items():

    topic_ids.append(
        int(topic_id)
    )

    topic_names.append(
        info["name"]
    )

    topic_keywords.append(
        info["keywords"]
    )

    sentence = " ".join(
        info["keywords"]
    )

    topic_sentences.append(
        sentence
    )

print("\n==============================================")
print("TOPIC SUMMARY")
print("==============================================")

for idx in range(len(topic_ids)):

    print(f"\nTopic {topic_ids[idx]}")

    print(f"Name : {topic_names[idx]}")

    print(
        f"Keywords : {', '.join(topic_keywords[idx])}"
    )

# ==========================================================
# Load Sentence Transformer
# ==========================================================

print("\nLoading Sentence Transformer...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    device=device
)

print("Embedding Model Loaded Successfully.")

print(f"\nEmbedding Model : {EMBEDDING_MODEL_NAME}")

print(f"Embedding Dimension : {embedding_model.get_embedding_dimension()}")

print("\nModel Ready.")



# ==========================================================
# Generate Topic Embeddings
# ==========================================================

print("\nGenerating Topic Embeddings...")

topic_embeddings = embedding_model.encode(

    topic_sentences,

    convert_to_tensor=True,

    normalize_embeddings=True,

    show_progress_bar=False

)

topic_embeddings = topic_embeddings.cpu()

print("Topic Embeddings Generated Successfully.")

# ==========================================================
# Embedding Summary
# ==========================================================

print("\n==============================================")
print("TOPIC EMBEDDING SUMMARY")
print("==============================================")

print(f"Total Topics          : {len(topic_ids)}")

print(f"Embedding Shape       : {tuple(topic_embeddings.shape)}")

print(f"Embedding Dimension   : {topic_embeddings.shape[1]}")

print("\nEmbedding Statistics")

print(f"Minimum Value : {topic_embeddings.min().item():.6f}")

print(f"Maximum Value : {topic_embeddings.max().item():.6f}")

print(f"Mean Value    : {topic_embeddings.mean().item():.6f}")

print(f"Std Deviation : {topic_embeddings.std().item():.6f}")

# ==========================================================
# Preview
# ==========================================================

print("\nTopic Embedding Preview")

for i in range(len(topic_ids)):

    print(
        f"Topic {topic_ids[i]} "
        f"({topic_names[i]}) "
        f"-> Shape {tuple(topic_embeddings[i].shape)}"
    )



# ==========================================================
# Save Topic Embeddings (.pt)
# ==========================================================

print("\nSaving Topic Embeddings...")

torch.save(
    {
        "model": EMBEDDING_MODEL_NAME,
        "embedding_dim": embedding_model.get_embedding_dimension(),
        "num_topics": len(topic_ids),
        "embeddings": topic_embeddings,
        "topic_ids": topic_ids,
        "topic_names": topic_names
    },
    EMBEDDING_PT_FILE
)

print(f"PyTorch File Saved : {EMBEDDING_PT_FILE}")

# ==========================================================
# Save Topic Embeddings (.csv)
# ==========================================================

print("\nCreating Topic Embedding CSV...")

embedding_dimension = topic_embeddings.shape[1]

column_names = [

    f"dim_{i}"

    for i in range(embedding_dimension)

]

embedding_df = pd.DataFrame(

    topic_embeddings.numpy(),

    columns=column_names

)

embedding_df.insert(

    0,

    "topic_name",

    topic_names

)

embedding_df.insert(

    0,

    "topic_id",

    topic_ids

)

embedding_df.to_csv(

    EMBEDDING_CSV_FILE,

    index=False

)

print(f"CSV File Saved : {EMBEDDING_CSV_FILE}")

# ==========================================================
# Save Topic Metadata
# ==========================================================

print("\nSaving Topic Metadata...")

metadata = []

for i in range(len(topic_ids)):

    metadata.append(

        {

            "topic_id": topic_ids[i],

            "topic_name": topic_names[i],

            "keywords": topic_keywords[i]

        }

    )

with open(

    METADATA_FILE,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        metadata,

        f,

        indent=4,

        ensure_ascii=False

    )

print(f"Metadata File Saved : {METADATA_FILE}")

# ==========================================================
# Preview
# ==========================================================

print("\n==============================================")
print("TOPIC EMBEDDING PREVIEW")
print("==============================================")

print(

    embedding_df.iloc[:, :12]

)


# ==========================================================
# Validate Saved Files
# ==========================================================

print("\nValidating Saved Files...")

assert EMBEDDING_PT_FILE.exists(), \
    "Topic embedding PT file not found."

assert EMBEDDING_CSV_FILE.exists(), \
    "Topic embedding CSV file not found."

assert METADATA_FILE.exists(), \
    "Topic metadata file not found."

print("Validation Successful.")

# ==========================================================
# Final Summary
# ==========================================================

print("\n==============================================")
print("TOPIC EMBEDDING GENERATION COMPLETED")
print("==============================================")

print(f"Total Topics          : {len(topic_ids)}")

print(f"Embedding Dimension   : {topic_embeddings.shape[1]}")

print(f"Embedding Shape       : {tuple(topic_embeddings.shape)}")

print(f"Embedding Model       : {EMBEDDING_MODEL_NAME}")

print("\nGenerated Files")
print("----------------------------------------------")

print(f"Keywords JSON   : {KEYWORD_FILE}")

print(f"Embeddings PT   : {EMBEDDING_PT_FILE}")

print(f"Embeddings CSV  : {EMBEDDING_CSV_FILE}")

print(f"Metadata JSON   : {METADATA_FILE}")

print("\nNext Step")
print("----------------------------------------------")

print("Run : step14.2_generate_image_caption.py")

print("==============================================")