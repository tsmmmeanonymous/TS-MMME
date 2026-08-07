#!/usr/bin/env python3
"""
===========================================================
MME-TSE
Step 1 : BERTopic Topic Generation
===========================================================

Input
-----
Dataset/Experiment/full_dataset.json

Output
------
Dataset/Experiment/topic_dataset.json
Resources/BERTopic/topic_model
Resources/BERTopic/topic_info.csv

===========================================================
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import jieba
from sklearn.feature_extraction.text import CountVectorizer


# ==========================================================
# CHINESE STOPWORDS
# ==========================================================


# ==========================================================
# PATHS
# ==========================================================

CURRENT_DIR = Path(__file__).resolve().parent

BASELINE_DIR = CURRENT_DIR.parents[2]

DATASET_FILE = (
    BASELINE_DIR
    / "Dataset"
    / "Experiment"
    / "full_dataset.json"
)

OUTPUT_FILE = (
    BASELINE_DIR
    / "Dataset"
    / "Experiment"
    / "topic_dataset.json"
)

MODEL_DIR = (
    BASELINE_DIR
    / "Resources"
    / "BERTopic"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

records = []

with open(DATASET_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if line:

            records.append(json.loads(line))

df = pd.DataFrame(records)

documents = df["content"].fillna("").astype(str).tolist()

print(f"Documents : {len(documents)}")

# ==========================================================
# EMBEDDING MODEL
# ==========================================================

print("\nLoading SentenceTransformer...")

embedding_model = SentenceTransformer(
    "paraphrase-multilingual-mpnet-base-v2"
)

print("Embedding documents...")

embeddings = embedding_model.encode(
    documents,
    show_progress_bar=True,
    batch_size=32
)

# ==========================================================
# BERTopic
# ==========================================================

print("\nRunning BERTopic...")


# ----------------------------------------------------------
# Chinese Tokenizer
# ----------------------------------------------------------

def jieba_tokenizer(text):
    return jieba.lcut(text)

vectorizer_model = CountVectorizer(
    tokenizer=jieba.lcut,
    token_pattern=None,
    lowercase=False,
    min_df=2
)



topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=False,
    verbose=True
)

topics, probs = topic_model.fit_transform(
    documents,
    embeddings
)

# ==========================================================
# SAVE MODEL
# ==========================================================


#topic_model.save( MODEL_DIR / "topic_model")

# ==========================================================
# TOPIC INFO
# ==========================================================

topic_info = topic_model.get_topic_info()

topic_info.to_csv(
    MODEL_DIR / "topic_info.csv",
    index=False
)

# ==========================================================
# BUILD LOOKUP
# ==========================================================

topic_lookup = {}

for topic_id in topic_info["Topic"]:

    if topic_id == -1:

        topic_lookup[topic_id] = "Outlier"

    else:

        words = topic_model.get_topic(topic_id)

        if words:

            topic_lookup[topic_id] = ", ".join(
                [w for w, _ in words[:5]]
            )

        else:

            topic_lookup[topic_id] = "Unknown"

# ==========================================================
# SAVE DATASET
# ==========================================================

print("\nWriting dataset...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for record, topic_id in zip(records, topics):

        record["cluster_id"] = int(topic_id)

        record["topic"] = topic_lookup[topic_id]

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("BERTopic COMPLETED")
print("=" * 60)

print(f"Documents      : {len(records)}")

print(f"Topics Found   : {len(topic_info)-1}")

print(f"Output Dataset : {OUTPUT_FILE}")

print(f"Topic Info     : {MODEL_DIR/'topic_info.csv'}")

print(f"Model Saved    : {MODEL_DIR/'topic_model'}")
