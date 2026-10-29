import sys
from pathlib import Path
import csv
import json

import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )

# ==========================================================
# Import Image Captioning Module
# ==========================================================

from Modules.image_captioning import generate_caption

# ==========================================================
# Project Paths
# ==========================================================

RESOURCE_DIR = (
    PROJECT_ROOT
    / "Resources"
    / "Topic"
)

# ==========================================================
# Files
# ==========================================================

TOPIC_EMBEDDING_FILE = (
    RESOURCE_DIR
    / "topic_embeddings.pt"
)

TOPIC_METADATA_FILE = (
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
# Fusion Weights
# ==========================================================

TEXT_WEIGHT = 0.7

IMAGE_WEIGHT = 0.3

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
print("TOPIC PREDICTION")
print("==============================================")

print(f"Device : {device}")

if device.type == "cuda":

    print(
        f"GPU : {torch.cuda.get_device_name(0)}"
    )

else:

    print("Running on CPU")

# ==========================================================
# Load Sentence Transformer
# ==========================================================

print("\nLoading Sentence Transformer...")

embedding_model = SentenceTransformer(

    EMBEDDING_MODEL_NAME,

    device=device

)

print("Sentence Transformer Loaded Successfully.")

# ==========================================================
# Load Image Captioning Module
# ==========================================================

print("\nLoading Image Captioning Module...")

print("Image Captioning Module Ready.")



# ==========================================================
# Load Topic Embedding Bank
# ==========================================================

print("\nLoading Topic Embedding Bank...")

topic_data = torch.load(
    TOPIC_EMBEDDING_FILE,
    map_location=device
)

# Support both new and old formats
if isinstance(topic_data, dict):

    topic_embeddings = topic_data["embeddings"].to(device)

    topic_ids = topic_data["topic_ids"]

    topic_names = topic_data["topic_names"]

    embedding_dim = topic_data["embedding_dim"]

    print(f"Embedding Model : {topic_data['model']}")
    print(f"Embedding Dim   : {embedding_dim}")
    print(f"Total Topics    : {topic_data['num_topics']}")

else:
    # Backward compatibility
    topic_embeddings = topic_data.to(device)

print("Topic Embeddings Loaded Successfully.")


###############################################################
# Cache Directory
###############################################################

CACHE_DIR = (
    PROJECT_ROOT /
    "Resources" /
    "Cache"
)

TEXT_EMBEDDING_FILE = CACHE_DIR / "text_embeddings.pt"
CAPTION_FILE = CACHE_DIR / "captions.csv"
CAPTION_EMBEDDING_FILE = CACHE_DIR / "caption_embeddings.pt"

###############################################################
# Load Offline Caches
###############################################################

print("\nLoading Offline Embedding Cache...")

text_embedding_cache = torch.load(
    TEXT_EMBEDDING_FILE,
    map_location=device
)

text_embedding_cache = text_embedding_cache["embeddings"]

print(f"Text Embeddings : {len(text_embedding_cache)}")

print("\nLoading Caption Cache...")

caption_cache = {}

with open(
    CAPTION_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        article_id = row.get("id", "").strip()

        if article_id == "":
            continue

        # Keyed as a string. The cache is read back from CSV, so every id
        # arrives as text, while article ids themselves are integers on some
        # datasets and non-numeric strings on others (Fakeddit, Weibo).
        caption_cache[str(article_id)] = row.get("caption", "")

print(f"Captions : {len(caption_cache)}")

print("\nLoading Caption Embedding Cache...")

caption_embedding_cache = torch.load(
    CAPTION_EMBEDDING_FILE,
    map_location=device
)
caption_embedding_cache = caption_embedding_cache["embeddings"]
print(f"Caption Embeddings : {len(caption_embedding_cache)}")

# ==========================================================
# Load Topic Metadata
# ==========================================================

print("\nLoading Topic Metadata...")

with open(

    TOPIC_METADATA_FILE,

    "r",

    encoding="utf-8"

) as f:

    topic_metadata = json.load(f)

print("Topic Metadata Loaded Successfully.")

# ==========================================================
# Build Lookup Dictionaries
# ==========================================================

topic_id_to_name = {}

topic_name_to_id = {}

for item in topic_metadata:

    topic_id_to_name[

        item["topic_id"]

    ] = item["topic_name"]

    topic_name_to_id[

        item["topic_name"]

    ] = item["topic_id"]

# ==========================================================
# Topic Bank Summary
# ==========================================================

print("\n==============================================")
print("TOPIC EMBEDDING BANK")
print("==============================================")

print(

    f"Total Topics          : {len(topic_metadata)}"

)

print(

    f"Embedding Shape       : {tuple(topic_embeddings.shape)}"

)

print(

    f"Embedding Dimension   : {topic_embeddings.shape[1]}"

)

print("\nAvailable Topics")

print("----------------------------------------------")

for item in topic_metadata:

    print(

        f'{item["topic_id"]} : {item["topic_name"]}'

    )

print("\nTopic Bank Ready.")



# ==========================================================
# Predict Topic
# ==========================================================

import os as _os_ft
_FT = torch.load(CACHE_DIR / "ft_text_features.pt") if _os_ft.environ.get("FT_TEXT") == "1" else None


def predict_topic(
    article_text,
    article_id=None,
    image_path=None
):
    """
    Predict Top-3 Topics for a new article.

    Parameters
    ----------
    article_text : str

    image_path : str or None

    Returns
    -------
    dict
    """
    cache_key = article_id
    # ======================================================
    # Generate Text Embedding
    # ======================================================

    print("\nGenerating Text Embedding...")

    print("Encoding started...")

    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()), \
        contextlib.redirect_stderr(io.StringIO()):

        if cache_key in text_embedding_cache:

            text_embedding = text_embedding_cache[cache_key]

            if not isinstance(text_embedding, torch.Tensor):
                text_embedding = torch.tensor(
                    text_embedding,
                    dtype=torch.float32
                )

            text_embedding = text_embedding.to(device)

        else:

            text_embedding = embedding_model.encode(
                article_text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        
    print("Encoding finished...")

    text_embedding = text_embedding.float().to(device)

    print("Text Embedding Generated.")

    # ======================================================
    # Optional Image
    # ======================================================

    from pathlib import Path

    image_used = False
    caption = None
    caption_embedding = None
    import os as _os2
    if _os2.environ.get("TEXT_ONLY") == "1":   # text-only variant never needs a caption
        image_path = None

    if image_path is not None and str(image_path).strip() != "":

        image_file = Path(image_path)

        if image_file.is_file():

            print("\nGenerating Image Caption...")

            caption = caption_cache.get(str(cache_key))

            if caption is None:

                try:   # MMM-Trend: unreadable/corrupt image -> fall back to text only instead of crashing
                    caption = generate_caption(str(image_file))
                except Exception as _e:
                    print(f"Caption failed ({_e}); using text only.")
                    caption = None

            print(f"Caption : {caption}")

            if caption is not None and str(caption).strip() != "":

                print("\nGenerating Caption Embedding...")

                if cache_key in caption_embedding_cache:

                    caption_embedding = caption_embedding_cache[cache_key]

                    if not isinstance(caption_embedding, torch.Tensor):
                        caption_embedding = torch.tensor(
                            caption_embedding,
                            dtype=torch.float32
                        )

                    caption_embedding = caption_embedding.to(device)

                else:

                    caption_embedding = embedding_model.encode(
                        caption,
                        convert_to_tensor=True,
                        normalize_embeddings=True
                    )

                caption_embedding = caption_embedding.float().to(device)

                image_used = True

            else:
                print("Caption generation failed. Using text only.")

        else:
            print(f"Image not found: {image_file}")
            print("Using text only.")

    # Modality variants (TS-MMME_T / TS-MMME_I): drop the other channel online too
    import os as _os
    if _os.environ.get("TEXT_ONLY") == "1":
        image_used = False
    if _os.environ.get("IMAGE_ONLY") == "1":
        text_embedding = torch.zeros_like(text_embedding)

    # ======================================================
    # Fusion
    # ======================================================

    if image_used:

        print("\nFusing Text and Image Embeddings...")

        article_embedding = (

            TEXT_WEIGHT * text_embedding +

            IMAGE_WEIGHT * caption_embedding

        )

        article_embedding = F.normalize(

            article_embedding,

            dim=0

        )

    else:

        print("\nNo Image Found.")

        print("Using Text Embedding Only.")

        article_embedding = text_embedding

    # ======================================================
    # Return Intermediate Results
    # ======================================================

    '''return {

        "article_embedding": article_embedding,

        "text_embedding": text_embedding,

        "caption_embedding": caption_embedding,

        "caption": caption,

        "image_used": image_used

    }'''



    # ======================================================
    # Intermediate Results
    # ======================================================

    prediction_output = {

        "article_embedding": article_embedding,

        "text_embedding": text_embedding,

        "caption_embedding": caption_embedding,

        "caption": caption,

        "image_used": image_used

    }

    # ======================================================
    # Rank Topics
    # ======================================================

    results = rank_topics(

        prediction_output

    )
    if _FT is not None:   # FT_TEXT=1: topic is ranked with the sentence encoder, the article is represented by the fine-tuned BERT_T feature
        results["article_embedding"] = _FT[int(cache_key)].to(device)

    return results




# ==========================================================
# Rank Topics
# ==========================================================

def rank_topics(

    prediction_output

):
    """
    Rank all topics using cosine similarity.

    Parameters
    ----------
    prediction_output : dict

    Returns
    -------
    dict
    """

    article_embedding = prediction_output[

        "article_embedding"

    ]

    # ------------------------------------------------------
    # Cosine Similarity
    # ------------------------------------------------------

    print("\nComputing Topic Similarity...")

    similarity_scores = torch.matmul(

        topic_embeddings,

        article_embedding

    )

    # ------------------------------------------------------
    # Top-3 Topics
    # ------------------------------------------------------

    top_scores, top_indices = torch.topk(

        similarity_scores,

        k=3

    )

    top_topics = []

    for score, topic_idx in zip(

        top_scores,

        top_indices

    ):

        topic_id = int(topic_idx.item())

        top_topics.append(

            {

                "topic_id": topic_id,

                "topic_name": topic_id_to_name[topic_id],

                "similarity": float(score.item())

            }

        )

    # ------------------------------------------------------
    # Best Topic
    # ------------------------------------------------------

    best_topic = top_topics[0]

    print("Topic Ranking Completed.")

    # ------------------------------------------------------
    # Return
    # ------------------------------------------------------

    return {

        "predicted_topic": best_topic["topic_id"],

        "predicted_topic_name": best_topic["topic_name"],

        "confidence": best_topic["similarity"],

        "top3_topics": top_topics,

        "article_embedding": prediction_output["article_embedding"],

        "text_embedding": prediction_output["text_embedding"],

        "caption_embedding": prediction_output["caption_embedding"],

        "caption": prediction_output["caption"],

        "image_used": prediction_output["image_used"]

    }



# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("NEW ARTICLE")
    print("==============================================")

    article_text = input(

        "\nEnter Article Text :\n\n"

    ).strip()

    image_path = input(

        "\nEnter Image Path (Press Enter if none) : "

    ).strip()

    if image_path == "":

        image_path = None

    '''
    # ======================================================
    # Generate Article Representation
    # ======================================================

    prediction_output = predict_topic(

        article_text,

        image_path

    )

    # ======================================================
    # Rank Topics
    # ======================================================

    results = rank_topics(

        prediction_output

    )'''



    results = predict_topic(

        article_text,

        image_path

    )

    # ======================================================
    # Display Results
    # ======================================================

    print("\n==============================================")
    print("TOPIC PREDICTION RESULTS")
    print("==============================================")

    print(f"Image Used : {results['image_used']}")

    if results["image_used"]:

        print(f"Caption    : {results['caption']}")

    print("\nPredicted Topic")
    print("----------------------------------------------")

    print(f"Topic ID   : {results['predicted_topic']}")

    print(f"Topic Name : {results['predicted_topic_name']}")

    print(f"Confidence : {results['confidence']:.4f}")

    print("\nTop-3 Topics")
    print("----------------------------------------------")

    for rank, topic in enumerate(

        results["top3_topics"],

        start=1

    ):

        print(

            f"{rank}. "

            f"Topic {topic['topic_id']} | "

            f"{topic['topic_name']} | "

            f"Similarity = {topic['similarity']:.4f}"

        )

    print("\n==============================================")
    print("Topic Prediction Completed.")
    print("==============================================")