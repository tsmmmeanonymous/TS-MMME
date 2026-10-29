import sys
from pathlib import Path
import json

import torch

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
# Paths
# ==========================================================

GRAPH_DIR = (
    PROJECT_ROOT
    / "Graph"
    / "Experiment"
    / "Training"
)

MEMORY_DIR = (

    GRAPH_DIR

    / "Memory"

)

ZMM_DIR = (

    GRAPH_DIR

    / "ZMM"

)

# ==========================================================
# Files
# ==========================================================

MEMORY_LOOKUP_FILE = (

    MEMORY_DIR

    / "memory_lookup.json"

)

ZMM_FILE = (

    ZMM_DIR

    / "memory_zmm.pt"

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

print("TEMPORAL MEMORY RETRIEVAL")

print("==============================================")

print(f"Device : {device}")

if device.type == "cuda":

    print(

        f"GPU : {torch.cuda.get_device_name(0)}"

    )

else:

    print(

        "Running on CPU"

    )
# ==========================================================
# Retrieval Hyperparameters
# ==========================================================

TOP_K_MEMORIES = 5

TOPIC_WEIGHT = 0.40
TIME_WEIGHT = 0.20
LOCATION_WEIGHT = 0.10
EMBEDDING_WEIGHT = 0.30
# ==========================================================
# Load Memory Lookup
# ==========================================================

print("\nLoading Memory Lookup...")

with open(

    MEMORY_LOOKUP_FILE,

    "r",

    encoding="utf-8"

) as f:

    memory_lookup = json.load(f)

print(

    f"Memory Lookup Entries : {len(memory_lookup)}"

)

print("Memory Lookup Loaded Successfully.")

# ==========================================================
# Load ZMM
# ==========================================================

print("\nLoading ZMM Embeddings...")

memory_zmm = torch.load(

    ZMM_FILE,

    map_location=device

)

memory_zmm = memory_zmm.to(device)

memory_zmm = torch.nn.functional.normalize(
    memory_zmm,
    dim=1
)

print("ZMM Loaded Successfully.")

print(

    f"Embedding Shape : {tuple(memory_zmm.shape)}"

)

print(

    f"Embedding Dimension : {memory_zmm.shape[1]}"

)

print("\nTemporal Memory Retrieval Ready.")

# ==========================================================
# Retrieve Temporal Memory
# ==========================================================

import os as _os
_ANCHOR = _os.environ.get("ANCHOR_TIME") == "1"
_LATEST_MEMORY_MONTH = max(k.split("|")[1] for k in memory_lookup)
print("Latest memory month:", _LATEST_MEMORY_MONTH, "| anchor beyond horizon:", _ANCHOR)


def retrieve_temporal_memory(
    top3_topics,
    month,
    location,
    article_embedding
):
    """
    Retrieve Top-K Temporal Memories using
    hybrid semantic retrieval.
    """
    candidate_memories = []

    # --------------------------------------------------
    # Normalize article embedding
    # --------------------------------------------------

    article_embedding = article_embedding.to(device)

    article_embedding = torch.nn.functional.normalize(
        article_embedding,
        dim=0
    )

    # --------------------------------------------------
    # Search every memory
    # --------------------------------------------------

    for lookup_key, memory_info in memory_lookup.items():

        topic_id_str, memory_month, memory_location = lookup_key.split("|")

        # =====================================================
        # Parse Memory Lookup Entry
        # =====================================================

        if isinstance(memory_info, int):

            # Old lookup format
            memory_index = memory_info

            truth_ratio = 0.0
            rumor_ratio = 0.0

            truth_count = 0
            rumor_count = 0

        else:

            # New lookup format
            memory_index = memory_info["memory_index"]

            truth_ratio = memory_info.get("truth_ratio", 0.0)
            rumor_ratio = memory_info.get("rumor_ratio", 0.0)

            truth_count = memory_info.get("truth_count", 0)
            rumor_count = memory_info.get("rumor_count", 0)

        memory_topic = int(topic_id_str)

        topic_score = 0.0

        # ------------------------------------------
        # Topic similarity
        # ------------------------------------------

        for topic in top3_topics:

            if topic["topic_id"] == memory_topic:

                topic_score = max(
                    topic_score,
                    topic["similarity"]
                )

        if topic_score == 0:
            continue

        # ------------------------------------------
        # Month similarity
        # ------------------------------------------

        if _ANCHOR and isinstance(month, str) and month > _LATEST_MEMORY_MONTH:
            month = _LATEST_MEMORY_MONTH   # MMM-Trend: article beyond the memory horizon -> measure recency from the newest memory

        if memory_month == month:
            time_score = 1.0
        else:

            try:

                y1, m1 = map(int, month.split("-"))
                y2, m2 = map(int, memory_month.split("-"))

                month_gap = abs(
                    (y1 - y2) * 12 +
                    (m1 - m2)
                )

                time_score = max(
                    0.0,
                    1.0 - month_gap / 12.0
                )

            except Exception:

                time_score = 0.0

        # ----------------------------------------------------------
        # Spatial Matching
        # ----------------------------------------------------------

        if location is None:

            # NO_SPATIAL ablation:
            # Ignore spatial filtering completely.
            location_score = 1.0

        else:

            if memory_location.lower() == location.lower():
                location_score = 1.0
            else:
                location_score = 0.0

        # ------------------------------------------
        # Embedding similarity
        # ------------------------------------------

        if not isinstance(memory_index, int):
            raise TypeError(
                f"memory_index must be int, got {type(memory_index)} "
                f"value={memory_index}"
            )

        memory_embedding = memory_zmm[memory_index]
        embedding_score = torch.matmul(
            article_embedding,
            memory_embedding
        ).item()

        # ------------------------------------------
        # Hybrid Retrieval Score
        # ------------------------------------------

        score = (
            TOPIC_WEIGHT * topic_score +
            TIME_WEIGHT * time_score +
            LOCATION_WEIGHT * location_score +
            EMBEDDING_WEIGHT * embedding_score
        )

        candidate_memories.append({

            "memory_index": memory_index,

            "lookup_key": lookup_key,

            "topic_id": memory_topic,

            "month": memory_month,

            "year_month": memory_month,

            "location": memory_location,

            "topic_score": topic_score,

            "time_score": time_score,

            "location_score": location_score,

            "embedding_score": embedding_score,

            "retrieval_score": float(score),

            "memory_embedding": memory_zmm[memory_index],

            # -------- Required by TS-VEA --------

            "truth_ratio": float(truth_ratio),

            "rumor_ratio": float(rumor_ratio),

            "truth_count": int(truth_count),

            "rumor_count": int(rumor_count)

        })

    # --------------------------------------------------
    # No Candidate
    # --------------------------------------------------

    if len(candidate_memories) == 0:

        return {

            "memory_found": False,

            "retrieved_memories": []
        }

    # --------------------------------------------------
    # Sort by Retrieval Score
    # --------------------------------------------------

    candidate_memories = sorted(

        candidate_memories,

        key=lambda x: x["retrieval_score"],

        reverse=True

    )

    retrieved = candidate_memories[:TOP_K_MEMORIES]

    return {

        "memory_found": True,

        "num_memories": len(retrieved),

        "retrieved_memories": retrieved,

        "best_memory": retrieved[0]

    }

# ==========================================================
# Display Retrieval Result
# ==========================================================

def display_retrieval_result(

    retrieval_result

):
    """
    Display retrieved Temporal Veracity Memory.
    """

    print("\n==============================================")
    print("TEMPORAL MEMORY RETRIEVAL RESULT")
    print("==============================================")

    if not retrieval_result["memory_found"]:

        print("\nMemory Found : False")

        print("\nNo Temporal Memory exists for the given")

        print("Topic / Month / Location combination.")

        return

    print(f"\nMemory Found        : {retrieval_result['memory_found']}")
    print(f"Retrieved Memories  : {retrieval_result['num_memories']}")

    best = retrieval_result["best_memory"]

    print(f"Memory Index        : {best['memory_index']}")
    print(f"Topic ID            : {best['topic_id']}")
    print(f"Month               : {best.get('year_month', best['month'])}")
    print(f"Location            : {best['location']}")

    print(f"Topic Score         : {best['topic_score']:.4f}")
    print(f"Time Score          : {best['time_score']:.4f}")
    print(f"Location Score      : {best['location_score']:.4f}")
    print(f"Embedding Score     : {best['embedding_score']:.4f}")
    print(f"Retrieval Score     : {best['retrieval_score']:.4f}")

    embedding = best["memory_embedding"]

    print("\nEmbedding Summary")
    print("----------------------------------------------")

    print(f"Embedding Shape     : {tuple(embedding.shape)}")

    print(f"Embedding Dimension : {embedding.shape[0]}")

    print(f"Minimum Value       : {embedding.min().item():.6f}")

    print(f"Maximum Value       : {embedding.max().item():.6f}")

    print(f"Mean Value          : {embedding.mean().item():.6f}")

    print(f"Std Deviation       : {embedding.std().item():.6f}")

    print("\nFirst 10 Dimensions")
    print("----------------------------------------------")

    for i in range(10):

        print(

            f"Dim {i:02d} : "

            f"{embedding[i].item():.6f}"

        )

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("TEMPORAL MEMORY RETRIEVAL TEST")
    print("==============================================")

    print("\nEnter Top-3 Predicted Topics")

    top3_topics = []

    for rank in range(1, 4):

        print(f"\nRank {rank}")

        topic_id = int(

            input("Topic ID : ")

        )

        similarity = float(

            input("Similarity : ")

        )

        top3_topics.append(

            {

                "topic_id": topic_id,

                "similarity": similarity

            }

        )

    month = input(

        "\nEnter Month (YYYY-MM) : "

    ).strip()

    location = input(

        "Enter Location : "

    ).strip()

    retrieval_result = retrieve_temporal_memory(

        top3_topics,

        month,

        location

    )

    display_retrieval_result(

        retrieval_result

    )

    print("\n==============================================")
    print("TEMPORAL MEMORY RETRIEVAL COMPLETED")
    print("Memory Bank : Training")
    print("==============================================")