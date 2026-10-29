from pathlib import Path
import json
import pandas as pd

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "Graph"
    / "Experiment"
    / "Training"
)

NODE_DIR = EXPERIMENT_DIR / "Nodes"

MEMORY_DIR = EXPERIMENT_DIR / "Memory"

EDGE_DIR = EXPERIMENT_DIR / "Edges"

EDGE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# Load Memory Nodes
# =====================================================

print("Loading Training Memory Nodes...")

memory_nodes = pd.read_csv(
    MEMORY_DIR / "memory_nodes.csv"
)

# =====================================================
# Load Lookup Tables
# =====================================================

with open(NODE_DIR / "topic_lookup.json","r",encoding="utf-8") as f:
    topic_lookup = json.load(f)

with open(NODE_DIR / "time_lookup.json","r",encoding="utf-8") as f:
    time_lookup = json.load(f)

with open(NODE_DIR / "location_lookup.json","r",encoding="utf-8") as f:
    location_lookup = json.load(f)

# =====================================================
# Create Edge Lists
# =====================================================

article_memory_edges = []

topic_memory_edges = []

time_memory_edges = []

location_memory_edges = []

print("Creating memory edges...")

for _, row in memory_nodes.iterrows():

    memory_index = int(row["memory_index"])

    # -------------------------------------------------
    # Topic -> Memory
    # -------------------------------------------------

    topic_index = topic_lookup[str(row["topic"])]

    topic_memory_edges.append({

        "source_index": topic_index,

        "target_index": memory_index

    })

    # -------------------------------------------------
    # Time -> Memory
    # -------------------------------------------------

    time_index = time_lookup[row["year_month"]]

    time_memory_edges.append({

        "source_index": time_index,

        "target_index": memory_index

    })

    # -------------------------------------------------
    # Location -> Memory
    # -------------------------------------------------

    location_index = location_lookup[row["location"]]

    location_memory_edges.append({

        "source_index": location_index,

        "target_index": memory_index

    })

    # -------------------------------------------------
    # Article -> Memory
    # -------------------------------------------------

    articles = str(row["article_indices"]).split(",")

    for article in articles:

        article_memory_edges.append({

            "source_index": int(article),

            "target_index": memory_index

        })

# =====================================================
# Convert to DataFrames
# =====================================================

article_memory_edges = pd.DataFrame(article_memory_edges)

topic_memory_edges = pd.DataFrame(topic_memory_edges)

time_memory_edges = pd.DataFrame(time_memory_edges)

location_memory_edges = pd.DataFrame(location_memory_edges)

# =====================================================
# Save
# =====================================================

article_memory_edges.to_csv(
    EDGE_DIR / "article_memory_edges.csv",
    index=False
)

topic_memory_edges.to_csv(
    EDGE_DIR / "topic_memory_edges.csv",
    index=False
)

time_memory_edges.to_csv(
    EDGE_DIR / "time_memory_edges.csv",
    index=False
)

location_memory_edges.to_csv(
    EDGE_DIR / "location_memory_edges.csv",
    index=False
)

# =====================================================
# Statistics
# =====================================================

stats = {

    "Split": "Training",

    "Memory Nodes": len(memory_nodes),

    "Article -> Memory": len(article_memory_edges),

    "Topic -> Memory": len(topic_memory_edges),

    "Time -> Memory": len(time_memory_edges),

    "Location -> Memory": len(location_memory_edges),

    "Total Memory Edges":
        len(article_memory_edges)
        + len(topic_memory_edges)
        + len(time_memory_edges)
        + len(location_memory_edges)
}

with open(
    EDGE_DIR / "memory_edge_statistics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        stats,
        f,
        indent=4
    )

# =====================================================
# Summary
# =====================================================

print("\n======================================")
print("MEMORY EDGE CREATION COMPLETED")
print("Split                   : Training")
print("======================================")
print(f"Memory Nodes            : {len(memory_nodes)}")
print(f"Article -> Memory   : {len(article_memory_edges)}")

print(f"Topic -> Memory     : {len(topic_memory_edges)}")

print(f"Time -> Memory      : {len(time_memory_edges)}")

print(f"Location -> Memory  : {len(location_memory_edges)}")

print(f"Total Memory Edges  : {stats['Total Memory Edges']}")

print("\nSaved Files")

print(EDGE_DIR / "article_memory_edges.csv")

print(EDGE_DIR / "topic_memory_edges.csv")

print(EDGE_DIR / "time_memory_edges.csv")

print(EDGE_DIR / "location_memory_edges.csv")

print(EDGE_DIR / "memory_edge_statistics.json")