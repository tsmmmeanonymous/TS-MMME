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

EDGE_DIR = EXPERIMENT_DIR / "Edges"

EDGE_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Node Tables
# =====================================================

print("Loading Training Graph Nodes...")

article_nodes = pd.read_csv(NODE_DIR / "article_nodes.csv")

# =====================================================
# Load Lookup Tables
# =====================================================

with open(NODE_DIR / "topic_lookup.json", "r", encoding="utf-8") as f:
    topic_lookup = json.load(f)

with open(NODE_DIR / "time_lookup.json", "r", encoding="utf-8") as f:
    time_lookup = json.load(f)

with open(NODE_DIR / "location_lookup.json", "r", encoding="utf-8") as f:
    location_lookup = json.load(f)

# =====================================================
# Create Edge Lists
# =====================================================

article_topic_edges = []
article_time_edges = []
article_location_edges = []

validation_rows = []

print("Creating edges...")

for _, row in article_nodes.iterrows():

    article_index = int(row["article_index"])

    # -------------------------------------------------
    # Article ---> Topic
    # -------------------------------------------------

    topic_index = topic_lookup[str(row["topic"])]

    article_topic_edges.append({

        "source_index": article_index,

        "target_index": topic_index

    })

    # -------------------------------------------------
    # Article ---> Time
    # -------------------------------------------------

    time_index = time_lookup[row["year_month"]]

    article_time_edges.append({

        "source_index": article_index,

        "target_index": time_index

    })

    # -------------------------------------------------
    # Article ---> Location
    # -------------------------------------------------

    location_index = location_lookup[row["location"]]

    article_location_edges.append({

        "source_index": article_index,

        "target_index": location_index

    })

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------

    validation_rows.append({

        "graph_split": row["graph_split"],

        "article_id": row["article_id"],

        "topic": row["topic"],

        "topic_index": topic_index,

        "year_month": row["year_month"],

        "time_index": time_index,

        "location": row["location"],

        "location_index": location_index

    })

# =====================================================
# Convert to DataFrames
# =====================================================

article_topic_edges = pd.DataFrame(article_topic_edges)

article_time_edges = pd.DataFrame(article_time_edges)

article_location_edges = pd.DataFrame(article_location_edges)

validation_df = pd.DataFrame(validation_rows)

# =====================================================
# Save Edge Files
# =====================================================

article_topic_edges.to_csv(
    EDGE_DIR / "article_topic_edges.csv",
    index=False
)

article_time_edges.to_csv(
    EDGE_DIR / "article_time_edges.csv",
    index=False
)

article_location_edges.to_csv(
    EDGE_DIR / "article_location_edges.csv",
    index=False
)

validation_df.to_csv(
    EDGE_DIR / "edge_validation.csv",
    index=False
)

# =====================================================
# Statistics
# =====================================================

statistics = {

    "Split": "Training",

    "Article Nodes": len(article_nodes),

    "Article->Topic": len(article_topic_edges),

    "Article->Time": len(article_time_edges),

    "Article->Location": len(article_location_edges),

    "Total Edges": (
        len(article_topic_edges)
        + len(article_time_edges)
        + len(article_location_edges)
    )
}

with open(
    EDGE_DIR / "edge_statistics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        statistics,
        f,
        indent=4
    )

# =====================================================
# Summary
# =====================================================

print("\n======================================")
print("GRAPH EDGE CREATION COMPLETED")
print("======================================")

print(f"Articles             : {len(article_nodes)}")
print(f"Article -> Topic     : {len(article_topic_edges)}")
print(f"Article -> Time      : {len(article_time_edges)}")
print(f"Article -> Location  : {len(article_location_edges)}")
print(f"Total Edges          : {statistics['Total Edges']}")

print("\nSaved Files")

print(EDGE_DIR / "article_topic_edges.csv")

print(EDGE_DIR / "article_time_edges.csv")

print(EDGE_DIR / "article_location_edges.csv")

print(EDGE_DIR / "edge_validation.csv")

print(EDGE_DIR / "edge_statistics.json")