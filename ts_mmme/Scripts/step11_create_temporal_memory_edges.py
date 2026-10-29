from pathlib import Path
import json
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

MEMORY_DIR = GRAPH_DIR / "Memory"

EDGE_DIR = GRAPH_DIR / "Edges"

EDGE_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Memory Nodes
# =====================================================

print("Loading memory nodes...")

memory_df = pd.read_csv(
    MEMORY_DIR / "memory_nodes.csv"
)

print(f"Memory Nodes : {len(memory_df)}")

# =====================================================
# Sort
# =====================================================

memory_df = memory_df.sort_values(
    by=[
        "topic",
        "location",
        "year_month"
    ]
)

# =====================================================
# Build Temporal Memory Edges
# =====================================================

edges = []

edge_id = 0

groups = memory_df.groupby(
    ["topic", "location"]
)

for (topic, location), group in groups:

    group = group.sort_values(
        "year_month"
    ).reset_index(drop=True)

    if len(group) < 2:
        continue

    for i in range(len(group) - 1):

        source = group.iloc[i]

        target = group.iloc[i + 1]

        edges.append({

            "edge_id": edge_id,

            "source_index": int(source["memory_index"]),
            "target_index": int(target["memory_index"]),

            "source_memory_id": source["memory_id"],
            "source_topic": source["topic"],
            "source_time": source["year_month"],
            "source_location": source["location"],

            "target_memory_id": target["memory_id"],
            "target_topic": target["topic"],
            "target_time": target["year_month"],
            "target_location": target["location"]

        })

        edge_id += 1

# =====================================================
# Save
# =====================================================

edges_df = pd.DataFrame(edges)

OUTPUT_FILE = EDGE_DIR / "memory_memory_edges.csv"

edges_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# =====================================================
# Statistics
# =====================================================

stats = {

    "Split": "Training",

    "memory_nodes": len(memory_df),

    "temporal_edges": len(edges_df),

    "groups": int(len(groups))

}

with open(

    EDGE_DIR / "memory_memory_statistics.json",

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
print("TEMPORAL MEMORY EDGE CREATION")
print("Split              : Training")
print("======================================")

print(f"Memory Nodes     : {len(memory_df)}")

print(f"Temporal Edges   : {len(edges_df)}")

print(f"Topic-Location Groups : {len(groups)}")

print("\nSample Edges")

print(edges_df.head(10))

print("\nSaved Files")

print(OUTPUT_FILE)

print(EDGE_DIR / "memory_memory_statistics.json")



