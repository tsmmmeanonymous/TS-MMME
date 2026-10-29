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

MEMORY_DIR = EXPERIMENT_DIR / "Memory"

INPUT_FILE = MEMORY_DIR / "temporal_groups.csv"

OUTPUT_FILE = MEMORY_DIR / "memory_nodes.csv"

LOOKUP_FILE = MEMORY_DIR / "memory_lookup.json"

STATS_FILE = MEMORY_DIR / "memory_statistics.json"

# =====================================================
# Load Temporal Groups
# =====================================================

print("Loading Training Temporal Groups...")

groups = pd.read_csv(INPUT_FILE)

print(f"Groups Loaded : {len(groups)}")

# =====================================================
# Create Memory Nodes
# =====================================================

print("Creating memory nodes...")

memory_nodes = pd.DataFrame({

    "graph_split": groups["graph_split"],

    "memory_index": groups["memory_index"],

    "memory_id": groups["memory_id"],

    "topic": groups["topic"],

    "year_month": groups["year_month"],

    "location": groups["location"],

    "article_count": groups["article_count"],

    "truth_count": groups["truth_count"],

    "rumor_count": groups["rumor_count"],

    "truth_ratio": groups["truth_ratio"],

    "rumor_ratio": groups["rumor_ratio"],

    "article_indices": groups["article_indices"]

})

# =====================================================
# Save Memory Nodes
# =====================================================

memory_nodes.to_csv(
    OUTPUT_FILE,
    index=False
)

# =====================================================
# Create Rich Memory Lookup Dictionary
# =====================================================

memory_lookup = {}

for _, row in memory_nodes.iterrows():

    key = (
        f"{row['topic']}|"
        f"{row['year_month']}|"
        f"{row['location']}"
    )

    memory_lookup[key] = {

        # Existing lookup
        "memory_index": int(row["memory_index"]),

        # Temporal-Spatial Information
        "topic": int(row["topic"]),
        "year_month": str(row["year_month"]),
        "location": str(row["location"]),

        # Evolution Statistics
        "article_count": int(row["article_count"]),
        "truth_count": int(row["truth_count"]),
        "rumor_count": int(row["rumor_count"]),

        "truth_ratio": float(row["truth_ratio"]),
        "rumor_ratio": float(row["rumor_ratio"]),

        # Evolution Score
        "evolution_score":
            float(row["truth_ratio"]) -
            float(row["rumor_ratio"]),

        # Dominant Historical State
        "dominant_state":
            (
                "Truth"
                if row["truth_ratio"] > 0.60
                else "Rumor"
                if row["rumor_ratio"] > 0.60
                else "Mixed"
            ),

        # Original Articles
        "article_indices":
            str(row["article_indices"])
    }

with open(
    LOOKUP_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        memory_lookup,
        f,
        indent=4,
        ensure_ascii=False
    )
# =====================================================
# Statistics
# =====================================================

stats = {

    "Split": "Training",

    "Memory Nodes": len(memory_nodes),

    "Average Articles per Memory":
        round(memory_nodes["article_count"].mean(), 2),

    "Largest Memory":
        int(memory_nodes["article_count"].max()),

    "Smallest Memory":
        int(memory_nodes["article_count"].min()),

    "Total Articles Represented":
        int(memory_nodes["article_count"].sum())
}

with open(
    STATS_FILE,
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
print("MEMORY NODE CREATION COMPLETED")
print("Split                    : Training")
print("======================================")

print(f"Memory Nodes             : {stats['Memory Nodes']}")
print(f"Average Articles/Memory  : {stats['Average Articles per Memory']}")
print(f"Largest Memory           : {stats['Largest Memory']}")
print(f"Smallest Memory          : {stats['Smallest Memory']}")
print(f"Total Articles Covered   : {stats['Total Articles Represented']}")

print("\nSaved Files")

print(OUTPUT_FILE)
print(LOOKUP_FILE)
print(STATS_FILE)