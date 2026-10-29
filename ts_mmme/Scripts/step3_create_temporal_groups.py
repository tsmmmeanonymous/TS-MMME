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

MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Article Nodes
# =====================================================

print("Loading Training Article Nodes...")

article_nodes = pd.read_csv(
    NODE_DIR / "article_nodes.csv"
)

print(f"Articles Loaded : {len(article_nodes)}")

# =====================================================
# Group Articles
# =====================================================

print("Creating temporal groups...")

grouped = article_nodes.groupby(
    ["topic", "year_month", "location"],
    sort=True
)

memory_rows = []

for memory_index, (group_key, group_df) in enumerate(grouped):

    topic, year_month, location = group_key

    truth_count = (group_df["label"] == "truth").sum()

    rumor_count = (group_df["label"] == "rumor").sum()

    total = len(group_df)

    truth_ratio = truth_count / total

    rumor_ratio = rumor_count / total

    memory_rows.append({

        "graph_split": "train",

        "memory_index": memory_index,

        "memory_id": f"M{memory_index}",

        "topic": topic,

        "year_month": year_month,

        "location": location,

        "article_count": total,

        "truth_count": int(truth_count),

        "rumor_count": int(rumor_count),

        "truth_ratio": round(truth_ratio, 4),

        "rumor_ratio": round(rumor_ratio, 4),

        "article_indices":
            ",".join(
                group_df["article_index"]
                .astype(str)
                .tolist()
            ),

        "article_ids":
            ",".join(
                group_df["article_id"]
                .tolist()
            )

    })

memory_df = pd.DataFrame(memory_rows)

# =====================================================
# Save CSV
# =====================================================

OUTPUT_FILE = MEMORY_DIR / "temporal_groups.csv"

memory_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# =====================================================
# Statistics
# =====================================================

statistics = {

    "Split": "Training",

    "Memory Groups": len(memory_df),

    "Total Articles": int(memory_df["article_count"].sum()),

    "Average Articles per Group":
        round(memory_df["article_count"].mean(), 2),

    "Maximum Group Size":
        int(memory_df["article_count"].max()),

    "Minimum Group Size":
        int(memory_df["article_count"].min())
}

with open(
    MEMORY_DIR / "temporal_group_statistics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        statistics,
        f,
        indent=4
    )

# =====================================================
# Preview
# =====================================================

print("\n======================================")
print("TEMPORAL GROUP CREATION COMPLETED")
print("======================================")

print(f"Memory Groups              : {len(memory_df)}")
print(f"Total Articles             : {statistics['Total Articles']}")
print(f"Average Articles / Group   : {statistics['Average Articles per Group']}")
print(f"Largest Group              : {statistics['Maximum Group Size']}")
print(f"Smallest Group             : {statistics['Minimum Group Size']}")

print("\nSample Groups")

print(
    memory_df[
        [
            "memory_id",
            "topic",
            "year_month",
            "location",
            "article_count",
            "truth_count",
            "rumor_count"
        ]
    ].head(10)
)

print("\nSaved Files")

print(OUTPUT_FILE)
print(MEMORY_DIR / "temporal_group_statistics.json")