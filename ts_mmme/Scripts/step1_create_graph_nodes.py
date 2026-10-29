from pathlib import Path
import json
import pandas as pd

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = PROJECT_ROOT / "Dataset" / "Experiment"

INPUT_FILE = EXPERIMENT_DIR / "train_dataset.json"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Graph"
    / "Experiment"
    / "Training"
    / "Nodes"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

print("Loading Training Dataset...")

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if line:
            records.append(json.loads(line))

df = pd.DataFrame(records)

print(f"Articles Loaded : {len(df)}")

# =====================================================
# ARTICLE NODES
# =====================================================

article_nodes = pd.DataFrame({

    "article_index": range(len(df)),

    "dataset_index": df.index,

    "article_id": [
        f"A{i}" for i in range(len(df))
    ],

    "graph_split": "train",

    "topic": df["cluster_id"],

    "year": df["year"],

    "month": df["month"],

    "year_month": df["year_month"],

    "location": df["location"],

    "label": df["label"]

})

article_nodes.to_csv(
    OUTPUT_DIR / "article_nodes.csv",
    index=False
)

article_lookup = {
    row["article_id"]: int(row["article_index"])
    for _, row in article_nodes.iterrows()
}

with open(
    OUTPUT_DIR / "article_lookup.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        article_lookup,
        f,
        indent=4
    )

# =====================================================
# TOPIC NODES
# =====================================================

topics = sorted(df["cluster_id"].unique())

topic_nodes = pd.DataFrame({

    "topic_index": range(len(topics)),

    "topic_id": [
        f"T{i}" for i in range(len(topics))
    ],

    "topic": topics

})

topic_nodes.to_csv(
    OUTPUT_DIR / "topic_nodes.csv",
    index=False
)

topic_lookup = {
    str(topic): idx
    for idx, topic in enumerate(topics)
}

with open(
    OUTPUT_DIR / "topic_lookup.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        topic_lookup,
        f,
        indent=4
    )

# =====================================================
# TIME NODES
# =====================================================

times = sorted(df["year_month"].unique())

time_nodes = pd.DataFrame({

    "time_index": range(len(times)),

    "time_id": [
        f"TM{i}" for i in range(len(times))
    ],

    "year_month": times

})

time_nodes.to_csv(
    OUTPUT_DIR / "time_nodes.csv",
    index=False
)

time_lookup = {
    time: idx
    for idx, time in enumerate(times)
}

with open(
    OUTPUT_DIR / "time_lookup.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        time_lookup,
        f,
        indent=4
    )

# =====================================================
# LOCATION NODES
# =====================================================

locations = sorted(df["location"].unique())

location_nodes = pd.DataFrame({

    "location_index": range(len(locations)),

    "location_id": [
        f"L{i}" for i in range(len(locations))
    ],

    "location": locations

})

location_nodes.to_csv(
    OUTPUT_DIR / "location_nodes.csv",
    index=False
)

location_lookup = {
    location: idx
    for idx, location in enumerate(locations)
}

with open(
    OUTPUT_DIR / "location_lookup.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        location_lookup,
        f,
        indent=4
    )

# =====================================================
# Statistics
# =====================================================

statistics = {

    "Split": "Training",

    "Article Nodes": len(article_nodes),

    "Topic Nodes": len(topic_nodes),

    "Time Nodes": len(time_nodes),

    "Location Nodes": len(location_nodes),

    "Truth Articles": int(
        (article_nodes["label"] == "truth").sum()
    ),

    "Rumor Articles": int(
        (article_nodes["label"] == "rumor").sum()
    )
}

with open(
    OUTPUT_DIR / "node_statistics.json",
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

print("\n====================================")
print("GRAPH NODE CREATION COMPLETED")
print("====================================")

print(f"Article Nodes  : {len(article_nodes)}")
print(f"Topic Nodes    : {len(topic_nodes)}")
print(f"Time Nodes     : {len(time_nodes)}")
print(f"Location Nodes : {len(location_nodes)}")

print(f"Truth Articles : {(article_nodes['label']=='truth').sum()}")

print(f"Rumor Articles : {(article_nodes['label']=='rumor').sum()}")

print("\nLookup Files")

print("article_lookup.json")
print("topic_lookup.json")
print("time_lookup.json")
print("location_lookup.json")

print("\nSaved Files")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(file.name)