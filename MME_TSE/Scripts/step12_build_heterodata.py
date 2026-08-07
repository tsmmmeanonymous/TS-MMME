from pathlib import Path
import json
import torch
import pandas as pd

from torch_geometric.data import HeteroData

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

NODE_DIR = GRAPH_DIR / "Nodes"
EDGE_DIR = GRAPH_DIR / "Edges"
MEMORY_DIR = GRAPH_DIR / "Memory"
FEATURE_DIR = GRAPH_DIR / "Features"

OUTPUT_DIR = GRAPH_DIR / "Hetero"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# Load Node Tables
# =====================================================

print("Loading node tables...")

article_nodes = pd.read_csv(
    NODE_DIR / "article_nodes.csv"
)

topic_nodes = pd.read_csv(
    NODE_DIR / "topic_nodes.csv"
)

time_nodes = pd.read_csv(
    NODE_DIR / "time_nodes.csv"
)

location_nodes = pd.read_csv(
    NODE_DIR / "location_nodes.csv"
)

memory_nodes = pd.read_csv(
    MEMORY_DIR / "memory_nodes.csv"
)

print(f"Article Nodes  : {len(article_nodes)}")
print(f"Topic Nodes    : {len(topic_nodes)}")
print(f"Time Nodes     : {len(time_nodes)}")
print(f"Location Nodes : {len(location_nodes)}")
print(f"Memory Nodes   : {len(memory_nodes)}")

# =====================================================
# Load Edge Tables
# =====================================================

print("\nLoading edge tables...")

article_topic_edges = pd.read_csv(
    EDGE_DIR / "article_topic_edges.csv"
)

article_time_edges = pd.read_csv(
    EDGE_DIR / "article_time_edges.csv"
)

article_location_edges = pd.read_csv(
    EDGE_DIR / "article_location_edges.csv"
)

article_memory_edges = pd.read_csv(
    EDGE_DIR / "article_memory_edges.csv"
)

topic_memory_edges = pd.read_csv(
    EDGE_DIR / "topic_memory_edges.csv"
)

time_memory_edges = pd.read_csv(
    EDGE_DIR / "time_memory_edges.csv"
)

location_memory_edges = pd.read_csv(
    EDGE_DIR / "location_memory_edges.csv"
)

memory_memory_edges = pd.read_csv(
    EDGE_DIR / "memory_memory_edges.csv"
)

print("Edge tables loaded successfully.")

# =====================================================
# Load Features
# =====================================================

print("\nLoading feature tensors...")

article_features = torch.load(
    FEATURE_DIR / "article_multimodal.pt"
)

memory_features = torch.load(
    FEATURE_DIR / "memory_multimodal.pt"
)

print(f"Article Feature Shape : {tuple(article_features.shape)}")
print(f"Memory Feature Shape  : {tuple(memory_features.shape)}")

# =====================================================
# Build Labels
# =====================================================

print("\nCreating article labels...")

label_map = {
    "rumor": 0,
    "truth": 1
}

article_labels = torch.tensor(

    article_nodes["label"]
    .map(label_map)
    .values,

    dtype=torch.long

)

print(f"Article Labels : {article_labels.shape}")

# =====================================================
# Create HeteroData
# =====================================================

print("\nCreating HeteroData...")

data = HeteroData()

# =====================================================
# Add Node Features
# =====================================================

# -------------------------
# Article Nodes
# -------------------------

data["article"].x = article_features

data["article"].y = article_labels

data["article"].num_nodes = len(article_nodes)

# -------------------------
# Topic Nodes
# -------------------------

data["topic"].num_nodes = len(topic_nodes)

# -------------------------
# Time Nodes
# -------------------------

data["time"].num_nodes = len(time_nodes)

# -------------------------
# Location Nodes
# -------------------------

data["location"].num_nodes = len(location_nodes)

# -------------------------
# Memory Nodes
# -------------------------

data["memory"].x = memory_features

data["memory"].num_nodes = len(memory_nodes)

print("\nNode features added successfully.")



# =====================================================
# Build Edge Index
# =====================================================

print("\nCreating graph edges...")

# -----------------------------------------------------
# Article -> Topic
# -----------------------------------------------------

data[
    "article",
    "belongs_to",
    "topic"
].edge_index = torch.tensor(

    article_topic_edges[
        [
            "source_index",
            "target_index"
        ]
    ].values.T,

    dtype=torch.long

)

# -----------------------------------------------------
# Article -> Time
# -----------------------------------------------------

data[
    "article",
    "occurs_in",
    "time"
].edge_index = torch.tensor(

    article_time_edges[
        [
            "source_index",
            "target_index"
        ]
    ].values.T,

    dtype=torch.long

)

# -----------------------------------------------------
# Article -> Location
# -----------------------------------------------------

data[
    "article",
    "located_in",
    "location"
].edge_index = torch.tensor(

    article_location_edges[
        [
            "source_index",
            "target_index"
        ]
    ].values.T,

    dtype=torch.long

)

# -----------------------------------------------------
# Article -> Memory
# -----------------------------------------------------

data[
    "article",
    "stored_in",
    "memory"
].edge_index = torch.tensor(

    article_memory_edges[
        [
            "source_index",
            "target_index"
        ]
    ].values.T,

    dtype=torch.long

)

# -----------------------------------------------------
# Topic -> Memory
# -----------------------------------------------------

data[
    "topic",
    "has_memory",
    "memory"
].edge_index = torch.tensor(

    topic_memory_edges[
        [
            "source_index",
            "target_index"
        ]
    ].values.T,

    dtype=torch.long

)

# -----------------------------------------------------
# Time -> Memory
# -----------------------------------------------------

data[
    "time",
    "has_memory",
    "memory"
].edge_index = torch.tensor(

    time_memory_edges[
        [
            "source_index",
            "target_index"
        ]
    ].values.T,

    dtype=torch.long

)

# -----------------------------------------------------
# Location -> Memory
# -----------------------------------------------------

data[
    "location",
    "has_memory",
    "memory"
].edge_index = torch.tensor(

    location_memory_edges[
        [
            "source_index",
            "target_index"
        ]
    ].values.T,

    dtype=torch.long

)

# -----------------------------------------------------
# Memory -> Memory
# -----------------------------------------------------

data[
    "memory",
    "evolves_to",
    "memory"
].edge_index = torch.tensor(

    memory_memory_edges[
        [
            "source_index",
            "target_index"
        ]
    ].values.T,

    dtype=torch.long

)

print("Edge construction completed.")

# =====================================================
# Validate Graph
# =====================================================

print("\nValidating graph...")

assert data["article"].x.shape[0] == len(article_nodes)

assert data["memory"].x.shape[0] == len(memory_nodes)

assert data["article"].y.shape[0] == len(article_nodes)

edge_types = [

    ("article","belongs_to","topic"),

    ("article","occurs_in","time"),

    ("article","located_in","location"),

    ("article","stored_in","memory"),

    ("topic","has_memory","memory"),

    ("time","has_memory","memory"),

    ("location","has_memory","memory"),

    ("memory","evolves_to","memory")

]

for edge_type in edge_types:

    edge_index = data[edge_type].edge_index

    assert edge_index.shape[0] == 2

    assert edge_index.dtype == torch.long

print("Graph validation passed.")

# =====================================================
# Graph Summary
# =====================================================

print("\n======================================")

print("HETEROGENEOUS GRAPH SUMMARY")

print("======================================")

print(f"Article Nodes  : {data['article'].num_nodes}")

print(f"Topic Nodes    : {data['topic'].num_nodes}")

print(f"Time Nodes     : {data['time'].num_nodes}")

print(f"Location Nodes : {data['location'].num_nodes}")

print(f"Memory Nodes   : {data['memory'].num_nodes}")

print()

print(f"Article Feature : {tuple(data['article'].x.shape)}")

print(f"Memory Feature  : {tuple(data['memory'].x.shape)}")

print()

for edge_type in edge_types:

    print(

        f"{edge_type} :",

        data[edge_type].edge_index.shape[1]

    )

print()

print("Graph successfully created.")



# =====================================================
# Save HeteroData
# =====================================================

print("\nSaving HeteroData...")

GRAPH_FILE = OUTPUT_DIR / "heterodata.pt"

torch.save(
    data,
    GRAPH_FILE
)

# =====================================================
# Graph Statistics
# =====================================================

statistics = {

    "Split": "Training",

    "node_types": {

        "article": int(data["article"].num_nodes),

        "topic": int(data["topic"].num_nodes),

        "time": int(data["time"].num_nodes),

        "location": int(data["location"].num_nodes),

        "memory": int(data["memory"].num_nodes)

    },

    "edge_types": {

        "article_topic":
            int(data["article","belongs_to","topic"].edge_index.shape[1]),

        "article_time":
            int(data["article","occurs_in","time"].edge_index.shape[1]),

        "article_location":
            int(data["article","located_in","location"].edge_index.shape[1]),

        "article_memory":
            int(data["article","stored_in","memory"].edge_index.shape[1]),

        "topic_memory":
            int(data["topic","has_memory","memory"].edge_index.shape[1]),

        "time_memory":
            int(data["time","has_memory","memory"].edge_index.shape[1]),

        "location_memory":
            int(data["location","has_memory","memory"].edge_index.shape[1]),

        "memory_memory":
            int(data["memory","evolves_to","memory"].edge_index.shape[1])

    },

    "feature_dimensions": {

        "article":
            int(data["article"].x.shape[1]),

        "memory":
            int(data["memory"].x.shape[1])

    },

    "labels": {

        "truth":
            int((data["article"].y == 1).sum()),

        "rumor":
            int((data["article"].y == 0).sum())

    }

}

with open(

    OUTPUT_DIR / "graph_statistics.json",

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        statistics,

        f,

        indent=4

    )

# =====================================================
# Save Human Readable Summary
# =====================================================

summary_file = OUTPUT_DIR / "heterodata_summary.txt"

with open(

    summary_file,

    "w",

    encoding="utf-8"

) as f:

    f.write("=" * 50 + "\n")

    f.write("HETEROGENEOUS GRAPH SUMMARY\n")

    f.write("=" * 50 + "\n\n")

    f.write("Node Types\n")

    f.write("-------------------------\n")

    for k, v in statistics["node_types"].items():

        f.write(f"{k:15s}: {v}\n")

    f.write("\n")

    f.write("Edge Types\n")

    f.write("-------------------------\n")

    for k, v in statistics["edge_types"].items():

        f.write(f"{k:20s}: {v}\n")

    f.write("\n")

    f.write("Feature Dimensions\n")

    f.write("-------------------------\n")

    for k, v in statistics["feature_dimensions"].items():

        f.write(f"{k:15s}: {v}\n")

    f.write("\n")

    f.write("Article Labels\n")

    f.write("-------------------------\n")

    f.write(f"Truth : {statistics['labels']['truth']}\n")

    f.write(f"Rumor : {statistics['labels']['rumor']}\n")

# =====================================================
# Print Final Graph
# =====================================================

print("\n======================================")
print("FINAL HETEROGENEOUS GRAPH")
print("Split : Training")
print("======================================\n")

print(data)

print("\n======================================")
print("GRAPH SAVED SUCCESSFULLY")
print("======================================")

print(f"\nHeteroData : {GRAPH_FILE}")

print(f"Statistics : {OUTPUT_DIR / 'graph_statistics.json'}")

print(f"Summary    : {summary_file}")

print("\nPipeline Completed Successfully.")