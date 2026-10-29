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

NODE_DIR = GRAPH_DIR / "Nodes"
EDGE_DIR = GRAPH_DIR / "Edges"
DEBUG_DIR = GRAPH_DIR / "Debug"

DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Nodes
# =====================================================

print("Loading node tables...")

article_nodes = pd.read_csv(NODE_DIR / "article_nodes.csv")
topic_nodes = pd.read_csv(NODE_DIR / "topic_nodes.csv")
time_nodes = pd.read_csv(NODE_DIR / "time_nodes.csv")
location_nodes = pd.read_csv(NODE_DIR / "location_nodes.csv")

# =====================================================
# Load Edges
# =====================================================

print("Loading edge tables...")

article_topic = pd.read_csv(EDGE_DIR / "article_topic_edges.csv")
article_time = pd.read_csv(EDGE_DIR / "article_time_edges.csv")
article_location = pd.read_csv(EDGE_DIR / "article_location_edges.csv")

# =====================================================
# Statistics
# =====================================================

stats = {}

stats["Article Nodes"] = len(article_nodes)
stats["Topic Nodes"] = len(topic_nodes)
stats["Time Nodes"] = len(time_nodes)
stats["Location Nodes"] = len(location_nodes)

stats["Article->Topic"] = len(article_topic)
stats["Article->Time"] = len(article_time)
stats["Article->Location"] = len(article_location)

# =====================================================
# Duplicate Edge Check
# =====================================================

dup_topic = article_topic.duplicated().sum()
dup_time = article_time.duplicated().sum()
dup_location = article_location.duplicated().sum()

# =====================================================
# Invalid Index Check
# =====================================================

invalid_topic = (
    article_topic["target_index"] >= len(topic_nodes)
).sum()

invalid_time = (
    article_time["target_index"] >= len(time_nodes)
).sum()

invalid_location = (
    article_location["target_index"] >= len(location_nodes)
).sum()

# =====================================================
# Missing Edge Check
# =====================================================

topic_degree = (
    article_topic
    .groupby("source_index")
    .size()
)

time_degree = (
    article_time
    .groupby("source_index")
    .size()
)

location_degree = (
    article_location
    .groupby("source_index")
    .size()
)

missing_topic = (
    len(article_nodes) - len(topic_degree)
)

missing_time = (
    len(article_nodes) - len(time_degree)
)

missing_location = (
    len(article_nodes) - len(location_degree)
)

# =====================================================
# Orphan Node Check
# =====================================================

orphan_topics = (
    len(
        set(topic_nodes["topic_index"])
        -
        set(article_topic["target_index"])
    )
)

orphan_times = (
    len(
        set(time_nodes["time_index"])
        -
        set(article_time["target_index"])
    )
)

orphan_locations = (
    len(
        set(location_nodes["location_index"])
        -
        set(article_location["target_index"])
    )
)

# =====================================================
# Validation Report
# =====================================================

report = {

    "Split": "Training",

    "Article Nodes": len(article_nodes),

    "Topic Nodes": len(topic_nodes),

    "Time Nodes": len(time_nodes),

    "Location Nodes": len(location_nodes),

    "Topic Edges": len(article_topic),

    "Time Edges": len(article_time),

    "Location Edges": len(article_location),

    "Duplicate Topic Edges": int(dup_topic),

    "Duplicate Time Edges": int(dup_time),

    "Duplicate Location Edges": int(dup_location),

    "Invalid Topic Indices": int(invalid_topic),

    "Invalid Time Indices": int(invalid_time),

    "Invalid Location Indices": int(invalid_location),

    "Missing Topic Edges": int(missing_topic),

    "Missing Time Edges": int(missing_time),

    "Missing Location Edges": int(missing_location),

    "Orphan Topic Nodes": int(orphan_topics),

    "Orphan Time Nodes": int(orphan_times),

    "Orphan Location Nodes": int(orphan_locations)

}

# =====================================================
# Save Report
# =====================================================

with open(
    DEBUG_DIR / "graph_validation_report.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )

# =====================================================
# Console Summary
# =====================================================

print("\n======================================")
print("GRAPH VALIDATION REPORT")
print("Split : Training")
print("======================================")

for k, v in report.items():
    print(f"{k:<30}: {v}")

graph_ok = (
    dup_topic == 0
    and dup_time == 0
    and dup_location == 0
    and invalid_topic == 0
    and invalid_time == 0
    and invalid_location == 0
    and missing_topic == 0
    and missing_time == 0
    and missing_location == 0
)

print("\n======================================")

if graph_ok:
    print("GRAPH STATUS : VALID")
else:
    print("GRAPH STATUS : INVALID")

print("======================================")

print("\nSaved:")
print(DEBUG_DIR / "graph_validation_report.json")