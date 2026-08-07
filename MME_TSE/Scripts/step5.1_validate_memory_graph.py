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
MEMORY_DIR = GRAPH_DIR / "Memory"
EDGE_DIR = GRAPH_DIR / "Edges"
DEBUG_DIR = GRAPH_DIR / "Debug"

DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Nodes
# =====================================================

print("Loading graph...")

article_nodes = pd.read_csv(NODE_DIR / "article_nodes.csv")
topic_nodes = pd.read_csv(NODE_DIR / "topic_nodes.csv")
time_nodes = pd.read_csv(NODE_DIR / "time_nodes.csv")
location_nodes = pd.read_csv(NODE_DIR / "location_nodes.csv")
memory_nodes = pd.read_csv(MEMORY_DIR / "memory_nodes.csv")

# =====================================================
# Load Edges
# =====================================================

article_memory = pd.read_csv(
    EDGE_DIR / "article_memory_edges.csv"
)

topic_memory = pd.read_csv(
    EDGE_DIR / "topic_memory_edges.csv"
)

time_memory = pd.read_csv(
    EDGE_DIR / "time_memory_edges.csv"
)

location_memory = pd.read_csv(
    EDGE_DIR / "location_memory_edges.csv"
)

# =====================================================
# Duplicate Edge Check
# =====================================================

dup_article = article_memory.duplicated().sum()
dup_topic = topic_memory.duplicated().sum()
dup_time = time_memory.duplicated().sum()
dup_location = location_memory.duplicated().sum()

# =====================================================
# Invalid Source Index Check
# =====================================================

invalid_article = (
    article_memory["source_index"] >= len(article_nodes)
).sum()

invalid_topic = (
    topic_memory["source_index"] >= len(topic_nodes)
).sum()

invalid_time = (
    time_memory["source_index"] >= len(time_nodes)
).sum()

invalid_location = (
    location_memory["source_index"] >= len(location_nodes)
).sum()

# =====================================================
# Invalid Target (Memory) Check
# =====================================================

invalid_memory_target = (

    (article_memory["target_index"] >= len(memory_nodes)).sum()

    +

    (topic_memory["target_index"] >= len(memory_nodes)).sum()

    +

    (time_memory["target_index"] >= len(memory_nodes)).sum()

    +

    (location_memory["target_index"] >= len(memory_nodes)).sum()

)

# =====================================================
# Degree Check
# =====================================================

article_degree = article_memory.groupby(
    "source_index"
).size()

topic_degree = topic_memory.groupby(
    "target_index"
).size()

time_degree = time_memory.groupby(
    "target_index"
).size()

location_degree = location_memory.groupby(
    "target_index"
).size()

missing_article = len(article_nodes) - len(article_degree)

missing_topic = len(memory_nodes) - len(topic_degree)

missing_time = len(memory_nodes) - len(time_degree)

missing_location = len(memory_nodes) - len(location_degree)

# =====================================================
# Orphan Memory Nodes
# =====================================================

memory_article = set(article_memory["target_index"])

memory_topic = set(topic_memory["target_index"])

memory_time = set(time_memory["target_index"])

memory_location = set(location_memory["target_index"])

orphan_memory = set(memory_nodes["memory_index"])

orphan_memory = (
    orphan_memory
    - memory_article
    - memory_topic
    - memory_time
    - memory_location
)

# =====================================================
# Validation Report
# =====================================================

report = {

    "Article Nodes": len(article_nodes),

    "Topic Nodes": len(topic_nodes),

    "Time Nodes": len(time_nodes),

    "Location Nodes": len(location_nodes),

    "Memory Nodes": len(memory_nodes),

    "Article->Memory": len(article_memory),

    "Topic->Memory": len(topic_memory),

    "Time->Memory": len(time_memory),

    "Location->Memory": len(location_memory),

    "Duplicate Article->Memory": int(dup_article),

    "Duplicate Topic->Memory": int(dup_topic),

    "Duplicate Time->Memory": int(dup_time),

    "Duplicate Location->Memory": int(dup_location),

    "Invalid Article Index": int(invalid_article),

    "Invalid Topic Index": int(invalid_topic),

    "Invalid Time Index": int(invalid_time),

    "Invalid Location Index": int(invalid_location),

    "Invalid Memory Index": int(invalid_memory_target),

    "Articles Without Memory": int(missing_article),

    "Memory Without Topic": int(missing_topic),

    "Memory Without Time": int(missing_time),

    "Memory Without Location": int(missing_location),

    "Orphan Memory Nodes": len(orphan_memory)

}
report["Split"] = "Training"

# =====================================================
# Save Report
# =====================================================

REPORT_FILE = DEBUG_DIR / "memory_graph_validation_report.json"

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )

# =====================================================
# Console Output
# =====================================================

print("\n======================================")
print("MEMORY GRAPH VALIDATION REPORT")
print("======================================")

for k, v in report.items():

    print(f"{k:<35}: {v}")

graph_ok = (

    dup_article == 0

    and dup_topic == 0

    and dup_time == 0

    and dup_location == 0

    and invalid_article == 0

    and invalid_topic == 0

    and invalid_time == 0

    and invalid_location == 0

    and invalid_memory_target == 0

    and missing_article == 0

    and missing_topic == 0

    and missing_time == 0

    and missing_location == 0

    and len(orphan_memory) == 0

)

print("\n======================================")

if graph_ok:
    print("GRAPH STATUS : VALID")
else:
    print("GRAPH STATUS : INVALID")

print("======================================")

print("\nSaved Report:")
print(REPORT_FILE)
print("Split : Training")