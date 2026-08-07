from pathlib import Path
import json
import torch
import pandas as pd

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HETERO_DIR = PROJECT_ROOT / "Graph" / "Experiment" / "Training" / "Hetero"

REPORT_DIR = PROJECT_ROOT / "Graph" / "Experiment" / "Training" / "Validation"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# Load Graph
# =====================================================

print("Loading HeteroData...")

graph_file = HETERO_DIR / "heterodata.pt"

data = torch.load(
    graph_file,
    weights_only=False
)

print("Graph Loaded Successfully")

# =====================================================
# Node Validation
# =====================================================

print("\nValidating Nodes...")

node_statistics = {}

node_statistics["article"] = int(data["article"].num_nodes)
node_statistics["topic"] = int(data["topic"].num_nodes)
node_statistics["time"] = int(data["time"].num_nodes)
node_statistics["location"] = int(data["location"].num_nodes)
node_statistics["memory"] = int(data["memory"].num_nodes)

print(node_statistics)

# =====================================================
# Feature Validation
# =====================================================

print("\nChecking Features...")

article_feature = data["article"].x
memory_feature = data["memory"].x

feature_statistics = {

    "article_shape": list(article_feature.shape),

    "memory_shape": list(memory_feature.shape),

    "article_nan":

        int(torch.isnan(article_feature).sum()),

    "memory_nan":

        int(torch.isnan(memory_feature).sum()),

    "article_inf":

        int(torch.isinf(article_feature).sum()),

    "memory_inf":

        int(torch.isinf(memory_feature).sum()),

    "article_zero_vectors":

        int((article_feature.abs().sum(dim=1)==0).sum()),

    "memory_zero_vectors":

        int((memory_feature.abs().sum(dim=1)==0).sum())

}

print(feature_statistics)

# =====================================================
# Label Validation
# =====================================================

print("\nChecking Labels...")

labels = data["article"].y

label_statistics = {

    "truth":

        int((labels==1).sum()),

    "rumor":

        int((labels==0).sum()),

    "min_label":

        int(labels.min()),

    "max_label":

        int(labels.max())

}

print(label_statistics)

# =====================================================
# Edge Validation
# =====================================================

print("\nChecking Edge Types...")

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

edge_statistics = {}

for edge_type in edge_types:

    edge = data[edge_type].edge_index

    edge_statistics[str(edge_type)] = {

        "edges":

            int(edge.shape[1]),

        "min_index":

            int(edge.min()),

        "max_index":

            int(edge.max())

    }

print(edge_statistics)






# =====================================================
# Advanced Graph Validation
# =====================================================

print("\nChecking Edge Integrity...")

validation = {}

for edge_type in edge_types:

    edge = data[edge_type].edge_index

    src = edge[0]
    dst = edge[1]

    # -----------------------------------------
    # Source Node Count
    # -----------------------------------------

    if edge_type[0] == "article":
        src_nodes = data["article"].num_nodes

    elif edge_type[0] == "topic":
        src_nodes = data["topic"].num_nodes

    elif edge_type[0] == "time":
        src_nodes = data["time"].num_nodes

    elif edge_type[0] == "location":
        src_nodes = data["location"].num_nodes

    else:
        src_nodes = data["memory"].num_nodes

    # -----------------------------------------
    # Target Node Count
    # -----------------------------------------

    if edge_type[2] == "article":
        dst_nodes = data["article"].num_nodes

    elif edge_type[2] == "topic":
        dst_nodes = data["topic"].num_nodes

    elif edge_type[2] == "time":
        dst_nodes = data["time"].num_nodes

    elif edge_type[2] == "location":
        dst_nodes = data["location"].num_nodes

    else:
        dst_nodes = data["memory"].num_nodes

    invalid_source = int((src >= src_nodes).sum())

    invalid_target = int((dst >= dst_nodes).sum())

    negative_source = int((src < 0).sum())

    negative_target = int((dst < 0).sum())

    duplicate_edges = int(
        edge.t().unique(dim=0).shape[0] != edge.shape[1]
    )

    self_loops = int((src == dst).sum())

    validation[str(edge_type)] = {

        "invalid_source": invalid_source,

        "invalid_target": invalid_target,

        "negative_source": negative_source,

        "negative_target": negative_target,

        "duplicate_edges": duplicate_edges,

        "self_loops": self_loops

    }

print("Edge Integrity Check Completed.")

# =====================================================
# Degree Statistics
# =====================================================

print("\nComputing Degree Statistics...")

degree_statistics = {}

for edge_type in edge_types:

    edge = data[edge_type].edge_index

    source_degree = torch.bincount(
        edge[0],
        minlength=int(edge[0].max()) + 1
    )

    target_degree = torch.bincount(
        edge[1],
        minlength=int(edge[1].max()) + 1
    )

    degree_statistics[str(edge_type)] = {

        "source_min":

            int(source_degree.min()),

        "source_max":

            int(source_degree.max()),

        "source_mean":

            float(source_degree.float().mean()),

        "target_min":

            int(target_degree.min()),

        "target_max":

            int(target_degree.max()),

        "target_mean":

            float(target_degree.float().mean())

    }

print("Degree Statistics Completed.")

# =====================================================
# Overall Graph Status
# =====================================================

graph_valid = True

for result in validation.values():

    if (
        result["invalid_source"] > 0 or
        result["invalid_target"] > 0 or
        result["negative_source"] > 0 or
        result["negative_target"] > 0
    ):
        graph_valid = False

status = "VALID" if graph_valid else "INVALID"

print("\nGraph Status :", status)

# =====================================================
# Save Validation Report
# =====================================================

report = {

    "graph_status": status,

    "node_statistics": node_statistics,

    "feature_statistics": feature_statistics,

    "label_statistics": label_statistics,

    "edge_statistics": edge_statistics,

    "edge_validation": validation,

    "degree_statistics": degree_statistics

}

REPORT_JSON = REPORT_DIR / "heterodata_validation_report.json"

with open(
    REPORT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )

# =====================================================
# Save Human Readable Report
# =====================================================

REPORT_TXT = REPORT_DIR / "heterodata_validation_summary.txt"

with open(
    REPORT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 60 + "\n")
    f.write("HETERODATA VALIDATION REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write("GRAPH STATUS\n")
    f.write("----------------------------\n")
    f.write(f"{status}\n\n")

    f.write("NODE TYPES\n")
    f.write("----------------------------\n")

    for k, v in node_statistics.items():

        f.write(f"{k:15s}: {v}\n")

    f.write("\n")

    f.write("FEATURES\n")
    f.write("----------------------------\n")

    for k, v in feature_statistics.items():

        f.write(f"{k:25s}: {v}\n")

    f.write("\n")

    f.write("LABELS\n")
    f.write("----------------------------\n")

    for k, v in label_statistics.items():

        f.write(f"{k:20s}: {v}\n")

    f.write("\n")

    f.write("EDGE TYPES\n")
    f.write("----------------------------\n")

    for k, v in edge_statistics.items():

        f.write(f"{k}: {v}\n")

    f.write("\n")

    f.write("EDGE VALIDATION\n")
    f.write("----------------------------\n")

    for k, v in validation.items():

        f.write(f"{k}\n")

        for key, value in v.items():

            f.write(f"    {key:20s}: {value}\n")

        f.write("\n")

# =====================================================
# Final Summary
# =====================================================

print("\n==============================================")
print("HETERODATA VALIDATION REPORT")
print("==============================================")

print(f"Graph Status : {status}")

print("\nNode Types")

for k, v in node_statistics.items():

    print(f"{k:12s}: {v}")

print("\nFeature Shapes")

print(feature_statistics["article_shape"])

print(feature_statistics["memory_shape"])

print("\nTruth :", label_statistics["truth"])

print("Rumor :", label_statistics["rumor"])




print("\nValidation Report")

print(REPORT_JSON)

print(REPORT_TXT)

print("\nValidation Completed Successfully.")