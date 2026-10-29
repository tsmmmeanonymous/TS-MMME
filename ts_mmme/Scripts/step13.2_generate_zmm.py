from pathlib import Path
import json

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import torch

from Models.temporal_memory_graph_transformer import (
    TemporalMemoryGraphTransformer
)

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPH_DIR = (
    PROJECT_ROOT
    / "Graph"
    / "Experiment"
    / "Training"
)

HETERO_DIR = GRAPH_DIR / "Hetero"

MODEL_DIR = GRAPH_DIR / "Models"

ZMM_DIR = GRAPH_DIR / "ZMM"

ZMM_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# Files
# ==========================================================

HETERODATA_FILE = HETERO_DIR / "heterodata.pt"

MODEL_FILE = MODEL_DIR / "best_graph_transformer.pth"

ZMM_PT_FILE = ZMM_DIR / "memory_zmm.pt"

ZMM_CSV_FILE = ZMM_DIR / "memory_zmm.csv"

STATISTICS_FILE = ZMM_DIR / "zmm_statistics.json"

# ==========================================================
# Device
# ==========================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

print("\n==============================================")
print("GENERATE GRAPH-ENHANCED MEMORY EMBEDDINGS")
print("==============================================")

print(f"Using Device : {device}")

if device.type == "cuda":

    print(f"GPU : {torch.cuda.get_device_name(0)}")

else:

    print("Running on CPU")


# ==========================================================
# Load Heterogeneous Graph
# ==========================================================

print("\nLoading Heterogeneous Graph...")

data = torch.load(
    HETERODATA_FILE,
    map_location=device,
    weights_only=False
)

data = data.to(device)

print("Graph Loaded Successfully.")

# ==========================================================
# Load Trained Model Checkpoint
# ==========================================================

print("\nLoading Trained Graph Transformer...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

print("Checkpoint Loaded Successfully.")

# ==========================================================
# Build Model
# ==========================================================

model = TemporalMemoryGraphTransformer(

    metadata=checkpoint["metadata"],

    num_topics=checkpoint["num_topics"],

    num_times=checkpoint["num_times"],

    num_locations=checkpoint["num_locations"],

    article_feature_dim=checkpoint["article_feature_dim"],

    memory_feature_dim=checkpoint["memory_feature_dim"],

    hidden_dim=checkpoint["hidden_dim"],

    num_heads=checkpoint["num_heads"],

    num_layers=checkpoint["num_layers"],

    dropout=checkpoint["dropout"]

).to(device)

# ==========================================================
# Load Learned Parameters
# ==========================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model Restored Successfully.")

# ==========================================================
# Graph Summary
# ==========================================================

print("\n==============================================")
print("GRAPH SUMMARY")
print("==============================================")

print(f"Article Nodes   : {data['article'].num_nodes}")
print(f"Topic Nodes     : {data['topic'].num_nodes}")
print(f"Time Nodes      : {data['time'].num_nodes}")
print(f"Location Nodes  : {data['location'].num_nodes}")
print(f"Memory Nodes    : {data['memory'].num_nodes}")

print("\nFeature Dimensions")

print(f"Article Features : {tuple(data['article'].x.shape)}")
print(f"Memory Features  : {tuple(data['memory'].x.shape)}")

print("\n==============================================")
print("MODEL SUMMARY")
print("==============================================")

print(f"Hidden Dimension : {checkpoint['hidden_dim']}")
print(f"Attention Heads  : {checkpoint['num_heads']}")
print(f"HGT Layers       : {checkpoint['num_layers']}")
print(f"Dropout          : {checkpoint['dropout']}")

print("\nModel Ready for ZMM Generation.")



# ==========================================================
# Generate Graph-Enhanced Memory Embeddings (ZMM)
# ==========================================================

print("\nGenerating Graph-Enhanced Memory Embeddings (ZMM)...")

with torch.no_grad():

    # ---------------------------------------------
    # Forward Pass
    # ---------------------------------------------

    node_embeddings = model(data)

    # ---------------------------------------------
    # Extract Memory Embeddings
    # ---------------------------------------------

    zmm = node_embeddings["memory"]

print("ZMM Generated Successfully.")

# ==========================================================
# Display ZMM Information
# ==========================================================

print("\n==============================================")
print("ZMM SUMMARY")
print("==============================================")

print(f"Memory Embedding Shape : {tuple(zmm.shape)}")

print(f"Embedding Dimension    : {zmm.shape[1]}")

print(f"Memory Nodes           : {zmm.shape[0]}")

print("\nEmbedding Statistics")

print(f"Minimum Value : {zmm.min().item():.6f}")

print(f"Maximum Value : {zmm.max().item():.6f}")

print(f"Mean Value    : {zmm.mean().item():.6f}")

print(f"Std Deviation : {zmm.std().item():.6f}")

# ==========================================================
# Move to CPU
# ==========================================================

zmm = zmm.cpu()



# ==========================================================
# Save ZMM (.pt)
# ==========================================================

print("\nSaving ZMM...")

torch.save(

    zmm,

    ZMM_PT_FILE

)

print(f"PyTorch File Saved : {ZMM_PT_FILE}")

# ==========================================================
# Save ZMM (.csv)
# ==========================================================

print("\nCreating CSV...")

embedding_dimension = zmm.shape[1]

column_names = [

    f"dim_{i}"

    for i in range(embedding_dimension)

]

zmm_df = pd.DataFrame(

    zmm.numpy(),

    columns=column_names

)

zmm_df.insert(

    0,

    "memory_index",

    range(len(zmm_df))

)

zmm_df.to_csv(

    ZMM_CSV_FILE,

    index=False

)

print(f"CSV File Saved : {ZMM_CSV_FILE}")

# ==========================================================
# Save Statistics
# ==========================================================

statistics = {

    "Split": "Training",

    "memory_nodes": int(zmm.shape[0]),

    "embedding_dimension": int(zmm.shape[1]),

    "minimum": float(zmm.min()),

    "maximum": float(zmm.max()),

    "mean": float(zmm.mean()),

    "std": float(zmm.std())

}

with open(

    STATISTICS_FILE,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        statistics,

        f,

        indent=4

    )

print(f"Statistics Saved : {STATISTICS_FILE}")


# ==========================================================
# Preview
# ==========================================================

print("\n==============================================")
print("ZMM PREVIEW")
print("Split : Training")
print("==============================================")

print(

    zmm_df.head()

)