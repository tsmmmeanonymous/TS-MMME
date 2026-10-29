from pathlib import Path
import json
import random

import pandas as pd

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

EDGE_DIR = GRAPH_DIR / "Edges"

MEMORY_DIR = GRAPH_DIR / "Memory"

MODEL_DIR = GRAPH_DIR / "Models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# Hyperparameters
# ==========================================================

HIDDEN_DIM = 768

NUM_HEADS = 4

NUM_LAYERS = 2

DROPOUT = 0.2

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

TEMPERATURE = 0.1

EPOCHS = 100

EARLY_STOPPING_PATIENCE = 15

RANDOM_SEED = 42

# ==========================================================
# Reproducibility
# ==========================================================

random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

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
print("TEMPORAL MEMORY GRAPH TRANSFORMER TRAINING")
print("==============================================")

print(f"Using Device : {device}")

if device.type == "cuda":
    print("Running on NVIDIA GPU")
elif device.type == "mps":
    print("Running on Apple Silicon GPU (MPS)")
else:
    print("Running on CPU")





## part2


# ==========================================================
# Load Heterogeneous Graph
# ==========================================================

print("\nLoading Heterogeneous Graph...")

heterodata_file = HETERO_DIR / "heterodata.pt"

data = torch.load(
    heterodata_file,
    map_location=device,
    weights_only=False
)

data = data.to(device)

print("Graph Loaded Successfully.")



print(data.metadata())

print(data.node_types)

print(data.edge_types)


# ==========================================================
# Load Memory Nodes
# ==========================================================

print("\nLoading Memory Nodes...")

memory_df = pd.read_csv(
    MEMORY_DIR / "memory_nodes.csv"
)

print(f"Memory Nodes : {len(memory_df)}")

# ==========================================================
# Load Memory-Memory Edges
# ==========================================================

print("\nLoading Temporal Memory Edges...")

memory_edges_df = pd.read_csv(
    EDGE_DIR / "memory_memory_edges.csv"
)

print(f"Temporal Memory Edges : {len(memory_edges_df)}")

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

# ==========================================================
# Build Graph Transformer
# ==========================================================

print("\nBuilding Temporal Memory Graph Transformer...")

model = TemporalMemoryGraphTransformer(

    metadata=data.metadata(),

    num_topics=data["topic"].num_nodes,

    num_times=data["time"].num_nodes,

    num_locations=data["location"].num_nodes,

    article_feature_dim=data["article"].x.shape[1],

    memory_feature_dim=data["memory"].x.shape[1],

    hidden_dim=HIDDEN_DIM,

    num_heads=NUM_HEADS,

    num_layers=NUM_LAYERS,

    dropout=DROPOUT

).to(device)

print("Model Created Successfully.")

# ==========================================================
# Optimizer
# ==========================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999)
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

print("Optimizer Created Successfully.")

# ==========================================================
# Model Summary
# ==========================================================

total_parameters = sum(

    p.numel()

    for p in model.parameters()

)

trainable_parameters = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad

)

print("\n==============================================")
print("MODEL SUMMARY")
print("==============================================")

print(f"Hidden Dimension      : {HIDDEN_DIM}")
print(f"Attention Heads       : {NUM_HEADS}")
print(f"HGT Layers            : {NUM_LAYERS}")
print(f"Dropout               : {DROPOUT}")

print(f"\nTotal Parameters      : {total_parameters:,}")
print(f"Trainable Parameters  : {trainable_parameters:,}")

print("\nModel Ready.")


# part3

# ==========================================================
# Create Positive Pairs
# ==========================================================

print("\nCreating Positive Pairs...")

positive_pairs = []

positive_set = set()

for _, row in memory_edges_df.iterrows():

    source = int(row["source_index"])

    target = int(row["target_index"])

    positive_pairs.append(
        (
            source,
            target
        )
    )

    positive_set.add(
        (
            source,
            target
        )
    )

print(f"Positive Pairs : {len(positive_pairs)}")

# ==========================================================
# Memory Metadata
# ==========================================================

memory_metadata = {}

for _, row in memory_df.iterrows():

    memory_metadata[
        int(row["memory_index"])
    ] = {

        "topic": int(row["topic"]),

        "location": row["location"]

    }

# ==========================================================
# Create Hard Negative Pairs
# ==========================================================

print("\nCreating Hard Negative Pairs...")

num_memory_nodes = data["memory"].num_nodes

hard_negative_pairs = []

for source, target in positive_pairs:

    source_topic = memory_metadata[source]["topic"]

    source_location = memory_metadata[source]["location"]

    while True:

        candidate = random.randint(
            0,
            num_memory_nodes - 1
        )

        if candidate == source:
            continue

        if (source, candidate) in positive_set:
            continue

        if (candidate, source) in positive_set:
            continue

        candidate_topic = memory_metadata[candidate]["topic"]

        candidate_location = memory_metadata[candidate]["location"]

        if (

            candidate_topic != source_topic

            or

            candidate_location != source_location

        ):

            hard_negative_pairs.append(

                (
                    source,
                    candidate
                )

            )

            break

print(f"Hard Negative Pairs : {len(hard_negative_pairs)}")

# ==========================================================
# Pair Statistics
# ==========================================================

print("\n==============================================")
print("PAIR STATISTICS")
print("==============================================")

print(f"Positive Pairs : {len(positive_pairs)}")

print(f"Negative Pairs : {len(hard_negative_pairs)}")

print("\nSample Positive Pairs")

for pair in positive_pairs[:10]:

    print(pair)

print("\nSample Negative Pairs")

for pair in hard_negative_pairs[:10]:

    print(pair)


# part4 

# ==========================================================
# Contrastive Loss
# ==========================================================

print("\nBuilding Contrastive Loss...")

def contrastive_loss(
    embeddings,
    positive_pairs,
    negative_pairs,
    temperature=0.1
):
    """
    Vectorized InfoNCE Loss.
    """

    device = embeddings.device

    anchor_idx = torch.tensor(
        [p[0] for p in positive_pairs],
        device=device
    )

    positive_idx = torch.tensor(
        [p[1] for p in positive_pairs],
        device=device
    )

    negative_idx = torch.tensor(
        [n[1] for n in negative_pairs],
        device=device
    )

    anchor = F.normalize(
        embeddings[anchor_idx],
        dim=1
    )

    positive = F.normalize(
        embeddings[positive_idx],
        dim=1
    )

    negative = F.normalize(
        embeddings[negative_idx],
        dim=1
    )

    positive_logits = torch.sum(
        anchor * positive,
        dim=1,
        keepdim=True
    )

    negative_logits = torch.sum(
        anchor * negative,
        dim=1,
        keepdim=True
    )

    logits = torch.cat(
        [
            positive_logits,
            negative_logits
        ],
        dim=1
    )

    logits = logits / temperature

    labels = torch.zeros(
        logits.size(0),
        dtype=torch.long,
        device=device
    )

    return F.cross_entropy(
        logits,
        labels
    )

print("Contrastive Loss Ready.")



#part 5

# ==========================================================
# Train Graph Transformer
# ==========================================================

print("\n==============================================")
print("TRAINING GRAPH TRANSFORMER")
print("==============================================")

best_loss = float("inf")
patience_counter = 0

history = {

    "epoch": [],

    "loss": []

}

for epoch in range(EPOCHS):

    # ------------------------------------------------------
    # Train Mode
    # ------------------------------------------------------

    model.train()

    optimizer.zero_grad()

    # ------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------

    node_embeddings = model(data)

    memory_embeddings = node_embeddings["memory"]

    # ------------------------------------------------------
    # Compute Contrastive Loss
    # ------------------------------------------------------

    loss = contrastive_loss(

        embeddings=memory_embeddings,

        positive_pairs=positive_pairs,

        negative_pairs=hard_negative_pairs,

        temperature=TEMPERATURE

    )

    # ------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
    )

    optimizer.step()

    scheduler.step()

    # ------------------------------------------------------
    # Save Training History
    # ------------------------------------------------------

    history["epoch"].append(
        epoch + 1
    )

    history["loss"].append(
        float(loss.item())
    )

    # ------------------------------------------------------
    # Save Best Model
    # ------------------------------------------------------

    if loss.item() < best_loss:

        best_loss = loss.item()
        patience_counter = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "loss": best_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hidden_dim": HIDDEN_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "metadata": data.metadata(),
                "num_topics": data["topic"].num_nodes,
                "num_times": data["time"].num_nodes,
                "num_locations": data["location"].num_nodes,
                "article_feature_dim": data["article"].x.shape[1],
                "memory_feature_dim": data["memory"].x.shape[1]
            },
            MODEL_DIR / "best_graph_transformer.pth"
        )

    else:
        patience_counter += 1

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

    # ------------------------------------------------------
    # Progress
    # ------------------------------------------------------

    if (

        epoch == 0

        or

        (epoch + 1) % 10 == 0

        or

        epoch == EPOCHS - 1

    ):

        print(

            f"Epoch [{epoch+1:03d}/{EPOCHS}]"

            f" | Loss = {loss.item():.6f}"

        )

print("\nTraining Finished.")

print(f"Best Loss : {best_loss:.6f}")


# part 6


# ==========================================================
# Save Training History
# ==========================================================

history_file = MODEL_DIR / "training_history.json"

with open(

    history_file,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        history,

        f,

        indent=4

    )

print("\nTraining history saved.")

# ==========================================================
# Save Training Statistics
# ==========================================================

training_statistics = {

    "Split": "Training",

    "epochs": EPOCHS,

    "best_loss": float(best_loss),

    "learning_rate": LEARNING_RATE,

    "weight_decay": WEIGHT_DECAY,

    "temperature": TEMPERATURE,

    "hidden_dimension": HIDDEN_DIM,

    "attention_heads": NUM_HEADS,

    "hgt_layers": NUM_LAYERS,

    "dropout": DROPOUT,

    "device": str(device),

    "article_nodes": int(data["article"].num_nodes),

    "topic_nodes": int(data["topic"].num_nodes),

    "time_nodes": int(data["time"].num_nodes),

    "location_nodes": int(data["location"].num_nodes),

    "memory_nodes": int(data["memory"].num_nodes),

    "positive_pairs": len(positive_pairs),

    "negative_pairs": len(hard_negative_pairs)

}

statistics_file = MODEL_DIR / "training_statistics.json"

with open(

    statistics_file,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        training_statistics,

        f,

        indent=4

    )

print("Training statistics saved.")

# ==========================================================
# Final Summary
# ==========================================================

print("\n==============================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("Split               : Training")
print("==============================================")

print(f"Epochs              : {EPOCHS}")

print(f"Best Loss           : {best_loss:.6f}")

print(f"Memory Nodes        : {data['memory'].num_nodes}")

print(f"Positive Pairs      : {len(positive_pairs)}")

print(f"Hard Negative Pairs : {len(hard_negative_pairs)}")

print("\nSaved Files")

print("----------------------------------------------")

print(f"Best Model          : {MODEL_DIR / 'best_graph_transformer.pth'}")

print(f"Training History    : {history_file}")

print(f"Training Statistics : {statistics_file}")

print("\nNext Step")

print("----------------------------------------------")

print("Run : step13.2_generate_zmm.py")

print("\n==============================================")