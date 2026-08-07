from pathlib import Path
import random
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPH_DIR = PROJECT_ROOT / "Graph" / "Experiment" / "Training"

HETERO_DIR = GRAPH_DIR / "Hetero"

NODE_DIR = GRAPH_DIR / "Nodes"

MEMORY_DIR = GRAPH_DIR / "Memory"

OUTPUT_DIR = GRAPH_DIR / "Visualization"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =====================================================
# Parameters
# =====================================================

NUM_SAMPLE_ARTICLES = 20

random.seed(42)

# =====================================================
# Load Graph
# =====================================================

print("Loading graph...")

data = torch.load(
    HETERO_DIR / "heterodata.pt",
    weights_only=False
)

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

print("Graph Loaded.")

# =====================================================
# Create NetworkX Graph
# =====================================================

G = nx.DiGraph()

# =====================================================
# Node Colors
# =====================================================

NODE_COLORS = {

    "article":"royalblue",

    "topic":"red",

    "time":"orange",

    "location":"green",

    "memory":"purple"

}

EDGE_COLORS = {

    "belongs_to":"gray",

    "occurs_in":"dodgerblue",

    "located_in":"green",

    "stored_in":"purple",

    "has_memory":"orange",

    "evolves_to":"red"

}

# =====================================================
# Sample Articles
# =====================================================

sample_articles = random.sample(

    list(range(data["article"].num_nodes)),

    NUM_SAMPLE_ARTICLES

)

sample_articles = set(sample_articles)

print(f"Sample Articles : {len(sample_articles)}")



# =====================================================
# Add Sample Article Nodes
# =====================================================

for idx in sample_articles:

    G.add_node(
        f"A{idx}",
        node_type="article",
        color=NODE_COLORS["article"]
    )

# =====================================================
# Article -> Topic
# =====================================================

edge = data["article", "belongs_to", "topic"].edge_index

sample_topics = set()

for src, dst in edge.t().tolist():

    if src not in sample_articles:
        continue

    topic_name = f"T{dst}"

    sample_topics.add(dst)

    G.add_node(
        topic_name,
        node_type="topic",
        color=NODE_COLORS["topic"]
    )

    G.add_edge(
        f"A{src}",
        topic_name,
        relation="belongs_to",
        color=EDGE_COLORS["belongs_to"]
    )

# =====================================================
# Article -> Time
# =====================================================

edge = data["article", "occurs_in", "time"].edge_index

sample_times = set()

for src, dst in edge.t().tolist():

    if src not in sample_articles:
        continue

    time_name = time_nodes.iloc[dst]["year_month"]

    sample_times.add(dst)

    G.add_node(
        f"TM{dst}",
        node_type="time",
        label=time_name,
        color=NODE_COLORS["time"]
    )

    G.add_edge(
        f"A{src}",
        f"TM{dst}",
        relation="occurs_in",
        color=EDGE_COLORS["occurs_in"]
    )

# =====================================================
# Article -> Location
# =====================================================

edge = data["article", "located_in", "location"].edge_index

sample_locations = set()

for src, dst in edge.t().tolist():

    if src not in sample_articles:
        continue

    location_name = location_nodes.iloc[dst]["location"]

    sample_locations.add(dst)

    G.add_node(
        f"L{dst}",
        node_type="location",
        label=location_name,
        color=NODE_COLORS["location"]
    )

    G.add_edge(
        f"A{src}",
        f"L{dst}",
        relation="located_in",
        color=EDGE_COLORS["located_in"]
    )

# =====================================================
# Article -> Memory
# =====================================================

edge = data["article", "stored_in", "memory"].edge_index

sample_memories = set()

for src, dst in edge.t().tolist():

    if src not in sample_articles:
        continue

    sample_memories.add(dst)

    G.add_node(
        f"M{dst}",
        node_type="memory",
        color=NODE_COLORS["memory"]
    )

    G.add_edge(
        f"A{src}",
        f"M{dst}",
        relation="stored_in",
        color=EDGE_COLORS["stored_in"]
    )

print()

print("Sample Graph")

print("-------------------------")

print(f"Articles  : {len(sample_articles)}")

print(f"Topics    : {len(sample_topics)}")

print(f"Times     : {len(sample_times)}")

print(f"Locations : {len(sample_locations)}")

print(f"Memories  : {len(sample_memories)}")



# =====================================================
# Topic -> Memory
# =====================================================

edge = data["topic", "has_memory", "memory"].edge_index

for src, dst in edge.t().tolist():

    if dst not in sample_memories:
        continue

    G.add_edge(
        f"T{src}",
        f"M{dst}",
        relation="has_memory",
        color=EDGE_COLORS["has_memory"]
    )

# =====================================================
# Time -> Memory
# =====================================================

edge = data["time", "has_memory", "memory"].edge_index

for src, dst in edge.t().tolist():

    if dst not in sample_memories:
        continue

    G.add_edge(
        f"TM{src}",
        f"M{dst}",
        relation="has_memory",
        color=EDGE_COLORS["has_memory"]
    )

# =====================================================
# Location -> Memory
# =====================================================

edge = data["location", "has_memory", "memory"].edge_index

for src, dst in edge.t().tolist():

    if dst not in sample_memories:
        continue

    G.add_edge(
        f"L{src}",
        f"M{dst}",
        relation="has_memory",
        color=EDGE_COLORS["has_memory"]
    )

# =====================================================
# Memory -> Memory
# =====================================================

edge = data["memory", "evolves_to", "memory"].edge_index

for src, dst in edge.t().tolist():

    if src not in sample_memories:
        continue

    if dst not in sample_memories:
        continue

    G.add_edge(
        f"M{src}",
        f"M{dst}",
        relation="evolves_to",
        color=EDGE_COLORS["evolves_to"]
    )

# =====================================================
# Draw Graph
# =====================================================

print("\nDrawing Graph...")

plt.figure(figsize=(18,16))

pos = nx.spring_layout(
    G,
    seed=42,
    k=1.2
)

node_colors = [
    G.nodes[n]["color"]
    for n in G.nodes()
]

edge_colors = [
    G.edges[e]["color"]
    for e in G.edges()
]

nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=700,
    alpha=0.95
)

nx.draw_networkx_edges(
    G,
    pos,
    edge_color=edge_colors,
    arrows=True,
    arrowsize=12,
    width=2
)

# =====================================================
# Labels
# =====================================================

labels = {}

for node in G.nodes():

    if node.startswith("A"):

        labels[node] = node

    elif node.startswith("T"):

        labels[node] = node

    elif node.startswith("TM"):

        labels[node] = G.nodes[node].get("label", node)

    elif node.startswith("L"):

        labels[node] = G.nodes[node].get("label", node)

    elif node.startswith("M"):

        labels[node] = node

nx.draw_networkx_labels(
    G,
    pos,
    labels,
    font_size=8,
    font_weight="bold"
)

# =====================================================
# Legend
# =====================================================

from matplotlib.lines import Line2D

legend = [

    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='Article',
        markerfacecolor='royalblue',
        markersize=10
    ),

    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='Topic',
        markerfacecolor='red',
        markersize=10
    ),

    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='Time',
        markerfacecolor='orange',
        markersize=10
    ),

    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='Location',
        markerfacecolor='green',
        markersize=10
    ),

    Line2D(
        [0],
        [0],
        marker='o',
        color='w',
        label='Memory',
        markerfacecolor='purple',
        markersize=10
    )

]

plt.legend(
    handles=legend,
    loc="upper left"
)

'''
plt.title(
    "Sample Heterogeneous Temporal Memory Graph",
    fontsize=18,
    weight="bold"
)

plt.axis("off")

OUTPUT_FILE = OUTPUT_DIR / "sample_heterogeneous_graph.png"

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()'''


plt.title(
    "Sample Heterogeneous Temporal Memory Graph",
    fontsize=18,
    weight="bold",
    pad=20
)

plt.axis("off")

OUTPUT_FILE = OUTPUT_DIR / "sample_heterogeneous_graph.png"

# Leave some space for the title
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.25
)

plt.show()



print("\n======================================")
print("GRAPH VISUALIZATION COMPLETED")
print("======================================")

print(f"Nodes : {G.number_of_nodes()}")
print(f"Edges : {G.number_of_edges()}")

print("\nSaved")

print(OUTPUT_FILE)


print("\nSample Memory IDs")
print(sorted(sample_memories))

edge = data["memory", "evolves_to", "memory"].edge_index

count = 0

for src, dst in edge.t().tolist():

    if src in sample_memories and dst in sample_memories:

        print(f"M{src} -> M{dst}")

        count += 1

print("\nVisible Memory→Memory Edges:", count)