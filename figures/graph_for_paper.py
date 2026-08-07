import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D


RUN = Path(__file__).resolve().parent.parent / "code for pheme" / "MME_TSE"

TRAINING = RUN / "Graph" / "Experiment" / "Training"

HERE = Path(__file__).resolve().parent

CHAINS = [
    ["M362", "M609"],
    ["M1302", "M1351"],
]

ARTICLES_PER_MEMORY = 2

COLUMN = 3.0

CHAIN_GAP = 4.0

COLORS = {
    "article": "#4472C4",
    "topic": "#C00000",
    "time": "#ED7D31",
    "location": "#2E8B57",
    "memory": "#663399",
}

ROW = {
    "article": 4.0,
    "topic": 3.0,
    "location": 2.0,
    "time": 1.0,
    "memory": 0.0,
}

EDGE_STYLE = {
    "belongs_to": ("#C00000", "solid", 1.2),
    "occurs_in": ("#ED7D31", "solid", 1.2),
    "located_in": ("#2E8B57", "solid", 1.2),
    "stored_in": ("#663399", "solid", 1.2),
    "evolves_to": ("black", "dashed", 2.6),
}


def load():
    articles = pd.read_csv(TRAINING / "Nodes" / "article_nodes.csv")
    memories = pd.read_csv(TRAINING / "Memory" / "memory_nodes.csv")
    meta = json.loads(
        (RUN / "Resources" / "Topic" / "topic_metadata.json").read_text()
    )
    return articles, memories.set_index("memory_id"), meta


def build(show_label):
    articles, memories, meta = load()

    G = nx.DiGraph()
    position = {}
    labels = {}
    time_members = {}

    def add(node, kind, label, x):
        if node not in position:
            G.add_node(node, kind=kind)
            position[node] = (x, ROW[kind])
            labels[node] = label

    offset = 0.0

    for chain in CHAINS:
        chain = sorted(chain, key=lambda m: memories.loc[m].year_month)

        topic_id = int(memories.loc[chain[0]].topic)
        location = str(memories.loc[chain[0]].location)

        topic_node = f"TOPIC_{topic_id}"
        location_node = f"LOC_{location}"

        for column, memory_id in enumerate(chain):
            row = memories.loc[memory_id]
            x = offset + column * COLUMN

            add(memory_id, "memory", f"$M_{{k,\\tau}}$\n{row.year_month}", x)

            indices = [
                int(v) for v in str(row.article_indices).split(",")
            ][:ARTICLES_PER_MEMORY]

            for slot, index in enumerate(indices):
                article = articles.iloc[index]
                node = f"A{index}"
                spread = (slot - (len(indices) - 1) / 2) * 1.2
                text = (
                    f"Article\n({article.label})" if show_label else "Article"
                )
                add(node, "article", text, x + spread)

                time_node = f"TIME_{row.year_month}"
                time_members.setdefault(time_node, []).append(x + spread)
                if time_node not in position:
                    add(time_node, "time", row.year_month, x + spread)

                G.add_edge(node, topic_node, relation="belongs_to")
                G.add_edge(node, time_node, relation="occurs_in")
                G.add_edge(node, location_node, relation="located_in")
                G.add_edge(node, memory_id, relation="stored_in")

        for a, b in zip(chain, chain[1:]):
            G.add_edge(a, b, relation="evolves_to")

        centre = offset + (len(chain) - 1) * COLUMN / 2
        add(topic_node, "topic", f"Topic {topic_id}", centre)
        add(location_node, "location", location, centre)

        offset += (len(chain) - 1) * COLUMN + CHAIN_GAP

    for node, xs in time_members.items():
        position[node] = (sum(xs) / len(xs), ROW["time"])

    return G, position, labels


def draw(G, position, labels, output):
    figure, axis = plt.subplots(figsize=(15, 8.5))

    for relation, (color, style, width) in EDGE_STYLE.items():
        edges = [e for e in G.edges if G.edges[e]["relation"] == relation]
        if not edges:
            continue
        nx.draw_networkx_edges(
            G, position, edgelist=edges, edge_color=color, style=style,
            width=width, arrowsize=22 if relation == "evolves_to" else 11,
            alpha=0.85, ax=axis, node_size=4000,
        )

    for kind, color in COLORS.items():
        nodes = [n for n in G.nodes if G.nodes[n]["kind"] == kind]
        nx.draw_networkx_nodes(
            G, position, nodelist=nodes, node_color=color,
            node_size=4000, edgecolors="white", linewidths=2.5, ax=axis,
        )

    nx.draw_networkx_labels(
        G, position, labels=labels,
        font_size=8.5, font_color="white", font_weight="bold", ax=axis,
    )

    left = min(x for x, _ in position.values())

    for kind, y in ROW.items():
        axis.text(
            left - 2.2, y, kind.capitalize(), fontsize=11, fontweight="bold",
            color=COLORS[kind], va="center", ha="right",
        )

    handles = [
        Line2D([0], [0], color=color, linestyle=style, lw=2,
               label=relation.replace("_", " "))
        for relation, (color, style, _) in EDGE_STYLE.items()
    ]
    axis.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.09),
        ncol=5, frameon=False, fontsize=10,
    )

    right = max(x for x, _ in position.values())
    axis.set_xlim(left - 3.2, right + 1.4)
    axis.set_ylim(-0.7, 4.7)
    axis.set_axis_off()
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"{output.name:42s} nodes {G.number_of_nodes():3d}  edges {G.number_of_edges():3d}")


if __name__ == "__main__":
    for show_label, name in [
        (True, "heterogeneous_graph_with_labels.png"),
        (False, "heterogeneous_graph_without_labels.png"),
    ]:
        G, position, labels = build(show_label)
        draw(G, position, labels, HERE / name)
