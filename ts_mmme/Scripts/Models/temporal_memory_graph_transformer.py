import torch
import torch.nn as nn

from torch_geometric.nn import HGTConv

class FeedForward(nn.Module):
    """
    Transformer Feed Forward Network.
    """

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class TemporalMemoryGraphTransformer(nn.Module):
    """
    Heterogeneous Temporal Memory Graph Transformer

    Input:
        HeteroData

    Output:
        Updated heterogeneous node embeddings.
        Memory embeddings are used as ZMM.
    """

    def __init__(
        self,
        metadata,
        num_topics,
        num_times,
        num_locations,
        article_feature_dim=1280,
        memory_feature_dim=1280,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    ):

        super().__init__()

        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # ==================================================
        # Input Projection
        # ==================================================

        self.article_projection = nn.Linear(
            article_feature_dim,
            hidden_dim
        )

        self.memory_projection = nn.Linear(
            memory_feature_dim,
            hidden_dim
        )

        # ==================================================
        # Structural Node Embeddings
        # ==================================================

        self.topic_embedding = nn.Embedding(
            num_topics,
            hidden_dim
        )

        self.time_embedding = nn.Embedding(
            num_times,
            hidden_dim
        )

        self.location_embedding = nn.Embedding(
            num_locations,
            hidden_dim
        )

        # ==================================================
        # HGT Layers
        # ==================================================

        self.layers = nn.ModuleList()

        self.pre_norms = nn.ModuleList()
        self.post_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for _ in range(num_layers):

            self.layers.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads
                )
            )

            self.pre_norms.append(
                nn.LayerNorm(hidden_dim)
            )

            self.post_norms.append(
                nn.LayerNorm(hidden_dim)
            )

            self.ffns.append(
                FeedForward(
                    hidden_dim,
                    dropout
                )
            )

        # ==================================================
        # Dropout
        # ==================================================

        self.dropout_layer = nn.Dropout(
            dropout
        )
        self._init_weights()

    def _init_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, data):

        device = data["memory"].x.device

        x_dict = {}

        # --------------------------------------------------
        # Article Nodes
        # --------------------------------------------------

        x_dict["article"] = self.article_projection(
            data["article"].x
        )

        # --------------------------------------------------
        # Memory Nodes
        # --------------------------------------------------

        x_dict["memory"] = self.memory_projection(
            data["memory"].x
        )

        # --------------------------------------------------
        # Topic Nodes
        # --------------------------------------------------

        topic_index = torch.arange(
            data["topic"].num_nodes,
            device=device
        )

        x_dict["topic"] = self.topic_embedding(
            topic_index
        )

        # --------------------------------------------------
        # Time Nodes
        # --------------------------------------------------

        time_index = torch.arange(
            data["time"].num_nodes,
            device=device
        )

        x_dict["time"] = self.time_embedding(
            time_index
        )

        # --------------------------------------------------
        # Location Nodes
        # --------------------------------------------------

        location_index = torch.arange(
            data["location"].num_nodes,
            device=device
        )

        x_dict["location"] = self.location_embedding(
            location_index
        )

        # ==================================================
        # Graph Transformer
        # ==================================================

        for layer, pre_norm, post_norm, ffn in zip(
            self.layers,
            self.pre_norms,
            self.post_norms,
            self.ffns
        ):

            # -------------------------------
            # Pre-LayerNorm
            # -------------------------------
            normalized_dict = {
                node_type: pre_norm(x)
                for node_type, x in x_dict.items()
            }

            # -------------------------------
            # HGT
            # -------------------------------
            updated_dict = layer(
                normalized_dict,
                data.edge_index_dict
            )

            # -------------------------------
            # Residual + FFN
            # -------------------------------
            for node_type, embedding in updated_dict.items():

                # First residual
                embedding = x_dict[node_type] + self.dropout_layer(embedding)

                # FFN with second residual
                embedding = embedding + ffn(post_norm(embedding))

                x_dict[node_type] = embedding

        return x_dict

    # ======================================================
    # Utility
    # ======================================================

    def get_memory_embeddings(self, data):
        """
        Returns graph-enhanced memory embeddings (ZMM).
        """

        outputs = self.forward(data)

        return outputs["memory"]

    def get_article_embeddings(self, data):
        """
        Returns graph-enhanced article embeddings.
        """

        outputs = self.forward(data)

        return outputs["article"]