import math
from pathlib import Path

import torch.nn.functional as F
import torch
import torch.nn as nn

BEST_MODEL = (
    Path(__file__).resolve().parents[2]
    / "Graph"
    / "Experiment"
    / "InferenceModels"
    / "best_inference_model.pt"
)

class CrossAttention(nn.Module):
    """
    Single-Head Cross Attention

    Query
        Incoming article embedding

    Key
        Retrieved temporal memory embedding

    Value
        Retrieved temporal memory embedding

    Output
        Context embedding
    """

    def __init__(

        self,

        article_dim=768,

        memory_dim=768,

        hidden_dim=768

    ):

        super().__init__()

        self.hidden_dim = hidden_dim

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.article_projection = nn.Linear(
            article_dim,
            hidden_dim
        )

        self.memory_projection = nn.Linear(
        memory_dim,
        hidden_dim
        )

        self.output_projection = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.LayerNorm(hidden_dim)

        )

        print("\nCross Attention Created Successfully.")

    # =====================================================
    # Forward
    # =====================================================

    def forward(
        self,
        article_embedding,
        memory_embeddings,
        retrieval_scores=None
    ):
        """
        Multi-memory cross attention.

        Parameters
        ----------
        article_embedding : Tensor
            Shape: (768,) or (B,768)

        memory_embeddings : Tensor
            Shape: (K,768) or (B,K,768)
        """

        # --------------------------------------------------
        # Batch dimensions
        # --------------------------------------------------

        if article_embedding.dim() == 1:
            article_embedding = article_embedding.unsqueeze(0)

        if memory_embeddings.dim() == 2:
            memory_embeddings = memory_embeddings.unsqueeze(0)

        # --------------------------------------------------
        # Project embeddings
        # --------------------------------------------------

        query = self.article_projection(article_embedding)
        query = query.unsqueeze(1)                     # (B,1,H)

        key_value = self.memory_projection(memory_embeddings)   # (B,K,H)

        # ------------------------------------------
        # Retrieval Score Weighting
        # ------------------------------------------

        if retrieval_scores is not None:

            if retrieval_scores.dim() == 1:
                retrieval_scores = retrieval_scores.unsqueeze(0)

            retrieval_scores = F.softmax(
                retrieval_scores,
                dim=-1
            )

            key_value = (
                key_value *
                retrieval_scores.unsqueeze(-1)
            )

        # --------------------------------------------------
        # Multi-head Cross Attention
        # --------------------------------------------------

        context, attention_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True
        )

        # --------------------------------------------------
        # Residual Connection
        # --------------------------------------------------

        context = context + query

        # --------------------------------------------------
        # Output Projection
        # --------------------------------------------------

        context = self.output_projection(context)

        context = context.squeeze(1)

        return (
            context.squeeze(0),
            attention_weights.squeeze(0)
        )
    
# =====================================================
# Display Cross Attention Result
# =====================================================

def display_attention_result(

    context_embedding,

    attention_weight

):
    """
    Display Cross Attention output.
    """

    print("\n==============================================")
    print("CROSS ATTENTION RESULT")
    print("==============================================")

    print("\nContext Embedding Summary")
    print("----------------------------------------------")

    print(f"Shape               : {tuple(context_embedding.shape)}")

    print(f"Dimension           : {context_embedding.shape[0]}")

    print(f"Minimum Value       : {context_embedding.min().item():.6f}")

    print(f"Maximum Value       : {context_embedding.max().item():.6f}")

    print(f"Mean Value          : {context_embedding.mean().item():.6f}")

    print(f"Std Deviation       : {context_embedding.std().item():.6f}")

    print("\nFirst 10 Dimensions")
    print("----------------------------------------------")

    for i in range(

        min(10, context_embedding.shape[0])

    ):

        print(

            f"Dim {i:02d} : "

            f"{context_embedding[i].item():.6f}"

        )

    print("\nAttention Weight")
    print("----------------------------------------------")

    print(attention_weight.detach().cpu().numpy())

    print("\nAttention Shape")

    print(f"Shape : {tuple(attention_weight.shape)}")

# =====================================================
# Singleton Cross Attention
# =====================================================

_cross_attention = None

def get_cross_attention():

    global _cross_attention

    if _cross_attention is None:

        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        _cross_attention = CrossAttention().to(device)

        checkpoint = torch.load(
            BEST_MODEL,
            map_location=device
        )

        _cross_attention.load_state_dict(
            checkpoint["cross_attention_state_dict"]
        )

        _cross_attention.eval()

    return _cross_attention

# =====================================================
# Public API
# =====================================================

def generate_context_embedding(
    article_embedding,
    memory_embeddings,
    retrieval_scores=None
):
    """
    Generate Context Embedding using Cross Attention.

    Parameters
    ----------
    article_embedding : Tensor

    memory_embedding : Tensor

    Returns
    -------
    dict
    """

    model = get_cross_attention()
    model.eval()

    # Convert inference tensors into normal tensors
    article_embedding = article_embedding.clone().detach()

    memory_embeddings = memory_embeddings.clone().detach()

    if retrieval_scores is not None:
        retrieval_scores = retrieval_scores.clone().detach()

    with torch.no_grad():
        context_embedding, attention_weight = model(
            article_embedding,
            memory_embeddings,
            retrieval_scores
        )

    return {

        "context_embedding": context_embedding,

        "attention_weight": attention_weight

    }

# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("CROSS ATTENTION TEST")
    print("==============================================")

    # ==================================================
    # Dummy Embeddings
    # ==================================================

    article_embedding = torch.randn(

        768

    )

    memory_embedding = torch.randn(

        768

    )

    print("\nInput Summary")
    print("----------------------------------------------")

    print(

        f"Article Embedding Shape : {tuple(article_embedding.shape)}"

    )

    print(

        f"Memory Embedding Shape  : {tuple(memory_embedding.shape)}"

    )

    # ==================================================
    # Build Model
    # ==================================================

    '''model = CrossAttention(

        article_dim=768,

        memory_dim=768,

        hidden_dim=768

    )'''

    model = get_cross_attention()

    # ==================================================
    # Forward
    # ==================================================

    print("\nRunning Cross Attention...")

    article_embedding = article_embedding.clone()
    memory_embeddings = memory_embeddings.clone()

    context_embedding, attention_weight = model(
        article_embedding,
        memory_embeddings,
        retrieval_scores
    )

    print("Cross Attention Completed.")

    # ==================================================
    # Display Results
    # ==================================================

    display_attention_result(

        context_embedding,

        attention_weight

    )

    print("\n==============================================")
    print("CROSS ATTENTION COMPLETED")
    print("==============================================")
    

