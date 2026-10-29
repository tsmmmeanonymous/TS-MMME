"""CLIP image projection. Active when IMG_CLIP_PROJ=1 (image-only variant I and the TI_ftclip variant).
The article is represented by its caption embedding plus a trained linear projection of its CLIP image
feature (512 -> 768). Retrieval is unchanged (caption embedding); the projected feature is added to the
query that enters cross-attention. Variants TI, T and T_ft do not set the flag and are unaffected."""
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ENABLED = os.environ.get("IMG_CLIP_PROJ") == "1"
ROOT = Path(__file__).resolve().parents[2]
FEATURE_FILE = ROOT / "Resources" / "Cache" / "clip_image_features.pt"
CHECKPOINT = ROOT / "Graph" / "Experiment" / "InferenceModels" / "best_inference_model.pt"
_feats = None
_trained = None


class ImageProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(512, 768)

    def forward(self, x):
        return self.proj(x)


def feats():
    global _feats
    if _feats is None:
        _feats = torch.load(FEATURE_FILE, map_location="cpu")
        print(f"CLIP image features loaded: {len(_feats)} articles")
    return _feats


def has_image(article_id):
    try:
        return int(article_id) in feats()
    except (TypeError, ValueError):
        return False


def apply(article_embedding, article_id, proj):
    """article_embedding (768,) + proj(CLIP feature); unchanged if the article has no readable image."""
    if not has_image(article_id):
        return article_embedding
    f = feats()[int(article_id)].to(article_embedding.device, dtype=torch.float32)
    return F.normalize(article_embedding + proj(f), dim=0)


def load_trained(device):
    global _trained
    if _trained is None:
        ck = torch.load(CHECKPOINT, map_location=device)
        _trained = ImageProj().to(device)
        _trained.load_state_dict(ck["image_proj_state_dict"])
        _trained.eval()
    return _trained
