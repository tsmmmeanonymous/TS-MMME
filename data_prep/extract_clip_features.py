#!/usr/bin/env python3
"""Writes ts_mmme/Resources/Cache/clip_image_features.pt (TS-MMME article id -> 512-d CLIP image feature, train + val + test),
used by the TS-MMME_I and TS-MMME_TI_ftclip variants."""
import os, json, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); M = f"{ROOT}/ts_mmme"
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

load = lambda f: [json.loads(l) for l in open(f"{M}/Dataset/Experiment/{f}.json") if l.strip()]
tr = load("train_dataset"); va = load("val_dataset"); te = {k: load(f"test_2024_Q{k}") for k in range(1, 5)}

# ---- CLIP features: train from Phase-1 features, val/test computed the same way (step7) ----
clip_tr = torch.load(f"{M}/Graph/Experiment/Training/Features/article_image.pt", map_location="cpu")
assert clip_tr.shape[0] == len(tr)
feats = {x["id"]: clip_tr[i].clone() for i, x in enumerate(tr) if clip_tr[i].abs().sum() > 0}
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"); clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
for x in va + [a for k in te for a in te[k]]:
    if not x["image_path"]: continue
    try:
        im = Image.open(x["image_path"]).convert("RGB")
        with torch.no_grad():
            f = clip.get_image_features(**proc(images=im, return_tensors="pt"))
            f = f if isinstance(f, torch.Tensor) else f.pooler_output
        feats[x["id"]] = torch.nn.functional.normalize(f, p=2, dim=1).squeeze(0)
    except Exception:
        pass
torch.save(feats, f"{M}/Resources/Cache/clip_image_features.pt")
print("clip features:", len(feats), "| train", sum(x["id"] in feats for x in tr), "val", sum(x["id"] in feats for x in va),
      "test", sum(a["id"] in feats for k in te for a in te[k]), flush=True)
