#!/usr/bin/env python3
"""Fine-tuned text features for TS-MMME_T(+FT encoder): the BERT_T encoder trained on <=2023-Q4
(bench_models/bert_text_2024.pt: mBERT with trained layer 11 + MaskAttention pooling), applied to every
TS-MMME article. Output: dict {TS-MMME article id -> L2-normalised 768-d tensor}. TS-MMME ids are
rebuilt exactly as build_full_master.py assigns them (train, val, then 2024 Q1..Q4, from bench_common.splits)."""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F
from transformers import BertModel, BertTokenizer
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "baselines"))
import bench_common as C

BERTP = "bert-base-multilingual-cased"; MAXLEN = 170; BATCH = 64
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = os.path.join(REPO, "ts_mmme", "Resources", "Cache", "ft_text_features.pt")


class MaskAtt(nn.Module):
    def __init__(s): super().__init__(); s.a = nn.Linear(768, 1)
    def forward(s, h, m):
        sc = s.a(h).view(-1, h.size(1)).masked_fill(m == 0, float("-inf"))
        return torch.matmul(torch.softmax(sc, -1).unsqueeze(1), h).squeeze(1)


bert = BertModel.from_pretrained(BERTP, attn_implementation="eager"); att = MaskAtt()
sd = torch.load(os.path.join(C.ROOT, "bench_models", "bert_text_2024.pt"), map_location="cpu")
mb = bert.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("bert.")}, strict=False)
assert not mb.unexpected_keys and any(".layer.11." in k for k in sd), "fine-tuned layer 11 missing"
att.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("att.")})
bert.to(DEV).eval(); att.to(DEV).eval()

R = C.load_rows(); tr, va, te = C.splits(R)
order = tr + va + [i for k in range(1, 5) for i in te[k]]          # TS-MMME id = position + 1
tok = BertTokenizer.from_pretrained(BERTP); feats = {}
with torch.no_grad():
    for s in range(0, len(order), BATCH):
        b = order[s:s+BATCH]
        ids = torch.tensor([tok.encode(R[i]["text"], max_length=MAXLEN, padding="max_length", truncation=True) for i in b])
        m = (ids != tok.pad_token_id).long()
        f = F.normalize(att(bert(ids.to(DEV), attention_mask=m.to(DEV))[0], m.to(DEV)), dim=1).cpu()
        for j, v in enumerate(f): feats[s + j + 1] = v.clone()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save(feats, OUT); print("saved", OUT, len(feats))
