"""Shared pieces of the zero-shot LLM baselines (Table 5): test data, prompts, answer parsing and output files.

All runners give each model the same input: the first 800 characters of the article text, its publication month and
location and, for the image settings, the article image. The model answers {"label": "real"} or {"label": "fake"}.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEST_DIR = REPO / "ts_mmme" / "Dataset" / "Experiment"      # written by data_prep/build_full_master.py
OUT_DIR = REPO / "results" / "llm"
ENV_FILE = REPO / ".env"
MAX_CHARS = 800

SYSTEM = ("You are a fact-checking assistant. Decide whether a news article is real or fake (misinformation). "
          "Answer with a JSON object only: {\"label\": \"real\"} or {\"label\": \"fake\"}.")


def articles():
    """The 1,061 test articles of the four 2024 quarters, each tagged with its quarter."""
    items = []
    for k in range(1, 5):
        with open(TEST_DIR / f"test_2024_Q{k}.json", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    x = json.loads(line)
                    x["quarter"] = k
                    items.append(x)
    return items


def text_prompt(x):
    return (f"Publication month: {x.get('year_month', '')}\nLocation: {x.get('location', '')}\n"
            f"Article:\n{x['content'][:MAX_CHARS]}\n\nIs this article real or fake? Answer with the JSON object only.")


def image_only_prompt(x):
    return (f"Publication month: {x.get('year_month', '')}\nLocation: {x.get('location', '')}\n"
            "The image attached to a news article is provided; the article text is not available.\n\n"
            "Is the news shown in this image real or fake? Answer with the JSON object only.")


def label_of(text):
    """'real' or 'fake' from a model answer, or None if the answer cannot be parsed."""
    t = (text or "").lower()
    m = re.search(r'"label"\s*:\s*"(real|fake)"', t)
    if m:
        return m.group(1)
    hits = re.findall(r"\b(real|fake)\b", t)
    return hits[-1] if hits else None


def output_file(model, mode):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{model.replace('/', '_')}__{mode}.jsonl"


def done_ids(path):
    """Ids already answered, so that an interrupted run resumes where it stopped."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def record(x, pred, raw, image_used=False, tokens=0, latency=None):
    rec = {"id": str(x["id"]), "quarter": x["quarter"], "gold": "fake" if x["label"] == "rumor" else "real",
           "pred": pred, "image_used": image_used, "tokens": tokens, "raw": str(raw)[:300]}
    if latency is not None:
        rec["latency_s"] = round(latency, 3)
    return rec


def append(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_keys(var):
    """Comma-separated API keys from .env (see .env.example)."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.strip().startswith(var + "="):
                keys = [k.strip() for k in line.split("=", 1)[1].split(",") if k.strip()]
                if keys:
                    return keys
    raise SystemExit(f"No keys found: set {var}=... in {ENV_FILE.name} (see .env.example)")
