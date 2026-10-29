#!/usr/bin/env python3
"""Zero-shot small multimodal LLMs run locally with MLX on Apple silicon (Table 5): Gemma 4 E4B and Qwen3.5-4B,
4-bit quantised, in the text (T), image-only (I) and text+image (TI) settings.

The per-article latency (prompt construction to answer, model loading excluded) is stored with each answer.
In the image-only setting, articles without a readable image are counted as real, as for TS-MMME_I.

Usage:
  python llm_baselines/llm_local.py --model gemma-4-e4b --mode TI
  python llm_baselines/llm_local.py --model qwen3.5-4b --mode I --limit 10
Requires: pip install -r requirements-llm.txt (mlx-vlm)
"""
import argparse
import os
import time

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

import common

MODELS = {"gemma-4-e4b": "mlx-community/gemma-4-e4b-it-4bit", "qwen3.5-4b": "mlx-community/Qwen3.5-4B-MLX-4bit"}


def answer(model, proc, cfg, prompt, image):
    fp = apply_chat_template(proc, cfg, prompt, num_images=1 if image else 0)
    res = generate(model, proc, fp, [image] if image else None, max_tokens=24, temperature=0.0, verbose=False)
    return getattr(res, "text", res)


def run(name, mode, limit=None):
    out = common.output_file(name, mode)
    done = common.done_ids(out)
    todo = [x for x in common.articles() if str(x["id"]) not in done][: limit or None]
    model, proc = load(MODELS[name])
    cfg = load_config(MODELS[name])
    print(f"{name} {mode}: {len(done)} done, {len(todo)} to go", flush=True)
    for n, x in enumerate(todo, 1):
        has_image = bool(x.get("image_path")) and os.path.exists(x["image_path"])
        if mode == "I" and not has_image:
            common.append(out, common.record(x, "real", "NO_IMAGE (counted as real)"))
            continue
        image = x["image_path"] if mode in ("I", "TI") and has_image else None
        prompt = common.SYSTEM + "\n\n" + (common.image_only_prompt(x) if mode == "I" else common.text_prompt(x))
        t = time.time()
        try:
            text = answer(model, proc, cfg, prompt, image)
        except Exception:                           # unreadable image
            if mode == "I":
                text, image = "NO_IMAGE (unreadable, counted as real)", None
            else:
                image = None
                text = answer(model, proc, cfg, common.SYSTEM + "\n\n" + common.text_prompt(x), None)
        common.append(out, common.record(x, common.label_of(text), text, bool(image), latency=time.time() - t))
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--mode", choices=["T", "I", "TI"], required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.model, a.mode, a.limit)
