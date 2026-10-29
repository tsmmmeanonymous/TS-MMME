#!/usr/bin/env python3
"""Zero-shot DeepSeek-R1-Distill-Qwen-7B (4-bit, MLX) on the text of the 2024 test articles (Table 5).

The answer is prefilled with an empty reasoning block and the start of the JSON object, so that the model gives the
label directly. The per-article latency is stored with each answer.

Usage:  python llm_baselines/llm_local_deepseek.py [--limit N]
Requires: pip install -r requirements-llm.txt (mlx-lm)
"""
import argparse
import time

from mlx_lm import generate, load

import common

MODEL_REPO = "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"
NAME = "deepseek-r1-7b"


def run(limit=None):
    out = common.output_file(NAME, "T")
    done = common.done_ids(out)
    todo = [x for x in common.articles() if str(x["id"]) not in done][: limit or None]
    model, tokenizer = load(MODEL_REPO)
    print(f"{NAME} T: {len(done)} done, {len(todo)} to go", flush=True)
    for n, x in enumerate(todo, 1):
        user = common.text_prompt(x).replace("Answer with the JSON object only.",
                                              'Answer with a JSON object only: {"label": "real"} or {"label": "fake"}.')
        prompt = f'<｜User｜>{user}<｜Assistant｜><think>\n</think>\n{{"label": "'
        t = time.time()
        try:
            text = '{"label": "' + generate(model, tokenizer, prompt=prompt, max_tokens=15, verbose=False)
        except Exception as e:
            text = f"ERROR: {e}"
        common.append(out, common.record(x, common.label_of(text), text, latency=time.time() - t))
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    run(ap.parse_args().limit)
