#!/usr/bin/env python3
"""Zero-shot LLM baselines through hosted APIs (Table 5): Qwen3.8-27B and gpt-oss-120b on Groq, Sarvam-105B on Sarvam.

Answers are appended to results/llm/<model>__<mode>.jsonl, so a run can be stopped and resumed at any time.
When several Groq keys are given, each request goes to the least-used key that is not rate-limited; on a 429 answer
the key cools down for the time the provider reports.

Usage:
  python llm_baselines/llm_api.py --model sarvam-105b --mode T
  python llm_baselines/llm_api.py --model qwen/qwen3.8-27b --mode T --limit 10
"""
import argparse
import base64
import io
import os
import re
import time

import requests
from PIL import Image

import common

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_KEYS"),
    "sarvam": ("https://api.sarvam.ai/v1/chat/completions", "SARVAM_KEY"),
}
VISION_MODELS = {"qwen/qwen3.8-27b"}
IMG_SIDE = 512


class KeyRouter:
    """Least-used key that is not cooling down; waits for the earliest reset when all keys are limited."""

    def __init__(self, keys):
        self.keys = keys
        self.cool_until = [0.0] * len(keys)
        self.tokens = [0] * len(keys)

    def pick(self):
        while True:
            now = time.time()
            free = [i for i in range(len(self.keys)) if self.cool_until[i] <= now]
            if free:
                return min(free, key=lambda i: self.tokens[i])
            time.sleep(max(1.0, min(self.cool_until) - now + 1))

    def cool(self, i, seconds):
        self.cool_until[i] = time.time() + seconds


def wait_seconds(resp):
    """Wait time after a 429: retry-after header, else the 'try again in ...' text, else 60 s."""
    h = resp.headers.get("retry-after")
    if h:
        try:
            return float(h) + 1
        except ValueError:
            pass
    m = re.search(r"try again in ((?:\d+h)?(?:\d+m)?(?:[\d.]+s)?)", resp.text)
    if m:
        return sum(float(v) * {"h": 3600, "m": 60, "s": 1}[u] for v, u in re.findall(r"([\d.]+)([hms])", m.group(1))) + 1
    return 3600 if re.search(r"per ?day|daily", resp.text, re.I) else 60


def image_data_url(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((IMG_SIDE, IMG_SIDE))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def messages(x, mode):
    user = common.text_prompt(x)
    if mode == "TI" and x.get("image_path") and os.path.exists(x["image_path"]):
        try:
            content = [{"type": "text", "text": user + "\nThe image attached to the article is provided."},
                       {"type": "image_url", "image_url": {"url": image_data_url(x["image_path"])}}]
            return [{"role": "system", "content": common.SYSTEM}, {"role": "user", "content": content}], True
        except OSError:
            pass                                    # unreadable image: text only, flagged in the output
    return [{"role": "system", "content": common.SYSTEM}, {"role": "user", "content": user}], False


def request_body(model, msgs):
    body = {"model": model, "messages": msgs, "temperature": 0, "max_completion_tokens": 400}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    elif model.startswith("qwen/"):
        body["reasoning_effort"] = "none"
    elif model.startswith("sarvam"):
        body["reasoning_effort"] = None
        body["max_tokens"] = body.pop("max_completion_tokens")
    return body


def run(model, mode, limit=None):
    provider = "sarvam" if model.startswith("sarvam") else "groq"
    url, key_var = PROVIDERS[provider]
    router = KeyRouter(common.load_keys(key_var))
    out = common.output_file(model, mode)
    done = common.done_ids(out)
    todo = [x for x in common.articles() if str(x["id"]) not in done][: limit or None]
    print(f"{model} {mode}: {len(done)} done, {len(todo)} to go", flush=True)
    for n, x in enumerate(todo, 1):
        msgs, image_used = messages(x, mode)
        payload, tries = request_body(model, msgs), 0
        while True:
            i = router.pick()
            tries += 1
            headers = ({"api-subscription-key": router.keys[i]} if provider == "sarvam"
                       else {"Authorization": f"Bearer {router.keys[i]}"})
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=120)
            except requests.RequestException:
                time.sleep(5)
                continue
            if r.status_code == 200:
                data = r.json()
                text = data["choices"][0]["message"].get("content", "")
                tokens = data.get("usage", {}).get("total_tokens", 0)
                router.tokens[i] += tokens
                break
            if r.status_code == 429:
                router.cool(i, wait_seconds(r))
                continue
            if r.status_code == 400 and "reasoning" in r.text.lower() and "reasoning_effort" in payload:
                payload.pop("reasoning_effort")
                continue
            if r.status_code in (500, 502, 503) and tries < 6:
                time.sleep(5 * tries)
                continue
            text, tokens = f"ERROR {r.status_code}: {r.text[:200]}", 0
            break
        common.append(out, common.record(x, common.label_of(text), text, image_used, tokens))
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="qwen/qwen3.8-27b | openai/gpt-oss-120b | sarvam-105b")
    ap.add_argument("--mode", choices=["T", "TI"], default="T")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.mode == "TI" and a.model not in VISION_MODELS:
        raise SystemExit(f"{a.model} does not accept images")
    run(a.model, a.mode, a.limit)
