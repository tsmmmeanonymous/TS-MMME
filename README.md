# TS-MMME

Code, predictions and evaluation scripts for **TS-MMME**, a temporal–spatial multimodal memory-enhanced model for misinformation detection, evaluated on MMM-Trend with a temporal split (train ≤ 2023-Q4, test on each 2024 quarter, metrics averaged over the four quarters).

## Repository layout

```
ts_mmme/            model code
  Scripts/          Phase 1 (graph + memory construction), Phase 2 (training), Phase 3 (inference), step scripts
  Resources/Topic   topic taxonomy and metadata
data_prep/          builds train/val/test files from MMM-Trend; fine-tuned text and CLIP image feature caches
baselines/          BERT_T, EANN_T, FTT_T, CLIP/BLIP/Qwen image and vision-language baselines
gating/             image-gated TS-MMME_TI (Table 4): make_gated.py + gate_ids.json
llm_baselines/      zero-shot LLM baselines (Groq / Sarvam API and local MLX models)
other_datasets/     TS-MMME on Weibo, FakeNewsNet, PHEME, Fakeddit
run/                end-to-end shell scripts
results/            predictions and metrics reported in the paper (summary.csv: Tables 3, 4, 6)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-llm.txt   # only for llm_baselines/
```

Download MMM-Trend and set `MMMTREND_DATA` to the folder containing `complete_dataset_v1.2.xlsx`, the article texts and images:

```bash
export MMMTREND_DATA=/path/to/mmm-trend
```

## Reproducing the results

```bash
run/01_prepare_data.sh            # train / val / test files (seed 2378, 10% validation)
run/04_baselines.sh               # baselines + fine-tuned text / CLIP feature caches
run/02_phase1.sh                  # heterogeneous graph, memory cells, Graph Transformer, memory bank
run/03_ts_mmme_variant.sh TI      # full model and ablations
run/03_ts_mmme_variant.sh T_ft    # text only (fine-tuned encoder)
run/03_ts_mmme_variant.sh I       # image only
run/03_ts_mmme_variant.sh TI_ftclip
python gating/make_gated.py       # Table 4 TS-MMME_TI (gated): Macro-F1 0.7652
```

Each step writes into `results/`. The predictions shipped in `results/` are the ones reported in the paper, so `gating/make_gated.py` and `llm_baselines/score.py` can be run directly without retraining.

| Variant | Text encoder | Image | Used for |
|---|---|---|---|
| `TI` | frozen | frozen | Tables 3, 6, 7 |
| `T_ft` | fine-tuned BERT_T | – | Tables 3, 4 (T) |
| `I` | – | CLIP projection | Tables 3, 4 (I) |
| `TI_ftclip` | fine-tuned BERT_T | CLIP projection | input to gating |

### Other datasets (Table 2)

```bash
python other_datasets/prepare.py --name pheme --train train.csv --test test.csv [--images DIR]
other_datasets/run.sh pheme
```

Results: `results/other_datasets/<name>/`.

### LLM baselines (Table 5)

```bash
cp llm_baselines/.env.example .env          # fill in GROQ_KEYS (comma-separated) and SARVAM_KEY
python llm_baselines/llm_api.py --model openai/gpt-oss-120b --mode T
python llm_baselines/llm_local.py --model gemma-4-e4b --mode TI      # Apple Silicon (MLX)
python llm_baselines/llm_local_deepseek.py
python llm_baselines/score.py               # results/llm/scores.json
```

All models receive the first 800 characters of the article, its month and location (and the image in TI mode). Unparseable answers count as *real*. Runs resume from the existing output file.

## Results

| Model | Macro-F1 |
|---|---|
| TS-MMME_T (fine-tuned) | 0.7628 |
| TS-MMME_I | 0.6468 |
| TS-MMME_TI (frozen) | 0.7559 |
| TS-MMME_TI (gated) | 0.7652 |
