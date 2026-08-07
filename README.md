# TS-MMME

Reference implementation of **TS-MMME** — a Tempo-Spatial Multimodal Multilingual
framework for misinformation detection.

The framework builds a heterogeneous temporal-spatial graph over historical
articles, encodes it into a frozen veracity memory bank indexed by topic and time
interval, and classifies each incoming article by retrieving the memories that
match its predicted topic and temporal-spatial context.

The same pipeline runs on five datasets — MMM-TREND, FakeNewsNet, PHEME,
Fakeddit and Weibo. Only the converter that produces the master file differs;
the graph schema, the model and all three phases are identical across datasets.

---

## Master flow

```
                 converter  ──►  Dataset/<dataset>_master.json
                                          │
                                          ▼
                              step0_create_sample_dataset.py
                                          │
                                          ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  PHASE 1   Offline Graph Construction                        │
    │  24 steps: caching → nodes → edges → memory → features       │
    │            → HeteroData → Graph Transformer → memory bank    │
    │  Output: frozen memory bank Z_MM                             │
    └──────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  PHASE 2   Inference Model Training                          │
    │  Trains cross-attention fusion + veracity classifier.        │
    │  Graph Transformer, topic predictor and memory bank frozen.  │
    │  Output: best_inference_model.pt                             │
    └──────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  PHASE 3   End-to-End Inference                              │
    │  Full online path per article: topic prediction → memory     │
    │  retrieval → cross-attention → classification → evolution.   │
    │  Output: predictions, metrics, evolution analysis            │
    └──────────────────────────────────────────────────────────────┘
```

Phase 1 must complete before Phase 2. Phase 2 must complete before Phase 3:
Phase 3 loads the checkpoint that Phase 2 writes.

---

## How it works

The premise: an article's veracity is better judged against **what was known to be
true or false for that topic, at that time, in that place** than in isolation.

1. **Convert.** Each dataset is turned into one master file with 16 fixed fields.
   The converter is the only dataset-specific code; everything below is identical
   for all five datasets.

2. **Build memory (Phase 1).** Articles become a heterogeneous graph with five
   node types — `article`, `topic`, `time` (month), `location`, `memory`.
   A `memory` node `M_{k,τ}` is one `(topic, month, location)` cell holding that
   slice's truth/rumor statistics; memory cells of a topic are chained in time by
   `evolves_to`. A Graph Transformer encodes this graph into a frozen memory bank
   `Z_MM` — the model's veracity history. It is never modified again.

3. **Learn to compare (Phase 2).** With the graph encoder and memory bank frozen,
   two small modules are trained: **cross-attention** (article = query, retrieved
   memories = keys/values) fuses the article with its historical context, and a
   **veracity classifier** maps the fused vector to truth vs rumor.

4. **Classify (Phase 3).** For a new article: embed it, predict its topic, then
   **retrieve** the most relevant memory cells by a weighted score —
   `0.40·topic + 0.20·time + 0.10·location + 0.30·embedding` — fuse via
   cross-attention, and classify. Ablation modes zero out individual terms
   (time, location, fusion) to measure each one's contribution.

The retrieval step in 4 is the novelty: classification is conditioned on
retrieved topic-time-location veracity memory, not on the article alone.

---

## Installation

Python 3.11 or 3.12. Python 3.14 is not supported — `torch-geometric` is
unreliable there.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Device selection is automatic, in the order MPS, CUDA, CPU. A GPU is strongly
recommended; CPU works but Phase 1 training is slow.

Verify the installation before running anything:

```bash
python MME_TSE/Scripts/smoketest.py
```

This checks every dependency, resolves the Phase 1 step manifest and confirms the
topic resources are readable. It does not need a dataset and is expected to
report the master file as absent on a fresh checkout.

---

## Datasets

No dataset ships with this repository.

Select the active dataset by editing one line in `MME_TSE/Scripts/config.py`:

```python
DATASET = "mmm_trend"   # mmm_trend | fakenewsnet | pheme | fakeddit | weibo
```

Every path used by the pipeline is derived from that value. No other file needs
editing when switching datasets.

| Dataset | Master file | How to obtain |
|---|---|---|
| MMM-TREND | `final_model_dataset_images_normalized_location.json` | Download from the [MMM-TREND dataset repository](https://github.com/elephant-trunks/MMM-TREND) and place it in `MME_TSE/Dataset/` |
| FakeNewsNet | `fakenewsnet_master.json` | `converters/fnn0_build_master_dataset.py` |
| PHEME | `pheme_master.json` | `converters/pheme0_build_master_dataset.py` |
| Fakeddit | `fakeddit_master.json` | `converters/fakeddit0_build_master_dataset.py` |
| Weibo | `rumdect_master.json` | `converters/rumdect0_build_master_dataset.py` |

Converters live in `MME_TSE/Scripts/converters/` and are run from `MME_TSE/`.
The FakeNewsNet and PHEME converters download their source release themselves
and need no API credentials. Fakeddit and Weibo (RumDect) require the raw
release to be downloaded first and its path set at the top of the converter.
Sampling size, truth ratio, image limit and seed are constants at the head of
each file.

`converters/README.md` documents the target format field by field, with two
worked examples and the failure modes that free-text location strings cause.
Read it before writing a converter for a new corpus.

Every master file, whatever its source, has the same sixteen fields:

```
content  image_path  cluster_id  time  label  year  season  month
id  weight  day  year_month  year_season  location  location_bengali
original_location
```

A converter for any other corpus needs only to emit this schema.

### Weibo topic discovery

Weibo is Chinese-language, so the seven English keyword topics transfer poorly.
`MME_TSE/Scripts/weibo_topics/` holds an alternative route that discovers topics
from the corpus instead of matching fixed ones:

```bash
python Scripts/weibo_topics/step0_create_event_dataset.py    # RumDect -> events
python Scripts/weibo_topics/step1_generate_bertopic.py       # BERTopic + jieba
python Scripts/weibo_topics/step2_create_train_test_split.py # 80/20 split
```

This path writes its own split directly and bypasses
`step0_create_sample_dataset.py`. It needs `bertopic` and `jieba`, which are not
in `requirements.txt` because the main pipeline does not use them.

It is exploratory and is **not** the route used for the reported Weibo result.
That run used `rumdect0_build_master_dataset.py` with the seven fixed topics,
like every other dataset.

Validate before committing to a long run:

```bash
python MME_TSE/Scripts/validate_dataset.py
```

This checks every record for required fields, valid labels, well-formed
timestamps and readable images, and exits non-zero on failure. It takes a few
minutes and avoids discovering data problems an hour into Phase 1.

---

## Running

All commands are issued from inside `MME_TSE/`.

```bash
cd MME_TSE
```

### Split

```bash
python Scripts/step0_create_sample_dataset.py
```

Writes an 80/20 stratified split over topic and label into
`Dataset/Experiment/`, seeded at 42.

Run this before Phase 1. The caching steps at the start of Phase 1 read
`train_dataset.json`, which does not exist on a fresh checkout. Phase 1 runs the
same script again in its normal position; the fixed seed reproduces the split.

### Phase 1 — offline graph construction

```bash
python Scripts/Phase1_Offline_Graph_Construction_Pipeline.py
```

Twenty-four steps in sequence:

| Stage | Steps |
|---|---|
| Preprocessing | image captioning, text and caption embeddings, topic embeddings |
| Graph | article, topic, temporal and spatial nodes and their edges |
| Memory | temporal groups, memory nodes, memory edges |
| Features | article and memory features, multimodal fusion |
| Heterograph | temporal memory edges, `HeteroData` assembly |
| Training | Graph Transformer, memory bank export |

The pipeline stops at the first failing step and does not resume. Three
validators must pass: `step2.1`, `step5.1` and `step12.1`.

After this phase the memory bank is frozen and is never modified again.

### Phase 2 — inference model training

```bash
python Scripts/Phase2_Inference_Model_Training_Pipeline.py
```

Trains the cross-attention fusion module and the veracity classifier while the
Graph Transformer, the topic predictor and the memory bank stay frozen.

Outputs:

```
Graph/Experiment/InferenceModels/best_inference_model.pt
Graph/Experiment/InferenceResults/training_statistics.json
Graph/Experiment/InferenceResults/training_history.json
```

The first epoch is substantially slower than the rest. It populates an inference
cache by running the full retrieval path for every training article; later epochs
reuse it.

### Phase 3 — end-to-end inference

```bash
python Scripts/Phase3_End_to_End_Inference_Pipeline.py
```

Two prompts appear. The first selects the ablation mode, the second selects
manual prediction or test-set evaluation. To evaluate the full test set with the
complete model, answer `1` then `2`.

Without a terminal:

```bash
printf '1\n2\n' | python Scripts/Phase3_End_to_End_Inference_Pipeline.py
```

Writes `Results/FULL_predictions.csv`, `Results/FULL_metrics.json` and
per-article evolution analyses under `Results/FULL/Evolution/`.

---

## Graph schema

Five node types and eight edge types, identical on every dataset.

| Node | Meaning |
|---|---|
| `article` | one news item, text and optional image |
| `topic` | one of seven topic clusters |
| `time` | a month interval |
| `location` | normalised geographic string |
| `memory` | a veracity memory `M_{k,τ}`, one per topic × interval |

Articles connect to their topic (`belongs_to`), interval (`occurs_in`), location
(`located_in`) and memory (`stored_in`). Memories of the same topic form a
chronological `evolves_to` chain, which is what makes veracity evolution
observable.

---

## Ablations

Phase 3 prompts for a mode at startup:

| Mode | Effect |
|---|---|
| `FULL` | complete model |
| `NO_EVOLUTION` | temporal-spatial veracity evolution analysis disabled |
| `NO_CROSS_ATTENTION` | article embedding passed directly to the classifier |
| `NO_TEMPORAL` | retrieval ignores the temporal interval |
| `NO_SPATIAL` | retrieval ignores location |

Each writes `Results/<MODE>_predictions.csv` and `Results/<MODE>_metrics.json`,
so modes never overwrite one another.

---

## Repository layout

```
TS-MMME/
├── MME_TSE/
│   ├── Scripts/
│   │   ├── config.py                       active dataset and all paths
│   │   ├── converters/                     dataset-specific master builders
│   │   │                                   + format guide and examples
│   │   ├── weibo_topics/                   BERTopic topic discovery (Weibo)
│   │   ├── Phase1_…, Phase2_…, Phase3_…    the three master pipelines
│   │   ├── step00 … step14_1               Phase 1 steps
│   │   ├── Phase2/                         retrieval, attention, classifier,
│   │   │                                   evolution, evaluation
│   │   ├── Models/                         Graph Transformer
│   │   ├── Modules/                        BLIP image captioning
│   │   ├── validate_dataset.py
│   │   └── smoketest.py
│   ├── Resources/Topic/                    seven topic definitions (required)
│   ├── Dataset/    Graph/    Results/    Logs/      generated at run time
├── figures/                                paper figure generators + output
├── results/                                published metrics and predictions
├── requirements.txt
└── README.md
```

Only `Scripts/` and `Resources/Topic/` ship with content. Every other directory
under `MME_TSE/` is populated by running the pipeline and is git-ignored.

---

## Published results

`results/` holds the metrics and per-article predictions from the runs reported
in the paper. Each `*_metrics.json` accompanies the `*_predictions.csv` produced
by the same run, so any metric can be recomputed from the predictions directly.

| Directory | Run |
|---|---|
| `mmm-trend/` | all five ablation modes |
| `fakenewsnet/` | full model |
| `pheme/` | full model |
| `pheme-ablation-a/` | memory contribution zeroed |
| `pheme-ablation-b/` | cross-attention removed |
| `fakeddit/` | full model |
| `weibo/` | full model |

Test splits for reference:

| Dataset | Test articles | Truth | Rumor | Topics | Years | Locations |
|---|---|---|---|---|---|---|
| MMM-TREND | 5,703 | 3,637 | 2,066 | 7 | 13 | 155 |
| FakeNewsNet | 4,334 | 3,229 | 1,105 | 7 | 9 | 1 |
| PHEME | 1,285 | 804 | 481 | 7 | 2 | 365 |
| Fakeddit | 1,998 | 911 | 1,087 | 7 | 9 | 1 |
| Weibo | 933 | 470 | 463 | 7 | 6 | 242 |

FakeNewsNet and Fakeddit ship no per-article geography, so location is a single
constant; their spatial component is present but carries no signal. Counts for
these two, PHEME and the remaining rows are computed directly from the
prediction files in `results/`.

Averaging is not uniform across these files: they were produced at different
times by runs configured differently. Recompute from the predictions CSV rather
than comparing the stored scalars across directories.

---

## Reproducibility

All random seeds are fixed at 42. The train/test split, the sampling order and
the model initialisation are deterministic.

Two sources of run-to-run variation remain and are unavoidable:

- Converter image downloads depend on external URLs being reachable at the time
  of the run, so the number of retrieved images may differ.
- GPU non-determinism in attention kernels produces small floating-point
  differences between runs on different hardware.

Phase 2 and Phase 3 evaluate the same test articles and should agree to within
about one percentage point. Phase 2 runs in batches over cached embeddings while
Phase 3 runs the full online path, which accounts for the small difference. A
larger gap indicates a problem.
