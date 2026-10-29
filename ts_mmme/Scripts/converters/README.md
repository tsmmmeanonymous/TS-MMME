# Dataset converters for the MME-TSE pipeline

The pipeline reads exactly one input format. A converter turns a raw dataset
into that format. Nothing else in the pipeline changes between datasets.

```
raw dataset  ->  [converter]  ->  master JSONL  ->  step0 -> Phase 1 -> Phase 2 -> Phase 3
                                  (16 fields)         ^
                                            pipeline starts here, unchanged
```

Two converters are included. Between them they cover the two shapes most
datasets take: flat tabular files, and one folder per item.

---

## The target format

JSONL. One JSON object per line. Exactly these 16 fields:

```json
{
  "content": "...",
  "image_path": "dataset/article_images/img_0.jpg",
  "cluster_id": 0,
  "time": 1531872000000,
  "label": "rumor",
  "year": 2018,
  "season": 3,
  "month": 7,
  "id": 0,
  "weight": 1,
  "day": 18,
  "year_month": "2018-07",
  "year_season": "2018-03",
  "location": "USA",
  "location_bengali": "",
  "original_location": "USA"
}
```

Rules that are easy to get wrong:

| Field | Rule |
|---|---|
| `label` | exactly `"truth"` or `"rumor"`, lowercase |
| `time` | epoch **milliseconds**, normalised to 00:00 UTC |
| `season` | `(month - 1) // 3 + 1`, values 1 to 4 |
| `year_month` | `"YYYY-MM"` zero-padded |
| `year_season` | `"YYYY-0S"` zero-padded season, not month |
| `image_path` | `""` or `"dataset/article_images/img_<N>.jpg"`, lowercase `dataset/` |
| `location` | no `\|` character, and never a string pandas reads as NaN |
| `weight` | constant `1` |
| `location_bengali` | constant `""` |

The `location` rule caused two crashes on PHEME. See "Two failures worth
knowing about" below.

---

## Example 1 : FakeNewsNet, a flat tabular dataset

### What the source looks like

`examples/fakenewsnet/01_raw_csv_row.txt`

```
Source file : gossipcop_fake.csv
CSV header  : id,news_url,title,tweet_ids

id        : gossipcop-6940044526
news_url  : kellyandryan.com/uncategorized/ryan-bio/
title     : LIVE with Kelly and Ryan
tweet_ids : 1202 ids, tab-separated
```

Four columns. No date, no image, no topic, no location. The label is not a
column at all, it is implied by which of the four CSV files the row is in.

### What it becomes

`examples/fakenewsnet/02_converted_record.json`

```json
{
  "content": "LIVE with Kelly and Ryan",
  "image_path": "dataset/article_images/img_0.jpg",
  "cluster_id": 0,
  "time": 1531872000000,
  "label": "rumor",
  "year": 2018, "season": 3, "month": 7, "day": 18,
  "year_month": "2018-07", "year_season": "2018-03",
  "id": 0, "weight": 1,
  "location": "USA", "location_bengali": "", "original_location": "USA"
}
```

### How each field was produced

| Field | Method |
|---|---|
| `content` | `title`, verbatim |
| `label` | file is `gossipcop_fake.csv`, so `"rumor"` |
| `time` | all 1,202 tweet ids decoded as Twitter snowflakes, earliest taken (2018-07-18), normalised to midnight |
| date parts | derived from `time` |
| `image_path` | fetched `news_url`, read its `og:image` meta tag, downloaded the image |
| `cluster_id` | title encoded with a sentence transformer, matched to the nearest of 7 fixed topics |
| `location` | constant `"USA"`, the dataset has no geography |

### The timestamp trick

FakeNewsNet ships no publication dates. Twitter IDs embed their creation time
in the high bits, so a date can be recovered with arithmetic alone:

```python
epoch_ms = (int(tweet_id) >> 22) + 1288834974657
```

No API key, no network call. The converter decodes every id attached to an
article and keeps the earliest, as a proxy for publication time.

---

## Example 2 : PHEME, one folder per item

### What the source looks like

`examples/pheme/01_raw_thread_folder.txt`

```
ottawashooting-all-rnr-threads/non-rumours/524944146025312256/
  annotation.json
  source-tweets/524944146025312256.json
  reactions/        (unused)
  structure.json    (unused)

annotation.json : {"is_rumour": "nonrumour"}

source tweet has 24 fields; the converter reads four:
  text          : LIVE: Ongoing @CTVNews-@ABC coverage of active shooting
                  incident near Canadian Parliament: http://t.co/0nuzHcvr2J
  created_at    : Wed Oct 22 15:23:32 +0000 2014
  user.location : 'New York City'
  media_url     : http://pbs.twimg.com/media/B0j56xpIgAAkSRv.png
```

### What it becomes

`examples/pheme/02_converted_record.json`

```json
{
  "content": "LIVE: Ongoing @CTVNews-@ABC coverage of active shooting incident    near Canadian Parliament: http://t.co/0nuzHcvr2J http://t.co/wxB5UphP4l",
  "image_path": "dataset/article_images/img_0.jpg",
  "cluster_id": 0,
  "time": 1413936000000,
  "label": "truth",
  "year": 2014, "season": 4, "month": 10, "day": 22,
  "year_month": "2014-10", "year_season": "2014-04",
  "id": 0, "weight": 1,
  "location": "New York City", "location_bengali": "", "original_location": "New York City"
}
```

### How each field was produced

| Field | Method |
|---|---|
| `content` | `text`, verbatim |
| `label` | `annotation.json` -> `is_rumour`: `rumour` becomes `"rumor"`, `nonrumour` becomes `"truth"` |
| `time` | `created_at` parsed with `"%a %b %d %H:%M:%S %z %Y"`, normalised to midnight |
| `image_path` | `entities.media[0].media_url`, downloaded directly |
| `cluster_id` | same nearest-topic method as FakeNewsNet |
| `location` | `user.location`, last comma-separated part, capped at 40 characters |

---

## Topic assignment

Both converters assign `cluster_id` the same way, so the topic space stays
identical across datasets and nothing is refitted:

1. Read the 7 keyword lists from `reference/topic_keywords.json`
2. Join each list into one string and encode it
3. Encode every article's text
4. Cosine similarity, take the argmax

Encoder: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, the same
model the pipeline's own topic step uses.

---

## Two failures worth knowing about

Both came from PHEME's free-text user locations. FakeNewsNet never hit them
because its location is a hardcoded constant.

**1. A location that reads `n/a`**

`step1` writes nodes to CSV, `step2` reads them back with pandas. Pandas
converts `n/a`, `NA`, `null`, `none`, `nan` and friends into `NaN` on read, so
the dictionary lookup failed:

```
KeyError: nan
```

**2. A location containing a pipe**

Memory cells are keyed as `topic|month|location`. Seven users had locations like
`"London | Najaf"` and `"USA | AUS | UK | PH"`, so the key split into more than
three parts:

```
ValueError: too many values to unpack (expected 3)
```

Both are handled in `normalise_location()` in the PHEME converter. If you write
a converter for a dataset with free-text locations, copy that function.

---

## Files here

```
converters/
  fnn0_build_master_dataset.py      flat tabular source
  pheme0_build_master_dataset.py    folder-per-item source, free-text locations

examples/
  fakenewsnet/01_raw_csv_row.txt
  fakenewsnet/02_converted_record.json
  pheme/01_raw_thread_folder.txt
  pheme/01b_raw_tweet_fields.json
  pheme/02_converted_record.json

reference/
  target_schema_sample.jsonl        three real lines of a working master file
  topic_keywords.json               the 7 topic definitions
```

---

## Writing a converter for a new dataset

1. Read `reference/target_schema_sample.jsonl` and match those 16 fields exactly.
2. Start from whichever converter matches your source shape.
3. Decide three things before writing code:
   - **Unit of analysis.** One record per article, per post, per thread?
   - **Timestamp source.** A real date field, or something derived?
   - **Location.** Real values, or a constant if the dataset has none?
4. Write the converted train and test files, then run
   `python other_datasets/prepare.py --name <name> --train <train> --test <test> [--images DIR]`
   and `other_datasets/run.sh <name>`.

Before trusting any result, compute the majority-class baseline for your split.
An accuracy at or below it means nothing was learned.
