import csv
import io
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


POOL_SIZE = 100000

TRUTH_RATIO = 0.6377

IMAGE_LIMIT = 100000

IMAGE_WORKERS = 12

IMAGE_TIMEOUT = 6

LOCATION = "USA"

RANDOM_SEED = 42

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

SOURCE_BASE = (
    "https://raw.githubusercontent.com/KaiDMML/"
    "FakeNewsNet/master/dataset"
)

SOURCE_FILES = [
    ("politifact_fake.csv", "rumor"),
    ("politifact_real.csv", "truth"),
    ("gossipcop_fake.csv", "rumor"),
    ("gossipcop_real.csv", "truth"),
]

TWITTER_EPOCH = 1288834974657

random.seed(RANDOM_SEED)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "Dataset"

RAW_DIR = DATASET_DIR / "FakeNewsNet"

IMAGE_DIR = DATASET_DIR / "article_images"

TOPIC_FILE = PROJECT_ROOT / "Resources" / "Topic" / "topic_keywords.json"

OUTPUT_FILE = DATASET_DIR / "fakenewsnet_master.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


OG_IMAGE = re.compile(
    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)

HEADERS = {"User-Agent": "Mozilla/5.0"}


def download_sources():

    for filename, _ in SOURCE_FILES:

        target = RAW_DIR / filename

        if target.exists():
            print(f"Present : {filename}")
            continue

        url = f"{SOURCE_BASE}/{filename}"

        print(f"Fetching : {filename}")

        response = requests.get(url, timeout=60)

        response.raise_for_status()

        target.write_bytes(response.content)


def decode_snowflake(tweet_id):

    return (int(tweet_id) >> 22) + TWITTER_EPOCH


def first_tweet_time(tweet_field):

    if not tweet_field:
        return None

    ids = re.split(r"[\s,\t]+", tweet_field.strip())

    stamps = []

    for value in ids:

        if not value.isdigit():
            continue

        stamp = decode_snowflake(value)

        if 1104537600000 < stamp < 1893456000000:
            stamps.append(stamp)

    if not stamps:
        return None

    return min(stamps)


def read_sources():

    records = []

    csv.field_size_limit(sys.maxsize)

    for filename, label in SOURCE_FILES:

        path = RAW_DIR / filename

        with open(path, "r", encoding="utf-8", errors="replace") as f:

            reader = csv.DictReader(f)

            for row in reader:

                title = (row.get("title") or "").strip()

                url = (row.get("news_url") or "").strip()

                stamp = first_tweet_time(row.get("tweet_ids"))

                if not title or stamp is None:
                    continue

                records.append({
                    "content": title,
                    "url": url,
                    "time": stamp,
                    "label": label,
                })

    print(f"Usable articles : {len(records)}")

    return records


def select_pool(records):

    truth = [r for r in records if r["label"] == "truth"]

    rumor = [r for r in records if r["label"] == "rumor"]

    random.shuffle(truth)

    random.shuffle(rumor)

    truth_target = int(POOL_SIZE * TRUTH_RATIO)

    rumor_target = POOL_SIZE - truth_target

    truth_target = min(truth_target, len(truth))

    rumor_target = min(rumor_target, len(rumor))

    pool = truth[:truth_target] + rumor[:rumor_target]

    random.shuffle(pool)

    print(f"Pool : {len(pool)} ({truth_target} truth / {rumor_target} rumor)")

    return pool


def fetch_image(task):

    index, url = task

    if not url:
        return index, ""

    if not url.startswith("http"):
        url = "http://" + url

    try:

        page = requests.get(
            url,
            timeout=IMAGE_TIMEOUT,
            headers=HEADERS,
        )

        match = OG_IMAGE.search(page.text)

        if not match:
            return index, ""

        image_url = match.group(1)

        if image_url.startswith("//"):
            image_url = "https:" + image_url

        blob = requests.get(
            image_url,
            timeout=IMAGE_TIMEOUT,
            headers=HEADERS,
        )

        if blob.status_code != 200 or len(blob.content) < 2048:
            return index, ""

        try:
            probe = Image.open(io.BytesIO(blob.content))
            probe.verify()
            probe = Image.open(io.BytesIO(blob.content))
            if probe.width * probe.height > 40000000:
                return index, ""
        except Exception:
            return index, ""

        name = f"img_{index}.jpg"

        (IMAGE_DIR / name).write_bytes(blob.content)

        return index, f"dataset/article_images/{name}"

    except Exception:
        return index, ""


def attach_images(pool):

    tasks = [
        (i, pool[i]["url"])
        for i in range(min(IMAGE_LIMIT, len(pool)))
    ]

    for record in pool:
        record["image_path"] = ""

    print(f"Retrieving up to {len(tasks)} images...")

    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool_executor:

        results = list(
            tqdm(
                pool_executor.map(fetch_image, tasks),
                total=len(tasks),
            )
        )

    hits = 0

    for index, path in results:

        pool[index]["image_path"] = path

        if path:
            hits += 1

    print(f"Images retrieved : {hits}")

    return pool


def assign_topics(pool):

    with open(TOPIC_FILE, "r", encoding="utf-8") as f:
        topics = json.load(f)

    topic_ids = sorted(topics.keys(), key=int)

    sentences = [
        " ".join(topics[key]["keywords"])
        for key in topic_ids
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading SentenceTransformer ({MODEL_NAME})...")
    print(f"Device : {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    topic_vectors = model.encode(
        sentences,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    article_vectors = model.encode(
        [r["content"] for r in pool],
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )

    similarity = article_vectors @ topic_vectors.T

    assignment = similarity.argmax(dim=1).tolist()

    for record, index in zip(pool, assignment):
        record["cluster_id"] = int(topic_ids[index])

    return pool


def build_record(record, index):

    stamp = record["time"]

    moment = datetime.fromtimestamp(stamp / 1000, tz=timezone.utc)

    midnight = datetime(
        moment.year,
        moment.month,
        moment.day,
        tzinfo=timezone.utc,
    )

    normalised = int(midnight.timestamp() * 1000)

    season = (moment.month - 1) // 3 + 1

    return {
        "content": record["content"],
        "image_path": record["image_path"],
        "cluster_id": record["cluster_id"],
        "time": normalised,
        "label": record["label"],
        "year": moment.year,
        "season": season,
        "month": moment.month,
        "id": index,
        "weight": 1,
        "day": moment.day,
        "year_month": f"{moment.year}-{moment.month:02d}",
        "year_season": f"{moment.year}-{season:02d}",
        "location": LOCATION,
        "location_bengali": "",
        "original_location": LOCATION,
    }


def write_master(pool):

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for index, record in enumerate(pool):

            line = build_record(record, index)

            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nMaster dataset written : {OUTPUT_FILE}")


def main():

    download_sources()

    records = read_sources()

    pool = select_pool(records)

    pool = attach_images(pool)

    pool = assign_topics(pool)

    write_master(pool)

    years = sorted({
        datetime.fromtimestamp(r["time"] / 1000, tz=timezone.utc).year
        for r in pool
    })

    topics = sorted({r["cluster_id"] for r in pool})

    images = sum(1 for r in pool if r["image_path"])

    print("")
    print(f"Articles : {len(pool)}")
    print(f"Truth    : {sum(1 for r in pool if r['label'] == 'truth')}")
    print(f"Rumor    : {sum(1 for r in pool if r['label'] == 'rumor')}")
    print(f"Topics   : {len(topics)}")
    print(f"Years    : {len(years)} ({years[0]}-{years[-1]})")
    print(f"Images   : {images}")


if __name__ == "__main__":
    main()
