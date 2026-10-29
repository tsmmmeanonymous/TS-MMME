import io
import json
import random
import re
import tarfile
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

IMAGE_WORKERS = 8

IMAGE_TIMEOUT = 10

RANDOM_SEED = 42

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

SOURCE_URL = "https://ndownloader.figshare.com/files/11767817"

ARCHIVE_NAME = "PHEME_veracity.tar.bz2"

ROOT_DIR_NAME = "all-rnr-annotated-threads"

TWITTER_FORMAT = "%a %b %d %H:%M:%S %z %Y"

random.seed(RANDOM_SEED)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "Dataset"

RAW_DIR = DATASET_DIR / "PHEME"

IMAGE_DIR = DATASET_DIR / "article_images"

TOPIC_FILE = PROJECT_ROOT / "Resources" / "Topic" / "topic_keywords.json"

OUTPUT_FILE = DATASET_DIR / "pheme_master.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


HEADERS = {"User-Agent": "Mozilla/5.0"}

PANDAS_NA_STRINGS = {
    "", "#n/a", "#n/a n/a", "#na", "-1.#ind", "-1.#qnan", "-nan",
    "1.#ind", "1.#qnan", "<na>", "n/a", "na", "null", "nan", "none",
}


def download_source():

    archive = RAW_DIR / ARCHIVE_NAME

    threads_dir = RAW_DIR / ROOT_DIR_NAME

    if threads_dir.exists():
        print(f"Present : {ROOT_DIR_NAME}")
        return threads_dir

    if not archive.exists():

        print(f"Fetching : {ARCHIVE_NAME}")

        response = requests.get(SOURCE_URL, timeout=600, stream=True)

        response.raise_for_status()

        with open(archive, "wb") as f:

            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    print("Extracting archive...")

    with tarfile.open(archive, "r:*") as tar:
        tar.extractall(RAW_DIR)

    return threads_dir


def normalise_location(value):

    if not value:
        return "Unknown"

    value = value.replace("|", " ")

    value = re.sub(r"\s+", " ", value).strip()

    if not value:
        return "Unknown"

    parts = [p.strip() for p in value.split(",") if p.strip()]

    if not parts:
        return "Unknown"

    result = parts[-1][:40]

    if result.strip().lower() in PANDAS_NA_STRINGS:
        return "Unknown"

    return result


def read_threads(threads_dir):

    records = []

    events = sorted(
        d for d in threads_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    for event in events:

        for bucket, label in (("rumours", "rumor"), ("non-rumours", "truth")):

            bucket_dir = event / bucket

            if not bucket_dir.exists():
                continue

            for thread in sorted(bucket_dir.iterdir()):

                if not thread.is_dir() or thread.name.startswith("."):
                    continue

                annotation = thread / "annotation.json"

                if not annotation.exists():
                    continue

                try:
                    meta = json.load(open(annotation, encoding="utf-8"))
                except Exception:
                    continue

                if meta.get("is_rumour") not in ("rumour", "nonrumour"):
                    continue

                sources = [
                    f for f in (thread / "source-tweets").glob("*.json")
                    if not f.name.startswith("._")
                ]

                if not sources:
                    continue

                try:
                    tweet = json.load(open(sources[0], encoding="utf-8"))
                except Exception:
                    continue

                text = (tweet.get("text") or "").strip()

                created = tweet.get("created_at")

                if not text or not created:
                    continue

                try:
                    moment = datetime.strptime(created, TWITTER_FORMAT)
                except Exception:
                    continue

                entities = tweet.get("entities") or {}

                extended = tweet.get("extended_entities") or {}

                media = entities.get("media") or extended.get("media") or []

                media_url = ""

                if media:
                    media_url = (
                        media[0].get("media_url_https")
                        or media[0].get("media_url")
                        or ""
                    )

                raw_location = (tweet.get("user") or {}).get("location")

                records.append({
                    "content": text,
                    "media_url": media_url,
                    "time": int(moment.timestamp() * 1000),
                    "label": label,
                    "location": normalise_location(raw_location),
                    "event": event.name.replace("-all-rnr-threads", ""),
                })

    print(f"Usable threads : {len(records)}")

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

    try:

        blob = requests.get(url, timeout=IMAGE_TIMEOUT, headers=HEADERS)

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

    for record in pool:
        record["image_path"] = ""

    tasks = [
        (i, pool[i]["media_url"])
        for i in range(len(pool))
        if pool[i]["media_url"]
    ][:IMAGE_LIMIT]

    print(f"Retrieving up to {len(tasks)} images...")

    with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as executor:

        results = list(
            tqdm(
                executor.map(fetch_image, tasks),
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

    moment = datetime.fromtimestamp(record["time"] / 1000, tz=timezone.utc)

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
        "location": record["location"],
        "location_bengali": "",
        "original_location": record["location"],
    }


def write_master(pool):

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for index, record in enumerate(pool):

            line = build_record(record, index)

            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"\nMaster dataset written : {OUTPUT_FILE}")


def main():

    threads_dir = download_source()

    records = read_threads(threads_dir)

    pool = select_pool(records)

    pool = attach_images(pool)

    pool = assign_topics(pool)

    write_master(pool)

    years = sorted({
        datetime.fromtimestamp(r["time"] / 1000, tz=timezone.utc).year
        for r in pool
    })

    print("")
    print(f"Articles  : {len(pool)}")
    print(f"Truth     : {sum(1 for r in pool if r['label'] == 'truth')}")
    print(f"Rumor     : {sum(1 for r in pool if r['label'] == 'rumor')}")
    print(f"Topics    : {len({r['cluster_id'] for r in pool})}")
    print(f"Years     : {len(years)} ({years[0]}-{years[-1]})")
    print(f"Locations : {len({r['location'] for r in pool})}")
    print(f"Images    : {sum(1 for r in pool if r['image_path'])}")


if __name__ == "__main__":
    main()
