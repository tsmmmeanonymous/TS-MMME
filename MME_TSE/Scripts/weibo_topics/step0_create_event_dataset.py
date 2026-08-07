#!/usr/bin/env python3
"""
===========================================================
MME-TSE
Step 0 : Create Event Dataset from RumDect Weibo Dataset
===========================================================
"""

import json
import re

from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# ==========================================================
# PATHS
# ==========================================================

CURRENT_DIR = Path(__file__).resolve().parent

BASELINE_DIR = CURRENT_DIR.parents[2]

RUMDECT_DIR = BASELINE_DIR / "rumdect"

LABEL_FILE = RUMDECT_DIR / "Weibo.txt"

WEIBO_DIR = RUMDECT_DIR / "Weibo"

OUTPUT_DIR = BASELINE_DIR / "Dataset" / "Experiment"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "full_dataset.json"

# ==========================================================
# LABEL MAP
# ==========================================================

LABEL_MAP = {
    0: "truth",
    1: "rumor"
}

# ==========================================================
# LOW INFORMATION PHRASES
# ==========================================================

STOP_PHRASES = {

    "转发微博",
    "转发微博。",
    "轉發微博",
    "轉發微博。",
    "repost",
    "Repost",

    "哈哈",
    "呵呵",
    "哈哈哈",
    "呵呵呵",

    "支持",
    "同意",
    "顶",
    "赞",

    "学习",
    "收藏",
    "mark",
    "沙发",
    "路过",

    "...",
    "……",
    "。。。",

    "[哈哈]",
    "[嘻嘻]",
    "[偷笑]",
    "[心]",
    "[good]"
}

# ==========================================================
# LOAD LABELS
# ==========================================================

def load_labels():

    labels = {}

    with open(LABEL_FILE, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            eid = None
            label = None

            for token in line.split():

                if token.startswith("eid:"):
                    eid = token.replace("eid:", "")

                elif token.startswith("label:"):
                    label = int(token.replace("label:", ""))

            if eid is not None:

                labels[eid] = LABEL_MAP[label]

    return labels
    
    
    
    
# ==========================================================
# FIND ORIGINAL POST
# ==========================================================

def find_original(posts, event_id):

    for post in posts:

        if str(post.get("id")) == str(event_id):
            return post

    for post in posts:

        if post.get("parent") is None:
            return post

    return posts[0]


# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\S+", " ", text)

    # Remove hashtags
    text = re.sub(r"#.*?#", " ", text)

    # Remove multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# CHECK INFORMATIVE COMMENT
# ==========================================================

def is_informative(text):

    if len(text) < 5:
        return False

    if text in STOP_PHRASES:
        return False

    # Only punctuation
    if re.fullmatch(r"[\W_]+", text):
        return False

    return True


# ==========================================================
# BUILD EVENT DOCUMENT
# ==========================================================

def merge_posts(posts, max_comments=30):

    merged = []

    seen = set()

    # ------------------------------------------------------
    # Keep Original Post
    # ------------------------------------------------------

    original_text = clean_text(
        posts[0].get("text", "")
    )

    if original_text:

        merged.append(original_text)

        seen.add(original_text)

    # ------------------------------------------------------
    # Keep Informative Comments
    # ------------------------------------------------------

    count = 0

    for post in posts[1:]:

        text = clean_text(
            post.get("text", "")
        )

        if not is_informative(text):
            continue

        if text in seen:
            continue

        seen.add(text)

        merged.append(text)

        count += 1

        if count >= max_comments:
            break

    return "\n".join(merged)
    
    
    
    
# ==========================================================
# CREATE DATASET
# ==========================================================

def create_dataset():

    labels = load_labels()

    files = sorted(WEIBO_DIR.glob("*.json"))

    rumor = 0
    truth = 0

    max_posts = 0
    min_posts = float("inf")
    total_posts = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as writer:

        for file in tqdm(files, desc="Processing Events"):

            event_id = file.stem

            if event_id not in labels:
                continue

            try:

                with open(file, "r", encoding="utf-8") as f:
                    posts = json.load(f)

            except Exception:
                continue

            if len(posts) == 0:
                continue

            # ------------------------------------------
            # Original Post
            # ------------------------------------------

            original = find_original(posts, event_id)

            timestamp = int(original.get("t", 0))

            dt = datetime.fromtimestamp(timestamp)

            location = original.get("user_location", "")

            if location is None:
                location = ""

            location = str(location).strip()

            # ------------------------------------------
            # Build Event Document
            # ------------------------------------------

            content = merge_posts(
                posts,
                max_comments=30
            )

            num_posts = len(posts)

            total_posts += num_posts

            max_posts = max(max_posts, num_posts)

            min_posts = min(min_posts, num_posts)

            label = labels[event_id]

            if label == "rumor":
                rumor += 1
            else:
                truth += 1

            sample = {

                "id": event_id,

                "event_id": event_id,

                "content": content,

                "image_path": "",

                "location": location,

                "year": dt.year,

                "month": dt.month,

                "year_month": f"{dt.year}-{dt.month:02d}",

                "label": label,

                "num_posts": num_posts

            }

            writer.write(
                json.dumps(
                    sample,
                    ensure_ascii=False
                ) + "\n"
            )

    # ======================================================
    # SUMMARY
    # ======================================================

    print("\n" + "=" * 60)
    print("EVENT DATASET CREATED")
    print("=" * 60)

    print(f"Total Events      : {rumor + truth}")
    print(f"Rumor Events      : {rumor}")
    print(f"Truth Events      : {truth}")

    print(f"Average Posts     : {total_posts / (rumor + truth):.2f}")
    print(f"Maximum Posts     : {max_posts}")
    print(f"Minimum Posts     : {min_posts}")

    print("\nOutput File")
    print(OUTPUT_FILE)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    create_dataset()
