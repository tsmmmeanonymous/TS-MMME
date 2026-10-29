import json
import re
import sys
import warnings
from pathlib import Path

from PIL import Image

import config

warnings.simplefilter("ignore")
Image.MAX_IMAGE_PIXELS = None

ROOT = config.PROJECT_ROOT

MASTER = config.MASTER_FILE

FILES = [
    MASTER,
    ROOT / "Dataset" / "Experiment" / "train_dataset.json",
    ROOT / "Dataset" / "Experiment" / "test_dataset.json",
]

REQUIRED = [
    "content", "image_path", "cluster_id", "time", "label", "year",
    "season", "month", "id", "weight", "day", "year_month",
    "year_season", "location", "location_bengali", "original_location",
]

RESERVED = {
    "", "#n/a", "#n/a n/a", "#na", "-1.#ind", "-1.#qnan", "-nan",
    "1.#ind", "1.#qnan", "<na>", "n/a", "na", "null", "nan", "none",
}


def check(record, issues):

    for field in REQUIRED:
        if field not in record:
            issues.append(f"missing field {field} in record {record.get('id')}")

    if not str(record.get("content", "")).strip():
        issues.append(f"empty content in record {record.get('id')}")

    if record.get("label") not in ("truth", "rumor"):
        issues.append(f"invalid label in record {record.get('id')}")

    for field in ("location", "original_location"):
        value = record.get(field)
        if not isinstance(value, str):
            issues.append(f"non-string {field} in record {record.get('id')}")
        elif value.strip().lower() in RESERVED:
            issues.append(f"reserved {field} value in record {record.get('id')}")
        elif "|" in value:
            issues.append(f"pipe in {field} of record {record.get('id')}")

    if not isinstance(record.get("time"), int):
        issues.append(f"non-integer time in record {record.get('id')}")

    if record.get("season") not in (1, 2, 3, 4):
        issues.append(f"invalid season in record {record.get('id')}")

    for field in ("year_month", "year_season"):
        if not re.fullmatch(r"\d{4}-\d{2}", str(record.get(field, ""))):
            issues.append(f"malformed {field} in record {record.get('id')}")


def main():

    issues = []
    records = 0
    images = 0

    for path in FILES:

        if not path.exists():
            continue

        with path.open(encoding="utf-8") as handle:

            for line in handle:

                if not line.strip():
                    continue

                record = json.loads(line)
                records += 1

                check(record, issues)

                relative = record.get("image_path", "")

                if not relative:
                    continue

                image = ROOT / relative.replace("dataset/", "Dataset/")

                if not image.exists():
                    issues.append(f"missing image {relative}")
                    continue

                try:
                    with Image.open(image) as handle_image:
                        handle_image.verify()
                    images += 1
                except Exception:
                    issues.append(f"unreadable image {relative}")

    print(f"records checked : {records}")
    print(f"images verified : {images}")
    print(f"issues          : {len(issues)}")

    for issue in issues[:40]:
        print(f"  {issue}")

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
