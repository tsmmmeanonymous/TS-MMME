import importlib
import json
import sys
from pathlib import Path

import config

SCRIPTS = Path(__file__).resolve().parent

failures = []


def check(label, condition, detail=""):
    status = "ok" if condition else "FAIL"
    print(f"[{status}] {label}{('  ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


print("=" * 60)
print("SMOKE TEST")
print("=" * 60)

print(f"\nPython {sys.version.split()[0]}")
print(f"Dataset : {config.DATASET}")
print(f"Root    : {config.PROJECT_ROOT}\n")

# -----------------------------------------------------
# Dependencies
# -----------------------------------------------------

for package in [
    "numpy", "pandas", "sklearn", "torch", "torch_geometric",
    "transformers", "sentence_transformers", "PIL", "networkx",
    "matplotlib", "tqdm", "requests",
]:
    try:
        importlib.import_module(package)
        check(f"import {package}", True)
    except Exception as error:
        check(f"import {package}", False, str(error))

# -----------------------------------------------------
# Device
# -----------------------------------------------------

try:
    import torch

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    check("torch device", True, device)
except Exception as error:
    check("torch device", False, str(error))

# -----------------------------------------------------
# Pipeline scripts referenced by the Phase 1 manifest
# -----------------------------------------------------

manifest = SCRIPTS / "Phase1_Offline_Graph_Construction_Pipeline.py"
text = manifest.read_text(encoding="utf-8")

referenced = sorted(set(
    line.split('"')[1]
    for line in text.splitlines()
    if '"Scripts/' in line and line.count('"') >= 2
))

missing = [
    name for name in referenced
    if not (config.PROJECT_ROOT / name).exists()
]

check(
    f"Phase 1 manifest ({len(referenced)} steps)",
    not missing,
    "" if not missing else f"missing: {', '.join(missing)}",
)

# -----------------------------------------------------
# Required inputs
# -----------------------------------------------------

for name in ["topic_keywords.json", "topic_metadata.json"]:
    path = config.RESOURCE_DIR / "Topic" / name
    ok = path.exists()
    detail = ""
    if ok:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            detail = f"{len(payload)} topics"
        except Exception as error:
            ok, detail = False, str(error)
    check(f"resource {name}", ok, detail)

# -----------------------------------------------------
# Dataset presence (informational)
# -----------------------------------------------------

print()
if config.MASTER_FILE.exists():
    size = config.MASTER_FILE.stat().st_size / 1_048_576
    print(f"[ok] master dataset present  {config.MASTER_FILE.name}  {size:.1f} MB")
else:
    print(f"[--] master dataset absent   {config.MASTER_FILE.name}")
    print("     Build it with the matching converter, or place it in Dataset/.")
    print("     This is expected on a fresh checkout and is not a failure.")

# -----------------------------------------------------

print()
print("=" * 60)

if failures:
    print(f"FAILED : {len(failures)}")
    for name in failures:
        print(f"  - {name}")
    sys.exit(1)

print("All checks passed.")
print("=" * 60)
