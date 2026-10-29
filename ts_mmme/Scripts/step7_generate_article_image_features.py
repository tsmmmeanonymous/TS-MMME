from pathlib import Path
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Images are stored outside MME_TSE
DATASET_ROOT = PROJECT_ROOT.parent

EXPERIMENT_DIR = PROJECT_ROOT / "Dataset" / "Experiment"

DATASET_FILE = EXPERIMENT_DIR / "train_dataset.json"

GRAPH_DIR = (
    PROJECT_ROOT
    / "Graph"
    / "Experiment"
    / "Training"
)

FEATURE_DIR = GRAPH_DIR / "Features"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# Load CLIP
# =====================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP model...")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model = CLIPModel.from_pretrained(MODEL_NAME)

model.eval()

# =====================================================
# Load Dataset
# =====================================================

print("Loading Training Dataset...")

records = []

with open(DATASET_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if line.strip():

            records.append(json.loads(line))

print(f"Articles : {len(records)}")

# =====================================================
# Generate Image Features
# =====================================================

embeddings = []

missing_images = []

print("Generating image embeddings...")

for idx, item in enumerate(
    tqdm(records, desc="Encoding Images")
):

    image_path = DATASET_ROOT / item["image_path"]

    try:

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        with torch.no_grad():

            image_features = model.get_image_features(
                **inputs
            )
            if not isinstance(image_features, torch.Tensor):   # transformers>=5 returns an output object
                image_features = image_features.pooler_output

            image_features = torch.nn.functional.normalize(
                image_features,
                p=2,
                dim=1
            )

        embeddings.append(
            image_features.squeeze(0)
        )

    except Exception as e:

        # Keep graph alignment
        embeddings.append(
            torch.zeros(512)
        )

        missing_images.append({

            "article_id": item.get("id", idx),

            "article_index": idx,

            "image_path": str(image_path),

            "error": str(e)
        })

# =====================================================
# Convert Tensor
# =====================================================

embeddings = torch.stack(embeddings)

# =====================================================
# Save Tensor
# =====================================================

FEATURE_FILE = FEATURE_DIR / "article_image.pt"

torch.save(
    embeddings,
    FEATURE_FILE
)

# =====================================================
# Metadata
# =====================================================

metadata = {

    "Split": "Training",

    "model": MODEL_NAME,

    "total_articles": len(records),

    "valid_images": len(records) - len(missing_images),

    "missing_images": len(missing_images),

    "embedding_dimension": 512,

    "tensor_shape": list(embeddings.shape),

    "missing_image_details": missing_images
}

METADATA_FILE = FEATURE_DIR / "article_image_metadata.json"

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

# =====================================================
# Summary
# =====================================================

print("\n======================================")
print("ARTICLE IMAGE FEATURES CREATED")
print("Split            : Training")
print("======================================")

print(f"Total Articles : {len(records)}")

print(f"Valid Images   : {len(records) - len(missing_images)}")

print(f"Missing Images : {len(missing_images)}")

print(f"Embedding Shape : {tuple(embeddings.shape)}")

print(f"Embedding Size  : {embeddings.shape[1]}")

print("\nSaved Files")

print(FEATURE_FILE)

print(METADATA_FILE)


