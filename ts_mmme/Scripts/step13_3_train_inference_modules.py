###############################################################
# step13.3_train_inference_modules.py
#
# Trainable Modules
# -----------------
# 1. Cross Attention
# 2. Veracity Classifier
#
# Frozen Modules
# --------------
# 1. Topic Prediction
# 2. Temporal Memory Retrieval
# 3. Graph Transformer (already used for ZMM generation)
#
###############################################################

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

###############################################################
# Project Root
###############################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

###############################################################
# Imports
###############################################################

from Phase2.step14_3_topic_prediction import predict_topic
from Phase2.step14_4_retrieve_temporal_memory import retrieve_temporal_memory
from Phase2.step14_5_cross_attention import CrossAttention
from Phase2.step14_6_veracity_classifier import VeracityClassifier

###############################################################
# Project Directories
###############################################################

GRAPH_DIR = (
    PROJECT_ROOT /
    "Graph" /
    "Experiment"
)

TRAINING_DIR = (
    GRAPH_DIR /
    "Training"
)

MODEL_DIR = (
    GRAPH_DIR /
    "InferenceModels"
)

RESULT_DIR = (
    GRAPH_DIR /
    "InferenceResults"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

###############################################################
# Dataset
###############################################################

DATASET_DIR = (
    PROJECT_ROOT /
    "Dataset" /
    "Experiment"
)

TRAIN_FILE = DATASET_DIR / "train_dataset.json"
TEST_FILE = DATASET_DIR / "test_dataset.json"
import os as _os
if _os.environ.get("VAL_HOLDOUT") == "1":   # MMM-Trend: validate/select on held-out pre-2024 data, never on the test set
    TEST_FILE = DATASET_DIR / "val_dataset.json"
    print("*** validation file:", TEST_FILE)

###############################################################
# Output Files
###############################################################

BEST_MODEL = (
    MODEL_DIR /
    "best_inference_model.pt"
)

LAST_MODEL = (
    MODEL_DIR /
    "last_inference_model.pt"
)

TRAIN_HISTORY = (
    RESULT_DIR /
    "training_history.json"
)

TRAIN_STATISTICS = (
    RESULT_DIR /
    "training_statistics.json"
)

###############################################################
# Hyperparameters
###############################################################

BATCH_SIZE = 32

NUM_EPOCHS = 20

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

EARLY_STOPPING = 5

MAX_GRAD_NORM = 1.0

ARTICLE_DIM = 768

MEMORY_DIM = 768

HIDDEN_DIM = 768

NUM_CLASSES = 2

###############################################################
# Reproducibility
###############################################################

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

###############################################################
# Device
###############################################################

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")

print()

print("===================================================")
print("TRAINING INFERENCE MODULES")
print("===================================================")
print("Device :", DEVICE)
print("===================================================")

###############################################################
# Dataset Class
###############################################################

class RumorDataset(Dataset):

    def __init__(self, json_file):

        self.samples = []

        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if len(line) == 0:
                    continue

                self.samples.append(
                    json.loads(line)
                )

        if _os.environ.get("IMAGE_ONLY") == "1" and _os.environ.get("IMG_ONLY_WITH_IMAGE") == "1":
            from Phase2 import image_proj as _ip
            _before = len(self.samples); self.samples = [x for x in self.samples if _ip.has_image(x.get("id"))]
            print(f"*** IMAGE_ONLY: kept {len(self.samples)} of {_before} articles that have a readable image")
        if _os.environ.get("SMOKE_N"):   # smoke test: tiny subset (keeps articles with unreadable images)
            _n = int(_os.environ["SMOKE_N"]); _keep = set(int(v) for v in _os.environ.get("SMOKE_IDS", "").split(",") if v)
            self.samples = self.samples[:_n] + [x for x in self.samples[_n:] if x.get("id") in _keep]
        print(
            f"{json_file.name} : {len(self.samples)} samples"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        label = (
            0
            if sample["label"].lower() == "truth"
            else 1
        )

        return {

            "article_id":
                sample["id"],

            "article_text":
                sample["content"],

            "image_path":
                sample["image_path"],

            "cluster_id":
                sample["cluster_id"],

            "year":
                sample["year"],

            "month":   # MMM-Trend fix: retrieval expects "YYYY-MM"; the integer month made every time score 0 during training
                sample.get("year_month", sample["month"]),

            "season":
                sample["season"],

            "location":
                sample["location"],

            "label":
                torch.tensor(
                    label,
                    dtype=torch.long
                )

        }

###############################################################
# Collate Function
###############################################################

def collate_fn(batch):

    return {

        "article_id":

            [x["article_id"] for x in batch],

        "article_text":

            [x["article_text"] for x in batch],

        "image_path":

            [x["image_path"] for x in batch],

        "cluster_id":

            [x["cluster_id"] for x in batch],

        "year":

            [x["year"] for x in batch],

        "month":

            [x["month"] for x in batch],

        "season":

            [x["season"] for x in batch],

        "location":

            [x["location"] for x in batch],

        "label":

            torch.stack(
                [x["label"] for x in batch]
            )

    }

###############################################################
# Dataset
###############################################################

print()

print("Loading Dataset...")

train_dataset = RumorDataset(TRAIN_FILE)

test_dataset = RumorDataset(TEST_FILE)

###############################################################
# DataLoader
###############################################################

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    collate_fn=collate_fn

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    collate_fn=collate_fn

)

print()

print("Train Samples :", len(train_dataset))

print("Test Samples  :", len(test_dataset))

print("Train Batches :", len(train_loader))

print("Test Batches  :", len(test_loader))

###############################################################
# Build Trainable Models
###############################################################

print()

print("Creating Cross Attention...")

cross_attention = CrossAttention(

    article_dim=ARTICLE_DIM,

    memory_dim=MEMORY_DIM,

    hidden_dim=HIDDEN_DIM

).to(DEVICE)

print("Done.")

print()

print("Creating Veracity Classifier...")

classifier = VeracityClassifier(

    input_dim=768,

    hidden_dim1=128,

    hidden_dim2=64,

    num_classes=2,

    dropout=0.30

).to(DEVICE)

print("Done.")

###############################################################
# Optimizer
###############################################################

from Phase2 import image_proj as _imgproj   # MMM-Trend: image-only variant, CLIP feature projection
_image_proj = _imgproj.ImageProj().to(DEVICE) if _imgproj.ENABLED else None
if _image_proj is not None:
    print("*** CLIP image feature projection enabled")

optimizer = AdamW(

    list(cross_attention.parameters())

    +

    list(classifier.parameters())

    +

    (list(_image_proj.parameters()) if _image_proj is not None else []),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)

scheduler = CosineAnnealingLR(

    optimizer,

    T_max=NUM_EPOCHS,

    eta_min=1e-6

)

criterion = nn.CrossEntropyLoss()
if _os.environ.get("CLASS_WEIGHTED") == "1":   # MMM-Trend: inverse-frequency class weights from the training labels
    _n1 = sum(1 for _x in train_dataset.samples if _x["label"].lower() != "truth"); _n0 = len(train_dataset.samples) - _n1
    _w = torch.tensor([len(train_dataset.samples) / (2 * _n0), len(train_dataset.samples) / (2 * _n1)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=_w.to(DEVICE))
    print("*** class-weighted loss, weights [truth, rumor] =", _w.tolist())

print()

print("Optimizer Ready.")
print("Scheduler Ready.")
print("Loss Ready.")

###############################################################
# Trainer
###############################################################

class InferenceTrainer:

    """
    Complete trainer for

    Topic Prediction
            ↓
    Temporal Memory Retrieval
            ↓
    Cross Attention
            ↓
    Veracity Classifier
    """

    def __init__(

        self,

        cross_attention,

        classifier,

        optimizer,

        scheduler,

        criterion,

        device

    ):

        self.cross_attention = cross_attention

        self.classifier = classifier

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.criterion = criterion

        self.device = device

        self.best_accuracy = 0.0

        self.early_counter = 0

        self.history = {

            "train_loss": [],
            "val_loss": [],

            "train_acc": [],
            "val_acc": [],

            "train_precision": [],
            "val_precision": [],

            "train_recall": [],
            "val_recall": [],

            "train_f1": [],
            "val_f1": []

        }
        ###############################################################
        # Frozen Inference Cache
        ###############################################################

        # Key:
        # (article_text, image_path, month, location)
        #
        # Value:
        # {
        #     "topic_result": topic_result,
        #     "retrieval_result": retrieval_result
        # }

        self.inference_cache = {}
    ###############################################################
    # Forward Pass for One Sample
    ###############################################################

    def forward_sample(
        self,
        article_id,
        article_text,
        image_path,
        month,
        location
    ):
        """
        Pipeline

        Article
        ↓
        Topic Prediction (Frozen)
        ↓
        Temporal Memory Retrieval (Frozen)
        ↓
        Cross Attention (Trainable)
        ↓
        Veracity Classifier (Trainable)
        """

        ###########################################################
        # Topic Prediction
        ###########################################################

        ###########################################################
        # Frozen Modules
        ###########################################################
        ###########################################################
        # Cache Key
        ###########################################################

        cache_key = (
            article_text,
            image_path,
            month,
            location
        )
        ###########################################################
        # Frozen Modules
        ###########################################################

        if cache_key in self.inference_cache:

            cached = self.inference_cache[cache_key]

            topic_result = cached["topic_result"]

            retrieval_result = cached["retrieval_result"]

        else:

            with torch.no_grad():

                ###################################################
                # Topic Prediction
                ###################################################

                topic_result = predict_topic(
                    article_text=article_text,
                    article_id=article_id,
                    image_path=image_path
                )

                article_embedding = topic_result["article_embedding"]

                top3_topics = topic_result["top3_topics"]

                ###################################################
                # Temporal Memory Retrieval
                ###################################################

                retrieval_result = retrieve_temporal_memory(
                    top3_topics=top3_topics,
                    month=month,
                    location=location,
                    article_embedding=article_embedding
                )

            self.inference_cache[cache_key] = {

                "topic_result": topic_result,

                "retrieval_result": retrieval_result

            }
        article_embedding = topic_result["article_embedding"]

        top3_topics = topic_result["top3_topics"]
        ###########################################################
        # No memory found
        ###########################################################

        if not retrieval_result["memory_found"]:

            context_embedding = torch.zeros(
                HIDDEN_DIM,
                dtype=torch.float32,
                device=self.device
            )

            logits, probabilities, prediction = self.classifier(
                context_embedding
            )

            return {
                "logits": logits,
                "probabilities": probabilities,
                "prediction": prediction,
                "attention": None,
                "context": context_embedding
            }

        ###########################################################
        # Build Memory Tensor
        ###########################################################

        memory_embeddings = []

        retrieval_scores = []

        for memory in retrieval_result["retrieved_memories"]:

            memory_embeddings.append(
                memory["memory_embedding"]
            )

            retrieval_scores.append(
                memory["retrieval_score"]
            )

        memory_embeddings = torch.stack(
            [m.to(self.device, dtype=torch.float32)
            for m in memory_embeddings],
            dim=0
        )

        retrieval_scores = torch.as_tensor(
            retrieval_scores,
            dtype=torch.float32,
            device=self.device
        )

        ###########################################################
        # Article Embedding
        ###########################################################

        if not isinstance(article_embedding, torch.Tensor):
            article_embedding = torch.as_tensor(
                article_embedding,
                dtype=torch.float32
            )

        article_embedding = article_embedding.to(
            self.device,
            dtype=torch.float32
        )

        if _image_proj is not None:
            article_embedding = _imgproj.apply(article_embedding, article_id, _image_proj)

        ###########################################################
        # Cross Attention
        ###########################################################

        context_embedding, attention_weights = \
            self.cross_attention(
                article_embedding,
                memory_embeddings,
                retrieval_scores
            )

        ###########################################################
        # Classifier
        ###########################################################

        logits, probabilities, prediction = \
            self.classifier(
                context_embedding
            )

        return {

            "logits": logits,

            "probabilities": probabilities,

            "prediction": prediction,

            "attention": attention_weights,

            "context": context_embedding
        }

    ###############################################################
    # Forward Batch
    ###############################################################

    def forward_batch(
        self,
        batch
    ):
        """
        Forward pass for an entire batch.

        Since topic prediction and temporal retrieval are Python
        inference pipelines, samples are processed sequentially.
        """

        logits_list = []

        prediction_list = []

        probability_list = []

        attention_list = []

        context_list = []

        batch_size = len(batch["article_text"])

        for i in range(batch_size):

            output = self.forward_sample(

                article_id=batch["article_id"][i],

                article_text=batch["article_text"][i],

                image_path=batch["image_path"][i],

                month=batch["month"][i],

                location=batch["location"][i]

            )

            logits_list.append(output["logits"])

            prediction_list.append(output["prediction"])

            probability_list.append(output["probabilities"])

            attention_list.append(output["attention"])

            context_list.append(output["context"])

        logits = torch.stack(logits_list, dim=0)

        return {

            "logits": logits,

            "predictions": prediction_list,

            "probabilities": probability_list,

            "attention": attention_list,

            "contexts": context_list

        }
    ###############################################################
    # Train One Epoch
    ###############################################################

    def train_epoch(
        self,
        train_loader,
        epoch
    ):

        self.cross_attention.train()

        self.classifier.train()

        running_loss = 0.0

        all_predictions = []

        all_labels = []

        progress = tqdm(

            train_loader,

            desc=f"Epoch {epoch} Training",

            leave=False

        )

        for batch in progress:

            #######################################################
            # Labels
            #######################################################

            labels = batch["label"].to(self.device)

            #######################################################
            # Reset Gradients
            #######################################################

            self.optimizer.zero_grad()

            #######################################################
            # Forward
            #######################################################

            outputs = self.forward_batch(batch)

            logits = outputs["logits"]

            #######################################################
            # Loss
            #######################################################

            loss = self.criterion(

                logits,

                labels

            )

            #######################################################
            # Backpropagation
            #######################################################

            loss.backward()

            #######################################################
            # Gradient Clipping
            #######################################################

            torch.nn.utils.clip_grad_norm_(

                list(self.cross_attention.parameters())

                +

                list(self.classifier.parameters()),

                MAX_GRAD_NORM

            )

            #######################################################
            # Optimizer Step
            #######################################################

            self.optimizer.step()

            #######################################################
            # Statistics
            #######################################################

            running_loss += loss.item()

            predictions = torch.argmax(

                logits,

                dim=1

            )

            all_predictions.extend(

                predictions.detach()

                .cpu()

                .numpy()

                .tolist()

            )

            all_labels.extend(

                labels.detach()

                .cpu()

                .numpy()

                .tolist()

            )

            #######################################################
            # Progress Bar
            #######################################################

            progress.set_postfix(

                loss=f"{loss.item():.4f}"

            )

        ###########################################################
        # Scheduler
        ###########################################################

        self.scheduler.step()

        ###########################################################
        # Epoch Metrics
        ###########################################################

        epoch_loss = (

            running_loss

            /

            len(train_loader)

        )

        epoch_accuracy = accuracy_score(

         all_labels,

            all_predictions

        )

        epoch_precision = precision_score(

            all_labels,

            all_predictions,

            zero_division=0

        )

        epoch_recall = recall_score(

            all_labels,

            all_predictions,

            zero_division=0

        )

        epoch_f1 = f1_score(

            all_labels,

            all_predictions,

            zero_division=0

        )

        ###########################################################
        # Store History
        ###########################################################

        self.history["train_loss"].append(

            epoch_loss

        )

        self.history["train_acc"].append(

            epoch_accuracy

        )

        self.history["train_precision"].append(epoch_precision)
        self.history["train_recall"].append(epoch_recall)
        self.history["train_f1"].append(epoch_f1)

        ###########################################################
        # Print Summary
        ###########################################################

        print()

        print("-------------------------------------------")

        print(

            f"Epoch {epoch} Training Summary"

        )

        print("-------------------------------------------")

        print(

            f"Loss      : {epoch_loss:.4f}"

        )

        print(

            f"Accuracy  : {epoch_accuracy:.4f}"

        )

        print(

        f"Precision : {epoch_precision:.4f}"

        )

        print(

        f"Recall    : {epoch_recall:.4f}"

        )

        print(

        f"F1 Score  : {epoch_f1:.4f}"

        )

        print("-------------------------------------------")

        ###########################################################
        # Return
        ###########################################################

        return {

        "loss": epoch_loss,

        "accuracy": epoch_accuracy,

        "precision": epoch_precision,

        "recall": epoch_recall,

        "f1": epoch_f1

        }
    ###############################################################
    # Validation
    ###############################################################

    def validate(
        self,
        test_loader,
        epoch
    ):

        self.cross_attention.eval()

        self.classifier.eval()

        running_loss = 0.0

        all_predictions = []

        all_labels = []

        all_probabilities = []

        with torch.no_grad():

            progress = tqdm(

                test_loader,

                desc=f"Epoch {epoch} Validation",

                leave=False

            )

            for batch in progress:

                ###################################################
                # Labels
                ###################################################

                labels = batch["label"].to(self.device)

                ###################################################
                # Forward
                ###################################################

                outputs = self.forward_batch(batch)

                logits = outputs["logits"]

                ###################################################
                # Loss
                ###################################################

                loss = self.criterion(

                    logits,

                    labels

                )

                running_loss += loss.item()

                ###################################################
                # Prediction
                ###################################################

                probabilities = torch.softmax(

                    logits,

                    dim=1

                )

                predictions = torch.argmax(

                    probabilities,

                    dim=1

                )

                ###################################################
                # Statistics
                ###################################################

                all_predictions.extend(

                    predictions.cpu()

                    .numpy()

                    .tolist()

                )

                all_labels.extend(

                    labels.cpu()

                    .numpy()

                    .tolist()

                )

                all_probabilities.extend(

                    probabilities.cpu()

                    .numpy()

                    .tolist()

                )

                ###################################################
                # Progress Bar
                ###################################################

                progress.set_postfix(

                    loss=f"{loss.item():.4f}"

                )

        ###########################################################
        # Epoch Metrics
        ###########################################################

        epoch_loss = (

            running_loss

            /

            len(test_loader)

        )

        epoch_accuracy = accuracy_score(

            all_labels,

            all_predictions

        )

        epoch_precision = precision_score(

            all_labels,

            all_predictions,

            zero_division=0

        )

        epoch_recall = recall_score(

            all_labels,

            all_predictions,

            zero_division=0

        )

        epoch_f1 = f1_score(

            all_labels,

            all_predictions,

            zero_division=0

        )

        ###########################################################
        # Store History
        ###########################################################

        self.history["val_loss"].append(

            epoch_loss

        )

        self.history["val_acc"].append(

            epoch_accuracy

        )
        self.history["val_precision"].append(epoch_precision)
        self.history["val_recall"].append(epoch_recall)
        self.history["val_f1"].append(epoch_f1)
        ###########################################################
        # Display Results
        ###########################################################

        print()

        print("-------------------------------------------")

        print(

            f"Epoch {epoch} Validation Summary"

        )

        print("-------------------------------------------")

        print(

            f"Loss      : {epoch_loss:.4f}"

        )

        print(

            f"Accuracy  : {epoch_accuracy:.4f}"

        )

        print(

            f"Precision : {epoch_precision:.4f}"

        )

        print(

            f"Recall    : {epoch_recall:.4f}"

        )

        print(

            f"F1 Score  : {epoch_f1:.4f}"

        )

        print("-------------------------------------------")

        ###########################################################
        # Return
        ###########################################################

        return {

            "loss": epoch_loss,

            "accuracy": epoch_accuracy,

            "precision": epoch_precision,

            "recall": epoch_recall,

            "f1": epoch_f1,

            "macro_f1": f1_score(all_labels, all_predictions, average="macro", zero_division=0),

            "predictions": all_predictions,

            "labels": all_labels,

            "probabilities": all_probabilities

        }
    ###############################################################
    # Save Checkpoint
    ###############################################################

    def save_checkpoint(
        self,
        epoch,
        val_metrics,
        filename
    ):

        checkpoint = {

            "epoch": epoch,

            "cross_attention_state_dict":

                self.cross_attention.state_dict(),

            "classifier_state_dict":

                self.classifier.state_dict(),

            **({"image_proj_state_dict": _image_proj.state_dict()} if _image_proj is not None else {}),

            "optimizer_state_dict":

                self.optimizer.state_dict(),

            "scheduler_state_dict":

                self.scheduler.state_dict(),

            "history":

                self.history,

            "validation_metrics":

                val_metrics

        }

        torch.save(

            checkpoint,

            filename

        )

        print()

        print(

            f"Checkpoint saved -> {filename}"

        )
    ###############################################################
    # Early Stopping
    ###############################################################

    def update_best_model(
        self,
        epoch,
        validation_metrics
    ):

        current_accuracy = validation_metrics["accuracy"]

        # MMM-Trend option A: select/early-stop on validation macro-F1 (as every baseline does)
        import os as _os
        if _os.environ.get("SELECT_MACRO_F1") == "1":
            current_accuracy = validation_metrics["macro_f1"]
            print(f"Val Macro-F1 (selection) : {current_accuracy:.4f}")

        if current_accuracy > self.best_accuracy:

            self.best_accuracy = current_accuracy

            self.early_counter = 0

            self.save_checkpoint(

                epoch,

                validation_metrics,

                BEST_MODEL

            )

            print()

            print(

                "New Best Model Saved."

            )

        else:

            self.early_counter += 1

            print()

            print(

                f"Early stopping counter : "

                f"{self.early_counter}/{EARLY_STOPPING}"

            )

        return (

            self.early_counter >= EARLY_STOPPING

        )
    ###############################################################
    # Training Loop
    ###############################################################

    def fit(
        self,
        train_loader,
        test_loader,
        num_epochs
    ):

        print()
        print("==================================================")
        print("Starting Training")
        print("==================================================")

        for epoch in range(1, num_epochs + 1):

            print()
            print("##################################################")
            print(f"Epoch {epoch}/{num_epochs}")
            print("##################################################")

            #######################################################
            # Train
            #######################################################

            train_metrics = self.train_epoch(
                train_loader=train_loader,
                epoch=epoch
            )

            #######################################################
            # Validation
            #######################################################

            validation_metrics = self.validate(
                test_loader=test_loader,
                epoch=epoch
            )

            #######################################################
            # Save Latest Checkpoint
            #######################################################

            self.save_checkpoint(
                epoch=epoch,
                val_metrics=validation_metrics,
                filename=LAST_MODEL
            )

            #######################################################
            # Save Best Model
            #######################################################

            stop_training = self.update_best_model(
                epoch=epoch,
                validation_metrics=validation_metrics
            )

            #######################################################
            # Epoch Summary
            #######################################################

            print()
            print("==================================================")
            print(f"Epoch {epoch} Completed")
            print("==================================================")
            print(f"Train Loss : {train_metrics['loss']:.4f}")
            print(f"Train Acc  : {train_metrics['accuracy']:.4f}")
            print(f"Val Loss   : {validation_metrics['loss']:.4f}")
            print(f"Val Acc    : {validation_metrics['accuracy']:.4f}")
            print(f"Best Acc   : {self.best_accuracy:.4f}")
            print("==================================================")

            #######################################################
            # Early Stop
            #######################################################

            if stop_training:

                print()
                print("**********************************************")
                print("Early stopping triggered.")
                print("**********************************************")

                break

        print()
        print("==================================================")
        print("Training Finished")
        print("==================================================")
    ###############################################################
    # Load Best Model
    ###############################################################

    def load_best_model(
        self
    ):

        if not BEST_MODEL.exists():

            print("Best model checkpoint not found.")

            return

        checkpoint = torch.load(
            BEST_MODEL,
            map_location=self.device
        )

        self.cross_attention.load_state_dict(
            checkpoint["cross_attention_state_dict"]
        )

        self.classifier.load_state_dict(
            checkpoint["classifier_state_dict"]
        )

        print()
        print("Best model loaded successfully.")
    ###############################################################
    # Evaluate Best Model
    ###############################################################

    def evaluate_best_model(
        self,
        test_loader
    ):

        print()
        print("==================================================")
        print("Evaluating Best Model")
        print("==================================================")

        self.load_best_model()

        metrics = self.validate(
            test_loader=test_loader,
            epoch=0
        )

        print()
        print("==================================================")
        print("FINAL RESULTS")
        print("==================================================")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1']:.4f}")
        print("==================================================")

        return metrics

    ###############################################################
    # Save Training History
    ###############################################################

    def save_training_history(self):

        with open(
            TRAIN_HISTORY,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                self.history,
                f,
                indent=4
            )

        print()
        print(f"Training history saved to:")
        print(TRAIN_HISTORY)


    ###############################################################
    # Save Final Statistics
    ###############################################################

    def save_statistics(
    self,
    metrics
    ):

        statistics = {

            "best_validation_accuracy":

                self.best_accuracy,

            "final_loss":

                metrics["loss"],

            "accuracy":

                metrics["accuracy"],

            "precision":

                metrics["precision"],

            "recall":

                metrics["recall"],

            "f1_score":

                metrics["f1"]

        }

        with open(
            TRAIN_STATISTICS,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                statistics,
                f,
                indent=4
            )

        print()
        print("Training statistics saved.")


    ###############################################################
    # Confusion Matrix
    ###############################################################

    def print_confusion_matrix(
        self,
        labels,
        predictions
    ):

        cm = confusion_matrix(
            labels,
            predictions
        )

        print()
        print("===================================================")
        print("CONFUSION MATRIX")
        print("===================================================")
        print(cm)
        print("===================================================")


    ###############################################################
    # Classification Report
    ###############################################################

    def print_classification_report(
        self,
        labels,
        predictions
    ):

        report = classification_report(

            labels,

            predictions,

            target_names=[
                "Truth",
                "Rumor"
            ],

            digits=4,

            zero_division=0

        )

        print()
        print("===================================================")
        print("CLASSIFICATION REPORT")
        print("===================================================")
        print(report)
        print("===================================================")


    ###############################################################
    # Final Evaluation
    ###############################################################

    def final_evaluation(
        self,
        test_loader
    ):

        metrics = self.evaluate_best_model(
            test_loader
        )

        self.print_confusion_matrix(

            metrics["labels"],

            metrics["predictions"]

        )

        self.print_classification_report(

            metrics["labels"],

            metrics["predictions"]

        )

        self.save_training_history()

        self.save_statistics(
            metrics
        )

        return metrics


    ###############################################################
    # Print History
    ###############################################################

    def print_history(self):

        print()

        print("===================================================")
        print("TRAINING HISTORY")
        print("===================================================")

        epochs = len(
            self.history["train_loss"]
        )

        for i in range(epochs):

            print(

                f"Epoch {i+1:02d}"

                f" | "

                f"Train Loss={self.history['train_loss'][i]:.4f}"

                f" | "

                f"Val Loss={self.history['val_loss'][i]:.4f}"

                f" | "

                f"Train Acc={self.history['train_acc'][i]:.4f}"

                f" | "

                f"Val Acc={self.history['val_acc'][i]:.4f}"

            )

        print("===================================================")


    ###############################################################
    # Model Summary
    ###############################################################

    def print_model_summary(self):

        print()

        print("===================================================")
        print("MODEL SUMMARY")
        print("===================================================")

        total_parameters = (

            sum(
                p.numel()
                for p in self.cross_attention.parameters()
            )

            +

            sum(
                p.numel()
                for p in self.classifier.parameters()
            )

        )

        trainable_parameters = (

            sum(
                p.numel()
                for p in self.cross_attention.parameters()
                if p.requires_grad
            )

            +

            sum(
                p.numel()
                for p in self.classifier.parameters()
                if p.requires_grad
            )

        )

        print(
            f"Total Parameters     : {total_parameters:,}"
        )

        print(
            f"Trainable Parameters : {trainable_parameters:,}"
        )

        print("Frozen Modules")

        print("  ✓ Topic Prediction")

        print("  ✓ Temporal Retrieval")

        print()

        print("Trainable Modules")

        print("  ✓ Cross Attention")

        print("  ✓ Veracity Classifier")

        print("===================================================")

    ###############################################################
    # Main
    ###############################################################

def main():

    print()

    print("==========================================================")
    print(" Rumor Veracity Training")
    print("==========================================================")

    ###########################################################
    # Trainer
    ###########################################################

    trainer = InferenceTrainer(

        cross_attention=cross_attention,

        classifier=classifier,

        optimizer=optimizer,

        scheduler=scheduler,

        criterion=criterion,

        device=DEVICE

    )

    ###########################################################
    # Model Summary
    ###########################################################

    trainer.print_model_summary()

    ###########################################################
    # Training
    ###########################################################

    trainer.fit(

        train_loader=train_loader,

        test_loader=test_loader,

        num_epochs=NUM_EPOCHS

    )

    ###########################################################
    # Final Evaluation
    ###########################################################

    metrics = trainer.final_evaluation(

        test_loader=test_loader

    )

    ###########################################################
    # History
    ###########################################################

    trainer.print_history()

    ###########################################################
    # Final Results
    ###########################################################

    print()

    print("==========================================================")
    print("TRAINING COMPLETED")
    print("==========================================================")

    print(
        f"Best Validation Accuracy : "
        f"{trainer.best_accuracy:.4f}"
    )

    print(
        f"Final Test Accuracy      : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Final Precision          : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Final Recall             : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Final F1 Score           : "
        f"{metrics['f1']:.4f}"
    )

    print()

    print("Saved Files")

    print("----------------------------")

    print(BEST_MODEL)

    print(LAST_MODEL)

    print(TRAIN_HISTORY)

    print(TRAIN_STATISTICS)

    print("==========================================================")

    return metrics


###############################################################
# Entry Point
###############################################################

if __name__ == "__main__":

    main()

