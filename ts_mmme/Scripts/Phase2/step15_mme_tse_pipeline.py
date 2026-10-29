import sys
from pathlib import Path
import torch
# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )

# ==========================================================
# Imports
# ==========================================================

from Modules.image_captioning import generate_caption

from .step14_3_topic_prediction import predict_topic

from .step14_4_retrieve_temporal_memory import retrieve_temporal_memory

from .step14_5_cross_attention import generate_context_embedding

from .step14_6_veracity_classifier import predict_veracity

from .step14_7_temporal_spatial_veracity_evolution import (
    analyse_temporal_spatial_evolution,
    display_evolution_result
)


# ==========================================================
# Ablation Configuration
# ==========================================================

print("\nSelect Ablation Mode")
print("1. FULL")
print("2. NO_EVOLUTION")
print("3. NO_CROSS_ATTENTION")
print("4. NO_TEMPORAL")
print("5. NO_SPATIAL")

choice = input("\nEnter Choice : ").strip()

ABLATION_MAP = {
    "1": "FULL",
    "2": "NO_EVOLUTION",
    "3": "NO_CROSS_ATTENTION",
    "4": "NO_TEMPORAL",
    "5": "NO_SPATIAL"
}

ABLATION_MODE = ABLATION_MAP.get(choice, "FULL")

print(f"\nRunning : {ABLATION_MODE}")

# ==========================================================
# MME-TSE Pipeline
# ==========================================================

class MMETSEPipeline:
    """
    End-to-End MME-TSE Prediction Pipeline

    Input
    -----
    Article Text

    Image Path (optional)

    Month

    Location

    Output
    ------
    Final Veracity Prediction
    """

    def __init__(self):

        print("\n==============================================")
        print("MME-TSE PIPELINE")
        print("==============================================")

        print("\nPipeline Initialized Successfully.")

        print("\nAvailable Modules")

        print("----------------------------------------------")

        print("✓ Image Captioning")

        print("✓ Topic Prediction")

        print("✓ Temporal Memory Retrieval")

        print("✓ Cross Attention")

        print("✓ Veracity Prediction")

        print("✓ Temporal-Spatial Veracity Evolution")

        print("\nPipeline Ready.")


    
    def _log(self, message="", logger=None):
        """
        Print to console and optionally write to log.
        """
        print(message)

        if logger is not None:
            logger.info(message)
    # ======================================================
    # Predict
    # ======================================================

    def predict(

        self,

        article_text,

        month,

        location,

        image_path=None,

        article_id=None

    ):
        """
        End-to-End MME-TSE Prediction.

        Parameters
        ----------
        article_text : str

        month : str

            Format : YYYY-MM

        location : str

        image_path : str or None

        Returns
        -------
        dict
        """

        print("\n==============================================")
        print("STARTING MME-TSE PIPELINE")
        print("==============================================")

        # ==================================================
        # Step 1
        # Topic Prediction
        # ==================================================

        print("\nStep 1 : Topic Prediction")

        topic_result = predict_topic(
            article_text,
            article_id=article_id,     # MMM-Trend fix: image_path was passed positionally into article_id
            image_path=image_path
        )

        # ==================================================
        # Step 2
        # Retrieve Temporal Memory
        # ==================================================

        print("\nStep 2 : Temporal Memory Retrieval")

        # Ablation
        retrieval_month = month
        retrieval_location = location

        if ABLATION_MODE == "NO_TEMPORAL":
            print("Ablation : Temporal information disabled")
            retrieval_month = None

        if ABLATION_MODE == "NO_SPATIAL":
            print("Ablation : Spatial information disabled")
            retrieval_location = None

        memory_result = retrieve_temporal_memory(

            topic_result["top3_topics"],

            retrieval_month,

            retrieval_location,

            topic_result["article_embedding"]

        )

        # --------------------------------------------------
        # No Matching Memory
        # --------------------------------------------------

        if not memory_result["memory_found"]:

            print("\nPipeline Stopped.")

            return {

                "success": False,

                "message": "No Temporal Memory Found.",

                "topic_result": topic_result,

                "memory_result": memory_result

            }

        # ==================================================
        # Step 3
        # Cross Attention
        # ==================================================

        print("\nStep 3 : Cross Attention")

        memory_embeddings = torch.stack(

            [

                m["memory_embedding"]

                for m in memory_result["retrieved_memories"]

            ]

        )

        retrieval_scores = torch.tensor(

            [

                m["retrieval_score"]

                for m in memory_result["retrieved_memories"]

            ],

            dtype=torch.float32,

            device=memory_embeddings.device

        )

        from . import image_proj as _imgproj   # MMM-Trend: image-only variant, CLIP feature projection
        if _imgproj.ENABLED:
            topic_result["article_embedding"] = _imgproj.apply(
                topic_result["article_embedding"], article_id,
                _imgproj.load_trained(topic_result["article_embedding"].device)).detach()

        if ABLATION_MODE == "NO_CROSS_ATTENTION":

            context_result = {

                "context_embedding": topic_result["article_embedding"].clone().detach(),

                "attention_weight": None

            }

        else:

            context_result = generate_context_embedding(

                topic_result["article_embedding"],

                memory_embeddings,

                retrieval_scores

            )

        # ==================================================
        # Step 4
        # Veracity Prediction
        # ==================================================

        print("\nStep 4 : Veracity Prediction")

        veracity_result = predict_veracity(

            context_result["context_embedding"]

        )

        # ==================================================
        # Step 5
        # Temporal-Spatial Veracity Evolution
        # ==================================================

        print("\nStep 5 : Temporal-Spatial Veracity Evolution")

        if ABLATION_MODE == "NO_EVOLUTION":

            print("Ablation : Evolution module disabled")

            evolution_result = None

        else:

            evolution_result = analyse_temporal_spatial_evolution(

                memory_result["retrieved_memories"]

            )

        # ==================================================
        # Final Result
        # ==================================================

        return {

            "success": True,

            "topic_result": topic_result,

            "memory_result": memory_result,

            "context_result": context_result,

            "veracity_result": veracity_result,

            "evolution_result": evolution_result

        }
    


    # ======================================================
    # Display Result
    # ======================================================

    def display_result(self, result, log_fn=print):
        """
        Display final MME-TSE prediction result.
        """

        log_fn("\n==============================================")
        log_fn("MME-TSE PREDICTION RESULT")
        log_fn("==============================================")

        # --------------------------------------------------
        # Prediction Failed
        # --------------------------------------------------

        if not result["success"]:

            log_fn("\nPrediction Status")

            log_fn("----------------------------------------------")

            log_fn("Failed")

            log_fn(f"\nReason : {result['message']}")

            return

        topic_result = result["topic_result"]

        memory_result = result["memory_result"]

        context_result = result["context_result"]

        veracity_result = result["veracity_result"]

        evolution_result = result["evolution_result"]

        # --------------------------------------------------
        # Topic Prediction
        # --------------------------------------------------

        log_fn("\nPredicted Topic")

        log_fn("----------------------------------------------")

        log_fn(
            f"Topic ID        : "
            f"{topic_result['predicted_topic']}"
        )

        log_fn(
            f"Topic Name      : "
            f"{topic_result['predicted_topic_name']}"
        )

        log_fn(
            f"Similarity      : "
            f"{topic_result['confidence']:.4f}"
        )

        # --------------------------------------------------
        # Top-3 Topics
        # --------------------------------------------------

        log_fn("\nTop-3 Candidate Topics")

        log_fn("----------------------------------------------")

        for i, topic in enumerate(

            topic_result["top3_topics"],

            start=1

        ):

            log_fn(

                f"Rank {i}"

            )

            log_fn(

                f"Topic ID     : {topic['topic_id']}"

            )

            log_fn(

                f"Topic Name   : {topic['topic_name']}"

            )

            log_fn(

                f"Similarity   : {topic['similarity']:.4f}"

            )

            log_fn()

        # --------------------------------------------------
        # Memory Retrieval
        # --------------------------------------------------

        log_fn("\nRetrieved Memories")

        log_fn("----------------------------------------------")

        for rank, memory in enumerate(

            memory_result["retrieved_memories"],

            start=1

        ):

            log_fn(f"\nRank {rank}")

            log_fn(f"Memory Index    : {memory['memory_index']}")

            log_fn(f"Topic           : {memory['topic_id']}")

            log_fn(f"Month           : {memory['month']}")

            log_fn(f"Location        : {memory['location']}")

            log_fn(f"Retrieval Score : {memory['retrieval_score']:.4f}")

        # --------------------------------------------------
        # Context Embedding
        # --------------------------------------------------

        log_fn("\nContext Embedding")

        log_fn("----------------------------------------------")

        log_fn(

            f"Dimension : "

            f"{context_result['context_embedding'].shape[0]}"

        )

        if context_result["attention_weight"] is not None:

            log_fn(
                f"Attention Shape : "
                f"{tuple(context_result['attention_weight'].shape)}"
            )

        else:

            log_fn("Attention Shape : Not Used")

        # --------------------------------------------------
        # Veracity Prediction
        # --------------------------------------------------

        log_fn("\nFinal Veracity Prediction")

        log_fn("----------------------------------------------")

        log_fn(

            f"Prediction : "

            f"{veracity_result['prediction']}"

        )

        log_fn(

            f"Confidence : "

            f"{veracity_result['confidence']:.4f}"

        )

        probabilities = veracity_result["probabilities"]

        log_fn(

            f"Real Probability   : "

            f"{probabilities[0].item():.4f}"

        )

        log_fn(

            f"Rumor Probability  : "

            f"{probabilities[1].item():.4f}"

        )

        # --------------------------------------------------
        # Temporal-Spatial Veracity Evolution
        # --------------------------------------------------

        if evolution_result is not None:

            display_evolution_result(

                evolution_result,

                log_fn

            )

        log_fn("\n==============================================")

        log_fn("PIPELINE COMPLETED SUCCESSFULLY")

        log_fn("==============================================")



# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("MME-TSE END-TO-END PREDICTION")
    print("==============================================")

    # ======================================================
    # User Input
    # ======================================================

    article_text = input(

        "\nEnter Article Text :\n\n"

    ).strip()

    image_path = input(

        "\nEnter Image Path (Optional) : "

    ).strip()

    if image_path == "":

        image_path = None

    month = input(

        "\nEnter Month (YYYY-MM) : "

    ).strip()

    location = input(

        "\nEnter Location : "

    ).strip()

    # ======================================================
    # Build Pipeline
    # ======================================================

    pipeline = MMETSEPipeline()

    # ======================================================
    # Prediction
    # ======================================================

    result = pipeline.predict(

        article_text=article_text,

        month=month,

        location=location,

        image_path=image_path

    )

    # ======================================================
    # Display Result
    # ======================================================

    pipeline.display_result(

        result

    )




