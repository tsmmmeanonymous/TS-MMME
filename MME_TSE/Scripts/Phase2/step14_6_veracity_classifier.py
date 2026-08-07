from pathlib import Path

import torch
import torch.nn as nn

BEST_MODEL = (
    Path(__file__).resolve().parents[2]
    / "Graph"
    / "Experiment"
    / "InferenceModels"
    / "best_inference_model.pt"
)


class VeracityClassifier(nn.Module):
    """
    Veracity Prediction Network

    Input
    -----
    Z_context (768)

    Output
    ------
    Real / Rumor logits
    """

    def __init__(

        self,

        input_dim=768,

        hidden_dim1=128,

        hidden_dim2=64,

        num_classes=2,

        dropout=0.30

    ):

        super().__init__()

        # ==========================================
        # Layer 1
        # ==========================================

        self.fc1 = nn.Linear(

            input_dim,

            hidden_dim1

        )

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(

            dropout

        )

        # ==========================================
        # Layer 2
        # ==========================================

        self.fc2 = nn.Linear(

            hidden_dim1,

            hidden_dim2

        )

        self.relu2 = nn.ReLU()

        self.dropout2 = nn.Dropout(

            dropout

        )

        # ==========================================
        # Output Layer
        # ==========================================

        self.fc3 = nn.Linear(

            hidden_dim2,

            num_classes

        )

        print("\nVeracity Classifier Created Successfully.")




            # =====================================================
    # Forward
    # =====================================================

    def forward(

        self,

        context_embedding

    ):
        """
        Parameters
        ----------
        context_embedding : Tensor

            Shape : (768,)

        Returns
        -------
        logits

        probabilities

        predicted_label
        """

        # ================================================
        # Add Batch Dimension
        # ================================================

        if context_embedding.dim() == 1:

            context_embedding = context_embedding.unsqueeze(0)
        # Move input to the model's device
        context_embedding = context_embedding.to(self.fc1.weight.device)

        # ================================================
        # Layer 1
        # ================================================

        x = self.fc1(

            context_embedding

        )

        x = self.relu1(

            x

        )

        x = self.dropout1(

            x

        )

        # ================================================
        # Layer 2
        # ================================================

        x = self.fc2(

            x

        )

        x = self.relu2(

            x

        )

        x = self.dropout2(

            x

        )

        # ================================================
        # Output Layer
        # ================================================

        logits = self.fc3(

            x

        )

        # ================================================
        # Softmax Probabilities
        # ================================================

        probabilities = torch.softmax(

            logits,

            dim=1

        )

        # ================================================
        # Prediction
        # ================================================

        predicted_label = torch.argmax(

            probabilities,

            dim=1

        )

        return (

            logits.squeeze(0),

            probabilities.squeeze(0),

            predicted_label.item()

        )
    

# =====================================================
# Singleton Veracity Classifier
# =====================================================

_classifier = None

def get_veracity_classifier():

    global _classifier

    if _classifier is None:

        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        _classifier = VeracityClassifier().to(device)

        checkpoint = torch.load(
            BEST_MODEL,
            map_location=device
        )

        _classifier.load_state_dict(
            checkpoint["classifier_state_dict"]
        )

        _classifier.eval()

    return _classifier
    

# =====================================================
# Display Prediction Result
# =====================================================

def display_prediction(

    logits,

    probabilities,

    predicted_label

):
    """
    Display Veracity Prediction Results.
    """

    LABEL_MAP = {

        0: "Real",

        1: "Rumor"

    }

    print("\n==============================================")
    print("VERACITY PREDICTION RESULT")
    print("==============================================")

    print("\nLogits")
    print("----------------------------------------------")

    print(logits.detach().cpu().numpy())

    print("\nClass Probabilities")
    print("----------------------------------------------")

    print(f"Real   : {probabilities[0].item():.4f}")

    print(f"Rumor  : {probabilities[1].item():.4f}")

    print("\nPrediction")
    print("----------------------------------------------")

    print(f"Predicted Label : {predicted_label}")

    print(f"Prediction      : {LABEL_MAP[predicted_label]}")

    confidence = probabilities[

        predicted_label

    ].item()

    print(f"Confidence      : {confidence:.4f}")

    print("\n==============================================")



# =====================================================
# Public API
# =====================================================

def predict_veracity(

    context_embedding

):
    """
    Predict article veracity.

    Parameters
    ----------
    context_embedding : Tensor

    Returns
    -------
    dict
    """

    model = get_veracity_classifier()

    logits, probabilities, predicted_label = model(

        context_embedding

    )

    LABEL_MAP = {

        0: "Real",

        1: "Rumor"

    }

    return {

        "prediction": LABEL_MAP[predicted_label],

        "predicted_label": predicted_label,

        "confidence": probabilities[

            predicted_label

        ].item(),

        "probabilities": probabilities,

        "logits": logits

    }



# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    print("\n==============================================")
    print("VERACITY CLASSIFIER TEST")
    print("==============================================")

    # ==================================================
    # Dummy Context Embedding
    # ==================================================

    context_embedding = torch.randn(

        768

    )

    print("\nInput Summary")
    print("----------------------------------------------")

    print(

        f"Context Embedding Shape : {tuple(context_embedding.shape)}"

    )

    # ==================================================
    # Build Classifier
    # ==================================================

    '''model = VeracityClassifier(

        input_dim=768,

        hidden_dim1=128,

        hidden_dim2=64,

        num_classes=2,

        dropout=0.30

    )'''


    model = get_veracity_classifier()

    # ==================================================
    # Forward
    # ==================================================

    print("\nRunning Veracity Classifier...")

    logits, probabilities, prediction = model(

        context_embedding

    )

    print("Prediction Completed.")

    # ==================================================
    # Display Result
    # ==================================================

    display_prediction(

        logits,

        probabilities,

        prediction

    )

    print("\n==============================================")
    print("VERACITY CLASSIFIER COMPLETED")
    print("==============================================")