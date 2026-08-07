import torch

from PIL import Image

from transformers import (

    BlipProcessor,

    BlipForConditionalGeneration

)


class ImageCaptioning:
    """
    BLIP Image Captioning Module

    Model:
        Salesforce/blip-image-captioning-base

    Input:
        Image path

    Output:
        Natural language caption
    """

    def __init__(

        self,

        model_name="Salesforce/blip-image-captioning-base",

        device=None

    ):

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"


        self.device = device

        print("\nLoading BLIP Image Captioning Model...")

        self.processor = BlipProcessor.from_pretrained(
            model_name
        )

        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print("BLIP Model Loaded Successfully.")


    # ======================================================
    # Generate Caption
    # ======================================================

    @torch.no_grad()
    def generate_caption(

        self,

        image_path,

        max_length=50,

        num_beams=5

    ):
        """
        Generate caption for an image.

        Parameters
        ----------
        image_path : str or Path

        max_length : int

        num_beams : int

        Returns
        -------
        caption : str
        """

        # ---------------------------------------------
        # Load Image
        # ---------------------------------------------

        image = Image.open(

            image_path

        ).convert(

            "RGB"

        )

        # ---------------------------------------------
        # Preprocess
        # ---------------------------------------------

        inputs = self.processor(

            images=image,

            return_tensors="pt"

        )

        inputs = {

            key: value.to(self.device)

            for key, value in inputs.items()

        }

        # ---------------------------------------------
        # Caption Generation
        # ---------------------------------------------

        outputs = self.model.generate(

            **inputs,

            max_length=max_length,

            num_beams=num_beams,

            early_stopping=True

        )

        # ---------------------------------------------
        # Decode Caption
        # ---------------------------------------------

        caption = self.processor.decode(

            outputs[0],

            skip_special_tokens=True

        )

        caption = caption.strip()

        return caption
    


    # ======================================================
    # Generate Caption from PIL Image
    # ======================================================

    @torch.no_grad()
    def generate_from_pil_image(

        self,

        image,

        max_length=50,

        num_beams=5

    ):
        """
        Generate caption directly from a PIL Image.

        Parameters
        ----------
        image : PIL.Image.Image

        max_length : int

        num_beams : int

        Returns
        -------
        caption : str
        """

        if image.mode != "RGB":

            image = image.convert("RGB")

        inputs = self.processor(

            images=image,

            return_tensors="pt"

        )

        inputs = {

            key: value.to(self.device)

            for key, value in inputs.items()

        }

        outputs = self.model.generate(

            **inputs,

            max_length=max_length,

            num_beams=num_beams,

            early_stopping=True

        )

        caption = self.processor.decode(

            outputs[0],

            skip_special_tokens=True

        )

        return caption.strip()
    

# ==========================================================
# Singleton Caption Model
# ==========================================================

_caption_model = None


def get_caption_model():
    """
    Returns a singleton BLIP caption model.

    The model is loaded only once.
    """

    global _caption_model

    if _caption_model is None:

        _caption_model = ImageCaptioning()

    return _caption_model


# ==========================================================
# Public API
# ==========================================================

def generate_caption(

    image_path

):
    """
    Generate caption from image path.

    Parameters
    ----------
    image_path : str

    Returns
    -------
    caption : str
    """

    if image_path is None:

        return None

    image_path = str(image_path).strip()

    if image_path == "":

        return None

    caption_model = get_caption_model()

    return caption_model.generate_caption(

        image_path

    )




# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":

    TEST_IMAGE = input(

        "\nEnter Image Path : "

    ).strip()

    caption_model = ImageCaptioning()

    caption = caption_model.generate_caption(

        TEST_IMAGE

    )

    print("\nGenerated Caption")

    print("------------------------------------")

    print(caption)