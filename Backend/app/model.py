import tensorflow as tf
from PIL import Image
import numpy as np
from app.custom_layers import ImprovedFusionLayer, CustomAttentionLayerWithPositionalEncoding
from app.utils import ssim_metric

def load_model():
    """
    Load the pre-trained TensorFlow model with custom layers.
    """
    model = tf.keras.models.load_model("saved_model/dehaze_model.keras",
                                       custom_objects={'ImprovedFusionLayer': ImprovedFusionLayer, 
                                                       'CustomAttentionLayerWithPositionalEncoding':CustomAttentionLayerWithPositionalEncoding, 
                                                       'ssim_metric':ssim_metric})

    return model

def dehaze_image(image_path, model, output_path):
    """
    Perform image dehazing using the loaded model and save the result.

    Args:
        image_path (str): Path to the input image.
        model (tf.keras.Model): Loaded TensorFlow model.
        output_path (str): Path to save the dehazed image.
    """
    from PIL import Image
    import numpy as np

    # Load and preprocess the image
    image = Image.open(image_path)

    # Convert to RGB if it has an alpha channel
    if image.mode == "RGBA" or image.mode == "LA":
        image = image.convert("RGB")

    image = image.resize((256, 256))  # Match the model input size
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict dehazed image
    dehazed_output = model.predict(image_array)

    # Postprocess and save the output
    dehazed_image = (dehazed_output[0] * 255).astype("uint8")
    Image.fromarray(dehazed_image).save(output_path)
