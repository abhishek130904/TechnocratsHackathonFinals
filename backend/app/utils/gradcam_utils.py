"""
Grad-CAM visualization utilities.

Grad-CAM (Gradient-weighted Class Activation Mapping) generates
heatmaps showing which parts of an image the model focuses on
when making predictions.

This helps explain why the model made a particular decision.
"""

import logging
import numpy as np
from typing import Optional
import io
import base64
from PIL import Image

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Grad-CAM will be disabled.")

logger = logging.getLogger(__name__)


def generate_heatmap(
    model: object,
    image: np.ndarray,
    layer_name: Optional[str] = None,
    class_idx: int = 0
) -> Optional[str]:
    """
    Generate a Grad-CAM heatmap for an image prediction.
    
    Grad-CAM works by:
    1. Running the image through the model
    2. Computing gradients of the prediction with respect to
       the feature maps of a convolutional layer
    3. Weighting the feature maps by these gradients
    4. Creating a heatmap showing important regions
    
    Args:
        model: Keras model
        image: Preprocessed image array with shape (1, H, W, 3)
        layer_name: Name of the convolutional layer to use.
                   If None, uses the last convolutional layer.
        class_idx: Class index to generate heatmap for (0 = fake, 1 = real)
        
    Returns:
        Base64-encoded image string of the heatmap, or None if error
    """
    if not TF_AVAILABLE:
        logger.warning("TensorFlow not available, cannot generate heatmap")
        return None
    
    try:
        # Find the target layer
        if layer_name is None:
            # Find the last convolutional layer
            for layer in reversed(model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            logger.warning("No convolutional layer found for Grad-CAM")
            return None
        
        # Create a model that outputs both the prediction and the feature maps
        grad_model = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            # Get feature maps and predictions
            conv_outputs, predictions = grad_model(image)
            # Get the score for the target class
            if predictions.shape[1] == 2:
                # Binary classification: use the "real" class (index 1)
                class_channel = predictions[:, 1]
            else:
                # Single output
                class_channel = predictions[:, 0]
        
        # Compute gradients of the class score with respect to feature maps
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Compute importance weights (global average pooling of gradients)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps by importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap to 0-1 range
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match original image size
        heatmap_resized = np.array(Image.fromarray(heatmap).resize(
            (image.shape[2], image.shape[1]),
            Image.Resampling.BILINEAR
        ))
        
        # Convert to RGB heatmap (red = important, blue = not important)
        heatmap_colored = np.zeros((*heatmap_resized.shape, 3), dtype=np.uint8)
        heatmap_colored[:, :, 0] = (heatmap_resized * 255).astype(np.uint8)  # Red channel
        
        # Overlay on original image (optional - for now just return heatmap)
        # You could blend: overlay = 0.4 * original + 0.6 * heatmap
        
        # Convert to base64
        pil_image = Image.fromarray(heatmap_colored)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        return None

