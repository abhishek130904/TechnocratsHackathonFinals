"""
Image preprocessing utilities.

This module contains functions for preprocessing images before
feeding them to the ML model.

The preprocessing must match what the model was trained on:
- Resize to 224x224
- Normalize pixel values
- Convert to RGB format
"""

import numpy as np
from PIL import Image
import io
import logging

from app.config import IMAGE_SIZE, IMAGE_CHANNELS

logger = logging.getLogger(__name__)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess an image for model input.
    
    Steps:
    1. Load image from bytes
    2. Convert to RGB (handles RGBA, grayscale, etc.)
    3. Resize to model input size (224x224)
    4. Convert to numpy array
    5. Normalize pixel values to 0-1 range
    6. Add batch dimension
    
    Args:
        image_bytes: Raw image file bytes
        
    Returns:
        Preprocessed image as numpy array with shape (1, 224, 224, 3)
        Ready to feed into the model
        
    Raises:
        ValueError: If image cannot be loaded or processed
    """
    try:
        # Step 1: Load image from bytes
        # PIL Image can handle various formats (JPEG, PNG, etc.)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Step 2: Convert to RGB
        # This ensures we have 3 channels (some images might be RGBA or grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Step 3: Resize to model input size
        # Most CNN models expect 224x224 images
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Step 4: Convert to numpy array
        # PIL Image -> numpy array (height, width, channels)
        image_array = np.array(image, dtype=np.float32)
        
        # Step 5: Normalize pixel values
        # Images are typically 0-255, but models expect 0-1 or normalized
        # We'll normalize to 0-1 range (can also use ImageNet normalization)
        image_array = image_array / 255.0
        
        # Step 6: Add batch dimension
        # Models expect shape (batch_size, height, width, channels)
        # We add batch_size=1 for single image
        image_array = np.expand_dims(image_array, axis=0)
        
        # Verify shape
        expected_shape = (1, IMAGE_SIZE[1], IMAGE_SIZE[0], IMAGE_CHANNELS)
        if image_array.shape != expected_shape:
            raise ValueError(
                f"Unexpected image shape: {image_array.shape}, "
                f"expected {expected_shape}"
            )
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")


def preprocess_image_imagenet(image_bytes: bytes) -> np.ndarray:
    """
    Alternative preprocessing using ImageNet normalization.
    
    Some models (like EfficientNet) are trained with ImageNet normalization,
    which uses specific mean and std values.
    
    Args:
        image_bytes: Raw image file bytes
        
    Returns:
        Preprocessed image normalized with ImageNet statistics
    """
    # ImageNet mean and std (RGB channels)
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    
    # First do basic preprocessing
    image_array = preprocess_image(image_bytes)
    
    # Remove batch dimension temporarily for normalization
    image_array = image_array[0]
    
    # Normalize each channel
    for i in range(3):
        image_array[:, :, i] = (image_array[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
    
    # Add batch dimension back
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

