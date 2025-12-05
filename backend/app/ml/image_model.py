"""
Image classification model interface.

This module handles loading and using the CNN model for image classification.

The model is typically:
- A transfer learning model (EfficientNetB0, ResNet50, etc.)
- Trained on fake vs real product images
- Outputs a probability score (0.0 = fake, 1.0 = real)
"""

import logging
import numpy as np
from typing import Optional
import os

# Try to import TensorFlow/Keras
# If not available, we'll use dummy predictions
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Image model will use dummy predictions.")

from app.config import IMAGE_MODEL_PATH

logger = logging.getLogger(__name__)


class ImageModel:
    """
    Wrapper class for the image classification CNN model.
    
    This class:
    - Loads the trained model from disk
    - Provides a simple interface for predictions
    - Handles model loading errors gracefully
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the image model.
        
        Args:
            model_path: Optional path to model file. If None, uses default from config.
        """
        self.model_path = model_path or str(IMAGE_MODEL_PATH)
        self.model: Optional[object] = None
        self._loaded = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """
        Load the trained model from disk.
        
        If the model file doesn't exist, we'll use dummy predictions
        until the model is trained.
        """
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Image model will use dummy predictions.")
            return
        
        if not os.path.exists(self.model_path):
            logger.warning(
                f"Image model not found at {self.model_path}. "
                "Using dummy predictions. Train the model first!"
            )
            return
        
        try:
            logger.info(f"Loading image model from {self.model_path}")
            # Load the saved Keras model
            self.model = keras.models.load_model(self.model_path)
            self._loaded = True
            logger.info("Image model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading image model: {e}")
            self.model = None
            self._loaded = False
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded and ready to use.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._loaded and self.model is not None
    
    def predict(self, image_array: np.ndarray) -> np.ndarray:
        """
        Predict authenticity score from image.
        
        Args:
            image_array: Preprocessed image array with shape (1, 224, 224, 3)
            
        Returns:
            Prediction array with shape (1, 1) containing authenticity score (0.0-1.0)
            - Values close to 1.0 = likely authentic
            - Values close to 0.0 = likely fake
        """
        if not self.is_loaded():
            # Return dummy prediction if model not loaded
            logger.warning("Image model not loaded, returning dummy prediction")
            return np.array([[0.5]])  # Neutral score
        
        try:
            # Run inference
            # The model expects batched input, so image_array should have batch dimension
            predictions = self.model.predict(image_array, verbose=0)
            
            # Ensure output is in correct format
            # Some models output (batch, 1), others output (batch, 2) for binary classification
            if predictions.shape[1] == 2:
                # Binary classification with 2 outputs: [fake_prob, real_prob]
                # We want the "real" probability (second column)
                return predictions[:, 1:2]  # Keep batch dimension
            else:
                # Single output: already the authenticity score
                return predictions
            
        except Exception as e:
            logger.error(f"Error during image prediction: {e}")
            # Return neutral score on error
            return np.array([[0.5]])
    
    def get_feature_vector(self, image_array: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from an intermediate layer.
        
        This is useful for:
        - Multimodal fusion (combining with text features)
        - Visualization (Grad-CAM)
        
        Args:
            image_array: Preprocessed image array
            
        Returns:
            Feature vector from the model's feature extraction layers
        """
        if not self.is_loaded():
            logger.warning("Image model not loaded, returning dummy features")
            # Return dummy feature vector (typical size for EfficientNetB0)
            return np.zeros((1, 1280))
        
        try:
            # Get the feature extraction part of the model
            # This is typically everything except the final classification layers
            # For a typical transfer learning model:
            # - Base model (feature extraction)
            # - GlobalAveragePooling
            # - Dense layers (classification)
            
            # We want to get features before the final classification
            # This depends on your model architecture
            # For now, we'll try to get the output of the base model
            
            # If the model has a 'base_model' attribute (common in transfer learning)
            if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                # Try to get features from the second-to-last layer
                # (before final classification)
                feature_model = keras.Model(
                    inputs=self.model.input,
                    outputs=self.model.layers[-2].output
                )
                features = feature_model.predict(image_array, verbose=0)
                return features
            else:
                # Fallback: use the full model output
                return self.model.predict(image_array, verbose=0)
                
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return np.zeros((1, 1280))  # Dummy features

