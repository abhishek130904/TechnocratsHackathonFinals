"""
Multimodal fusion model interface.

This module combines features from:
- Image model (CNN features)
- Text model (transformer embeddings)
- Metadata (seller rating, review count)

The fusion model learns to combine these different modalities
to make a final authenticity prediction.
"""

import logging
import numpy as np
from typing import Optional
import os

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Fusion model will use weighted average.")

from app.config import FUSION_MODEL_PATH

logger = logging.getLogger(__name__)


class FusionModel:
    """
    Wrapper class for the multimodal fusion model.
    
    This model takes:
    - Image feature vector (from CNN)
    - Text feature vector (from transformer)
    - Metadata features (normalized)
    
    And outputs:
    - Final authenticity score (0.0 to 1.0)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the fusion model.
        
        Args:
            model_path: Optional path to model file. If None, uses default from config.
        """
        self.model_path = model_path or str(FUSION_MODEL_PATH)
        self.model: Optional[object] = None
        self._loaded = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """
        Load the trained fusion model from disk.
        
        If the model doesn't exist, we'll use a simple weighted average
        as a fallback.
        """
        if not TF_AVAILABLE:
            logger.warning(
                "TensorFlow not available. "
                "Fusion model will use weighted average."
            )
            return
        
        if not os.path.exists(self.model_path):
            logger.warning(
                f"Fusion model not found at {self.model_path}. "
                "Using weighted average. Train the model first!"
            )
            return
        
        try:
            logger.info(f"Loading fusion model from {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            self._loaded = True
            logger.info("Fusion model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading fusion model: {e}")
            self.model = None
            self._loaded = False
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._loaded and self.model is not None
    
    def predict(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
        metadata_features: np.ndarray
    ) -> float:
        """
        Predict final authenticity score from combined features.
        
        This is the main fusion function that combines all modalities.
        
        Args:
            image_features: Feature vector from image model
                Shape: (1, image_feature_dim) or (image_feature_dim,)
            text_features: Embedding vector from text model
                Shape: (text_embedding_dim,) - typically 768 for DistilBERT
            metadata_features: Normalized metadata
                Shape: (2,) - [rating_norm, reviews_norm]
                
        Returns:
            Final authenticity score (0.0 to 1.0)
        """
        if not self.is_loaded():
            # Fallback: simple weighted average
            logger.warning("Fusion model not loaded, using weighted average")
            return self._weighted_average(image_features, text_features, metadata_features)
        
        try:
            # Prepare inputs for the model
            # The model expects specific shapes, so we need to ensure consistency
            
            # Flatten image features if needed
            if len(image_features.shape) > 1:
                image_features = image_features.flatten()
            if len(image_features.shape) == 1:
                image_features = image_features.reshape(1, -1)
            
            # Ensure text features are 1D
            if len(text_features.shape) > 1:
                text_features = text_features.flatten()
            if len(text_features.shape) == 1:
                text_features = text_features.reshape(1, -1)
            
            # Ensure metadata features are 1D
            if len(metadata_features.shape) > 1:
                metadata_features = metadata_features.flatten()
            if len(metadata_features.shape) == 1:
                metadata_features = metadata_features.reshape(1, -1)
            
            # The model architecture depends on how it was trained
            # Common approaches:
            # 1. Concatenate all features: [image_features, text_features, metadata_features]
            # 2. Separate inputs for each modality (multi-input model)
            
            # For now, we'll assume concatenation approach
            # If your model uses separate inputs, you'll need to adjust this
            combined_features = np.concatenate([
                image_features,
                text_features,
                metadata_features
            ], axis=1)
            
            # Run prediction
            prediction = self.model.predict(combined_features, verbose=0)
            
            # Extract score (handle different output shapes)
            if isinstance(prediction, np.ndarray):
                if prediction.shape[1] == 2:
                    # Binary classification: return probability of "real" class
                    score = float(prediction[0, 1])
                else:
                    # Single output: already the score
                    score = float(prediction[0, 0])
            else:
                score = float(prediction)
            
            # Ensure score is in valid range
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error during fusion prediction: {e}")
            # Fallback to weighted average on error
            return self._weighted_average(image_features, text_features, metadata_features)
    
    def _weighted_average(
        self,
        image_features: np.ndarray,
        text_features: np.ndarray,
        metadata_features: np.ndarray
    ) -> float:
        """
        Simple weighted average fallback when model is not loaded.
        
        This is a basic fusion strategy that can work reasonably well
        until the full model is trained.
        
        Args:
            image_features: Image feature vector
            text_features: Text embedding vector
            metadata_features: Metadata features
            
        Returns:
            Weighted average score
        """
        # Extract individual scores from feature vectors
        # This is a simplification - in reality, you'd need to map features to scores
        
        # For image: assume first value or mean represents score
        image_score = float(np.mean(image_features)) if image_features.size > 0 else 0.5
        
        # For text: use a simple heuristic (could be improved)
        text_score = 0.5  # Placeholder - would need text model's score
        
        # For metadata: use the rating (first element)
        metadata_score = float(metadata_features[0]) if metadata_features.size > 0 else 0.5
        
        # Weighted combination
        # These weights can be tuned or learned during training
        final_score = (
            0.5 * image_score +      # 50% weight on image
            0.3 * text_score +       # 30% weight on text
            0.2 * metadata_score     # 20% weight on metadata
        )
        
        return np.clip(final_score, 0.0, 1.0)

