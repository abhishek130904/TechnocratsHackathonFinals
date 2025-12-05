"""
Text embedding and classification model interface.

This module handles:
- Loading a pretrained transformer model (e.g., DistilBERT)
- Extracting text embeddings
- Optional: Text-only classification score

We use HuggingFace transformers for text processing.
"""

import logging
import numpy as np
from typing import Optional, List
import os

# Try to import transformers
# If not available, we'll use dummy embeddings
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Text model will use dummy embeddings.")

from app.config import TEXT_MODEL_NAME, MAX_TEXT_LENGTH

logger = logging.getLogger(__name__)


class TextModel:
    """
    Wrapper class for text embedding and classification.
    
    Uses a pretrained transformer model (DistilBERT) to:
    - Convert text to numerical embeddings
    - Optionally predict authenticity from text alone
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the text model.
        
        Args:
            model_name: HuggingFace model name. Defaults to DistilBERT.
        """
        self.model_name = model_name or TEXT_MODEL_NAME
        self.tokenizer: Optional[object] = None
        self.model: Optional[object] = None
        self._loaded = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """
        Load the pretrained transformer model.
        
        This downloads the model from HuggingFace if not cached.
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "Transformers library not available. "
                "Text model will use dummy embeddings."
            )
            return
        
        try:
            logger.info(f"Loading text model: {self.model_name}")
            # Load tokenizer and model from HuggingFace
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Set model to evaluation mode (no training)
            self.model.eval()
            
            self._loaded = True
            logger.info("Text model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading text model: {e}")
            self.tokenizer = None
            self.model = None
            self._loaded = False
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._loaded and self.tokenizer is not None and self.model is not None
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get text embeddings (feature vector) from input text.
        
        This is the main function for multimodal fusion.
        It converts text to a fixed-size numerical vector.
        
        Steps:
        1. Tokenize the text
        2. Run through transformer model
        3. Extract embeddings (typically from [CLS] token or mean pooling)
        
        Args:
            text: Input text (title + description)
            
        Returns:
            Feature vector as numpy array with shape (embedding_dim,)
            For DistilBERT, this is typically 768 dimensions
        """
        if not self.is_loaded():
            logger.warning("Text model not loaded, returning dummy embeddings")
            # Return dummy embedding vector (DistilBERT size)
            return np.zeros(768)
        
        try:
            # Step 1: Tokenize
            # Convert text to token IDs that the model understands
            encoded = self.tokenizer(
                text,
                max_length=MAX_TEXT_LENGTH,
                padding='max_length',  # Pad to max_length
                truncation=True,        # Truncate if too long
                return_tensors='pt'     # Return PyTorch tensors
            )
            
            # Step 2: Get embeddings
            # Run through the model (no gradient computation needed)
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Get the [CLS] token embedding (first token, represents entire sentence)
                # Shape: (batch_size, hidden_size)
                embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting text embeddings: {e}")
            return np.zeros(768)  # Return dummy embeddings on error
    
    def predict_score(self, text: str) -> float:
        """
        Predict authenticity score from text alone.
        
        This is a simple heuristic-based approach.
        For a real implementation, you would train a classifier
        on top of the embeddings.
        
        Args:
            text: Input text
            
        Returns:
            Authenticity score from 0.0 to 1.0
        """
        if not self.is_loaded():
            return 0.5  # Neutral score
        
        # Simple keyword-based scoring
        # In production, you would use a trained classifier
        text_lower = text.lower()
        
        # Positive indicators
        positive_keywords = [
            "authentic", "original", "genuine", "official", "certified",
            "warranty", "guaranteed", "brand new", "sealed"
        ]
        
        # Negative indicators
        negative_keywords = [
            "fake", "replica", "copy", "first copy", "duplicate",
            "unbranded", "generic", "no brand"
        ]
        
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        
        # More nuanced scoring based on keyword presence and text length
        # Longer, detailed descriptions with positive keywords score higher
        text_length_factor = min(len(text.split()) / 50.0, 1.0)  # Normalize by 50 words
        
        # Base score calculation
        if positive_count > 0 and negative_count == 0:
            base_score = 0.75 + (0.15 * min(positive_count / 3.0, 1.0))  # 0.75-0.90
        elif negative_count > 0 and positive_count == 0:
            base_score = 0.25 - (0.15 * min(negative_count / 3.0, 1.0))  # 0.10-0.25
        elif positive_count > negative_count:
            base_score = 0.55 + (0.15 * (positive_count - negative_count) / 3.0)  # 0.55-0.70
        elif negative_count > positive_count:
            base_score = 0.45 - (0.15 * (negative_count - positive_count) / 3.0)  # 0.30-0.45
        else:
            base_score = 0.5
        
        # Adjust based on text length (longer descriptions are often more trustworthy)
        score = base_score + (0.1 * text_length_factor)
        
        # Ensure score stays in valid range
        return max(0.0, min(1.0, score))

