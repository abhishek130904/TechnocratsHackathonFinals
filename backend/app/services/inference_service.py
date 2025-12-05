"""
Inference Service.

This service orchestrates the ML models to make predictions.

It:
1. Preprocesses the image
2. Extracts text embeddings
3. Combines features (multimodal fusion)
4. Generates explanations and heatmaps

This is the main business logic layer that sits between the API routes
and the ML model implementations.
"""

import logging
from typing import Dict, Optional, Any
import numpy as np

from app.ml.image_model import ImageModel
from app.ml.text_model import TextModel
from app.ml.fusion_model import FusionModel
from app.ml.preprocessing import preprocess_image
from app.utils.gradcam_utils import generate_heatmap

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Service for running inference on product data.
    
    This class coordinates all ML models to produce a final prediction.
    """
    
    def __init__(self):
        """
        Initialize the inference service.
        
        Loads all ML models into memory.
        Models are loaded lazily (only when first used) to avoid
        loading errors if models don't exist yet.
        """
        self.image_model: Optional[ImageModel] = None
        self.text_model: Optional[TextModel] = None
        self.fusion_model: Optional[FusionModel] = None
        self._models_loaded = False
        
    def _ensure_models_loaded(self):
        """
        Lazy loading of models.
        
        Only loads models when they're first needed.
        This prevents errors during startup if models don't exist yet.
        """
        if not self._models_loaded:
            try:
                logger.info("Loading ML models...")
                self.image_model = ImageModel()
                self.text_model = TextModel()
                self.fusion_model = FusionModel()
                self._models_loaded = True
                logger.info("ML models loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load models: {e}. Using dummy predictions.")
                self._models_loaded = False
    
    async def predict(
        self,
        image: bytes,
        title: str,
        description: str,
        seller_rating: Optional[float] = None,
        num_reviews: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main prediction method.
        
        This is the core function that:
        1. Processes the image through the CNN
        2. Processes the text through the NLP model
        3. Combines features through the fusion model
        4. Generates explanations
        
        Args:
            image: Raw image bytes
            title: Product title text
            description: Product description text
            seller_rating: Optional seller rating (0.0-5.0)
            num_reviews: Optional number of reviews
            
        Returns:
            Dictionary containing:
                - score: Authenticity score (0.0 to 1.0)
                - image_reason: Explanation from image analysis
                - text_reason: Explanation from text analysis
                - metadata_reason: Explanation from metadata
                - heatmap: Optional base64-encoded heatmap image
        """
        self._ensure_models_loaded()
        
        try:
            # Step 1: Preprocess image
            # Convert raw bytes to numpy array and resize/normalize
            processed_image = preprocess_image(image)
            
            # Step 2: Get image features
            # Run image through CNN to get feature vector
            if self.image_model and self.image_model.is_loaded():
                image_features = self.image_model.predict(processed_image)
                image_score = float(image_features[0])  # Assuming binary output
                image_reason = self._generate_image_reason(image_score)
            else:
                # Dummy prediction if model not loaded
                logger.warning("Image model not loaded, using dummy prediction")
                image_features = np.array([0.5])  # Neutral score
                image_score = 0.5
                image_reason = "Image model not available (using placeholder)"
            
            # Step 3: Get text features
            # Combine title and description, then get embeddings
            combined_text = f"{title}. {description}"
            if self.text_model and self.text_model.is_loaded():
                text_features = self.text_model.get_embeddings(combined_text)
                text_score = self.text_model.predict_score(combined_text)
                text_reason = self._generate_text_reason(text_score, title, description)
            else:
                # Dummy prediction if model not loaded
                logger.warning("Text model not loaded, using dummy prediction")
                text_features = np.zeros(768)  # DistilBERT embedding size
                text_score = 0.5
                text_reason = "Text model not available (using placeholder)"
            
            # Step 4: Prepare metadata features
            # Normalize metadata to 0-1 range for model input
            metadata_features = self._prepare_metadata_features(seller_rating, num_reviews)
            metadata_reason = self._generate_metadata_reason(seller_rating, num_reviews)
            
            # Step 5: Multimodal fusion
            # Combine all features and get final score
            if self.fusion_model and self.fusion_model.is_loaded():
                final_score = self.fusion_model.predict(
                    image_features=image_features,
                    text_features=text_features,
                    metadata_features=metadata_features
                )
            else:
                # Simple weighted average if fusion model not loaded
                logger.warning("Fusion model not loaded, using weighted average")
                # Use actual text_score and metadata, but image is dummy (0.5)
                # Adjust weights to give more importance to text and metadata when image model isn't available
                if not (self.image_model and self.image_model.is_loaded()):
                    # If image model not loaded, reduce its weight and increase text/metadata
                    final_score = (
                        0.2 * image_score +      # 20% weight on image (dummy)
                        0.5 * text_score +       # 50% weight on text (actual analysis)
                        0.3 * metadata_features[0]  # 30% weight on metadata (actual data)
                    )
                else:
                    # If image model is loaded, use standard weights
                    final_score = (
                        0.5 * image_score +      # 50% weight on image
                        0.3 * text_score +       # 30% weight on text
                        0.2 * metadata_features[0]  # 20% weight on metadata
                    )
            
            # Step 6: Generate heatmap (optional, can be slow)
            # This shows which parts of the image the model focused on
            heatmap = None
            try:
                if self.image_model and self.image_model.is_loaded():
                    heatmap = generate_heatmap(
                        model=self.image_model.model,
                        image=processed_image,
                        layer_name=None  # Will use last conv layer
                    )
            except Exception as e:
                logger.warning(f"Could not generate heatmap: {e}")
                heatmap = None
            
            return {
                "score": float(np.clip(final_score, 0.0, 1.0)),  # Ensure 0-1 range
                "image_reason": image_reason,
                "text_reason": text_reason,
                "metadata_reason": metadata_reason,
                "heatmap": heatmap
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            raise
    
    def _prepare_metadata_features(
        self,
        seller_rating: Optional[float],
        num_reviews: Optional[int]
    ) -> np.ndarray:
        """
        Prepare metadata features for the model.
        
        Normalizes metadata to 0-1 range:
        - seller_rating: Already 0-5, normalize to 0-1
        - num_reviews: Log-scale normalization (reviews can vary widely)
        
        Args:
            seller_rating: Rating from 0.0 to 5.0
            num_reviews: Number of reviews
            
        Returns:
            numpy array of normalized features
        """
        # Normalize seller rating (0-5 -> 0-1)
        rating_norm = (seller_rating / 5.0) if seller_rating is not None else 0.5
        
        # Normalize review count using log scale
        # This handles the wide range of review counts (0 to millions)
        if num_reviews is not None and num_reviews > 0:
            # Log scale: log(1 + reviews) / log(1 + max_reviews)
            # Using 10000 as a reasonable "high" number of reviews
            reviews_norm = np.log1p(num_reviews) / np.log1p(10000)
            reviews_norm = min(reviews_norm, 1.0)  # Cap at 1.0
        else:
            reviews_norm = 0.0
        
        return np.array([rating_norm, reviews_norm])
    
    def _generate_image_reason(self, score: float) -> str:
        """
        Generate explanation based on image analysis score.
        
        Args:
            score: Image authenticity score (0.0 to 1.0)
            
        Returns:
            Human-readable explanation
        """
        if score >= 0.7:
            return "Image shows characteristics consistent with authentic products"
        elif score >= 0.4:
            return "Image shows some inconsistencies that warrant review"
        else:
            return "Image shows significant inconsistencies with authentic products"
    
    def _generate_text_reason(self, score: float, title: str, description: str) -> str:
        """
        Generate explanation based on text analysis.
        
        Looks for suspicious keywords and patterns.
        
        Args:
            score: Text authenticity score (0.0 to 1.0)
            title: Product title
            description: Product description
            
        Returns:
            Human-readable explanation
        """
        # Simple keyword-based detection (can be enhanced)
        suspicious_keywords = [
            "first copy", "replica", "fake", "duplicate", "copy",
            "unbranded", "no brand", "generic"
        ]
        
        text_lower = (title + " " + description).lower()
        found_keywords = [kw for kw in suspicious_keywords if kw in text_lower]
        
        if found_keywords:
            return f"Description contains suspicious terms: {', '.join(found_keywords)}"
        elif score >= 0.7:
            return "Text description appears authentic and professional"
        elif score >= 0.4:
            return "Text description shows some inconsistencies"
        else:
            return "Text description shows significant red flags"
    
    def _generate_metadata_reason(
        self,
        seller_rating: Optional[float],
        num_reviews: Optional[int]
    ) -> str:
        """
        Generate explanation based on metadata.
        
        Args:
            seller_rating: Seller rating (0.0-5.0)
            num_reviews: Number of reviews
            
        Returns:
            Human-readable explanation
        """
        reasons = []
        
        if seller_rating is not None:
            if seller_rating < 2.0:
                reasons.append(f"Very low seller rating ({seller_rating:.1f}/5.0)")
            elif seller_rating < 3.5:
                reasons.append(f"Below-average seller rating ({seller_rating:.1f}/5.0)")
            elif seller_rating >= 4.5:
                reasons.append(f"High seller rating ({seller_rating:.1f}/5.0)")
        
        if num_reviews is not None:
            if num_reviews < 10:
                reasons.append(f"Very few reviews ({num_reviews})")
            elif num_reviews < 50:
                reasons.append(f"Limited review history ({num_reviews} reviews)")
            elif num_reviews >= 1000:
                reasons.append(f"Established seller with {num_reviews} reviews")
        
        if not reasons:
            return "Metadata information not provided"
        
        return ". ".join(reasons) + "."

