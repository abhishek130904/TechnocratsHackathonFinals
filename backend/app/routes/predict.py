"""
Prediction route handler.

This module handles the /predict endpoint, which is the main API endpoint
for detecting counterfeit products.

The endpoint:
1. Receives image, text, and metadata
2. Preprocesses the inputs
3. Runs them through ML models
4. Returns authenticity score and decision
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    DecisionEnum,
    ExplanationDetails,
    ErrorResponse
)
from app.services.inference_service import InferenceService
from app.config import (
    APPROVE_THRESHOLD,
    FLAG_THRESHOLD,
    MAX_UPLOAD_SIZE,
    ALLOWED_IMAGE_EXTENSIONS
)

logger = logging.getLogger(__name__)

# Create router for prediction endpoints
router = APIRouter()

# Initialize inference service
# This service handles all the ML model calls
inference_service = InferenceService()


@router.post("/predict", response_model=PredictionResponse)
async def predict_authenticity(
    image: UploadFile = File(..., description="Product image file"),
    title: str = Form(..., description="Product title"),
    description: str = Form(..., description="Product description"),
    seller_rating: Optional[float] = Form(None, description="Seller rating (0.0-5.0)"),
    num_reviews: Optional[int] = Form(None, description="Number of reviews")
):
    """
    Main prediction endpoint.
    
    This endpoint accepts:
    - An image file (multipart/form-data)
    - Text fields (title, description)
    - Optional metadata (seller_rating, num_reviews)
    
    It returns:
    - Authenticity score (0.0 to 1.0)
    - Decision (approve/flag/reject)
    - Explanations for the decision
    
    Args:
        image: Uploaded image file
        title: Product title text
        description: Product description text
        seller_rating: Optional seller rating (0.0 to 5.0)
        num_reviews: Optional number of reviews
        
    Returns:
        PredictionResponse: Contains score, decision, and explanations
        
    Raises:
        HTTPException: If validation fails or processing error occurs
    """
    try:
        # Validate image file
        # Check file extension
        file_extension = "." + image.filename.split(".")[-1].lower() if "." in image.filename else ""
        if file_extension not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            )
        
        # Check file size
        image_content = await image.read()
        if len(image_content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image too large. Maximum size: {MAX_UPLOAD_SIZE / 1024 / 1024} MB"
            )
        
        # Reset file pointer (we read it to check size)
        await image.seek(0)
        
        logger.info(f"Processing prediction request: {image.filename}")
        
        # Call inference service to get prediction
        # This is where the actual ML models are called
        result = await inference_service.predict(
            image=image_content,
            title=title,
            description=description,
            seller_rating=seller_rating,
            num_reviews=num_reviews
        )
        
        # Determine decision based on score and thresholds
        # This is a simple rule-based decision
        if result["score"] >= APPROVE_THRESHOLD:
            decision = DecisionEnum.APPROVE
        elif result["score"] >= FLAG_THRESHOLD:
            decision = DecisionEnum.FLAG
        else:
            decision = DecisionEnum.REJECT
        
        # Build response
        response = PredictionResponse(
            authenticity_score=result["score"],
            decision=decision,
            explanations=ExplanationDetails(
                image_reason=result.get("image_reason"),
                text_reason=result.get("text_reason"),
                metadata_reason=result.get("metadata_reason"),
                heatmap=result.get("heatmap")
            )
        )
        
        logger.info(f"Prediction complete: score={result['score']:.3f}, decision={decision}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions (they're already properly formatted)
        raise
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Error processing prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing prediction: {str(e)}"
        )


@router.post("/feedback")
async def submit_feedback(
    prediction_id: Optional[str] = None,
    was_correct: bool = Form(..., description="Was the prediction correct?"),
    actual_label: Optional[str] = Form(None, description="Actual label (real/fake)"),
    comments: Optional[str] = Form(None, description="Additional comments")
):
    """
    Feedback endpoint for improving the model.
    
    This allows users to provide feedback on predictions,
    which can be used to improve the dataset and retrain models.
    
    For now, this is a placeholder that logs feedback.
    In production, you would store this in a database.
    
    Args:
        prediction_id: Optional ID of the prediction being reviewed
        was_correct: Whether the prediction was correct
        actual_label: Actual label if prediction was wrong
        comments: Additional feedback comments
        
    Returns:
        JSON response confirming feedback was received
    """
    logger.info(f"Feedback received: prediction_id={prediction_id}, was_correct={was_correct}")
    
    # TODO: Store feedback in database or file
    # This data can be used to improve the training dataset
    
    return {
        "status": "success",
        "message": "Feedback received. Thank you for helping improve the model!",
        "prediction_id": prediction_id
    }

