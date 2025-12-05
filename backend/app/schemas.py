"""
Pydantic schemas for request/response validation.

Pydantic schemas ensure that:
1. Incoming data matches expected types and formats
2. Response data is properly structured
3. API documentation is automatically generated

This is important for type safety and preventing errors.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class DecisionEnum(str, Enum):
    """
    Enumeration for the authenticity decision.
    
    This ensures we only return valid decision values.
    """
    APPROVE = "approve"  # Product is likely authentic
    FLAG = "flag"        # Product is suspicious, needs review
    REJECT = "reject"    # Product is likely counterfeit


class PredictionRequest(BaseModel):
    """
    Request schema for the /predict endpoint.
    
    This defines what data the client must send.
    Note: The image file is handled separately via multipart/form-data,
    but we include it here for documentation purposes.
    """
    title: str = Field(
        ...,
        description="Product title",
        min_length=1,
        max_length=200,
        example="Nike Air Max 90 Running Shoes"
    )
    description: str = Field(
        ...,
        description="Product description",
        min_length=1,
        max_length=2000,
        example="Authentic Nike Air Max 90 with original box and tags"
    )
    seller_rating: Optional[float] = Field(
        None,
        description="Seller rating (0.0 to 5.0)",
        ge=0.0,
        le=5.0,
        example=4.5
    )
    num_reviews: Optional[int] = Field(
        None,
        description="Number of reviews for the seller",
        ge=0,
        example=1250
    )

    @validator('title', 'description')
    def validate_text_not_empty(cls, v):
        """Ensure text fields are not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Text field cannot be empty")
        return v.strip()


class ExplanationDetails(BaseModel):
    """
    Detailed explanations for the prediction.
    
    These help users understand why a product was flagged or rejected.
    """
    image_reason: Optional[str] = Field(
        None,
        description="Explanation based on image analysis",
        example="Logo region shows inconsistencies with authentic products"
    )
    text_reason: Optional[str] = Field(
        None,
        description="Explanation based on text analysis",
        example="Description contains suspicious terms like 'first copy'"
    )
    metadata_reason: Optional[str] = Field(
        None,
        description="Explanation based on metadata analysis",
        example="Low seller rating (2.1/5.0) and very few reviews (3)"
    )
    heatmap: Optional[str] = Field(
        None,
        description="Base64-encoded heatmap image showing important regions",
        example=None
    )


class PredictionResponse(BaseModel):
    """
    Response schema for the /predict endpoint.
    
    This defines what the API returns to the client.
    """
    authenticity_score: float = Field(
        ...,
        description="Authenticity score from 0.0 (fake) to 1.0 (real)",
        ge=0.0,
        le=1.0,
        example=0.83
    )
    decision: DecisionEnum = Field(
        ...,
        description="Decision: approve, flag, or reject",
        example=DecisionEnum.FLAG
    )
    explanations: ExplanationDetails = Field(
        ...,
        description="Detailed explanations for the prediction"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True  # Use enum values in JSON output


class HealthResponse(BaseModel):
    """
    Response schema for the health check endpoint.
    """
    status: str = Field(..., description="API status", example="ok")
    version: str = Field(..., description="API version", example="1.0.0")
    models_loaded: Dict[str, bool] = Field(
        ...,
        description="Status of loaded ML models",
        example={"image_model": True, "text_model": True, "fusion_model": True}
    )


class ErrorResponse(BaseModel):
    """
    Standard error response schema.
    
    All errors should follow this format for consistency.
    """
    error: str = Field(..., description="Error message", example="Invalid image format")
    detail: Optional[str] = Field(
        None,
        description="Additional error details",
        example="Only JPG, PNG, and WEBP formats are supported"
    )

