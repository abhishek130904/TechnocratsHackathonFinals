"""
Configuration settings for the Counterfeit Detection API.

This module contains all configuration constants, including:
- Model paths
- Decision thresholds
- API settings
- File upload limits

As a beginner, you can modify these values to adjust the behavior of the system.
"""

import os
from pathlib import Path
from typing import Optional

# Base directory paths
# These paths are relative to the project root
BASE_DIR = Path(__file__).parent.parent.parent  # Goes up to project root

# Model file paths
# These are the paths where trained models will be stored
MODELS_DIR = BASE_DIR / "ml_pipeline" / "models"
IMAGE_MODEL_PATH = MODELS_DIR / "image_model.h5"
TEXT_MODEL_NAME = "distilbert-base-uncased"  # HuggingFace model name
FUSION_MODEL_PATH = MODELS_DIR / "fusion_model.h5"

# Decision thresholds for authenticity scoring
# These determine when a product is approved, flagged, or rejected
# Score range: 0.0 (definitely fake) to 1.0 (definitely real)
APPROVE_THRESHOLD = 0.75  # If score >= 0.75, approve
FLAG_THRESHOLD = 0.50     # If score >= 0.50 but < 0.75, flag as suspicious
# If score < 0.50, reject

# Image preprocessing settings
# These match what the model was trained on
IMAGE_SIZE = (224, 224)  # Width x Height in pixels
IMAGE_CHANNELS = 3       # RGB color channels

# Text preprocessing settings
MAX_TEXT_LENGTH = 512    # Maximum tokens for text input (BERT/DistilBERT limit)

# API settings
API_TITLE = "Counterfeit Product Detection API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Multi-modal AI system for detecting counterfeit products from e-commerce listings.

This API accepts:
- Product images
- Product title and description text
- Optional metadata (seller rating, review count)

Returns an authenticity score and decision (approve/flag/reject).
"""

# File upload settings
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB maximum file size
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Optional: Database settings (for future use)
# For now, we'll use in-memory storage or JSON files
DATABASE_ENABLED = False
DATABASE_PATH = BASE_DIR / "data" / "predictions.json"

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "app.log"

# CORS settings (for frontend integration)
CORS_ORIGINS = [
    "http://localhost:3000",  # React dev server default
    "http://localhost:5173",  # Vite dev server default
]

# Optional: Celery/Redis settings (scaffolding for async tasks)
CELERY_ENABLED = False
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

