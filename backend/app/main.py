"""
FastAPI main application entry point.

This is where the FastAPI app is created and configured.
All routes are registered here.

To run the server:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.config import API_TITLE, API_VERSION, API_DESCRIPTION, CORS_ORIGINS
from app.routes import predict
from app.schemas import HealthResponse

# Configure logging
# This helps us debug issues and track what the API is doing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application instance
# FastAPI automatically generates API documentation at /docs
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",  # Swagger UI documentation
    redoc_url="/redoc"  # Alternative ReDoc documentation
)

# Configure CORS (Cross-Origin Resource Sharing)
# This allows the React frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Register route handlers
# These connect URL paths to Python functions
app.include_router(predict.router, prefix="/api", tags=["predictions"])


@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the API status and version.
    This is useful for monitoring and checking if the API is running.
    
    Returns:
        HealthResponse: Status information including model loading status
    """
    logger.info("Health check requested")
    
    # Check actual model loading status
    from app.routes.predict import inference_service
    
    models_status = {
        "image_model": inference_service.image_model.is_loaded() if inference_service.image_model else False,
        "text_model": inference_service.text_model.is_loaded() if inference_service.text_model else False,
        "fusion_model": inference_service.fusion_model.is_loaded() if inference_service.fusion_model else False
    }
    
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        models_loaded=models_status
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler.
    
    Catches any unhandled exceptions and returns a proper error response.
    This prevents the API from crashing and exposing internal errors.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if logger.level == logging.DEBUG else "An unexpected error occurred"
        }
    )


# Application startup event
# This runs when the server starts
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    
    This is where we:
    - Load ML models into memory
    - Initialize database connections
    - Set up other resources
    """
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info("Loading ML models...")
    
    # Load models through the inference service
    # The service will handle lazy loading and error handling
    from app.services.inference_service import InferenceService
    from app.routes.predict import inference_service
    
    # Trigger model loading
    inference_service._ensure_models_loaded()
    
    # Log model status
    models_status = {
        "image_model": inference_service.image_model.is_loaded() if inference_service.image_model else False,
        "text_model": inference_service.text_model.is_loaded() if inference_service.text_model else False,
        "fusion_model": inference_service.fusion_model.is_loaded() if inference_service.fusion_model else False
    }
    
    logger.info(f"Model loading status: {models_status}")
    
    if all(models_status.values()):
        logger.info("All models loaded successfully!")
    else:
        logger.warning("Some models are not loaded. The API will use dummy predictions.")
        logger.warning("Train models first using scripts in ml_pipeline/scripts/")
    
    logger.info("Server ready!")


# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.
    
    Clean up resources when the server shuts down.
    """
    logger.info("Shutting down server...")
    # TODO: Clean up resources (close DB connections, etc.)

