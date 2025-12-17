"""
API server for serving mushroom classification predictions.
This module provides a FastAPI application for serving predictions from the trained model.
"""

import os
import sys
import yaml
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import traceback

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add the project directory to the Python path to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Configure logging with file creation
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "api.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Log startup
logger.info("=" * 50)
logger.info("MUSHROOM CLASSIFICATION API STARTING UP")
logger.info(f"Project root: {project_root}")
logger.info("=" * 50)

# Load configuration with fallback
config = {"paths": {"models": "models"}}  # Default config
try:
    config_path = os.path.join(project_root, "config", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        logger.warning(f"Config file not found at {config_path}, using defaults")
except Exception as e:
    logger.error(f"Error loading configuration: {e}")

# Try to import model loader with error handling
try:
    from src.model_serving.model_loader import MushroomModel

    logger.info("Successfully imported MushroomModel")
    MODEL_LOADER_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import MushroomModel: {e}")
    MODEL_LOADER_AVAILABLE = False


# Define the application with CORS
app = FastAPI(
    title="Mushroom Classification API",
    description="API for predicting mushroom edibility based on various features",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model_cache = {}
api_stats = {
    "startup_time": datetime.now(),
    "total_predictions": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
}


# Input schema definition
class MushroomFeatures(BaseModel):
    """Mushroom features for prediction."""

    # Categorical features
    cap_shape: Optional[str] = Field(None, description="Shape of the cap")
    cap_color: Optional[str] = Field(None, description="Color of the cap")
    cap_surface: Optional[str] = Field(None, description="Surface texture of the cap")
    gill_color: Optional[str] = Field(None, description="Color of the gills")
    gill_attachment: Optional[str] = Field(None, description="Gill attachment type")
    stem_color: Optional[str] = Field(None, description="Color of the stem")
    ring_type: Optional[str] = Field(None, description="Type of ring on the stem")
    habitat: Optional[str] = Field(
        None, description="Habitat where the mushroom is found"
    )

    # Numeric features
    cap_diameter: Optional[float] = Field(None, description="Diameter of the cap in cm")
    stem_height: Optional[float] = Field(None, description="Height of the stem in cm")
    stem_width: Optional[float] = Field(None, description="Width of the stem in cm")

    # Binary features
    does_bruise_or_bleed: Optional[bool] = Field(
        None, description="Whether the mushroom bruises or bleeds"
    )
    has_ring: Optional[bool] = Field(
        None, description="Whether the mushroom has a ring"
    )

    class Config:
        schema_extra = {
            "example": {
                "cap_shape": "convex",
                "cap_color": "brown",
                "cap_surface": "smooth",
                "gill_color": "white",
                "gill_attachment": "free",
                "stem_color": "white",
                "ring_type": "pendant",
                "habitat": "woods",
                "cap_diameter": 5.7,
                "stem_height": 8.2,
                "stem_width": 1.5,
                "does_bruise_or_bleed": True,
                "has_ring": True,
            }
        }


# Output schema definition
class PredictionResult(BaseModel):
    """Result of mushroom classification prediction."""

    prediction: List[int] = Field(
        ..., description="Raw numerical prediction (0=Poisonous, 1=Edible)"
    )
    prediction_label: List[str] = Field(
        ..., description="Human-readable prediction labels"
    )
    probability: Optional[List[float]] = Field(
        None, description="Probability of the positive class (Edible)"
    )
    class_names: List[str] = Field(
        ..., description="Names of the classes (Poisonous, Edible)"
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("API startup event triggered")

    # Check for model files
    models_dir = os.path.join(project_root, "models")
    logger.info(f"Checking models directory: {models_dir}")

    if os.path.exists(models_dir):
        model_files = [
            f for f in os.listdir(models_dir) if f.endswith((".joblib", ".pkl"))
        ]
        logger.info(f"Found model files: {model_files}")
    else:
        logger.warning(f"Models directory does not exist: {models_dir}")
        os.makedirs(models_dir, exist_ok=True)

    # Try to load a default model
    try:
        default_model_path = os.path.join(models_dir, "latest_model.joblib")
        if os.path.exists(default_model_path):
            model_cache["default"] = MushroomModel(
                default_model_path, model_type="xgboost"
            )
            logger.info("Successfully loaded default model")
        else:
            logger.warning(f"Default model not found at {default_model_path}")
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Mushroom Classification API is running",
        "status": "healthy",
        "startup_time": api_stats["startup_time"].isoformat(),
        "total_predictions": api_stats["total_predictions"],
        "models_loaded": len(model_cache),
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(
    features: MushroomFeatures,
    ground_truth: Optional[int] = Query(
        None, description="Ground truth for model evaluation (0=Poisonous, 1=Edible)"
    ),
):
    """Make a prediction based on the provided mushroom features."""
    api_stats["total_predictions"] += 1

    try:
        logger.info(f"Received prediction request #{api_stats['total_predictions']}")

        # Get model (try cache first, then load)
        current_model = None
        if "default" in model_cache:
            current_model = model_cache["default"]
        else:
            # Try to load model dynamically
            models_dir = os.path.join(project_root, "models")
            model_files = []
            if os.path.exists(models_dir):
                model_files = [
                    f for f in os.listdir(models_dir) if f.endswith((".joblib", ".pkl"))
                ]

            if model_files:
                model_path = os.path.join(models_dir, model_files[0])
                current_model = MushroomModel(model_path, model_type="xgboost")
                model_cache["default"] = current_model
                logger.info(f"Dynamically loaded model: {model_files[0]}")
            else:
                raise HTTPException(
                    status_code=503,
                    detail="No trained model available. Please train a model first.",
                )

        # Convert Pydantic model to dict
        feature_dict = features.model_dump()
        logger.info(f"Features received: {feature_dict}")

        # Make prediction
        result = current_model.predict(feature_dict)

        # Add metadata
        result["model_type"] = getattr(current_model, "model_type", "unknown")
        result["prediction_id"] = api_stats["total_predictions"]
        result["timestamp"] = datetime.now().isoformat()

        # Try to store prediction in database (optional)
        try:
            from src.model_serving.database import DatabaseClient

            db_client = DatabaseClient("oltp")
            db_client.store_prediction(feature_dict, result)
            db_client.close()
            logger.info("Stored prediction in database")
        except Exception as db_error:
            logger.warning(f"Could not store prediction in database: {db_error}")

        api_stats["successful_predictions"] += 1
        logger.info(f"Successful prediction: {result['prediction_label']}")
        return result

    except HTTPException:
        api_stats["failed_predictions"] += 1
        raise
    except Exception as e:
        api_stats["failed_predictions"] += 1
        logger.error(f"Error making prediction: {e}")
        logger.error("Traceback: " + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint with detailed status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - api_stats["startup_time"]).total_seconds(),
        "models_loaded": len(model_cache),
        "statistics": api_stats.copy(),
    }

    # Check model availability
    if len(model_cache) == 0:
        models_dir = os.path.join(project_root, "models")
        if os.path.exists(models_dir):
            model_files = [
                f for f in os.listdir(models_dir) if f.endswith((".joblib", ".pkl"))
            ]
            health_status["available_models"] = model_files
            if not model_files:
                health_status["status"] = "degraded"
                health_status["warning"] = "No model files found"
        else:
            health_status["status"] = "degraded"
            health_status["warning"] = "Models directory not found"

    return health_status


@app.get("/models")
async def list_models():
    """List available models."""
    models_dir = os.path.join(project_root, "models")
    available_models = []

    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith((".joblib", ".pkl")):
                file_path = os.path.join(models_dir, file)
                file_stats = os.stat(file_path)
                available_models.append(
                    {
                        "filename": file,
                        "size_bytes": file_stats.st_size,
                        "modified_time": datetime.fromtimestamp(
                            file_stats.st_mtime
                        ).isoformat(),
                        "loaded": file
                        in [
                            model.model_path.split("/")[-1]
                            for model in model_cache.values()
                        ],
                    }
                )

    return {
        "models_directory": models_dir,
        "loaded_models": len(model_cache),
        "available_models": available_models,
    }


@app.post("/test-prediction")
async def test_prediction():
    """Test endpoint with sample data."""
    sample_features = MushroomFeatures(
        cap_shape="convex",
        cap_color="brown",
        cap_surface="smooth",
        gill_color="white",
        gill_attachment="free",
        stem_color="white",
        ring_type="pendant",
        habitat="woods",
        cap_diameter=5.7,
        stem_height=8.2,
        stem_width=1.5,
        does_bruise_or_bleed=True,
        has_ring=True,
    )

    return await predict(sample_features)


# Run the API with uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting API server with uvicorn...")
    logger.info("API will be available at: http://localhost:8000")
    logger.info("API documentation at: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,  # Set to True for development
    )
