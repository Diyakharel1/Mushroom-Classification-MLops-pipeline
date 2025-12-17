"""
Production-ready API server for mushroom classification predictions.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import traceback

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add the project directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Import model loader
try:
    from src.model_serving.model_loader import MushroomModel

    MODEL_LOADER_AVAILABLE = True
    logger.info("Successfully imported MushroomModel")
except ImportError as e:
    MODEL_LOADER_AVAILABLE = False
    logger.error(f"Failed to import MushroomModel: {e}")

    # Fallback dummy model for testing
    class MushroomModel:
        def __init__(self, model_path, model_type="dummy"):
            self.model_path = model_path
            self.model_type = model_type
            logger.warning("Using dummy model - train a real model first!")

        def predict(self, features):
            import random

            pred = random.choice([0, 1])
            return {
                "prediction": [pred],
                "prediction_label": ["Poisonous" if pred == 0 else "Edible"],
                "probability": [random.random()],
                "class_names": ["Poisonous", "Edible"],
            }


# Initialize FastAPI app
app = FastAPI(
    title="Mushroom Classification API",
    description="Production API for mushroom edibility prediction",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model_cache = {}
api_stats = {
    "startup_time": datetime.now(),
    "total_predictions": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
}


# Pydantic models
class MushroomFeatures(BaseModel):
    """Input features for mushroom classification."""

    # Categorical features
    cap_shape: Optional[str] = Field(None, description="Shape of the cap")
    cap_color: Optional[str] = Field(None, description="Color of the cap")
    cap_surface: Optional[str] = Field(None, description="Surface texture")
    gill_color: Optional[str] = Field(None, description="Gill color")
    gill_attachment: Optional[str] = Field(None, description="Gill attachment")
    stem_color: Optional[str] = Field(None, description="Stem color")
    ring_type: Optional[str] = Field(None, description="Ring type")
    habitat: Optional[str] = Field(None, description="Habitat")

    # Numerical features
    cap_diameter: Optional[float] = Field(None, description="Cap diameter (cm)")
    stem_height: Optional[float] = Field(None, description="Stem height (cm)")
    stem_width: Optional[float] = Field(None, description="Stem width (cm)")

    # Boolean features
    does_bruise_or_bleed: Optional[bool] = Field(None, description="Bruises/bleeds")
    has_ring: Optional[bool] = Field(None, description="Has ring")

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


class PredictionResult(BaseModel):
    """Prediction result schema."""

    prediction: List[int] = Field(..., description="Numerical prediction")
    prediction_label: List[str] = Field(..., description="Human-readable labels")
    probability: Optional[List[float]] = Field(None, description="Class probabilities")
    class_names: List[str] = Field(..., description="Class names")
    model_type: Optional[str] = Field(None, description="Model type used")
    prediction_id: Optional[int] = Field(None, description="Prediction ID")
    timestamp: Optional[str] = Field(None, description="Prediction timestamp")


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    logger.info("🚀 Starting Mushroom Classification API")

    # Load default model if available
    models_dir = os.path.join(project_root, "models")
    if os.path.exists(models_dir):
        model_files = [
            f for f in os.listdir(models_dir) if f.endswith((".joblib", ".pkl"))
        ]
        if model_files:
            try:
                default_model = os.path.join(models_dir, "latest_model.joblib")
                if os.path.exists(default_model):
                    model_cache["default"] = MushroomModel(default_model, "xgboost")
                    logger.info("✅ Default model loaded successfully")
                else:
                    # Load first available model
                    first_model = os.path.join(models_dir, model_files[0])
                    model_cache["default"] = MushroomModel(first_model, "xgboost")
                    logger.info(f"✅ Loaded model: {model_files[0]}")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
        else:
            logger.warning("⚠️ No model files found in models directory")
    else:
        logger.warning("⚠️ Models directory not found")
        os.makedirs(models_dir, exist_ok=True)


@app.get("/")
async def root():
    """Root endpoint with API status."""
    return {
        "message": "🍄 Mushroom Classification API",
        "version": "2.0.0",
        "status": "healthy",
        "startup_time": api_stats["startup_time"].isoformat(),
        "total_predictions": api_stats["total_predictions"],
        "models_loaded": len(model_cache),
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "models": "/models",
            "test": "/test-prediction",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    uptime = (datetime.now() - api_stats["startup_time"]).total_seconds()

    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": uptime,
        "models_loaded": len(model_cache),
        "model_loader_available": MODEL_LOADER_AVAILABLE,
        "statistics": api_stats.copy(),
    }

    # Check model availability
    if len(model_cache) == 0:
        status["status"] = "degraded"
        status["warning"] = "No models loaded"

    return status


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
                        "loaded": any(
                            file in model.model_path for model in model_cache.values()
                        ),
                    }
                )

    return {
        "models_directory": models_dir,
        "loaded_models": len(model_cache),
        "available_models": available_models,
    }


@app.post("/predict", response_model=PredictionResult)
async def predict(features: MushroomFeatures):
    """Make mushroom classification prediction."""
    api_stats["total_predictions"] += 1

    try:
        logger.info(f"🔮 Prediction request #{api_stats['total_predictions']}")

        # Get or load model
        if "default" not in model_cache:
            models_dir = os.path.join(project_root, "models")
            model_files = []
            if os.path.exists(models_dir):
                model_files = [
                    f for f in os.listdir(models_dir) if f.endswith((".joblib", ".pkl"))
                ]

            if not model_files:
                raise HTTPException(
                    status_code=503,
                    detail="No trained model available. Please train a model first.",
                )

            # Load first available model
            model_path = os.path.join(models_dir, model_files[0])
            model_cache["default"] = MushroomModel(model_path, "xgboost")
            logger.info(f"Dynamically loaded model: {model_files[0]}")

        current_model = model_cache["default"]

        # Convert input to dict and make prediction
        feature_dict = features.model_dump()
        result = current_model.predict(feature_dict)

        # Add metadata
        result.update(
            {
                "model_type": getattr(current_model, "model_type", "unknown"),
                "prediction_id": api_stats["total_predictions"],
                "timestamp": datetime.now().isoformat(),
            }
        )

        api_stats["successful_predictions"] += 1
        logger.info(f"✅ Prediction: {result['prediction_label']}")

        return result

    except HTTPException:
        api_stats["failed_predictions"] += 1
        raise
    except Exception as e:
        api_stats["failed_predictions"] += 1
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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


if __name__ == "__main__":
    logger.info("🚀 Starting API server...")
    logger.info("📖 API documentation: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", reload=False)
