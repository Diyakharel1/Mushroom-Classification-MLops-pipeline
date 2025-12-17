# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import joblib
import pandas as pd

# Reuse your transformer
from src.transform import transform_data

app = FastAPI(title="Mushroom Classifier API", version="1.0")

MODEL_PATH = Path("artifacts/mushroom_pipeline.joblib")
FEATURES_PATH = Path("artifacts/feature_names.json")

# --------- Load model + expected features ---------------------------------- #
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Train and save it first "
        "(see artifacts/mushroom_pipeline.joblib)."
    )

pipe = joblib.load(MODEL_PATH)

def _load_feature_names():
    # Prefer explicit list saved during setup
    if FEATURES_PATH.exists():
        try:
            return json.loads(FEATURES_PATH.read_text())
        except Exception as e:
            raise RuntimeError(f"Failed to read {FEATURES_PATH}: {e}")

    # Fallback: try sklearn attribute on the underlying estimator
    model = getattr(pipe, "named_steps", {}).get("model", pipe)
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return names.tolist()

    # Last resort: try xgboost booster (may be None if trained w/ numpy arrays)
    try:
        booster = model.get_booster()
        if getattr(booster, "feature_names", None):
            return booster.feature_names
    except Exception:
        pass

    raise RuntimeError(
        "Feature names unknown. Please create artifacts/feature_names.json "
        "with the training feature list."
    )

FEATURE_NAMES = _load_feature_names()

# --------- Schemas ---------------------------------------------------------- #
class RawItem(BaseModel):
    # Raw schema (subset ok; transformer will impute & one-hot)
    cap_diameter: Optional[float] = None
    stem_height: Optional[float] = None
    stem_width: Optional[float] = None
    cap_shape: Optional[str] = None
    cap_surface: Optional[str] = None
    cap_color: Optional[str] = None
    does_bruise_or_bleed: Optional[str] = Field(None, description="t/f/yes/no")
    gill_attachment: Optional[str] = None
    gill_color: Optional[str] = None
    stem_color: Optional[str] = None
    has_ring: Optional[str] = Field(None, description="t/f/yes/no")
    ring_type: Optional[str] = None
    habitat: Optional[str] = None
    season: Optional[str] = None

class RawBatch(BaseModel):
    items: List[RawItem]

class XItem(BaseModel):
    # Already-transformed (one-hot) row
    features: Dict[str, Any]

class XBatch(BaseModel):
    items: List[XItem]

# --------- Helpers ---------------------------------------------------------- #
def _align_columns(dfX: pd.DataFrame) -> pd.DataFrame:
    """Add missing columns as 0, drop unexpected ones, and order as during training."""
    for col in FEATURE_NAMES:
        if col not in dfX.columns:
            dfX[col] = 0
    extra = [c for c in dfX.columns if c not in FEATURE_NAMES]
    if extra:
        dfX = dfX.drop(columns=extra)
    return dfX[FEATURE_NAMES]

# --------- Routes ----------------------------------------------------------- #
@app.get("/")
def root():
    return {
        "message": "Mushroom Classifier API is running.",
        "endpoints": ["/health", "/docs", "/predict_raw", "/predict"],
    }

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": MODEL_PATH.exists(),
        "n_features": len(FEATURE_NAMES),
        "model_path": str(MODEL_PATH),
    }

@app.post("/predict_raw")
def predict_raw(batch: RawBatch):
    try:
        raw_df = pd.DataFrame([i.model_dump() for i in batch.items])
        if raw_df.empty:
            raise ValueError("Empty request payload.")
        # Transform raw → OHE
        tx_df = transform_data(raw_df)
        # Remove any accidental target columns
        for tcol in ("class", "class_encoded"):
            if tcol in tx_df.columns:
                tx_df = tx_df.drop(columns=tcol)
        X = _align_columns(tx_df)
        y = pipe.predict(X).tolist()
        try:
            proba = pipe.predict_proba(X).tolist()
        except Exception:
            proba = None
        return {"pred": y, "proba": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.post("/predict")
def predict(batch: XBatch):
    try:
        df = pd.DataFrame([it.features for it in batch.items])
        if df.empty:
            raise ValueError("Empty request payload.")
        X = _align_columns(df)
        y = pipe.predict(X).tolist()
        try:
            proba = pipe.predict_proba(X).tolist()
        except Exception:
            proba = None
        return {"pred": y, "proba": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
