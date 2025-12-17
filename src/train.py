"""
Model training entrypoint for Mushroom Classification.

Key features:
- ColumnStore-first data loader (same external behaviour), with a robust local fallback:
  * If ColumnStore + splits exist: load train/test/(optional val) by experiment_id.
  * Else: load data/raw/secondary_data.csv (semicolon-safe), run src.transform.transform_data,
          or read data/processed/secondary_transformed.parquet if present.
- XGBoost primary model with graceful fallback to RandomForest if xgboost is unavailable.
- MLflow experiment setup with stable params/metrics logging and model signature.
- Simple data sanity checks (row counts, null ratio). Great Expectations is optional.
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from .mlflow_utils import init_mlflow

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# Project root on path for imports like "from src.transform import transform_data"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.transform import transform_data  # our refactored transformer

# ------------------------- logging ------------------------------------------ #

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger('airflow.task')

# ------------------------- optional deps ------------------------------------ #

# Great Expectations (optional)
try:
    import great_expectations as gx  # noqa: F401
    GE_AVAILABLE = True
except Exception:
    GE_AVAILABLE = False

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Models
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb  # type: ignore
    XGB_OK = True
except Exception:
    XGB_OK = False

# ------------------------- DB manager (optional) ---------------------------- #

# ColumnStore manager (kept for DAG compatibility)
try:
    from config.database import db_manager
except Exception:
    db_manager = None
    logger.warning("config.database.db_manager not available; will use local data.")

# ------------------------- config ------------------------------------------ #

@dataclass
class TrainConfig:
    # Data
    experiment_name: str = "mushroom_xgboost"
    data_source: str = "auto"  # "auto" | "columnstore" | "local"
    parquet_path: str = "data/processed/secondary_transformed.parquet"
    csv_path: str = "data/raw/secondary_data.csv"

    # Model (XGBoost defaults aligned with small, fast baseline)
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    random_state: int = 42

    # MLflow
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5001")
    register_as: Optional[str] = "mushroom_classifier"

# ------------------------- data loading ------------------------------------- #

def _load_from_columnstore(experiment_id: str, split: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Mirrors your previous columnstore_trainer approach:
    SELECT cf.* FROM cleaned_features cf
    JOIN split_table ON cf.id = split_table.feature_id WHERE experiment_id = %s
    """
    if db_manager is None or not hasattr(db_manager, "mariadb_engine") or db_manager.mariadb_engine is None:
        raise RuntimeError("ColumnStore not available.")

    split_map = {"train": "train_data", "test": "test_data", "validation": "validation_data"}
    if split not in split_map:
        raise ValueError(f"Invalid split: {split}")

    query = f"""
        SELECT cf.*
        FROM cleaned_features cf
        INNER JOIN {split_map[split]} sd ON cf.id = sd.feature_id
        WHERE sd.experiment_id = %s
    """

    df = pd.read_sql(query, db_manager.mariadb_engine, params=[experiment_id])
    if df.empty:
        raise RuntimeError(f"No {split} rows for experiment {experiment_id}")

    target = "class"
    X = df.drop(columns=[c for c in ["id", "created_at", "data_version"] if c in df.columns] + [target])
    y = df[target]
    return X, y

def _try_local_frame(cfg: TrainConfig) -> pd.DataFrame:
    """Prefer processed parquet; else read CSV (semicolon-safe) + transform."""
    parquet_file = PROJECT_ROOT / cfg.parquet_path
    if parquet_file.exists():
        logger.info(f"Reading processed parquet: {parquet_file}")
        return pd.read_parquet(parquet_file)

    csv_file = PROJECT_ROOT / cfg.csv_path
    if not csv_file.exists():
        raise FileNotFoundError(f"Neither parquet nor CSV found: {parquet_file} | {csv_file}")

    logger.info(f"Reading raw CSV: {csv_file} (sep auto-detect)")
    # The secondary dataset is ';' delimited. Use engine to be robust.
    df_raw = pd.read_csv(csv_file, sep=";", engine="python")
    logger.info("Running transform_data on raw CSV...")
    df_t = transform_data(df_raw)
    # Persist for reuse
    (PROJECT_ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    try:
        df_t.to_parquet(parquet_file, index=False)
        logger.info(f"Saved processed parquet: {parquet_file}")
    except Exception as e:
        logger.warning(f"Parquet save failed ({e}); skipping persist.")
    return df_t

def load_data(cfg: TrainConfig, experiment_id: Optional[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Returns X_train, y_train, X_test, y_test.
    Priority:
      1) ColumnStore (if cfg.data_source in ['auto','columnstore'] and db available and experiment_id provided)
      2) Local processed/CSV (transform on the fly if needed)
    """
    use_col_store = (cfg.data_source in ("auto", "columnstore")) and (experiment_id is not None)
    if use_col_store:
        try:
            logger.info(f"Attempting ColumnStore load for experiment_id={experiment_id}")
            X_train, y_train = _load_from_columnstore(experiment_id, "train")
            X_test, y_test = _load_from_columnstore(experiment_id, "test")
            logger.info("Loaded ColumnStore splits successfully.")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.warning(f"ColumnStore path failed ({e}); falling back to local.")

    # Local path
    df_t = _try_local_frame(cfg)
    if "class_encoded" in df_t.columns:
        target = "class_encoded"
    elif "class" in df_t.columns:
        # If class present, map to 0/1
        target = "class"
        if df_t[target].dtype == object:
            df_t[target] = (df_t[target].astype(str).str.lower().isin(["p", "poison", "poisonous"])).astype(int)
    else:
        raise KeyError("Target column not found in transformed data.")

    X = df_t.drop(columns=[target])
    y = df_t[target].astype(int)

    # Simple split (stratified) for local workflow
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.random_state, stratify=y
    )
    logger.info(f"Local split -> train={X_train.shape}, test={X_test.shape}")
    return X_train, y_train, X_test, y_test

# ------------------------- validation --------------------------------------- #

def quick_sanity_checks(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    rows, cols = X.shape
    null_pct = float(X.isna().sum().sum()) / max(1, rows * cols) * 100.0
    res = {
        "rows": int(rows),
        "cols": int(cols),
        "null_pct": null_pct,
        "class_balance": y.value_counts(normalize=True).to_dict(),
        "ok": rows >= 100 and null_pct < 20.0,
    }
    logger.info(f"Sanity checks: {res}")
    return res

# ------------------------- training ----------------------------------------- #

def build_model(cfg: TrainConfig):
    if XGB_OK:
        model = xgb.XGBClassifier(
            n_estimators=cfg.xgb_n_estimators,
            max_depth=cfg.xgb_max_depth,
            learning_rate=cfg.xgb_learning_rate,
            subsample=cfg.xgb_subsample,
            colsample_bytree=cfg.xgb_colsample_bytree,
            eval_metric="logloss",
            random_state=cfg.random_state,
            tree_method="hist",
        )
        model_name = "xgboost"
    else:
        model = RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=cfg.random_state
        )
        model_name = "random_forest"
        logger.warning("xgboost not installed; using RandomForest instead.")
    # pipe = Pipeline([("model", model)])  # data already one-hot from transform
    return model_name, model

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

# ------------------------- MLflow ------------------------------------------- #

def setup_mlflow(experiment_name: str, tracking_uri: str) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    exp_id = exp.experiment_id if exp is not None else mlflow.create_experiment(experiment_name)
    logger.info(f"MLflow ready | uri={tracking_uri} | experiment={experiment_name} ({exp_id})")
    return str(exp_id)

def log_run(ml_run, cfg: TrainConfig, model_name: str, model,
            X_train: pd.DataFrame, X_test: pd.DataFrame,
            metrics: Dict[str, float]) -> None:
    # Params
    for k, v in asdict(cfg).items():
        if k not in ("tracking_uri",):  # avoid noisy values
            mlflow.log_param(k, v)
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("n_features", X_train.shape[1])

    # Metrics
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # Signature
    try:
        signature = infer_signature(X_train.head(20), pipe.predict_proba(X_train.head(20)))
    except Exception:
        signature = None

    # Input example
    input_example = X_train.head(5)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=cfg.register_as if cfg.register_as else None,
    )
    logger.info("Model + artifacts logged to MLflow.")

# ------------------------- main --------------------------------------------- #

def train(experiment_id: Optional[str] = None, cfg: Optional[TrainConfig] = None) -> Dict[str, Any]:
    cfg = cfg or TrainConfig()
    logger.info(f"Config: {cfg}")

    # Load data (ColumnStore or local)
    X_train, y_train, X_test, y_test = load_data(cfg, experiment_id)

    # Sanity checks (fast)
    sanity = quick_sanity_checks(X_train, y_train)
    if not sanity["ok"]:
        logger.warning("Sanity checks failed; continuing, but results may be unreliable.")

    # Build + fit
    model_name, pipe = build_model(cfg)
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    scores = evaluate(y_test, y_pred)
    logger.info(f"Scores: {scores}")

    # MLflow
    try:
        init_mlflow()
        # exp_id = setup_mlflow(cfg.experiment_name, cfg.tracking_uri)
        # run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_id=experiment_id):
            with mlflow.start_run(run_name="train", nested=True):
                log_run(mlflow.active_run(), cfg, model_name, pipe, X_train, X_test, scores)
    except Exception as e:
        logger.warning(f"MLflow logging skipped due to: {e}")

    return {
        "model": pipe,
        "metrics": scores,
        "sanity": sanity,
        "model_name": model_name,
        "n_features": int(X_train.shape[1]),
        "data_source": "columnstore" if (experiment_id and db_manager is not None) else "local",
    }

def main():
    # If provided, use EXPERIMENT_ID to load ColumnStore splits
    experiment_id = os.getenv("EXPERIMENT_ID")  # e.g., exp_20250915_103000
    try:
        res = train(experiment_id=experiment_id)
        logger.info(f"✅ Training complete: {res['metrics']}")
        return True
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    exit(0 if main() else 1)
