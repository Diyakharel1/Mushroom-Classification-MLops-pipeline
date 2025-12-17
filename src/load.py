"""
Load component of the ETL pipeline for Mushroom Classification.
Responsible for saving processed data and trained models with ColumnStore integration.
"""

import os
import pandas as pd
import logging
import joblib
import sys
from sklearn.model_selection import train_test_split
from datetime import datetime
from .redis_conn import redis_connection
import pickle

redis_store = redis_connection()

# Add project root to path
sys.path.append("/app")

# Import model versioning (with error handling)
try:
    from src.model_versioning import register_and_promote_model
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Model versioning module not available")

    def register_and_promote_model(*args, **kwargs):
        return "v1.0.0"


# Get a logger for this module
logger = logging.getLogger(__name__)


def load_data(test_size=0.3, random_state=42):
    """Prepare data for modeling and save to files"""
    logger.info("Preparing data for modeling")
    unserailized_df = redis_store.get('transformed_df')
    output_path = '/app/data/train_and_test_data'
    df = pickle.loads(unserailized_df)
    if "class_encoded" not in df.columns:
        # Try to find alternative target columns
        target_candidates = ["class", "edible", "target"]
        target_column = None

        for candidate in target_candidates:
            if candidate in df.columns:
                target_column = candidate
                break

        if target_column is None:
            # Use the last column as target
            target_column = df.columns[-1]
            logger.warning(
                f"No standard target column found, using last column: {target_column}"
            )

        # Rename target column to class_encoded for consistency
        df = df.rename(columns={target_column: "class_encoded"})
        logger.info(f"Renamed target column '{target_column}' to 'class_encoded'")

    X = df.drop(columns=["class_encoded"])
    y = df["class_encoded"]

    # Convert integer columns to float for consistency
    integer_cols = X.select_dtypes(include=["int64", "int32"]).columns
    if not integer_cols.empty:
        X[integer_cols] = X[integer_cols].astype("float64")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Split data into train ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) sets"
    )

    # Ensure output path exists
    os.makedirs(output_path, exist_ok=True)

    # Save data splits
    X_train.to_csv(os.path.join(output_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_path, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_path, "y_test.csv"), index=False)

    logger.info(f"Saved processed data to {output_path}")

    # return X_train, X_test, y_train, y_test
    return X_train.shape, X_test.shape, y_train.shape, y_test.shape


def save_model(
    model,
    model_name,
    output_path,
    metrics=None,
    promote=None,
    X_sample=None,
    y_sample=None,
):
    """
    Save a trained model and optionally register it with the model registry.

    Args:
        model: Trained model object
        model_name (str): Name of the model
        output_path (str): Path to save the model
        metrics (dict, optional): Model performance metrics
        promote (str, optional): Stage to promote model to ('staging', 'production')
        X_sample (pd.DataFrame, optional): Sample input data
        y_sample (pd.Series, optional): Sample target data

    Returns:
        str: Path to saved model
    """
    try:
        logger.info(f"Saving {model_name} model")
        os.makedirs(output_path, exist_ok=True)

        # Save model using joblib
        model_path = os.path.join(output_path, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Saved {model_name} model to {model_path}")

        # Save model metadata
        metadata = {
            "model_name": model_name,
            "saved_at": datetime.now().isoformat(),
            "model_path": model_path,
            "metrics": metrics or {},
            "promote": promote,
        }

        metadata_path = os.path.join(output_path, f"{model_name}_metadata.json")
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved {model_name} metadata to {metadata_path}")

        # Register with model versioning if available and metrics provided
        if metrics and hasattr(register_and_promote_model, "__call__"):
            try:
                version = register_and_promote_model(
                    model_path, model_name, metrics, promote, X_sample, y_sample
                )
                logger.info(f"Model {model_name} registered as version {version}")
                if promote:
                    logger.info(f"Model {model_name} promoted to {promote}")
            except Exception as e:
                logger.warning(
                    f"Error registering model {model_name} with versioning system: {e}"
                )

        return model_path

    except Exception as e:
        logger.error(f"Error saving model {model_name}: {e}")
        raise


def load_model(model_path):
    """
    Load a saved model from disk.

    Args:
        model_path (str): Path to the saved model

    Returns:
        object: Loaded model
    """
    try:
        logger.info(f"Loading model from {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")

        return model

    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise


def save_data_splits_to_columnstore(X_train, X_test, y_train, y_test, experiment_id):
    """
    Save data splits directly to ColumnStore database.

    Args:
        X_train, X_test: Feature datasets
        y_train, y_test: Target datasets
        experiment_id (str): Experiment identifier

    Returns:
        dict: Results of the save operation
    """
    try:
        from config.database import db_manager

        logger.info(f"Saving data splits to ColumnStore for experiment {experiment_id}")

        # Combine features and targets
        train_data = pd.concat([X_train, y_train.rename("class")], axis=1)
        test_data = pd.concat([X_test, y_test.rename("class")], axis=1)

        # Add data version
        data_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_data["data_version"] = data_version
        test_data["data_version"] = data_version

        # Insert training data
        train_success = db_manager.insert_cleaned_data(train_data, data_version)
        if not train_success:
            raise Exception("Failed to insert training data")

        # Insert test data
        test_success = db_manager.insert_cleaned_data(test_data, data_version)
        if not test_success:
            raise Exception("Failed to insert test data")

        # Create data splits references
        train_ids = list(range(1, len(train_data) + 1))  # Simplified ID assignment
        test_ids = list(
            range(len(train_data) + 1, len(train_data) + len(test_data) + 1)
        )

        splits_success = db_manager.create_data_splits(
            experiment_id, train_ids, test_ids, []
        )
        if not splits_success:
            raise Exception("Failed to create data split references")

        logger.info(f"Successfully saved data splits to ColumnStore")

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "data_version": data_version,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
        }

    except Exception as e:
        logger.error(f"Error saving data splits to ColumnStore: {e}")
        return {"status": "failed", "error": str(e)}


# Example of how to set up logging if this script is run standalone for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger.info("src/load.py executed as main script (for testing).")
