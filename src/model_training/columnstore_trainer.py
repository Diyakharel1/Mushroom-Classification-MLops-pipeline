"""Model training with ColumnStore data integration."""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import joblib
import sys
import os

# Add project root to path
sys.path.append("/app")
from config.database import db_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColumnStoreTrainer:
    """Model trainer using ColumnStore data."""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.models = {}
        self.metrics = {}

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load training data from ColumnStore."""
        try:
            logger.info(f"Loading training data for experiment {self.experiment_id}")

            query = """
            SELECT cf.* 
            FROM cleaned_features cf
            INNER JOIN train_data td ON cf.id = td.feature_id
            WHERE td.experiment_id = %s
            """

            df = pd.read_sql(
                query, db_manager.mariadb_engine, params=[self.experiment_id]
            )

            if df.empty:
                raise ValueError(
                    f"No training data found for experiment {self.experiment_id}"
                )

            # Separate features and target
            target_column = "class"
            feature_columns = [
                col
                for col in df.columns
                if col not in ["id", "created_at", "data_version", target_column]
            ]

            X = df[feature_columns]
            y = df[target_column]

            logger.info(f"Loaded training data: {X.shape} features, {len(y)} samples")
            return X, y

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise

    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load test data from ColumnStore."""
        try:
            logger.info(f"Loading test data for experiment {self.experiment_id}")

            query = """
            SELECT cf.* 
            FROM cleaned_features cf
            INNER JOIN test_data td ON cf.id = td.feature_id
            WHERE td.experiment_id = %s
            """

            df = pd.read_sql(
                query, db_manager.mariadb_engine, params=[self.experiment_id]
            )

            if df.empty:
                raise ValueError(
                    f"No test data found for experiment {self.experiment_id}"
                )

            # Separate features and target
            target_column = "class"
            feature_columns = [
                col
                for col in df.columns
                if col not in ["id", "created_at", "data_version", target_column]
            ]

            X = df[feature_columns]
            y = df[target_column]

            logger.info(f"Loaded test data: {X.shape} features, {len(y)} samples")
            return X, y

        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            raise

    def load_validation_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load validation data from ColumnStore."""
        try:
            logger.info(f"Loading validation data for experiment {self.experiment_id}")

            query = """
            SELECT cf.* 
            FROM cleaned_features cf
            INNER JOIN validation_data vd ON cf.id = vd.feature_id
            WHERE vd.experiment_id = %s
            """

            df = pd.read_sql(
                query, db_manager.mariadb_engine, params=[self.experiment_id]
            )

            if df.empty:
                logger.warning(
                    f"No validation data found for experiment {self.experiment_id}"
                )
                return pd.DataFrame(), pd.Series()

            # Separate features and target
            target_column = "class"
            feature_columns = [
                col
                for col in df.columns
                if col not in ["id", "created_at", "data_version", target_column]
            ]

            X = df[feature_columns]
            y = df[target_column]

            logger.info(f"Loaded validation data: {X.shape} features, {len(y)} samples")
            return X, y

        except Exception as e:
            logger.error(f"Failed to load validation data: {e}")
            raise

    def train_random_forest(
        self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs
    ) -> RandomForestClassifier:
        """Train Random Forest model."""
        try:
            logger.info("Training Random Forest model")

            # Set default parameters
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            }
            params.update(kwargs)

            # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            self.models["random_forest"] = model
            logger.info("Random Forest training completed")

            return model

        except Exception as e:
            logger.error(f"Failed to train Random Forest: {e}")
            raise

    def train_logistic_regression(
        self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs
    ) -> LogisticRegression:
        """Train Logistic Regression model."""
        try:
            logger.info("Training Logistic Regression model")

            # Set default parameters
            params = {"max_iter": 1000, "random_state": 42, "solver": "liblinear"}
            params.update(kwargs)

            # Train model
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)

            self.models["logistic_regression"] = model
            logger.info("Logistic Regression training completed")

            return model

        except Exception as e:
            logger.error(f"Failed to train Logistic Regression: {e}")
            raise

    def evaluate_model(
        self, model, model_name: str, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        try:
            logger.info(f"Evaluating {model_name} model")

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
            }

            # Store metrics
            self.metrics[model_name] = metrics

            # Log detailed results
            logger.info(f"{model_name} Metrics: {metrics}")
            logger.info(f"{model_name} Classification Report:")
            logger.info(f"\n{classification_report(y_test, y_pred)}")

            return metrics

        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            raise

    def log_experiment_to_mlflow(
        self, model_name: str, model, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> str:
        """Log experiment to MLflow."""
        try:
            with mlflow.start_run(run_name=f"{model_name}_{self.experiment_id}"):
                # Log parameters
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())

                # Log metrics
                if model_name in self.metrics:
                    mlflow.log_metrics(self.metrics[model_name])

                # Log model
                mlflow.sklearn.log_model(
                    model, model_name, registered_model_name=f"mushroom_{model_name}"
                )

                # Log additional info
                mlflow.log_param("experiment_id", self.experiment_id)
                mlflow.log_param("training_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("feature_count", X_train.shape[1])

                run_id = mlflow.active_run().info.run_id
                logger.info(f"Logged experiment to MLflow with run_id: {run_id}")

                return run_id

        except Exception as e:
            logger.error(f"Failed to log experiment to MLflow: {e}")
            raise

    def save_model_artifacts(self, model_name: str, model) -> str:
        """Save model artifacts to filesystem."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = f"/app/models/{model_name}_{self.experiment_id}_{timestamp}"
            os.makedirs(model_dir, exist_ok=True)

            # Save model
            model_path = os.path.join(model_dir, "model.pkl")
            joblib.dump(model, model_path)

            # Save metrics
            metrics_path = os.path.join(model_dir, "metrics.json")
            if model_name in self.metrics:
                import json

                with open(metrics_path, "w") as f:
                    json.dump(self.metrics[model_name], f, indent=2)

            logger.info(f"Saved model artifacts to {model_dir}")
            return model_dir

        except Exception as e:
            logger.error(f"Failed to save model artifacts: {e}")
            raise

    def run_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        try:
            start_time = datetime.now()
            logger.info(
                f"Starting training pipeline for experiment {self.experiment_id}"
            )

            # Load data
            X_train, y_train = self.load_training_data()
            X_test, y_test = self.load_test_data()
            X_val, y_val = self.load_validation_data()

            # Train models
            rf_model = self.train_random_forest(X_train, y_train)
            lr_model = self.train_logistic_regression(X_train, y_train)

            # Evaluate models
            rf_metrics = self.evaluate_model(rf_model, "random_forest", X_test, y_test)
            lr_metrics = self.evaluate_model(
                lr_model, "logistic_regression", X_test, y_test
            )

            # Log to MLflow
            rf_run_id = self.log_experiment_to_mlflow(
                "random_forest", rf_model, X_train, X_test
            )
            lr_run_id = self.log_experiment_to_mlflow(
                "logistic_regression", lr_model, X_train, X_test
            )

            # Save artifacts
            rf_model_dir = self.save_model_artifacts("random_forest", rf_model)
            lr_model_dir = self.save_model_artifacts("logistic_regression", lr_model)

            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()

            results = {
                "status": "success",
                "experiment_id": self.experiment_id,
                "training_time_seconds": training_time,
                "data_shapes": {
                    "train": X_train.shape,
                    "test": X_test.shape,
                    "validation": X_val.shape if not X_val.empty else (0, 0),
                },
                "models": {
                    "random_forest": {
                        "metrics": rf_metrics,
                        "mlflow_run_id": rf_run_id,
                        "model_dir": rf_model_dir,
                    },
                    "logistic_regression": {
                        "metrics": lr_metrics,
                        "mlflow_run_id": lr_run_id,
                        "model_dir": lr_model_dir,
                    },
                },
            }

            logger.info(f"Training pipeline completed successfully: {results}")
            return results

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "experiment_id": self.experiment_id,
            }


def main():
    """Main function for standalone execution."""
    logger.info("Starting ColumnStore Model Training")

    # Test database connection
    if not db_manager.test_mariadb_connection():
        logger.error("MariaDB connection test failed")
        return False

    # Get experiment ID (you might want to pass this as an argument)
    experiment_id = os.getenv(
        "EXPERIMENT_ID", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Initialize trainer
    trainer = ColumnStoreTrainer(experiment_id)

    # Run training pipeline
    results = trainer.run_training_pipeline()

    if results["status"] == "success":
        logger.info("Model training completed successfully")
        return True
    else:
        logger.error("Model training failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
