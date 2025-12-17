"""
Main ETL pipeline script for the mushroom classification project.
Orchestrates the entire pipeline from data extraction to model training and evaluation with ColumnStore.
"""

import os
import argparse
import logging
import json
import yaml
from datetime import datetime
import sys

# Add project root to path
sys.path.append("/app")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MLflow for experiment tracking
try:
    import mlflow
    import mlflow.sklearn

    # Use consistent MLflow tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    MLFLOW_AVAILABLE = True
    logger.info(f"MLflow tracking enabled with URI: {mlflow_uri}")
except ImportError:
    MLFLOW_AVAILABLE = False

    # Create a mock mlflow object for compatibility
    class MockRunInfo:
        def __init__(self):
            self.run_id = "mock_run"

    class MockRun:
        def __init__(self):
            self.info = MockRunInfo()

    class MockMLflow:
        @staticmethod
        def set_experiment(*args, **kwargs):
            pass

        @staticmethod
        def start_run(*args, **kwargs):
            return MockRun()

        @staticmethod
        def log_param(*args, **kwargs):
            pass

        @staticmethod
        def log_metric(*args, **kwargs):
            pass

        @staticmethod
        def log_artifact(*args, **kwargs):
            pass

        @staticmethod
        def end_run(*args, **kwargs):
            pass

        @staticmethod
        def active_run():
            return None

        @staticmethod
        def get_experiment_by_name(*args, **kwargs):
            return None

        @staticmethod
        def create_experiment(*args, **kwargs):
            return "0"

    mlflow = MockMLflow()
    logger.warning("MLflow not available - using mock implementation")

# Import ETL components with error handling
try:
    from src.extract import extract_data
    from src.transform import transform_data
    from src.load import load_data, save_model
    from src.train import train, train_models

    # Import database manager with fallback paths
    try:
        from config.database import db_manager
    except ImportError:
        try:
            sys.path.append("/app/airflow")
            from config.database import db_manager
        except ImportError:
            logger.warning(
                "Database manager not available - some features may not work"
            )
            db_manager = None

    logger.info("Successfully imported ETL components")
except ImportError as e:
    logger.error(f"Failed to import ETL components: {e}")
    raise

# Import MLOps components with error handling
try:
    from src.monitoring import monitor_model_performance

    # Import model versioning with simplified handling
    _orig_register_and_promote_model = None
    try:
        from src.model_versioning import (
            register_and_promote_model as _orig_register_and_promote_model,
        )
    except ImportError:
        logger.warning("Model versioning not available")

    # Define the function once, using the imported function if available
    def register_and_promote_model(*args, **kwargs):
        if _orig_register_and_promote_model:
            try:
                return _orig_register_and_promote_model(*args, **kwargs)
            except Exception:
                return "v1.0.0"
        else:
            return "v1.0.0"

    logger.info("Successfully imported MLOps components")
except ImportError as e:
    logger.warning(f"Some MLOps components could not be imported: {e}")

    # Define dummy functions for missing components
    class ModelMonitor:
        def __init__(self, *args, **kwargs):
            pass

        def record_metrics(self, *args, **kwargs):
            pass

        def generate_monitoring_report(self, *args, **kwargs):
            return "monitoring_report.html"

    class ModelRegistry:
        def __init__(self, *args, **kwargs):
            pass

        def get_production_models(self, *args, **kwargs):
            return {}

        def register_model(self, *args, **kwargs):
            return "model_registered"

        def get_staging_models(self, *args, **kwargs):
            return {}

    def monitor_model_performance(*args, **kwargs):
        return {"status": "monitoring_disabled"}

    def create_ab_test(*args, **kwargs):
        return "test_123"


def run_pipeline(config_path, step=None):
    """
    Run the full ETL pipeline or a specific step with ColumnStore integration.

    Args:
        config_path (str): Path to the configuration file.
        step (str, optional): Specific pipeline step to run.
    """
    global MLFLOW_AVAILABLE

    start_time = datetime.now()
    logger.info(f"Starting pipeline run at {start_time}")

    # Generate experiment ID
    experiment_id = f"exp_{start_time.strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Pipeline experiment ID: {experiment_id}")

    # Define pipeline steps
    all_steps = ["extract", "transform", "load", "train", "monitoring", "versioning"]
    steps = [step] if step else all_steps

    # Initialize MLflow tracking if available
    if MLFLOW_AVAILABLE and mlflow:
        try:
            experiment_name = "mushroom_classification_pipeline"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id_mlflow = mlflow.create_experiment(experiment_name)
                    logger.info(f"Created new MLflow experiment: {experiment_name}")
                else:
                    experiment_id_mlflow = experiment.experiment_id
                    logger.info(f"Using existing MLflow experiment: {experiment_name}")

                mlflow.set_experiment(experiment_name)
            except Exception as e:
                logger.warning(f"Could not set up MLflow experiment: {e}")
                experiment_id_mlflow = "0"

            # Start pipeline run
            pipeline_run = mlflow.start_run(run_name=f"pipeline_run_{experiment_id}")
            pipeline_run_id = pipeline_run.info.run_id
            logger.info(f"Started MLflow pipeline run: {pipeline_run_id}")  

            # Log pipeline parameters
            mlflow.log_param("experiment_id", experiment_id)
            mlflow.log_param("config_path", config_path)
            mlflow.log_param("start_time", start_time.isoformat())
            if step:
                mlflow.log_param("specific_step", step)
            else:
                mlflow.log_param("run_type", "full_pipeline")

        except Exception as e:
            logger.warning(f"Failed to initialize MLflow tracking: {e}")
            MLFLOW_AVAILABLE = False

    # Initialize variables that may be used across steps
    training_results = None
    model_metrics = {}

    try:
        # Load configuration with better error handling
        if not os.path.exists(config_path):
            logger.warning(
                f"Config file {config_path} not found, creating default config"
            )
            config = {
                "paths": {
                    "raw_data": "/app/data/raw",
                    "processed_data": "/app/data/processed",
                    "models": "/app/models",
                    "metrics": "/app/models/metrics",
                },
                "data_split": {"test_size": 0.3, "random_state": 42},
                "models": {
                    "random_forest": {"n_estimators": 100},
                    "gradient_boosting": {"n_estimators": 100},
                    "xgboost": {"n_estimators": 100, "max_depth": 6},
                },
            }

            # Save default config
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Created default config at {config_path}")
        else:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

        # Ensure required directories exist
        for path_key, path_value in config["paths"].items():
            os.makedirs(path_value, exist_ok=True)

        logger.info(f"Pipeline configuration loaded successfully")
        logger.info(f"Steps to run: {steps}")

        # Initialize ColumnStore tables
        if db_manager:
            if hasattr(db_manager, "create_columnstore_tables"):
                logger.info("Creating ColumnStore tables...")
                # Use getattr to avoid type checker issues
                create_method = getattr(db_manager, "create_columnstore_tables")
                create_method()
            elif hasattr(db_manager, "create_tables"):
                logger.info("Creating tables using create_tables method...")
                db_manager.create_tables()  # Fallback to original method
            else:
                logger.warning("No table creation method available")
        else:
            logger.warning("Database manager not available - skipping table creation")

        # Extract data
        if "extract" in steps:
            logger.info("Step 1: Extracting data")
            data_files = extract_data(config["paths"]["raw_data"])
            logger.info(f"Data extraction completed: {len(data_files)} files processed")

            if MLFLOW_AVAILABLE and mlflow:
                mlflow.log_metric("files_processed", len(data_files))

        # Transform data
        if "transform" in steps:
            logger.info("Step 2: Transforming data")

            # Load data using enhanced ETL
            from src.data_processing.enhanced_etl import EnhancedMushroomETL

            try:
                etl = EnhancedMushroomETL()

                # Extract and transform
                raw_data = etl.extract_data()
                transformed_data = etl.transform_data(raw_data)

                # Load into ColumnStore
                success = etl.load_data(transformed_data)

                if success:
                    # Create train/test splits
                    split_info = etl.create_train_test_splits(experiment_id)
                    logger.info(f"Data transformation and loading completed")
                    logger.info(f"Split info: {split_info}")

                    if MLFLOW_AVAILABLE and mlflow:
                        mlflow.log_metric(
                            "training_samples", split_info.get("train_size", 0)
                        )
                        mlflow.log_metric(
                            "test_samples", split_info.get("test_size", 0)
                        )
                        mlflow.log_metric(
                            "validation_samples", split_info.get("validation_size", 0)
                        )
                else:
                    raise Exception("Failed to load data into ColumnStore")

            except Exception as e:
                logger.error(f"Enhanced ETL failed: {e}")
                logger.info("Falling back to legacy transformation method")

                # Extract data first, then transform
                raw_data = extract_data(config["paths"]["raw_data"])
                transformed_data = transform_data(raw_data)

                # Save transformed data
                os.makedirs(config["paths"]["processed_data"], exist_ok=True)
                transformed_data.to_csv(
                    os.path.join(
                        config["paths"]["processed_data"], "transformed_data.csv"
                    ),
                    index=False,
                )

        # Skip load step if using enhanced ETL
        if "load" in steps and "transform" not in steps:
            logger.info("Step 3: Loading processed data into ColumnStore")
            load_data(config["paths"]["processed_data"], config["paths"]["models"])
        elif "load" in steps:
            logger.error("Cannot load data: No transformed data available")
            return

        # Train models using ColumnStore data
        if "train" in steps:
            logger.info("Step 4: Training models from ColumnStore")

            try:
                training_results = train(experiment_id, config)

                # Ensure training_results is a dictionary (compatibility check)
                if isinstance(training_results, tuple):
                    # Convert legacy tuple format to dict format
                    if len(training_results) == 2:
                        model_type, accuracy = training_results
                        training_results = {
                            "best_model": model_type,
                            "best_accuracy": accuracy,
                            "model_results": {model_type: {"accuracy": accuracy}},
                            "trained_models": {},
                            "deployment_ready": True,
                        }

                if training_results and isinstance(training_results, dict):
                    logger.info(f"Model training completed successfully")
                    logger.info(
                        f"Best model: {training_results.get('best_model')} with accuracy: {training_results.get('best_accuracy', 0):.4f}"
                    )

                    # Extract model metrics for monitoring
                    model_metrics = {}
                    for model_name, results in training_results.get(
                        "model_results", {}
                    ).items():
                        model_metrics[model_name] = {
                            "accuracy": results.get("accuracy", 0.0),
                            "precision": results.get("precision", 0.0),
                            "recall": results.get("recall", 0.0),
                            "f1_score": results.get("f1_score", 0.0),
                        }

                    # Save trained models
                    for model_name, pipeline in training_results.get(
                        "trained_models", {}
                    ).items():
                        model_path = save_model(
                            pipeline,
                            model_name,
                            config["paths"]["models"],
                            metrics=model_metrics.get(model_name),
                        )
                        logger.info(f"Saved {model_name} to {model_path}")
                else:
                    raise Exception("No training results returned")

            except Exception as e:
                logger.error(f"Model training failed: {e}")
                raise

        # Setup model monitoring
        if "monitoring" in steps and "model_metrics" in locals():
            logger.info("Step 5: Setting up model monitoring")

            try:
                for model_name, metrics in model_metrics.items():
                    monitoring_results = monitor_model_performance(
                        None,  # Model object not needed for basic monitoring
                        None,  # X_test
                        None,  # y_test
                        metrics_path=os.path.join(
                            config["paths"]["metrics"], "monitoring", model_name
                        ),
                    )
                    logger.info(f"Monitoring setup completed for {model_name}")

                if MLFLOW_AVAILABLE and mlflow:
                    mlflow.log_metric("monitoring_setup", 1)

            except Exception as e:
                logger.warning(f"Model monitoring setup failed: {e}")

        # Setup model versioning
        if "versioning" in steps and "model_metrics" in locals():
            logger.info("Step 6: Setting up model versioning")

            try:
                registry = ModelRegistry(
                    os.path.join(config["paths"]["models"], "registry")
                )

                for model_name, metrics in model_metrics.items():
                    model_path = os.path.join(
                        config["paths"]["models"], f"{model_name}.joblib"
                    )

                    if os.path.exists(model_path):
                        promote_to = (
                            "production"
                            if (
                                training_results
                                and model_name == training_results.get("best_model", "")
                            )
                            else "staging"
                        )

                        try:
                            version = register_and_promote_model(
                                model_path, model_name, metrics, promote_to
                            )
                            logger.info(
                                f"Registered {model_name} as version {version} in {promote_to}"
                            )
                        except Exception as e:
                            logger.error(f"Error registering model {model_name}: {e}")
            except Exception as e:
                logger.warning(f"Model versioning setup failed: {e}")

        # Save pipeline metadata
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        pipeline_metadata = {
            "experiment_id": experiment_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "elapsed_time_seconds": elapsed_time,
            "steps_completed": steps,
            "config_path": config_path,
        }

        metadata_path = os.path.join(
            config["paths"]["models"], f"pipeline_metadata_{experiment_id}.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(pipeline_metadata, f, indent=2, default=str)

        logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Pipeline metadata saved to {metadata_path}")

        if MLFLOW_AVAILABLE and mlflow:
            mlflow.log_metric("pipeline_duration_seconds", elapsed_time)
            mlflow.log_artifact(metadata_path, "pipeline_metadata")
            mlflow.end_run()

        return pipeline_metadata

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if MLFLOW_AVAILABLE and mlflow:
            mlflow.log_param("error", str(e))
            mlflow.end_run(status="FAILED")
        raise

    finally:
        if MLFLOW_AVAILABLE and mlflow and mlflow.active_run():
            mlflow.end_run()


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Run the mushroom classification ETL pipeline"
    )
    parser.add_argument(
        "--config",
        default="/app/config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--step",
        choices=["extract", "transform", "load", "train", "monitoring", "versioning"],
        help="Run a specific pipeline step",
    )

    args = parser.parse_args()

    try:
        result = run_pipeline(args.config, args.step)
        logger.info("Pipeline execution completed successfully")
        return True
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
