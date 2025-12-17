"""
Model versioning module for the mushroom classification pipeline.
Handles model versioning, storage, and retrieval.
"""

import os
import json
import shutil
import logging
import datetime
from pathlib import Path
import uuid

# Determine the project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Create logs directory relative to the project root
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "model_versioning.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Class for managing model versions in a registry."""

    def __init__(self, registry_path="models/registry"):
        """Initialize the model registry.

        Args:
            registry_path (str): Path to the model registry directory
        """
        self.registry_path = registry_path
        self.versions_file = os.path.join(registry_path, "versions.json")
        self.staging_path = os.path.join(registry_path, "staging")
        self.production_path = os.path.join(registry_path, "production")
        self.archive_path = os.path.join(registry_path, "archive")

        # Create registry directories if they don't exist
        for path in [
            self.registry_path,
            self.staging_path,
            self.production_path,
            self.archive_path,
        ]:
            os.makedirs(path, exist_ok=True)

        # Initialize or load versions registry
        self.versions = self._load_versions()

    def _load_versions(self):
        """Load versions from the versions file."""
        if os.path.exists(self.versions_file):
            try:
                with open(self.versions_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading versions file: {e}")

        # If file doesn't exist or error loading, create new structure
        return {"models": {}, "production": {}, "staging": {}}

    def _save_versions(self):
        """Save versions to the versions file."""
        try:
            with open(self.versions_file, "w") as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving versions file: {e}")

    def register_model(self, model_path, model_name, version=None, metadata=None):
        """Register a new model version.

        Args:
            model_path (str): Path to the model file
            model_name (str): Name of the model
            version (str, optional): Version string. If None, generates timestamp-based version.
            metadata (dict, optional): Additional metadata about the model

        Returns:
            str: The version ID of the registered model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Generate version ID if not provided
        if version is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"{timestamp}_v{uuid.uuid4().hex[:8]}"

        # Initialize model entry if it doesn't exist
        if model_name not in self.versions["models"]:
            self.versions["models"][model_name] = {}

        # Create metadata if not provided
        if metadata is None:
            metadata = {}

        # Add basic metadata
        metadata.update(
            {
                "registered_at": datetime.datetime.now().isoformat(),
                "original_path": model_path,
                "status": "registered",
            }
        )

        # Create destination path
        dest_path = os.path.join(self.registry_path, model_name, version)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Copy model file to registry
        try:
            shutil.copy2(model_path, dest_path)
            logger.info(f"Copied model from {model_path} to {dest_path}")
        except Exception as e:
            logger.error(f"Error copying model file: {e}")
            raise

        # Update versions registry
        self.versions["models"][model_name][version] = metadata
        self._save_versions()

        logger.info(f"Registered {model_name} version {version}")
        return version

    def promote_to_staging(self, model_name, version):
        """Promote a model version to staging.

        Args:
            model_name (str): Name of the model
            version (str): Version ID to promote
        """
        try:
            # Check if model and version exist
            if (
                model_name not in self.versions["models"]
                or version not in self.versions["models"][model_name]
            ):
                raise ValueError(f"Model {model_name} version {version} not found")

            # Get source path
            source_path = os.path.join(self.registry_path, model_name, version)
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Model file not found: {source_path}")

            # Set destination path
            dest_path = os.path.join(self.staging_path, f"{model_name}.joblib")

            # Copy model to staging
            shutil.copy2(source_path, dest_path)

            # Update staging registry
            self.versions["staging"][model_name] = {
                "version": version,
                "promoted_at": datetime.datetime.now().isoformat(),
            }

            # Update model status
            self.versions["models"][model_name][version]["status"] = "staging"

            self._save_versions()
            logger.info(f"Promoted {model_name} version {version} to staging")

        except Exception as e:
            logger.error(f"Error promoting model to staging: {e}")
            raise

    def promote_to_production(self, model_name, version=None):
        """Promote a model version to production.

        Args:
            model_name (str): Name of the model
            version (str, optional): Version ID to promote. If None, uses the current staging version.
        """
        try:
            # If version not specified, use staging version
            if version is None:
                if model_name not in self.versions["staging"]:
                    raise ValueError(f"No staging version found for {model_name}")
                version = self.versions["staging"][model_name]["version"]

            # Check if model and version exist
            if (
                model_name not in self.versions["models"]
                or version not in self.versions["models"][model_name]
            ):
                raise ValueError(f"Model {model_name} version {version} not found")

            # Get source path
            source_path = os.path.join(self.registry_path, model_name, version)
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Model file not found: {source_path}")

            # Set destination path
            dest_path = os.path.join(self.production_path, f"{model_name}.joblib")

            # Archive current production model if it exists
            if os.path.exists(dest_path) and model_name in self.versions["production"]:
                old_version = self.versions["production"][model_name]["version"]
                archive_path = os.path.join(
                    self.archive_path, f"{model_name}_{old_version}.joblib"
                )
                shutil.copy2(dest_path, archive_path)
                logger.info(
                    f"Archived production model {model_name} version {old_version}"
                )

            # Copy model to production
            shutil.copy2(source_path, dest_path)

            # Update production registry
            self.versions["production"][model_name] = {
                "version": version,
                "promoted_at": datetime.datetime.now().isoformat(),
            }

            # Update model status
            self.versions["models"][model_name][version]["status"] = "production"

            self._save_versions()
            logger.info(f"Promoted {model_name} version {version} to production")

        except Exception as e:
            logger.error(f"Error promoting model to production: {e}")
            raise

    def get_model_info(self, model_name, version=None):
        """Get information about a model version.

        Args:
            model_name (str): Name of the model
            version (str, optional): Version ID. If None, returns info for all versions.

        Returns:
            dict: Model information
        """
        if model_name not in self.versions["models"]:
            return {}

        if version is not None:
            if version not in self.versions["models"][model_name]:
                return {}
            return self.versions["models"][model_name][version]

        return self.versions["models"][model_name]

    def get_production_model_info(self, model_name):
        """Get information about the production model.

        Args:
            model_name (str): Name of the model

        Returns:
            dict: Production model information
        """
        if model_name not in self.versions["production"]:
            return {}

        version = self.versions["production"][model_name]["version"]

        return {
            "path": os.path.join(self.production_path, f"{model_name}.joblib"),
            "version": version,
            "info": self.versions["models"][model_name][version],
            "promoted_at": self.versions["production"][model_name]["promoted_at"],
        }

    def get_staging_model_info(self, model_name):
        """Get information about the staging model.

        Args:
            model_name (str): Name of the model

        Returns:
            dict: Staging model information
        """
        if model_name not in self.versions["staging"]:
            return {}

        version = self.versions["staging"][model_name]["version"]

        return {
            "path": os.path.join(self.staging_path, f"{model_name}.joblib"),
            "version": version,
            "info": self.versions["models"][model_name][version],
            "promoted_at": self.versions["staging"][model_name]["promoted_at"],
        }

    def list_model_versions(self, model_name):
        """List all versions of a model.

        Args:
            model_name (str): Name of the model

        Returns:
            list: List of version IDs
        """
        if model_name not in self.versions["models"]:
            return []

        return list(self.versions["models"][model_name].keys())

    def get_production_models(self):
        """Get information about all production models.

        Returns:
            dict: Dictionary of production models with model names as keys
        """
        result = {}
        for model_name in self.versions["production"]:
            result[model_name] = self.get_production_model_info(model_name)
        return result

    def get_staging_models(self):
        """Get information about all staging models.

        Returns:
            dict: Dictionary of staging models with model names as keys
        """
        result = {}
        for model_name in self.versions["staging"]:
            result[model_name] = self.get_staging_model_info(model_name)
        return result

    def compare_model_versions(self, model_name, version1, version2):
        """Compare two model versions.

        Args:
            model_name (str): Name of the model
            version1 (str): First version ID
            version2 (str): Second version ID

        Returns:
            dict: Comparison results
        """
        if model_name not in self.versions["models"]:
            return {}

        if (
            version1 not in self.versions["models"][model_name]
            or version2 not in self.versions["models"][model_name]
        ):
            return {}

        model1 = self.versions["models"][model_name][version1]
        model2 = self.versions["models"][model_name][version2]

        # Extract metrics for comparison if they exist
        metrics1 = model1.get("metrics", {})
        metrics2 = model2.get("metrics", {})

        comparison = {}

        # Compare common metrics
        for metric in set(metrics1.keys()).intersection(set(metrics2.keys())):
            try:
                # Calculate absolute and relative differences
                if isinstance(metrics1[metric], (int, float)) and isinstance(
                    metrics2[metric], (int, float)
                ):
                    abs_diff = metrics2[metric] - metrics1[metric]
                    rel_diff = (
                        abs_diff / metrics1[metric] * 100
                        if metrics1[metric] != 0
                        else float("inf")
                    )

                    comparison[metric] = {
                        "version1": metrics1[metric],
                        "version2": metrics2[metric],
                        "absolute_difference": abs_diff,
                        "relative_difference_percent": rel_diff,
                        "improved": abs_diff > 0,  # Assuming higher is better
                    }
            except Exception as e:
                logger.warning(f"Error comparing metric {metric}: {e}")

        return comparison


def register_and_promote_model(model_path, model_name, metrics=None, promote=None):
    """Register a model and optionally promote it to staging or production.

    Args:
        model_path (str): Path to the model file
        model_name (str): Name of the model
        metrics (dict, optional): Model performance metrics
        promote (str, optional): Where to promote the model ('staging' or 'production')

    Returns:
        str: The version ID of the registered model
    """
    try:
        # Create model registry
        registry = ModelRegistry()

        # Prepare metadata
        metadata = {"metrics": metrics or {}}

        # Register model
        version = registry.register_model(model_path, model_name, metadata=metadata)

        # Promote if requested
        if promote == "staging":
            registry.promote_to_staging(model_name, version)
        elif promote == "production":
            registry.promote_to_staging(
                model_name, version
            )  # Always go through staging
            registry.promote_to_production(model_name, version)

        return version

    except Exception as e:
        logger.error(f"Error registering/promoting model: {e}")
        raise


def get_production_model_path(model_name):
    """Get the path to the current production model.

    Args:
        model_name (str): Name of the model

    Returns:
        str: Path to the production model file
    """
    registry = ModelRegistry()
    model_info = registry.get_production_model_info(model_name)

    if not model_info:
        raise FileNotFoundError(f"No production model found for {model_name}")

    return model_info["path"]
