"""
Model loader utility for loading saved models and making predictions.
"""

import os
import joblib
import logging
import pandas as pd
import numpy as np
from typing import Optional, Union, Any, Dict, List
from sklearn.preprocessing import LabelEncoder

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/model_serving.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class MushroomModel:
    """Class for loading and using mushroom classification models."""

    def __init__(self, model_path, model_type="xgboost"):
        """
        Initialize the model loader.

        Args:
            model_path (str): Path to the saved model file.
            model_type (str): Type of model ('xgboost', 'logistic_regression', or 'decision_tree').
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
        self.categorical_encoders: Dict[str, LabelEncoder] = {}
        self.load_model()

    def load_model(self):
        """Load the saved model and its preprocessing artifacts."""
        try:
            logger.info(f"Loading {self.model_type} model from {self.model_path}")

            # Check if file exists
            if not os.path.exists(self.model_path):
                logger.error(f"Model file {self.model_path} does not exist")
                raise FileNotFoundError(f"Model file {self.model_path} does not exist")

            # Load the model
            self.model = joblib.load(self.model_path)

            # Load feature names if available
            model_dir = os.path.dirname(self.model_path)
            feature_file = os.path.join(model_dir, "feature_names.txt")
            if os.path.exists(feature_file):
                with open(feature_file, "r") as f:
                    self.feature_names = f.read().strip().split(",")
                logger.info(f"Loaded {len(self.feature_names)} feature names")

            # Load categorical encoders if available
            encoders_file = os.path.join(model_dir, "categorical_encoders.joblib")
            if os.path.exists(encoders_file):
                self.categorical_encoders = joblib.load(encoders_file)
                logger.info(
                    f"Loaded {len(self.categorical_encoders)} categorical encoders"
                )

            logger.info(f"Successfully loaded {self.model_type} model")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def preprocess_input(self, data):
        """
        Preprocess input data to match the expected model input format.

        Args:
            data (dict or pd.DataFrame): Input data to preprocess.

        Returns:
            pd.DataFrame: Preprocessed data ready for prediction.
        """
        try:
            logger.info("Preprocessing input data")

            # Convert dict to DataFrame if needed
            if isinstance(data, dict):
                data = pd.DataFrame([data])

            # Apply categorical encoding if encoders are available
            if self.categorical_encoders:
                for col, encoder in self.categorical_encoders.items():
                    if col in data.columns:
                        try:
                            data[col] = encoder.transform(data[col])
                        except ValueError:
                            # Handle unseen categories
                            logger.warning(
                                f"Unseen category in {col}, using default encoding"
                            )
                            data[col] = 0

            # Ensure all required features are present
            if self.feature_names:
                missing_features = set(self.feature_names) - set(data.columns)
                extra_features = set(data.columns) - set(self.feature_names)

                if missing_features:
                    logger.warning(f"Missing features: {missing_features}")
                    # Fill in missing features with default values (0)
                    for feature in missing_features:
                        data[feature] = 0

                if extra_features:
                    logger.warning(
                        f"Extra features that will be ignored: {extra_features}"
                    )

                # Keep only the features used by the model in the correct order
                data = data[self.feature_names]

            logger.info(f"Input data preprocessed. Shape: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Error preprocessing input data: {e}")
            raise

    def predict(self, data):
        """
        Make predictions using the loaded model.

        Args:
            data (dict or pd.DataFrame): Input data for prediction.

        Returns:
            dict: Prediction results.
        """
        try:
            logger.info("Making prediction")

            # Preprocess input data
            X = self.preprocess_input(data)

            # Make prediction
            if self.model is None:
                raise ValueError("Model not loaded")

            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X)
                predictions = self.model.predict(X)

                # Format results
                result = {
                    "prediction": predictions.tolist(),
                    "probability": probabilities[
                        :, 1
                    ].tolist(),  # Probability of positive class
                    "class_names": ["Poisonous", "Edible"],
                }

                # Add human-readable prediction
                result["prediction_label"] = [
                    "Poisonous" if p == 0 else "Edible" for p in predictions
                ]

            else:
                if self.model is None:
                    raise ValueError("Model not loaded")
                predictions = self.model.predict(X)
                result = {
                    "prediction": predictions.tolist(),
                    "class_names": ["Poisonous", "Edible"],
                }
                result["prediction_label"] = [
                    "Poisonous" if p == 0 else "Edible" for p in predictions
                ]

            logger.info(f"Prediction complete: {result['prediction_label']}")
            return result

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
