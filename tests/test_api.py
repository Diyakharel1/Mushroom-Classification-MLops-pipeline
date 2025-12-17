"""
Unit tests for the model serving API.
Tests FastAPI endpoints and model prediction functionality.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_serving.api import app
from src.model_serving.model_loader import MushroomModel


class TestAPI(unittest.TestCase):
    """Tests for the FastAPI application."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)

        # Mock response for successful prediction
        self.mock_prediction_response = {
            "prediction": [1],
            "prediction_label": ["Edible"],
            "probability": [0.85],
            "class_names": ["Poisonous", "Edible"],
        }

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"message": "Mushroom Classification API is running"}
        )

    @patch("src.model_serving.model_loader.MushroomModel.predict")
    def test_predict_endpoint_with_valid_data(self, mock_predict):
        """Test the prediction endpoint with valid data."""
        # Set up the mock to return our expected prediction
        mock_predict.return_value = self.mock_prediction_response

        # Test data for prediction
        request_data = {
            "cap_shape": "bell",
            "cap_surface": "smooth",
            "cap_color": "brown",
            "bruises": True,
            "odor": "none",
            "gill_attachment": "free",
            "gill_color": "white",
            "stalk_shape": "enlarging",
            "stalk_color_above_ring": "white",
            "stalk_color_below_ring": "white",
            "habitat": "woods",
        }

        # Make the request
        response = self.client.post("/predict", json=request_data)

        # Check the results
        self.assertEqual(response.status_code, 200)

        # Response should match our mock
        json_response = response.json()
        self.assertEqual(json_response["prediction"], [1])
        self.assertEqual(json_response["prediction_label"], ["Edible"])
        self.assertEqual(json_response["probability"], [0.85])
        self.assertEqual(json_response["class_names"], ["Poisonous", "Edible"])

        # Check that the prediction was called with our data
        mock_predict.assert_called_once()

    def test_predict_endpoint_with_invalid_data(self):
        """Test the prediction endpoint with invalid data."""
        # Missing required fields
        request_data = {"cap_shape": "bell"}  # missing other required fields

        # Make the request
        response = self.client.post("/predict", json=request_data)

        # It should return server error (since validation passes but model fails)
        self.assertEqual(response.status_code, 500)

    @patch("src.model_serving.model_loader.MushroomModel.predict")
    def test_predict_endpoint_with_model_error(self, mock_predict):
        """Test the prediction endpoint when model raises an exception."""
        # Set up the mock to raise an exception
        mock_predict.side_effect = Exception("Model prediction error")

        # Valid request data
        request_data = {
            "cap_shape": "bell",
            "cap_surface": "smooth",
            "cap_color": "brown",
            "bruises": True,
            "odor": "none",
            "gill_attachment": "free",
            "gill_color": "white",
            "stalk_shape": "enlarging",
            "stalk_color_above_ring": "white",
            "stalk_color_below_ring": "white",
            "habitat": "woods",
        }

        # Make the request
        response = self.client.post("/predict", json=request_data)

        # It should return an internal server error
        self.assertEqual(response.status_code, 500)
        self.assertIn("detail", response.json())

    @patch("src.model_serving.database.DatabaseClient")
    @patch("src.model_serving.model_loader.MushroomModel.predict")
    def test_predict_with_database_integration(self, mock_predict, mock_db_client):
        """Test the prediction endpoint with database integration."""
        # Set up the mock to return our expected prediction
        mock_predict.return_value = self.mock_prediction_response

        # Mock database client
        mock_db_instance = MagicMock()
        mock_db_client.return_value = mock_db_instance

        # Test data for prediction
        request_data = {
            "cap_shape": "bell",
            "cap_surface": "smooth",
            "cap_color": "brown",
            "bruises": True,
            "odor": "none",
            "gill_attachment": "free",
            "gill_color": "white",
            "stalk_shape": "enlarging",
            "stalk_color_above_ring": "white",
            "stalk_color_below_ring": "white",
            "habitat": "woods",
        }

        # Make the request
        response = self.client.post("/predict", json=request_data)

        # Check the results
        self.assertEqual(response.status_code, 200)

        # Verify that DatabaseClient was instantiated and store_prediction was called
        mock_db_client.assert_called_once_with("oltp")
        mock_db_instance.store_prediction.assert_called_once()
        mock_db_instance.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
