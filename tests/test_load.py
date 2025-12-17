"""
Unit tests for the load component of the ETL pipeline.
Tests data loading and model saving functionality.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from sklearn.tree import DecisionTreeClassifier

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.load import load_data, save_model


class TestLoad(unittest.TestCase):
    """Tests for the load module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()

        # Create a small test dataset with features and target
        self.test_data = pd.DataFrame(
            {
                "class_encoded": [0, 1, 0, 1],
                "feature1": [1, 2, 3, 4],
                "feature2": [5, 6, 7, 8],
                "feature3": [9, 10, 11, 12],
            }
        )

        # Create a real model that can be pickled instead of MagicMock
        self.mock_model = DecisionTreeClassifier(random_state=42)
        # Fit it with some dummy data so it can make predictions
        X_dummy = np.array([[1, 2], [3, 4]])
        y_dummy = np.array([0, 1])
        self.mock_model.fit(X_dummy, y_dummy)

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)

    def test_load_data_splits(self):
        """Test data splitting and saving."""
        # Test with specific test_size and random_state
        X_train, X_test, y_train, y_test = load_data(
            self.test_data.copy(), self.temp_dir, test_size=0.5, random_state=42
        )

        # Check if data was split correctly
        self.assertEqual(len(X_train), 2)
        self.assertEqual(len(X_test), 2)
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(y_test), 2)

        # Check if files were saved
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "X_train.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "X_test.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "y_train.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "y_test.csv")))

    def test_save_model(self):
        """Test model saving functionality."""
        model_name = "test_model"

        # Save the mock model
        save_model(self.mock_model, model_name, self.temp_dir)

        # Check if model file was created
        self.assertTrue(
            os.path.exists(os.path.join(self.temp_dir, f"{model_name}.joblib"))
        )

        # Try loading the model to ensure it was saved correctly
        loaded_model = joblib.load(os.path.join(self.temp_dir, f"{model_name}.joblib"))

        # The loaded model should be the same type and have similar properties
        self.assertIsInstance(loaded_model, DecisionTreeClassifier)
        self.assertEqual(loaded_model.random_state, 42)

    @patch("joblib.dump")
    def test_save_model_error_handling(self, mock_dump):
        """Test error handling when saving model fails."""
        # Mock joblib.dump to raise an exception
        mock_dump.side_effect = Exception("Error saving model")

        # The function should catch the exception and not raise it
        with self.assertRaises(Exception):
            save_model(self.mock_model, "test_model", self.temp_dir)


if __name__ == "__main__":
    unittest.main()
