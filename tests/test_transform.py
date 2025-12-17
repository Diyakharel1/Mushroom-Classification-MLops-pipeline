"""
Unit tests for the transform component of the ETL pipeline.
Tests data transformation functionality.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.transform import transform_data


class TestTransform(unittest.TestCase):
    """Tests for the transform module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a small test dataset with some missing values
        self.test_data = pd.DataFrame(
            {
                "class": ["p", "e", "p", "e"],
                "cap_shape": ["x", "b", "x", None],
                "cap_surface": ["s", "y", None, "y"],
                "cap_color": ["n", "w", "n", "w"],
                "has_ring": ["t", "f", "t", "f"],
                "habitat": ["g", "p", "g", "p"],
                "odor": ["n", "a", "n", "a"],
            }
        )

    def test_handle_missing_values(self):
        """Test handling of missing values in the transform function."""
        df_transformed = transform_data(self.test_data.copy())

        # Check if missing values were handled (no nulls in transformed data)
        self.assertEqual(df_transformed.isnull().sum().sum(), 0)

        # The transformation function should have removed the original columns
        self.assertNotIn("cap_shape", df_transformed.columns)
        self.assertNotIn("cap_surface", df_transformed.columns)

    def test_encode_categorical_variables(self):
        """Test encoding of categorical variables in the transform function."""
        df_transformed = transform_data(self.test_data.copy())

        # Check if class encoding was done
        self.assertIn("class_encoded", df_transformed.columns)

        # Check if categorical variables were encoded or removed
        for col in self.test_data.columns:
            if col != "class":
                self.assertNotIn(col, df_transformed.columns)

        # Check if we have one-hot encoded columns (should have more columns than input)
        self.assertGreater(len(df_transformed.columns), len(self.test_data.columns))

    def test_transform_data_end_to_end(self):
        """Test the entire transformation pipeline."""
        df_transformed = transform_data(self.test_data.copy())

        # Check if the final dataframe has the expected structure
        self.assertIn("class_encoded", df_transformed.columns)

        # No missing values should be present
        self.assertEqual(df_transformed.isnull().sum().sum(), 0)

        # The original categorical columns should be transformed
        for col in self.test_data.columns:
            if col != "class":
                self.assertNotIn(col, df_transformed.columns)


if __name__ == "__main__":
    unittest.main()
