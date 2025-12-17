"""
Unit tests for the extract component of the ETL pipeline.
Tests data extraction functionality.
"""

import os
import sys
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extract import extract_data


class TestExtract(unittest.TestCase):
    """Tests for the extract module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_path = "data/test/test_data.csv"
        os.makedirs(os.path.dirname(self.test_data_path), exist_ok=True)

        # Create a small test dataset
        self.test_data = pd.DataFrame(
            {
                "class": ["p", "e"],
                "cap-shape": ["x", "b"],
                "cap-surface": ["s", "y"],
                "cap-color": ["n", "w"],
            }
        )
        self.test_data.to_csv(self.test_data_path, sep=";", index=False)

    def tearDown(self):
        """Tear down test fixtures."""
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
        if os.path.exists(os.path.dirname(self.test_data_path)):
            os.rmdir(os.path.dirname(self.test_data_path))

    def test_extract_data_success(self):
        """Test successful data extraction."""
        df = extract_data(os.path.dirname(self.test_data_path))

        # Check if the data was loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(
            list(df.columns), ["class", "cap_shape", "cap_surface", "cap_color"]
        )

        # Check if column names were processed correctly
        self.assertIn("cap_shape", df.columns)
        self.assertNotIn("cap-shape", df.columns)

    def test_extract_data_file_not_found(self):
        """Test extraction with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            extract_data("non_existent_file.csv")

    @patch("pandas.read_csv")
    def test_extract_data_pandas_error(self, mock_read_csv):
        """Test handling of pandas errors."""
        # Mock the pandas read_csv method to raise an exception
        mock_read_csv.side_effect = Exception("Pandas error")

        with self.assertRaises(Exception):
            extract_data(self.test_data_path)


if __name__ == "__main__":
    unittest.main()
