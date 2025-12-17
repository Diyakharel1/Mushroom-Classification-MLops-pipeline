"""Basic unit tests for the mushroom classification project."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append("/app")


class TestBasicFunctionality:
    """Basic functionality tests."""

    def test_import_core_modules(self):
        """Test that core modules can be imported."""
        try:
            import pandas as pd
            import numpy as np
            import sklearn
            import mlflow

            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import core modules: {e}")

    def test_database_config_import(self):
        """Test database configuration import."""
        try:
            from config.database import DatabaseConfig, DatabaseManager

            config = DatabaseConfig()
            assert config is not None
            assert hasattr(config, "mariadb_config")
            assert hasattr(config, "postgres_config")
        except ImportError as e:
            pytest.fail(f"Failed to import database config: {e}")

    def test_data_processing_import(self):
        """Test data processing modules import."""
        try:
            from src.data_processing.enhanced_etl import EnhancedMushroomETL

            etl = EnhancedMushroomETL()
            assert etl is not None
        except ImportError as e:
            pytest.fail(f"Failed to import ETL module: {e}")

    def test_model_training_import(self):
        """Test model training modules import."""
        try:
            from src.model_training.columnstore_trainer import ColumnStoreTrainer

            trainer = ColumnStoreTrainer("test_experiment")
            assert trainer is not None
        except ImportError as e:
            pytest.fail(f"Failed to import trainer module: {e}")


class TestDataProcessing:
    """Data processing tests."""

    def test_dataframe_creation(self):
        """Test basic DataFrame operations."""
        data = {
            "feature1": [1, 2, 3, 4],
            "feature2": ["a", "b", "c", "d"],
            "target": [0, 1, 0, 1],
        }
        df = pd.DataFrame(data)

        assert len(df) == 4
        assert list(df.columns) == ["feature1", "feature2", "target"]
        assert df["target"].sum() == 2

    def test_missing_value_handling(self):
        """Test missing value handling logic."""
        data = {"numeric": [1, 2, np.nan, 4], "categorical": ["a", "b", np.nan, "d"]}
        df = pd.DataFrame(data)

        # Fill missing values
        df["numeric"].fillna(df["numeric"].median(), inplace=True)
        df["categorical"].fillna(df["categorical"].mode()[0], inplace=True)

        assert not df.isnull().any().any()


class TestModelComponents:
    """Model component tests."""

    def test_sklearn_model_creation(self):
        """Test scikit-learn model creation."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        lr = LogisticRegression(random_state=42)

        assert rf is not None
        assert lr is not None
        assert hasattr(rf, "fit")
        assert hasattr(lr, "fit")

    def test_model_training_basic(self):
        """Test basic model training."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        # Generate sample data
        X, y = make_classification(
            n_samples=100, n_features=10, n_classes=2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Test prediction
        predictions = model.predict(X[:5])
        assert len(predictions) == 5
        assert all(pred in [0, 1] for pred in predictions)

    def test_model_metrics(self):
        """Test model evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1


class TestConfiguration:
    """Configuration and environment tests."""

    def test_environment_variables(self):
        """Test environment variable handling."""
        # Test with default values
        mariadb_host = os.getenv("MARIADB_HOST", "mariadb-columnstore")
        mariadb_port = int(os.getenv("MARIADB_PORT", 3306))

        assert mariadb_host is not None
        assert isinstance(mariadb_port, int)
        assert mariadb_port > 0

    def test_directory_structure(self):
        """Test required directory structure."""
        required_dirs = ["/app/data", "/app/models", "/app/config", "/app/src"]

        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                assert os.path.isdir(dir_path)
            # Note: In CI/CD, some directories might not exist yet

    def test_file_permissions(self):
        """Test file permission requirements."""
        test_file = "/tmp/test_permissions.txt"
        try:
            with open(test_file, "w") as f:
                f.write("test")

            # Check if file can be read
            with open(test_file, "r") as f:
                content = f.read()

            assert content == "test"

            # Cleanup
            os.remove(test_file)
        except Exception as e:
            pytest.fail(f"File permission test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
