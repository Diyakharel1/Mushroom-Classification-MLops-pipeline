"""Enhanced ETL pipeline with MariaDB/ColumnStore integration."""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import sys
import inspect

# Add project root to path
sys.path.append("/app")
from config.database import db_manager

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Log when this module is imported
caller_frame = inspect.currentframe().f_back
if caller_frame:
    caller_file = caller_frame.f_globals.get("__file__", "unknown")
    logger.info(f"enhanced_etl.py imported by: {caller_file}")
else:
    logger.info("enhanced_etl.py loaded directly")


class EnhancedMushroomETL:
    """Enhanced ETL pipeline with ColumnStore integration."""

    def __init__(self, data_path: str = "/app/data/raw/mushroom_data.csv"):
        # Log instantiation
        caller_frame = inspect.currentframe().f_back
        caller_info = "unknown"
        if caller_frame:
            caller_info = f"{caller_frame.f_globals.get('__file__', 'unknown')}:{caller_frame.f_lineno}"

        logger.info(f"EnhancedMushroomETL instantiated by: {caller_info}")

        self.data_path = data_path
        self.data_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.encoders = {}
        self.scalers = {}
        self.usage_stats = {
            "created_at": datetime.now(),
            "methods_called": [],
            "caller_info": caller_info,
        }

    def extract_data(self) -> pd.DataFrame:
        """Extract data from source."""
        self.usage_stats["methods_called"].append(f"extract_data_{datetime.now()}")
        try:
            logger.info(f"Extracting data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            logger.info(f"Extracted {len(df)} rows and {len(df.columns)} columns")

            # Log data lineage
            self._log_data_lineage(
                source_file=os.path.basename(self.data_path),
                step="extract",
                transformation="Raw data extraction from CSV",
                rows_input=0,
                rows_output=len(df),
            )

            return df

        except Exception as e:
            logger.error(f"Failed to extract data: {e}")
            raise

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform and clean the data."""
        self.usage_stats["methods_called"].append(f"transform_data_{datetime.now()}")
        try:
            logger.info("Starting data transformation")
            initial_rows = len(df)

            # Create a copy for transformation
            df_transformed = df.copy()

            # Handle missing values
            df_transformed = self._handle_missing_values(df_transformed)

            # Encode categorical variables
            df_transformed = self._encode_categorical_variables(df_transformed)

            # Scale numerical features
            df_transformed = self._scale_numerical_features(df_transformed)

            # Feature engineering
            df_transformed = self._engineer_features(df_transformed)

            # Data validation
            self._validate_transformed_data(df_transformed)

            logger.info(f"Data transformation completed. Shape: {df_transformed.shape}")

            # Log data lineage
            self._log_data_lineage(
                source_file="mushroom_data.csv",
                step="transform",
                transformation="Data cleaning, encoding, scaling, feature engineering",
                rows_input=initial_rows,
                rows_output=len(df_transformed),
            )

            return df_transformed

        except Exception as e:
            logger.error(f"Failed to transform data: {e}")
            raise

    def load_data(self, df: pd.DataFrame) -> bool:
        """Load transformed data into ColumnStore."""
        self.usage_stats["methods_called"].append(f"load_data_{datetime.now()}")
        try:
            logger.info("Loading data into MariaDB/ColumnStore")

            # Ensure database tables exist
            if not db_manager.create_columnstore_tables():
                raise Exception("Failed to create ColumnStore tables")

            # Insert cleaned data
            success = db_manager.insert_cleaned_data(df, self.data_version)

            if success:
                logger.info(f"Successfully loaded {len(df)} rows into ColumnStore")

                # Log data lineage
                self._log_data_lineage(
                    source_file="transformed_data",
                    step="load",
                    transformation="Data loaded into MariaDB/ColumnStore",
                    rows_input=len(df),
                    rows_output=len(df),
                )

            return success

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def create_train_test_splits(
        self, experiment_id: str, test_size: float = 0.2, val_size: float = 0.1
    ) -> Dict[str, Any]:
        """Create train/test/validation splits in ColumnStore."""
        self.usage_stats["methods_called"].append(
            f"create_train_test_splits_{datetime.now()}"
        )
        try:
            logger.info("Creating train/test/validation splits")

            # Get all feature IDs from ColumnStore
            with db_manager.get_mariadb_connection() as conn:
                result = conn.execute("SELECT id FROM cleaned_features ORDER BY id")
                all_ids = [row[0] for row in result.fetchall()]

            # Create splits
            train_ids, temp_ids = train_test_split(
                all_ids, test_size=(test_size + val_size), random_state=42
            )

            test_ids, val_ids = train_test_split(
                temp_ids, test_size=(val_size / (test_size + val_size)), random_state=42
            )

            # Store splits in database
            success = db_manager.create_data_splits(
                experiment_id, train_ids, test_ids, val_ids
            )

            if success:
                split_info = {
                    "experiment_id": experiment_id,
                    "train_size": len(train_ids),
                    "test_size": len(test_ids),
                    "validation_size": len(val_ids),
                    "total_size": len(all_ids),
                }

                logger.info(f"Data splits created: {split_info}")
                return split_info
            else:
                raise Exception("Failed to create data splits")

        except Exception as e:
            logger.error(f"Failed to create train/test splits: {e}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values")

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(
                f"Found missing values: {missing_counts[missing_counts > 0].to_dict()}"
            )

            # Strategy: Fill categorical with mode, numerical with median
            for column in df.columns:
                if df[column].dtype == "object":
                    df[column].fillna(df[column].mode()[0], inplace=True)
                else:
                    df[column].fillna(df[column].median(), inplace=True)
        else:
            logger.info("No missing values found")

        return df

    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        logger.info("Encoding categorical variables")

        categorical_columns = df.select_dtypes(include=["object"]).columns
        categorical_columns = [col for col in categorical_columns if col != "class"]

        for column in categorical_columns:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))
            self.encoders[column] = encoder

        # Handle target variable separately
        if "class" in df.columns:
            target_encoder = LabelEncoder()
            df["class"] = target_encoder.fit_transform(df["class"])
            self.encoders["class"] = target_encoder

        logger.info(f"Encoded {len(categorical_columns)} categorical columns")
        return df

    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info("Scaling numerical features")

        numerical_columns = df.select_dtypes(include=[np.number]).columns
        numerical_columns = [col for col in numerical_columns if col != "class"]

        if len(numerical_columns) > 0:
            scaler = StandardScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            self.scalers["numerical_features"] = scaler
            logger.info(f"Scaled {len(numerical_columns)} numerical columns")

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features."""
        logger.info("Engineering additional features")

        # Example feature engineering (customize based on domain knowledge)
        if "cap_diameter" in df.columns and "stem_height" in df.columns:
            df["cap_to_stem_ratio"] = df["cap_diameter"] / (df["stem_height"] + 1e-8)

        if "stem_width" in df.columns and "stem_height" in df.columns:
            df["stem_volume_proxy"] = df["stem_width"] * df["stem_height"]

        logger.info("Feature engineering completed")
        return df

    def _validate_transformed_data(self, df: pd.DataFrame) -> None:
        """Validate transformed data quality."""
        logger.info("Validating transformed data")

        # Check for infinite values
        if np.isinf(df.select_dtypes(include=[np.number])).any().any():
            logger.warning("Found infinite values in data")

        # Check for null values after transformation
        if df.isnull().any().any():
            logger.warning("Found null values after transformation")

        # Check data types
        logger.info(f"Data types after transformation: {df.dtypes.to_dict()}")

        logger.info("Data validation completed")

    def _log_data_lineage(
        self,
        source_file: str,
        step: str,
        transformation: str,
        rows_input: int,
        rows_output: int,
        processing_time: float = 0.0,
    ) -> None:
        """Log data lineage information."""
        try:
            with db_manager.get_mariadb_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO data_lineage 
                    (source_file, processing_step, transformation_applied, 
                     rows_input, rows_output, processing_time_seconds, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        source_file,
                        step,
                        transformation,
                        rows_input,
                        rows_output,
                        processing_time,
                        "enhanced_etl",
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log data lineage: {e}")

    def save_artifacts(self) -> None:
        """Save transformation artifacts."""
        try:
            artifacts_dir = f"/app/models/artifacts_{self.data_version}"
            os.makedirs(artifacts_dir, exist_ok=True)

            # Save encoders
            if self.encoders:
                joblib.dump(self.encoders, f"{artifacts_dir}/encoders.pkl")
                logger.info("Saved encoders")

            # Save scalers
            if self.scalers:
                joblib.dump(self.scalers, f"{artifacts_dir}/scalers.pkl")
                logger.info("Saved scalers")

        except Exception as e:
            logger.error(f"Failed to save artifacts: {e}")

    def run_etl_pipeline(self, experiment_id: str) -> Dict[str, Any]:
        """Run the complete ETL pipeline."""
        self.usage_stats["methods_called"].append(f"run_etl_pipeline_{datetime.now()}")
        try:
            start_time = datetime.now()
            logger.info(
                f"Starting enhanced ETL pipeline - Usage stats: {self.usage_stats}"
            )

            # Extract
            raw_data = self.extract_data()

            # Transform
            transformed_data = self.transform_data(raw_data)

            # Load
            load_success = self.load_data(transformed_data)

            if not load_success:
                raise Exception("Failed to load data into ColumnStore")

            # Create data splits
            split_info = self.create_train_test_splits(experiment_id)

            # Save artifacts
            self.save_artifacts()

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            pipeline_results = {
                "status": "success",
                "data_version": self.data_version,
                "experiment_id": experiment_id,
                "processing_time_seconds": processing_time,
                "raw_data_shape": raw_data.shape,
                "transformed_data_shape": transformed_data.shape,
                "split_info": split_info,
                "artifacts_saved": True,
                "usage_stats": self.usage_stats,
            }

            logger.info(f"ETL pipeline completed successfully: {pipeline_results}")
            return pipeline_results

        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "data_version": self.data_version,
                "usage_stats": self.usage_stats,
            }

    def log_usage_summary(self):
        """Log a summary of how this instance was used."""
        logger.info(f"ETL Usage Summary: {self.usage_stats}")


def main():
    """Main function for standalone execution."""
    logger.info("Starting Enhanced ETL Pipeline - STANDALONE EXECUTION")

    # Test database connections
    if not db_manager.test_mariadb_connection():
        logger.error("MariaDB connection test failed")
        return False

    # Initialize ETL pipeline
    etl = EnhancedMushroomETL()

    # Generate experiment ID
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run pipeline
    results = etl.run_etl_pipeline(experiment_id)

    if results["status"] == "success":
        logger.info("Enhanced ETL pipeline completed successfully")
        return True
    else:
        logger.error("Enhanced ETL pipeline failed")
        return False


# Log when module is executed directly
if __name__ == "__main__":
    logger.info("enhanced_etl.py executed as main script")
    success = main()
    exit(0 if success else 1)
else:
    logger.info("enhanced_etl.py imported as module")
