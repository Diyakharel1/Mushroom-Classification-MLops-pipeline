"""
Extract component of the ETL pipeline for Mushroom Classification.
Responsible for reading data from the source files.
"""

import os
import pandas as pd
import logging
from .redis_conn import redis_connection
import pickle

redis_conn = redis_connection()

# Create logs directory using absolute paths
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)
os.makedirs(log_dir, exist_ok=True)

# Configure logging with safe absolute paths
log_path = os.path.join(log_dir, "extract.log")

# Only set up file handler if we have write permissions
try:
    handlers = [logging.StreamHandler()]
    if os.access(log_dir, os.W_OK):
        handlers.append(logging.FileHandler(log_path))
    else:
        print(f"Warning: No write access to {log_dir}, using console logging only")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
except Exception as e:
    # Fallback to console-only logging
    print(f"Error setting up file logging: {e}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

logger = logging.getLogger(__name__)


def extract_data():
    """
    Extract data from source files or directory.

    Args:
        data_path (str): Path to the data file or directory containing data files.

    Returns:
        pd.DataFrame: Extracted data.
    """
    try:
        data_path = "/app/data/raw/sample_data.csv"
        logger.info(f"Extracting data from {data_path}")

        # Check if path exists
        if not os.path.exists(data_path):
            logger.error(f"Path {data_path} does not exist")
            raise FileNotFoundError(f"Path {data_path} does not exist")

        # Handle both file and directory paths
        if os.path.isfile(data_path):
            # Direct file path
            file_path = data_path
            logger.info(f"Using file: {file_path}")
        elif os.path.isdir(data_path):
            # Directory path - find CSV files
            csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
            if not csv_files:
                logger.error(f"No CSV files found in {data_path}")
                raise FileNotFoundError(f"No CSV files found in {data_path}")

            # Use the first CSV file found
            file_path = os.path.join(data_path, csv_files[0])
            logger.info(f"Using file: {file_path}")
        else:
            raise ValueError(f"Path {data_path} is neither a file nor a directory")

        # Try different delimiters and encodings
        delimiters = [";", ",", "\t"]
        encodings = ["utf-8", "latin-1", "cp1252"]

        df = None
        for delimiter in delimiters:
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                    if not df.empty and len(df.columns) > 1:
                        logger.info(
                            f"Successfully read with delimiter='{delimiter}' and encoding='{encoding}'"
                        )
                        break
                except Exception as e:
                    logger.debug(
                        f"Failed with delimiter='{delimiter}' and encoding='{encoding}': {e}"
                    )
                    continue
            if df is not None and not df.empty:
                break

        if df is None or df.empty:
            # Fallback: try with default pandas settings
            try:
                df = pd.read_csv(file_path)
                logger.info("Successfully read with default pandas settings")
            except Exception as e:
                logger.error(f"Could not read file with any method: {e}")
                raise

        # Clean column names - replace hyphens with underscores and strip whitespace
        df.columns = df.columns.str.replace("-", "_").str.strip().str.lower()

        # Basic data validation
        if df.empty:
            raise ValueError("Extracted DataFrame is empty")

        if len(df.columns) == 0:
            raise ValueError("Extracted DataFrame has no columns")

        logger.info(f"Successfully extracted {len(df)} records from {file_path}")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        serialized_df = pickle.dumps(df)
        redis_conn.set('df', serialized_df)
        return {'shape': df.shape}

    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise
