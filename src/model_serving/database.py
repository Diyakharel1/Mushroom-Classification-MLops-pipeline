"""
Database utility for storing mushroom classification data and predictions.
This module connects to MariaDB databases for OLTP and OLAP operations.
"""

import os
import yaml
import logging
import pandas as pd
import pymysql
from datetime import datetime
from sqlalchemy import (
    create_engine,
    text,
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/database.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    config_path = "config/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    logger.info(f"Loaded configuration from {config_path}")
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    # Default database config if file loading fails
    config = {
        "database": {
            "oltp": {
                "host": "localhost",
                "port": 3307,
                "user": "root",
                "password": "example",
                "database": "mushroom_oltp_db",
            },
            "olap": {
                "host": "localhost",
                "port": 3308,
                "user": "root",
                "password": "example",
                "database": "mushroom_olap_db",
            },
        }
    }

# Define SQLAlchemy Base class
Base = declarative_base()


# Define database models
class MushroomData(Base):
    """Model for raw mushroom data in OLTP database."""

    __tablename__ = "mushrooms"

    id = Column(Integer, primary_key=True)
    cap_shape = Column(String(50))
    cap_color = Column(String(50))
    cap_surface = Column(String(50))
    gill_color = Column(String(50))
    gill_attachment = Column(String(50))
    stem_color = Column(String(50))
    ring_type = Column(String(50))
    habitat = Column(String(50))
    cap_diameter = Column(Float)
    stem_height = Column(Float)
    stem_width = Column(Float)
    does_bruise_or_bleed = Column(Boolean)
    has_ring = Column(Boolean)
    class_label = Column(String(20))  # 'Poisonous' or 'Edible'
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class PredictionLog(Base):
    """Model for prediction logs in OLTP database."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True)
    prediction_label = Column(String(20))  # 'Poisonous' or 'Edible'
    prediction_probability = Column(Float)  # Probability of the positive class
    model_type = Column(
        String(50)
    )  # 'xgboost', 'logistic_regression', or 'decision_tree'
    cap_shape = Column(String(50))
    cap_color = Column(String(50))
    cap_surface = Column(String(50))
    gill_color = Column(String(50))
    gill_attachment = Column(String(50))
    stem_color = Column(String(50))
    ring_type = Column(String(50))
    habitat = Column(String(50))
    cap_diameter = Column(Float)
    stem_height = Column(Float)
    stem_width = Column(Float)
    does_bruise_or_bleed = Column(Boolean)
    has_ring = Column(Boolean)
    created_at = Column(DateTime, default=datetime.now)


class DatabaseClient:
    """Client for interacting with the mushroom databases."""

    def __init__(self, db_type="oltp"):
        """
        Initialize the database client.

        Args:
            db_type (str): Type of database to connect to ('oltp' or 'olap').
        """
        self.db_type = db_type
        self.engine = None
        self.session = None
        self.connect()

    def connect(self):
        """Connect to the database."""
        try:
            logger.info(f"Connecting to {self.db_type.upper()} database")

            # Get database configuration
            db_config = config["database"][self.db_type]

            # Create database connection string
            connection_string = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

            # Create engine
            self.engine = create_engine(connection_string)

            # Create tables if they don't exist
            Base.metadata.create_all(self.engine)

            # Create session
            Session = sessionmaker(bind=self.engine)
            self.session = Session()

            logger.info(f"Connected to {self.db_type.upper()} database")

        except Exception as e:
            logger.error(f"Error connecting to {self.db_type.upper()} database: {e}")
            raise

    def close(self):
        """Close the database connection."""
        try:
            if self.session:
                self.session.close()
            logger.info(f"Closed connection to {self.db_type.upper()} database")

        except Exception as e:
            logger.error(
                f"Error closing {self.db_type.upper()} database connection: {e}"
            )

    def store_prediction(self, features, prediction):
        """
        Store a prediction result in the OLTP database.

        Args:
            features (dict): Features used for prediction.
            prediction (dict): Prediction result.

        Returns:
            int: ID of the inserted record.
        """
        try:
            logger.info("Storing prediction result")

            # Create prediction log
            prediction_log = PredictionLog(
                prediction_label=prediction["prediction_label"][
                    0
                ],  # Take first prediction
                prediction_probability=(
                    prediction.get("probability", [0.0])[0]
                    if prediction.get("probability")
                    else None
                ),
                model_type=prediction.get("model_type", "unknown"),
                **features,  # Unpack feature dict
            )

            # Add to session and commit
            self.session.add(prediction_log)
            self.session.commit()

            logger.info(f"Stored prediction result with ID {prediction_log.id}")
            return prediction_log.id

        except Exception as e:
            logger.error(f"Error storing prediction result: {e}")
            self.session.rollback()
            raise

    def store_mushroom_data(self, mushroom_data):
        """
        Store mushroom data in the OLTP database.

        Args:
            mushroom_data (dict): Mushroom data to store.

        Returns:
            int: ID of the inserted record.
        """
        try:
            logger.info("Storing mushroom data")

            # Create mushroom data record
            mushroom_record = MushroomData(**mushroom_data)

            # Add to session and commit
            self.session.add(mushroom_record)
            self.session.commit()

            logger.info(f"Stored mushroom data with ID {mushroom_record.id}")
            return mushroom_record.id

        except Exception as e:
            logger.error(f"Error storing mushroom data: {e}")
            self.session.rollback()
            raise

    def load_data_to_olap(self):
        """
        Transfer data from OLTP to OLAP database for analytical processing.
        """
        try:
            logger.info("Transferring data from OLTP to OLAP database")

            # Connect to OLAP database
            olap_client = DatabaseClient("olap")

            # Define query to get recent prediction data
            query = """
            SELECT 
                pl.id, pl.prediction_label, pl.prediction_probability, pl.model_type,
                pl.cap_shape, pl.cap_color, pl.cap_surface, pl.gill_color,
                pl.gill_attachment, pl.stem_color, pl.ring_type, pl.habitat,
                pl.cap_diameter, pl.stem_height, pl.stem_width,
                pl.does_bruise_or_bleed, pl.has_ring, pl.created_at
            FROM 
                prediction_logs pl
            WHERE 
                pl.created_at > DATE_SUB(NOW(), INTERVAL 1 DAY)
            """

            # Execute query
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()

            logger.info(f"Found {len(rows)} new records to transfer to OLAP")

            # If there are rows to transfer
            if rows:
                # Convert rows to DataFrame
                data = pd.DataFrame(rows, columns=result.keys())

                # Upload to OLAP database
                data.to_sql(
                    "analytics_predictions",
                    olap_client.engine,
                    if_exists="append",
                    index=False,
                )

                logger.info(f"Transferred {len(data)} records to OLAP database")

            # Close OLAP connection
            olap_client.close()

        except Exception as e:
            logger.error(f"Error transferring data to OLAP database: {e}")
            raise

    def execute_query(self, query):
        """
        Execute a SQL query against the database.

        Args:
            query (str): SQL query to execute.

        Returns:
            pd.DataFrame: Result of the query as a DataFrame.
        """
        try:
            logger.info(f"Executing query on {self.db_type.upper()} database")

            # Execute query
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchall()

            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=result.keys())

            logger.info(f"Query returned {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise


def save_prediction_to_db(features, prediction, confidence=None):
    """
    Save prediction to database - wrapper function for compatibility with tests.

    Args:
        features (dict): Input features used for prediction
        prediction (str or int): The prediction result
        confidence (float, optional): Prediction confidence score
    """
    try:
        db = DatabaseClient()
        db.store_prediction(features, prediction)
        db.close()
        logger.info("Prediction saved to database successfully")
    except Exception as e:
        logger.error(f"Error saving prediction to database: {e}")
        raise
