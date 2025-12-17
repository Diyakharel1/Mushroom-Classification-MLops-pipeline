"""
Database configuration and connection management for MariaDB and PostgreSQL.
"""

import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from typing import Optional, Dict, Any
import pymysql
import psycopg2
import time

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database connections for both MariaDB and PostgreSQL."""

    def __init__(self):
        self.mariadb_engine = None
        self.postgres_engine = None
        self._init_connections()

    def _init_connections(self):
        """Initialize database connections with retry logic."""
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                # MariaDB connection
                mariadb_host = os.getenv("MARIADB_HOST", "mariadb-columnstore")
                mariadb_port = os.getenv("MARIADB_PORT", "3306")
                mariadb_user = os.getenv("MARIADB_USER", "mushroom_user")
                mariadb_password = os.getenv("MARIADB_PASSWORD", "mushroom_pass")
                mariadb_database = os.getenv("MARIADB_DATABASE", "mushroom_analytics")

                mariadb_url = f"mysql+pymysql://{mariadb_user}:{mariadb_password}@{mariadb_host}:{mariadb_port}/{mariadb_database}?charset=utf8mb4"
                self.mariadb_engine = create_engine(
                    mariadb_url,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False,
                    connect_args={
                        "connect_timeout": 60,
                        "read_timeout": 30,
                        "write_timeout": 30,
                    },
                )

                # Test MariaDB connection
                with self.mariadb_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("MariaDB connection initialized successfully")

                # PostgreSQL connection (for Airflow metadata)
                postgres_url = os.getenv(
                    "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
                    "postgresql+psycopg2://airflow:airflow@postgres/airflow",
                )
                self.postgres_engine = create_engine(
                    postgres_url,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False,
                )

                # Test PostgreSQL connection
                with self.postgres_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("PostgreSQL connection initialized successfully")

                return  # Success, exit retry loop

            except Exception as e:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        "Failed to initialize database connections after all retries"
                    )
                    raise

    def test_mariadb_connection(self) -> bool:
        """Test MariaDB connection."""
        try:
            with self.mariadb_engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                row = result.fetchone()
                if row and row[0] == 1:
                    logger.info("MariaDB connection test successful")
                    return True
                return False
        except Exception as e:
            logger.error(f"MariaDB connection test failed: {e}")
            return False

    def test_postgres_connection(self) -> bool:
        """Test PostgreSQL connection."""
        try:
            with self.postgres_engine.connect() as conn:
                result = conn.execute(text("SELECT 1 as test"))
                row = result.fetchone()
                if row and row[0] == 1:
                    logger.info("PostgreSQL connection test successful")
                    return True
                return False
        except Exception as e:
            logger.error(f"PostgreSQL connection test failed: {e}")
            return False

    def create_tables(self) -> bool:
        """Create database tables if they don't exist."""
        try:
            create_tables_sql = """
            CREATE TABLE IF NOT EXISTS cleaned_features (
                id INT AUTO_INCREMENT PRIMARY KEY,
                cap_diameter DECIMAL(10,6),
                cap_shape VARCHAR(50),
                cap_surface VARCHAR(50),
                cap_color VARCHAR(50),
                does_bruise_or_bleed VARCHAR(50),
                gill_attachment VARCHAR(50),
                gill_spacing VARCHAR(50),
                gill_color VARCHAR(50),
                stem_height DECIMAL(10,6),
                stem_width DECIMAL(10,6),
                stem_root VARCHAR(50),
                stem_surface VARCHAR(50),
                stem_color VARCHAR(50),
                veil_type VARCHAR(50),
                veil_color VARCHAR(50),
                has_ring VARCHAR(50),
                ring_type VARCHAR(50),
                spore_print_color VARCHAR(50),
                habitat VARCHAR(50),
                season VARCHAR(50),
                class VARCHAR(10),
                data_version VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_class (class),
                INDEX idx_data_version (data_version)
            ) ENGINE=InnoDB;
            """

            with self.mariadb_engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
                logger.info("Database tables created successfully")
                return True
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            return False

    def create_columnstore_tables(self) -> bool:
        """Create ColumnStore tables - alias for create_tables() for compatibility."""
        return self.create_tables()

    def insert_cleaned_data(self, df: pd.DataFrame, data_version: str) -> bool:
        """Insert cleaned data into MariaDB."""
        try:
            # Add data_version column if not present
            if "data_version" not in df.columns:
                df = df.copy()
                df["data_version"] = data_version

            # Insert data using pandas to_sql with smaller chunks
            df.to_sql(
                "cleaned_features",
                self.mariadb_engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=500,
            )

            logger.info(f"Inserted {len(df)} records into cleaned_features table")
            return True

        except Exception as e:
            logger.error(f"Error inserting cleaned data: {e}")
            return False

    def create_data_splits(
        self, experiment_id: str, train_ids: list, test_ids: list, validation_ids: list
    ) -> bool:
        """Create data splits for an experiment."""
        try:
            with self.mariadb_engine.connect() as conn:
                # Insert train data references
                if train_ids:
                    for feature_id in train_ids:
                        conn.execute(
                            text(
                                "INSERT INTO train_data (feature_id, experiment_id) VALUES (:feature_id, :experiment_id)"
                            ),
                            {"feature_id": feature_id, "experiment_id": experiment_id},
                        )

                # Insert test data references
                if test_ids:
                    for feature_id in test_ids:
                        conn.execute(
                            text(
                                "INSERT INTO test_data (feature_id, experiment_id) VALUES (:feature_id, :experiment_id)"
                            ),
                            {"feature_id": feature_id, "experiment_id": experiment_id},
                        )

                # Insert validation data references
                if validation_ids:
                    for feature_id in validation_ids:
                        conn.execute(
                            text(
                                "INSERT INTO validation_data (feature_id, experiment_id) VALUES (:feature_id, :experiment_id)"
                            ),
                            {"feature_id": feature_id, "experiment_id": experiment_id},
                        )

                conn.commit()
                logger.info(f"Created data splits for experiment {experiment_id}")
                return True

        except Exception as e:
            logger.error(f"Error creating data splits: {e}")
            return False

    def get_experiment_data(
        self, experiment_id: str, data_type: str = "train"
    ) -> Optional[pd.DataFrame]:
        """Get experiment data from MariaDB."""
        try:
            table_map = {
                "train": "train_data",
                "test": "test_data",
                "validation": "validation_data",
            }

            if data_type not in table_map:
                raise ValueError(f"Invalid data_type: {data_type}")

            query = f"""
            SELECT cf.* 
            FROM cleaned_features cf
            INNER JOIN {table_map[data_type]} split_data ON cf.id = split_data.feature_id
            WHERE split_data.experiment_id = :experiment_id
            """

            df = pd.read_sql(
                query, self.mariadb_engine, params={"experiment_id": experiment_id}
            )
            logger.info(
                f"Retrieved {len(df)} records for experiment {experiment_id}, data_type {data_type}"
            )
            return df

        except Exception as e:
            logger.error(f"Error getting experiment data: {e}")
            return None

    def verify_database_setup(self) -> Dict[str, Any]:
        """Verify database setup and return status information."""
        status = {
            "mariadb_connected": False,
            "postgres_connected": False,
            "tables_exist": False,
            "sample_data_exists": False,
        }

        try:
            # Test connections
            status["mariadb_connected"] = self.test_mariadb_connection()
            status["postgres_connected"] = self.test_postgres_connection()

            if status["mariadb_connected"]:
                # Check if tables exist
                with self.mariadb_engine.connect() as conn:
                    result = conn.execute(text("SHOW TABLES LIKE 'cleaned_features'"))
                    status["tables_exist"] = len(result.fetchall()) > 0

                    if status["tables_exist"]:
                        # Check if sample data exists
                        result = conn.execute(
                            text("SELECT COUNT(*) FROM cleaned_features")
                        )
                        count = result.fetchone()[0]
                        status["sample_data_exists"] = count > 0
                        status["record_count"] = count

        except Exception as e:
            logger.error(f"Error verifying database setup: {e}")
            status["error"] = str(e)

        return status


# Global database manager instance
db_manager = DatabaseManager()
