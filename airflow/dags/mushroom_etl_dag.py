"""
Airflow DAG for the mushroom classification ETL pipeline.
Updated to use ColumnStore training from train.py
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import warnings
import os
import sys
import logging
import mlflow


warnings.filterwarnings("ignore")

# Environment configuration
PROJECT_ROOT = "/app"
sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger('airflow.task')

# Import modules
from src.mlflow_utils import init_mlflow
from src.extract import extract_data
from src.transform import transform_data
from src.load import load_data
from src.train import train
from src.redis_conn import redis_connection

redis_conn = redis_connection()

# Import database manager with fallback paths
try:
    from config.database import db_manager
except ImportError:
    try:
        sys.path.append("/app/airflow")
        from config.database import db_manager
    except ImportError:
        logger.warning(
            "Database manager not available - some features may not work"
        )
        db_manager = None

MODULES_AVAILABLE = True
logger.info("Successfully imported ETL components")
logger.info("Successfully imported all required modules")

def mlflow_head(**context):
    experiment_info = init_mlflow()
    if experiment_info['new_experiment']:
        logger.info(f"Created new MLflow experiment: {experiment_info['experiment_name']}")
    else:
        logger.info(f"Using existing MLflow experiment: {experiment_info['experiment_name']}")

    start_time = datetime.now()
    mlflow_run_name = f'pipeline_run_exp_{start_time}'

    # Create parent run but don’t end it here
    parent = mlflow.start_run(run_name=mlflow_run_name)
    context['ti'].xcom_push(key="parent_run_id", value=parent.info.run_id)
    context['ti'].xcom_push(key="experiment_id", value=parent.info.run_id)
    context['ti'].xcom_push(key="start_time", value=start_time)



def task_extract(**context):
    """Extract task - calls dedicated extract module"""
    try:
        init_mlflow()
        logger.info("Starting data extraction")
        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available")
        parent_run_id = context['ti'].xcom_pull(key="parent_run_id", task_ids="start_run")
        start_time = context['ti'].xcom_pull(key="start_time", task_ids="start_run")
        try:
            with mlflow.start_run(run_id=parent_run_id):
                with mlflow.start_run(run_name="extract", nested=True):
                    experiment_id = parent_run_id
                    # Log pipeline parameters
                    mlflow.log_param("experiment_id", experiment_id)
                    # mlflow.log_param("config_path", config_path)
                    mlflow.log_param("start_time", start_time)

        except Exception as e:
            logger.warning(f"Failed to initialize MLflow tracking: {e}")
        # Extract data
        data_shape = extract_data()
        logger.info(
            f"Extracted data shape: {data_shape if data_shape is not None else 'Unknown'}"
        )

        return {"status": "success", "message": "Data extraction completed"}

    except Exception as e:
        logger.error(f"Extract task failed: {e}")
        raise


def task_transform(**context):
    """Transform task - calls dedicated transform module"""
    try:
        logger.info("Starting data transformation")

        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available")

        # Get data from previous task
        ti = context["ti"]
        extract_result = ti.xcom_pull(task_ids="extract")

        if not extract_result or extract_result.get("status") != "success":
            raise ValueError("Extract task did not complete successfully")

        # Transform data
        transformed_data_size = transform_data()
        logger.info(f"Data transformation completed with size {transformed_data_size}")

        return {"status": "success", "message": "Data transformation completed"}

    except Exception as e:
        logger.error(f"Transform task failed: {e}")
        raise


def task_load(**context):
    """Load task - calls dedicated load module and returns experiment_id"""
    try:
        logger.info("Starting data loading to ColumnStore")

        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available")

        # Get data from previous task
        ti = context["ti"]
        transform_result = ti.xcom_pull(task_ids="transform")

        if not transform_result or transform_result.get("status") != "success":
            raise ValueError("Transform task did not complete successfully")

        # Load data and get experiment ID
        shapes = load_data()

        # if not experiment_id:
        #     experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Data loaded to ColumnStore with experiment_id: {shapes}")

        return {
            "status": "success",
            "Shapes": shapes,
            "message": "Data loading completed",
        }

    except Exception as e:
        logger.error(f"Load task failed: {e}")
        raise


def task_train(**context):
    """Train task - calls updated train.py with ColumnStore data loading"""
    try:
        logger.info("Starting model training with ColumnStore data")

        if not MODULES_AVAILABLE:
            raise ImportError("Required modules not available")

        # Get experiment_id from previous task
        ti = context["ti"]
        load_result = ti.xcom_pull(task_ids="load")

        if not load_result or load_result.get("status") != "success":
            raise ValueError("Load task did not complete successfully")

        experiment_id = context['ti'].xcom_pull(key="experiment_id", task_ids="start_run")

        if not experiment_id:
            raise ValueError("No experiment_id found from load task")

        logger.info(f"Training models for experiment_id: {experiment_id}")

        # Training configuration
        # config = {
        #     "xgboost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
        # }

        results = train(experiment_id=experiment_id)
        accuracy = results['metrics']['accuracy']

        logger.info(f"Training completed successfully")
        logger.info(f"Best model: {results.get('best_model')}")
        logger.info(f"Best accuracy: {results.get('best_accuracy', 0):.4f}")

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "best_accuracy": accuracy,
            "message": "Model training completed",
        }

    except Exception as e:
        logger.error(f"Train task failed: {e}")
        raise


def task_evaluate(**context):
    """Evaluate task - evaluates trained models"""
    try:
        logger.info("Starting model evaluation")

        # Get results from training task
        ti = context["ti"]
        train_result = ti.xcom_pull(task_ids="train")

        if not train_result or train_result.get("status") != "success":
            raise ValueError("Train task did not complete successfully")

        experiment_id = train_result.get("experiment_id")
        best_model = train_result.get("best_model")
        best_accuracy = train_result.get("best_accuracy", 0)

        logger.info(f"Evaluation for experiment {experiment_id}")
        logger.info(f"Best model: {best_model} with accuracy: {best_accuracy:.4f}")

        # Here you could add more detailed evaluation logic
        evaluation_results = {
            "experiment_id": experiment_id,
            "best_model": best_model,
            "best_accuracy": best_accuracy,
            "evaluation_passed": best_accuracy > 0.8,  # Example threshold
            "recommendation": "Deploy" if best_accuracy > 0.9 else "Review",
        }

        return {
            "status": "success",
            "evaluation_results": evaluation_results,
            "message": "Model evaluation completed",
        }

    except Exception as e:
        logger.error(f"Evaluate task failed: {e}")
        raise


def task_visualize(**context):
    """Visualize task - creates visualizations and reports"""
    try:
        logger.info("Starting visualization generation")

        # Get results from evaluation task
        ti = context["ti"]
        eval_result = ti.xcom_pull(task_ids="evaluate")

        if not eval_result or eval_result.get("status") != "success":
            raise ValueError("Evaluate task did not complete successfully")

        evaluation_results = eval_result.get("evaluation_results", {})

        logger.info("Creating visualization artifacts")
        logger.info(f"Model performance: {evaluation_results}")

        # Here you could add visualization generation logic
        # For now, just log the results

        return {
            "status": "success",
            "visualizations_created": True,
            "message": "Visualization generation completed",
        }

    except Exception as e:
        logger.error(f"Visualize task failed: {e}")
        raise


# DAG Definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    "mushroom_etl_columnstore_pipeline",
    default_args=default_args,
    description="Mushroom classification pipeline using ColumnStore data with train.py",
    schedule=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mushroom", "classification", "columnstore", "xgboost"],
) as dag:
    # Task definitions
    start_run = PythonOperator(
        task_id="start_run",
        python_callable=mlflow_head,
        provide_context=True,
    )

    extract = PythonOperator(
        task_id="extract",
        python_callable=task_extract,
        doc_md="Extract mushroom data from source",
        provide_context=True,
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=task_transform,
        doc_md="Transform and clean mushroom data",
        provide_context=True,
    )

    load = PythonOperator(
        task_id="load",
        python_callable=task_load,
        doc_md="Load processed data into ColumnStore database",
        provide_context=True,
    )

    _train = PythonOperator(
        task_id="train",
        python_callable=task_train,
        doc_md="Train XGBoost model using ColumnStore data via train.py",
        provide_context=True,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=task_evaluate,
        doc_md="Evaluate trained model performance",
        provide_context=True,
    )

    visualize = PythonOperator(
        task_id="visualize",
        python_callable=task_visualize,
        doc_md="Generate visualizations and reports",
        provide_context=True,
    )

    # Task dependencies
    start_run >> extract >> transform >> load >> _train >> evaluate >> visualize
