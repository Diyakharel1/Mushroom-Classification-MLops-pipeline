"""
Monitoring module for the mushroom classification pipeline.
Tracks model performance and data drift with ColumnStore integration.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append("/app")

# Configure logging
logger = logging.getLogger(__name__)


def monitor_model_performance(
    model, X_test, y_test, X_reference=None, metrics_path=None
):
    """
    Monitor model performance and check for data drift.

    Args:
        model: Trained model (can be None for basic monitoring)
        X_test: Test features (can be None)
        y_test: Test target (can be None)
        X_reference: Reference data to compare against for drift detection
        metrics_path: Path to save monitoring metrics and visualizations

    Returns:
        dict: Monitoring metrics
    """
    logger.info("Starting model performance monitoring")

    monitoring_results = {
        "feature_importance": [],
        "drifted_features": [],
        "num_drifted_features": 0,
        "report_path": "",
        "monitoring_timestamp": pd.Timestamp.now().isoformat(),
    }

    try:
        # Create output directory if specified
        if metrics_path:
            os.makedirs(metrics_path, exist_ok=True)
            monitoring_results["report_path"] = metrics_path

        # Model-specific monitoring (if model and data are provided)
        if model is not None and X_test is not None:
            # Check for feature importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_

                # Handle feature names
                if isinstance(X_test, pd.DataFrame):
                    feature_names = X_test.columns
                    indices = np.argsort(importances)[::-1]
                    top_features = [
                        (feature_names[i], importances[i]) for i in indices[:20]
                    ]
                else:
                    indices = np.argsort(importances)[::-1]
                    top_features = [
                        (f"feature_{i}", importances[i]) for i in indices[:20]
                    ]

                monitoring_results["feature_importance"] = top_features

                # Plot feature importance if path provided
                if metrics_path:
                    try:
                        plt.figure(figsize=(12, 8))
                        if isinstance(X_test, pd.DataFrame):
                            plt.bar(
                                range(len(indices[:20])),
                                [importances[i] for i in indices[:20]],
                                align="center",
                            )
                            plt.xticks(
                                range(len(indices[:20])),
                                [feature_names[i] for i in indices[:20]],
                                rotation=90,
                            )
                        else:
                            plt.bar(
                                range(20), importances[indices[:20]], align="center"
                            )
                            plt.xticks(
                                range(20),
                                [f"Feature {i}" for i in indices[:20]],
                                rotation=90,
                            )

                        plt.xlabel("Features")
                        plt.ylabel("Importance")
                        plt.title("Feature Importance")
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(metrics_path, "feature_importance.png"),
                            dpi=150,
                            bbox_inches="tight",
                        )
                        plt.close()
                        logger.info("Feature importance plot saved")
                    except Exception as e:
                        logger.warning(f"Could not save feature importance plot: {e}")

            # Model performance evaluation
            if y_test is not None:
                try:
                    predictions = model.predict(X_test)

                    # Calculate basic metrics
                    from sklearn.metrics import accuracy_score, classification_report

                    accuracy = accuracy_score(y_test, predictions)

                    monitoring_results["current_accuracy"] = accuracy
                    monitoring_results["classification_report"] = classification_report(
                        y_test, predictions, output_dict=True
                    )

                    logger.info(f"Current model accuracy: {accuracy:.4f}")

                except Exception as e:
                    logger.warning(f"Could not evaluate model performance: {e}")

        # Data drift detection (if reference data is provided)
        if X_reference is not None and X_test is not None:
            logger.info("Performing data drift detection")

            try:
                if isinstance(X_test, pd.DataFrame) and isinstance(
                    X_reference, pd.DataFrame
                ):
                    # Get common columns
                    common_columns = list(
                        set(X_test.columns) & set(X_reference.columns)
                    )
                    logger.info(
                        f"Analyzing drift for {len(common_columns)} common features"
                    )

                    drift_metrics = {}
                    for col in common_columns:
                        try:
                            # Handle different data types
                            if X_reference[col].dtype in ["object", "category"]:
                                # For categorical data, use different approach
                                ref_dist = X_reference[col].value_counts(normalize=True)
                                test_dist = X_test[col].value_counts(normalize=True)

                                # Simple distribution comparison
                                common_cats = set(ref_dist.index) & set(test_dist.index)
                                if len(common_cats) > 0:
                                    drift_score = sum(
                                        abs(
                                            ref_dist.get(cat, 0) - test_dist.get(cat, 0)
                                        )
                                        for cat in common_cats
                                    )
                                    p_value = (
                                        0.1 if drift_score > 0.2 else 0.9
                                    )  # Simplified
                                else:
                                    drift_score = 1.0
                                    p_value = 0.001

                                drift_metrics[col] = {
                                    "ks_stat": drift_score,
                                    "p_value": p_value,
                                }
                            else:
                                # For numerical data, use KS test
                                ks_stat, p_value = ks_2samp(
                                    X_reference[col].dropna(), X_test[col].dropna()
                                )
                                drift_metrics[col] = {
                                    "ks_stat": ks_stat,
                                    "p_value": p_value,
                                }

                            # Consider drift if p-value is small
                            if p_value < 0.05:
                                monitoring_results["drifted_features"].append(col)

                        except Exception as e:
                            logger.warning(
                                f"Could not calculate drift for column {col}: {e}"
                            )

                    monitoring_results["num_drifted_features"] = len(
                        monitoring_results["drifted_features"]
                    )
                    monitoring_results["drift_metrics"] = drift_metrics

                    logger.info(
                        f"Detected drift in {monitoring_results['num_drifted_features']} features"
                    )

                    # Create drift report if metrics path provided and drift detected
                    if metrics_path and monitoring_results["drifted_features"]:
                        try:
                            plt.figure(figsize=(12, 6))
                            drift_scores = [
                                drift_metrics[col]["ks_stat"]
                                for col in monitoring_results["drifted_features"]
                            ]
                            plt.bar(
                                monitoring_results["drifted_features"], drift_scores
                            )
                            plt.xlabel("Features")
                            plt.ylabel("Drift Score")
                            plt.title("Data Drift Detection")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            plt.savefig(
                                os.path.join(metrics_path, "data_drift.png"),
                                dpi=150,
                                bbox_inches="tight",
                            )
                            plt.close()

                            # Save drift metrics as CSV
                            pd.DataFrame(drift_metrics).T.to_csv(
                                os.path.join(metrics_path, "drift_metrics.csv")
                            )
                            logger.info("Data drift report saved")

                        except Exception as e:
                            logger.warning(f"Could not save drift report: {e}")

            except Exception as e:
                logger.warning(f"Data drift detection failed: {e}")
                monitoring_results["drift_error"] = str(e)

        # Generate summary report
        if metrics_path:
            try:
                summary_path = os.path.join(metrics_path, "monitoring_summary.json")
                import json

                with open(summary_path, "w") as f:
                    json.dump(monitoring_results, f, indent=2, default=str)
                logger.info(f"Monitoring summary saved to {summary_path}")
            except Exception as e:
                logger.warning(f"Could not save monitoring summary: {e}")

        logger.info("Model performance monitoring complete")
        return monitoring_results

    except Exception as e:
        logger.error(f"Error in model performance monitoring: {e}")
        monitoring_results["error"] = str(e)
        return monitoring_results


class ModelMonitor:
    """Model monitoring class for tracking performance over time."""

    def __init__(self, model_name, storage_path):
        self.model_name = model_name
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        logger.info(f"Initialized ModelMonitor for {model_name}")

    def record_metrics(self, metrics):
        """Record model metrics."""
        try:
            metrics_with_timestamp = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "model_name": self.model_name,
                **metrics,
            }

            # Save to JSON file
            metrics_file = os.path.join(
                self.storage_path, f"{self.model_name}_metrics.json"
            )

            # Load existing metrics if file exists
            existing_metrics = []
            if os.path.exists(metrics_file):
                try:
                    import json

                    with open(metrics_file, "r") as f:
                        existing_metrics = json.load(f)
                except:
                    existing_metrics = []

            # Append new metrics
            existing_metrics.append(metrics_with_timestamp)

            # Save updated metrics
            import json

            with open(metrics_file, "w") as f:
                json.dump(existing_metrics, f, indent=2)

            logger.info(f"Recorded metrics for {self.model_name}")

        except Exception as e:
            logger.error(f"Error recording metrics for {self.model_name}: {e}")

    def generate_monitoring_report(self):
        """Generate a monitoring report."""
        try:
            report_path = os.path.join(
                self.storage_path, f"{self.model_name}_monitoring_report.html"
            )

            # Simple HTML report
            html_content = f"""
            <html>
            <head><title>Model Monitoring Report - {self.model_name}</title></head>
            <body>
                <h1>Model Monitoring Report</h1>
                <h2>Model: {self.model_name}</h2>
                <p>Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Monitoring data stored in: {self.storage_path}</p>
            </body>
            </html>
            """

            with open(report_path, "w") as f:
                f.write(html_content)

            logger.info(
                f"Generated monitoring report for {self.model_name} at {report_path}"
            )
            return report_path

        except Exception as e:
            logger.error(
                f"Error generating monitoring report for {self.model_name}: {e}"
            )
            return None
