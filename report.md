# Mushroom Classification Project Report

## 1. Problem Statement

The objective of this project is to build a machine learning model that can accurately classify mushrooms as either edible or poisonous based on their physical characteristics. The dataset used contains various features describing the appearance and properties of mushrooms.

## 2. Dataset Description

- **Source:** UCI Machine Learning Repository - Mushroom Dataset
- **Features:** 22 categorical features such as cap-shape, cap-color, odor, gill-size, etc.
- **Target:** `class` (edible = e, poisonous = p)
- **Size:** 8124 instances

## 3. Data Preprocessing

- **Missing Values:** The dataset contains missing values in the 'stalk-root' feature, represented by '?'.
  - These were handled by replacing '?' with the mode value of the column.
- **Encoding:** All categorical features were label encoded to convert them into numerical format suitable for machine learning algorithms.
- **Splitting:** The dataset was split into training and testing sets (typically 80% train, 20% test).

## 4. Model Selection

- **Algorithms Tried:** Decision Tree, Random Forest, Logistic Regression
- **Final Model:** Random Forest Classifier was selected due to its superior performance and robustness to overfitting.

## 5. Model Training

- The Random Forest Classifier was trained on the preprocessed training data.
- Hyperparameters such as the number of estimators and maximum depth were tuned using grid search and cross-validation.

## 6. Evaluation Metrics

- **Accuracy:** Proportion of correctly classified instances.
- **Precision, Recall, F1-Score:** Evaluated to ensure balanced performance, especially since misclassifying a poisonous mushroom as edible is critical.
- **Confusion Matrix:** Used to visualize true positives, false positives, true negatives, and false negatives.

## 7. Results

- **Accuracy on Test Set:** ~99%
- **Precision (Poisonous):** High (close to 1.0)
- **Recall (Poisonous):** High (close to 1.0)
- **F1-Score:** High (close to 1.0)
- **Confusion Matrix:** Very few misclassifications, if any.

## 8. Model Interpretation

- **Feature Importance:** The most important features for classification were 'odor', 'spore-print-color', and 'gill-size'.
- **Interpretability:** Random Forest provides feature importance scores, helping to understand which features contribute most to the prediction.

## 9. Deployment

- The trained model can be saved using joblib or pickle for future inference.
- A simple API or web interface can be built to allow users to input mushroom features and receive predictions.

## 10. Conclusion

The project successfully demonstrates the use of machine learning for mushroom classification. The Random Forest model achieves high accuracy and reliability, making it suitable for practical use. Proper preprocessing and careful model selection were key to achieving these results.

## 11. References

- UCI Machine Learning Repository: Mushroom Dataset
- Scikit-learn Documentation

## 3.5.2 Experiment Tracking with MLflow

MLflow serves as the central experiment tracking system for this mushroom classification project, providing comprehensive monitoring and management of all machine learning experiments. This section details how MLflow is implemented to ensure reproducibility and systematic tracking of model development.

### 3.5.2.1 MLflow Setup and Configuration

The MLflow tracking server is configured to store experiment data locally with the following setup:

- **Tracking URI**: Local file system or remote MLflow server
- **Experiment Name**: "mushroom-classification-experiments"
- **Run Naming Convention**: timestamp-model_type-feature_set (e.g., "20241201_randomforest_all_features")

_[Placeholder: Figure 3.5.2.1 - MLflow UI Dashboard showing experiment overview]_

### 3.5.2.2 Parameter Logging

All hyperparameters for each model training run are systematically logged to ensure complete reproducibility. The current implementation logs the following parameters:

XGBoost Model Parameters

- `model_n_estimators`: Number of boosting rounds (100)
- `model_max_depth`: Maximum tree depth (6)
- `model_learning_rate`: Step size shrinkage (0.1)
- `model_random_state`: Random seed for reproducibility (42)
- `model_use_label_encoder`: Whether to use label encoder (False)
- `model_eval_metric`: Evaluation metric ('logloss')

XGBoost Internal Parameters

- `xgboost_objective`: Loss function to be optimized
- `xgboost_booster`: Type of booster (gbtree, gblinear, dart)
- `xgboost_tree_method`: Tree construction algorithm
- `xgboost_subsample`: Subsample ratio of training instances
- `xgboost_colsample_bytree`: Subsample ratio of columns when constructing each tree

**Data and Preprocessing Parameters:**

- `pipeline_experiment_id`: Unique identifier for the training experiment
- `data_training_samples`: Number of samples in training set
- `data_test_samples`: Number of samples in test set
- `data_feature_count`: Number of features used for training
- `preprocessing_categorical_features`: Number of categorical features
- `preprocessing_numerical_features`: Number of numerical features

**Pipeline Configuration Parameters:**

- `training_approach`: Training strategy ("streamlined_single_model")
- `ab_testing_required`: Whether A/B testing is needed (False)
- `deployment_strategy`: Deployment approach ("direct_production_deployment")

_[Placeholder: Figure 3.5.2.2 - MLflow Parameters tab showing properly logged XGBoost hyperparameters]_

**Troubleshooting Parameter Logging:**

If you're seeing test parameters like `test_param_number: 42` and `test_param_string: hello_world`, this indicates:

1. **Test Run Active**: You may be viewing a test MLflow run rather than the actual training run
2. **Wrong Experiment**: Check that you're viewing the correct experiment ("mushroom_classification_xgboost" or "mushroom_classification_pipeline")
3. **Run Selection**: Ensure you've selected the most recent training run, not a test or debug run

**Expected Parameter Structure in MLflow UI:**

```
model_n_estimators: 100
model_max_depth: 6
model_learning_rate: 0.1
model_random_state: 42
model_use_label_encoder: False
model_eval_metric: logloss
xgboost_objective: binary:logistic
xgboost_booster: gbtree
xgboost_tree_method: auto
xgboost_subsample: 0.8
xgboost_colsample_bytree: 0.8
pipeline_experiment_id: 123456
data_training_samples: 6499
data_test_samples: 1625
data_feature_count: 22
preprocessing_categorical_features: 22
preprocessing_numerical_features: 0
training_approach: streamlined_single_model
ab_testing_required: False
deployment_strategy: direct_production_deployment
```

### 3.5.2.3 Metrics Logging

Key performance metrics are logged for both training and validation sets to monitor model performance and detect overfitting:

**Classification Metrics:**

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for both edible and poisonous classes
- **Recall**: Recall for both edible and poisonous classes
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **Confusion Matrix**: True/False positives and negatives

**Training Metrics:**

- `train_accuracy`: Training set accuracy
- `train_precision_edible`: Training precision for edible class
- `train_precision_poisonous`: Training precision for poisonous class
- `train_recall_edible`: Training recall for edible class
- `train_recall_poisonous`: Training recall for poisonous class

**Validation Metrics:**

- `val_accuracy`: Validation set accuracy
- `val_precision_edible`: Validation precision for edible class
- `val_precision_poisonous`: Validation precision for poisonous class
- `val_recall_edible`: Validation recall for edible class
- `val_recall_poisonous`: Validation recall for poisonous class

_[Placeholder: Figure 3.5.2.3 - MLflow Metrics comparison chart showing performance across different runs]_

### 3.5.2.4 Artifact Logging

MLflow artifacts ensure that all important outputs from each experiment are preserved:

**Model Artifacts:**

- **Trained Models**: Serialized model objects (pickle/joblib format)
- **Preprocessing Pipeline**: Label encoders and feature transformers
- **Model Metadata**: Model signature and input/output schema

**Evaluation Artifacts:**

- **Confusion Matrix Plots**: Visual representation of classification results
- **ROC Curves**: ROC curve plots for model comparison
- **Feature Importance Plots**: Visualization of feature contributions
- **Classification Reports**: Detailed performance metrics in text format

**Data Artifacts:**

- **Feature Engineering Code**: Scripts used for data preprocessing
- **Dataset Snapshots**: Processed datasets used for training
- **Feature Descriptions**: Documentation of feature transformations

_[Placeholder: Figure 3.5.2.4 - MLflow Artifacts tab showing logged files and models]_

### 3.5.2.5 Code Versioning and Reproducibility

MLflow integrates with Git to ensure complete experiment reproducibility:

**Git Integration:**

- **Commit Hash**: Automatic logging of Git commit SHA
- **Repository URL**: Remote repository location
- **Branch Name**: Active branch during experiment
- **Dirty Flag**: Indication of uncommitted changes

**Environment Tracking:**

- **Python Version**: Python interpreter version
- **Package Dependencies**: Requirements.txt or conda environment
- **Hardware Specifications**: CPU, memory, and GPU information

_[Placeholder: Figure 3.5.2.5 - MLflow Run details showing Git commit information]_

### 3.5.2.6 Model Registry and Lifecycle Management

The MLflow Model Registry provides centralized model lifecycle management:

**Model Registration Process:**

1. **Staging**: New models are registered in "Staging" stage
2. **Validation**: Models undergo validation testing
3. **Production**: Approved models transition to "Production" stage
4. **Archived**: Outdated models are moved to "Archived" stage

**Model Versioning:**

- **Version Numbers**: Automatic incremental versioning (v1, v2, v3...)
- **Model Descriptions**: Detailed descriptions of model improvements
- **Performance Comparisons**: Side-by-side metrics comparison
- **Transition Logs**: Complete audit trail of stage transitions

**Model Metadata:**

- **Model Signature**: Input/output schema validation
- **Performance Benchmarks**: Standardized evaluation metrics
- **Deployment Information**: Serving endpoints and configurations

_[Placeholder: Figure 3.5.2.6 - MLflow Model Registry showing model versions and stages]_

### 3.5.2.7 Experiment Comparison and Analysis

MLflow facilitates comprehensive experiment analysis through:

**Run Comparison:**

- **Parallel Coordinates Plot**: Multi-dimensional parameter comparison
- **Scatter Plots**: Parameter vs. metric correlations
- **Table View**: Side-by-side run comparisons
- **Difference Highlighting**: Changes between experiment runs

**Best Model Selection:**

- **Automated Ranking**: Models ranked by specified metrics
- **Pareto Optimization**: Multi-objective optimization visualization
- **Statistical Significance**: Performance difference testing

_[Placeholder: Figure 3.5.2.7 - MLflow experiment comparison dashboard]_

### 3.5.2.8 Integration with Training Pipeline

MLflow is seamlessly integrated into the training pipeline through Python code:

```python
# Example MLflow logging implementation
import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="randomforest_experiment_1"):
    # Log parameters
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })

    # Train model and log metrics
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "precision_poisonous": precision_score
    })

    # Log model and artifacts
    mlflow.sklearn.log_model(model, "random_forest_model")
    mlflow.log_artifact("confusion_matrix.png")
```

This comprehensive MLflow implementation ensures that every aspect of the model development process is tracked, versioned, and reproducible, providing a solid foundation for model governance and continuous improvement.
