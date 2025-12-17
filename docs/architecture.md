# Mushroom Classification System Architecture

## System Components

```
+-------------------+    +-----------------------+    +------------------+
|                   |
|   Data Sources    +--->+   ETL Pipeline        +--->+  Model Training  |
|                   |    |                       |    |                  |
+-------------------+    +-----------------------+    +------------------+
                                                              |
                                                              v
+-------------------+    +-----------------------+    +------------------+
|                   |    |                       |    |                  |
|   API Consumers   +<-->+   API Server          +<-->+  Model Registry  |
|                   |    |                       |    |                  |
+-------------------+    +-----------------------+    +------------------+
       ^                           |                          |
       |                           v                          |
       |                 +-----------------------+            |
       |                 |                       |            |
       +---------------->+   A/B Testing         |<-----------+
                         |                       |
                         +-----------------------+
                                    |
                                    v
                         +-----------------------+
                         |                       |
                         |   Model Monitoring    |
                         |                       |
                         +-----------------------+
```

## Data Flow

1. **Data Collection**: Raw mushroom data is collected from CSV sources.

2. **ETL Pipeline**: The data goes through extraction, transformation, and loading:

   - **Extract**: Read and parse the raw data
   - **Transform**: Clean, normalize, and encode data for modeling
   - **Load**: Split data into train/test sets and save for model training

3. **Model Training**: Multiple models are trained on the prepared data:

   - Logistic Regression
   - Decision Tree
   - XGBoost

4. **Model Registry**: Trained models are registered, versioned, and deployed:

   - Models are saved with metadata and performance metrics
   - Models are promoted from staging to production
   - Multiple versions are managed with full history

5. **API Server**: The FastAPI server serves predictions:

   - Provides endpoints for mushroom classification
   - Integrates with A/B testing for model comparison
   - Logs predictions to the database for future analysis

6. **A/B Testing**: Tests different model versions in production:

   - Routes traffic between model variants
   - Collects performance metrics
   - Performs statistical significance testing
   - Helps select the best-performing model

7. **Model Monitoring**: Tracks model performance:
   - Detects data drift
   - Monitors prediction performance
   - Generates reports and visualizations
   - Triggers alerts on performance degradation

## Technology Stack

- **Python**: Core programming language
- **Pandas/NumPy**: Data processing
- **Scikit-learn/XGBoost**: Machine learning
- **FastAPI**: API framework
- **Flask**: (Optional) Alternative API framework
- **Airflow**: Workflow orchestration
- **MariaDB**: Data storage
- **Docker**: Containerization
- **Pytest**: Automated testing

## Development Workflow

```
  Development          CI/CD Pipeline            Deployment
+------------+      +----------------+       +----------------+
|            |      |                |       |                |
| Code       +----->+ Test           +------>+ Deploy Model   |
|            |      |                |       |                |
+------------+      +----------------+       +----------------+
      ^                    |                        |
      |                    v                        v
+------------+      +----------------+       +----------------+
|            |      |                |       |                |
| Feedback   +<-----+ Model Quality  |<------+ Monitor        |
|            |      | Evaluation     |       |                |
+------------+      +----------------+       +----------------+
```
Data Ingestion Workflow - Mushroom Classification System
========================================================

┌─────────────────┐    ┌────────────────────┐    ┌─────────────────────┐
│                 │    │                    │    │                     │
│  Data Sources   │───▶│  Step 1: Fetch    │───▶│  Step 2: Initial    │
│                 │    │  Data              │    │  Validation         │
│ • secondary_    │    │                    │    │                     │
│   data.csv      │    │ • File Reader      │    │ • Great Expectations│
│ • fraction_     │    │ • Error Handling   │    │ • Schema Check      │
│   dataset.csv   │    │ • Pandas CSV       │    │ • Format Check      │
│                 │    │   Parser           │    │ • Data Type Check   │
└─────────────────┘    └────────────────────┘    └─────────────────────┘
                                │                            │
                                │                            ▼
                                │                 ┌─────────────────────┐
                                │                 │                     │
                                │                 │  Validation Failed? │
                                │                 │                     │
                                │                 │ ┌─────────────────┐ │
                                │                 │ │ Alert & Stop    │ │
                                │                 │ │ Pipeline        │ │
                                │                 │ └─────────────────┘ │
                                │                 └─────────────────────┘
                                │                            │
                                │                            ▼ (Pass)
                                │                 ┌─────────────────────┐
                                │                 │                     │
                                │                 │  Step 3: Load to    │
                                │                 │  Staging Table      │
                                │                 │                     │
                                │                 │ • PostgreSQL        │
                                │                 │ • Staging Schema    │
                                │                 │ • Raw Data Storage  │
                                │                 └─────────────────────┘
                                │                            │
                                │                            ▼
                                │                 ┌─────────────────────┐
                                │                 │                     │
                                │                 │  Step 4: Detailed   │
                                │                 │  Data Validation    │
                                │                 │                     │
                                │                 │ • Content Checks    │
                                │                 │ • Range Validation  │
                                │                 │ • Set Membership    │
                                │                 │ • Custom Rules      │
                                │                 └─────────────────────┘
                                │                            │
                                │                            ▼
                                │                 ┌─────────────────────┐
                                │                 │                     │
                                │                 │  Validation Failed? │
                                │                 │                     │
                                │                 │ ┌─────────────────┐ │
                                │                 │ │ Generate Alerts │ │
                                │                 │ │ & Reports       │ │
                                │                 │ └─────────────────┘ │
                                │                 └─────────────────────┘
                                │                            │
                                │                            ▼ (Pass)
                                │                 ┌─────────────────────┐
                                │                 │                     │
                                │                 │  Step 5: Promote    │
                                │                 │  to Production      │
                                │                 │                     │
                                │                 │ • MariaDB/          │
                                │                 │   ColumnStore       │
                                │                 │ • mushroom_data     │
                                │                 │   table             │
                                │                 └─────────────────────┘
                                │                            │
                                ▼                            ▼
                    ┌─────────────────────┐    ┌─────────────────────┐
                    │                     │    │                     │
                    │  Airflow DAG        │    │  MLflow Tracking    │
                    │  Orchestration      │    │                     │
                    │                     │    │ • Data Lineage      │
                    │ • Task Dependencies │    │ • Validation Results│
                    │ • Retry Logic       │    │ • Pipeline Metrics  │
                    │ • Scheduling        │    │ • Artifact Storage  │
                    │ • Monitoring        │    │                     │
                    └─────────────────────┘    └─────────────────────┘