# 🍄 Mushroom Edibility Classification – MLOps Pipeline

This project implements a **full end-to-end MLOps pipeline** for **binary classification of mushrooms as edible or poisonous** using morphological features.  
It demonstrates the complete machine learning lifecycle with a strong focus on **automation, data quality, reproducibility, monitoring, and ethical responsibility** in a **safety-critical ML system**.

---

## 📁 Dataset
- **Source:** UCI Machine Learning Repository – *Secondary Mushroom Dataset*
- **Total Records:** 61,069
- **Features:** 20 (mixed data types)
  - **Numerical:** cap-diameter, stem-height, stem-width
  - **Categorical:** cap-shape, gill-color, habitat, season, ring-type, etc.
- **Target Variable:**
  - `e` → Edible
  - `p` → Poisonous

⚠️ High-stakes classification: misclassifying a poisonous mushroom as edible can result in serious health risks.

---

## 🧠 Machine Learning Model
- **XGBoost Classifier**
  - Handles mixed numerical and categorical data
  - Provides feature importance for interpretability
  - Optimized for high recall on the poisonous class

---

## 🏗️ MLOps Pipeline Architecture
**Pipeline Stages:**
1. Data Ingestion (CSV → Redis cache)
2. Data Validation (Great Expectations)
3. Data Preprocessing (imputation, encoding, outlier handling)
4. Analytical Storage (MariaDB ColumnStore – star schema)
5. Model Training & Experiment Tracking (MLflow)
6. Model Deployment (FastAPI + Docker)
7. Monitoring & Drift Detection (Evidently AI, Prometheus, Grafana)

---

## 📊 Evaluation Metrics
- **Primary:** Recall (Poisonous class)
- **Secondary:** Accuracy, F1-score, AUC-ROC

🎯 Targets:
- Poisonous recall ≥ 95%
- Overall accuracy ≥ 90%

---

## 🧪 Results
| Metric | Value |
|------|------|
| Poisonous Recall | 95% |
| Overall Accuracy | 90% |
| Data Completeness | 70% → 95% |

---

## 🧰 Tools & Technologies
- **Pipeline:** Airflow, Redis, Great Expectations, MariaDB ColumnStore
- **ML:** Python, pandas, scikit-learn, XGBoost, MLflow
- **Deployment:** FastAPI, Docker, Docker Compose
- **Monitoring:** Evidently AI, Prometheus, Grafana

---

## 📝 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Diyakharel1/mushroom-classification-mlops.git
cd mushroom-classification-mlops
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start MLOps stack
```bash
docker-compose up -d
```

### 4. Trigger pipeline
- Airflow UI: http://localhost:8080  
- Run DAG: `mushroom_etl_columnstore_pipeline`

### 5. Access API
- Swagger UI: http://localhost:8000/docs

---

## 📌 Key Highlights
- Safety-critical ML classification
- Robust handling of missing data
- Automated data validation
- High-recall optimized model
- Full experiment tracking
- Continuous drift monitoring
- Ethics-first design

---

## 🏗️ Project Structure
```
Mushroom-Classification-MLOps/
├── dags/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
├── models/
├── api/
├── monitoring/
├── docker/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## 🚀 Future Improvements
- Real-time streaming ingestion
- Automated feature engineering
- Explainable AI (SHAP/LIME)
- Canary deployments

---

## 👩‍💻 Contributor
**Diya Kharel**  
MSc Computing – Data Management & MLOps (BCU)

---

## 📬 Contact
For questions or suggestions, feel free to open an issue on GitHub.
# Mushroom-Classification-MLops-pipeline
