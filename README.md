# Enterprise Loan Default Risk Intelligence & Prediction Platform

Production-grade Machine Learning platform for predicting LendingClub loan defaults, quantifying financial expected loss ($), calculating SHAP explainability attributions, and monitoring statistical model drift (PSI & KS tests).

## 🚀 Key Features

- **Leakage-Free Feature Engineering (`src/features.py`)**:
  - `installment_to_inc_ratio`, `dti_installment_ratio`, `revol_bal_to_inc`, `credit_limit_est`, `open_acc_ratio`, `delinq_per_year`, `inq_per_year`, `utilization_x_dti`, `credit_risk_flag`.
- **Advanced Multi-Model Benchmark (`src/train.py`)**:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM (`LGBMClassifier`)
  - **Stacking Meta-Ensemble** (`StackingClassifier` with cross-validated Logistic Regression meta-learner).
- **Threshold Calibration**:
  - Optimizes classification decision boundary $\tau^*$ to maximize F1-score & Youden's J statistic.
- **Financial Risk & Loss Quantification (`src/explain.py`)**:
  - Probability of Default (PD) & Risk Grades (`A+` through `F`).
  - Financial Expected Loss calculation ($EL = \text{Loan Amount} \times PD \times LGD$).
  - Actionable Decision Recommendation (`APPROVE`, `MANUAL_REVIEW`, `REJECT`).
- **SHAP Explainability**:
  - Tree & Linear SHAP local feature attributions.
- **Model Drift & Monitoring (`src/monitoring.py`)**:
  - Population Stability Index (PSI) calculation.
  - Two-sample Kolmogorov-Smirnov (KS) distribution drift testing.
- **Enterprise REST API (`api/main.py`)**:
  - `POST /predict`: Single applicant risk assessment with SHAP top factors & expected loss.
  - `POST /predict/batch`: Bulk JSON batch processing with portfolio summary statistics.
  - `POST /predict/csv`: Upload raw CSV file for batch inference.
  - `POST /explain`: Full SHAP feature attribution waterfall values.
  - `POST /monitoring/drift`: Population Stability Index (PSI) & KS test feature drift analysis.
- **Publication-Ready Visualizations (`reports/`)**:
  - Comparative ROC curves (`roc_curves_all_models.png`)
  - Comparative Precision-Recall curves (`pr_curves_all_models.png`)
  - Probability Calibration curves (`calibration_curves.png`)
  - Confusion Matrix plot (`confusion_matrix_best_model.png`)
  - Model Comparison CSV table (`model_comparison.csv`)

---

## 🛠️ Project Structure

```text
loan_default_prediction/
├── api/
│   ├── main.py                     # FastAPI server with single, batch, CSV upload, SHAP, and drift endpoints
│   └── sample_request.json
├── data/
│   └── raw/
│       └── loan_data_demo.csv      # 12,000 synthetic LendingClub records
├── frontend/                       # React + Vite Interactive Dashboard
├── models/
│   ├── best_model_pipeline.joblib  # Production Stacking Ensemble Pipeline
│   ├── stacking_ensemble_pipeline.joblib
│   ├── xgboost_pipeline.joblib
│   ├── lightgbm_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   ├── logistic_regression_pipeline.joblib
│   └── metadata.json
├── reports/
│   ├── roc_curves_all_models.png
│   ├── pr_curves_all_models.png
│   ├── calibration_curves.png
│   ├── confusion_matrix_best_model.png
│   ├── model_comparison.csv
│   └── training_log.txt
├── scripts/
│   └── generate_demo_data.py
├── src/
│   ├── config.py                   # Paths, features, targets
│   ├── data.py                     # Data loader, canonical column aliases, target cleaning
│   ├── features.py                 # Scikit-learn custom DomainFeatureEngineer transformer
│   ├── preprocess.py               # Imputation, scaling, one-hot encoding pipeline
│   ├── train.py                    # Multi-model training, stacking, threshold calibration & plots
│   ├── explain.py                  # Expected loss ($), risk grading, SHAP attributions
│   └── monitoring.py               # PSI & KS statistical drift monitoring
├── tests/
│   ├── test_api.py                 # FastAPI endpoints integration test suite
│   ├── test_data.py                # Data loading unit tests
│   └── test_features.py            # Feature engineering transformer unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start Guide

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Synthetic LendingClub Demo Data
```bash
python scripts/generate_demo_data.py
```

### 3. Run Full Pipeline Training & Benchmarking
```bash
python -m src.train
```

### 4. Run Pytest Verification
```bash
python -m pytest
```

### 5. Start API Server
```bash
uvicorn api.main:app --reload
```
Interactive OpenAPI documentation is available at:
- `http://127.0.0.1:8000/docs`
