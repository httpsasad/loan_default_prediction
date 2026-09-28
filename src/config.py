from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

TARGET = "default"
RANDOM_STATE = 42

BASE_FEATURES = [
    "loan_amnt", "term", "int_rate", "installment", "annual_inc", "dti",
    "home_ownership", "verification_status", "purpose", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
    "delinq_2yrs", "inq_last_6mths", "credit_history_years"
]

MODEL_FILE = MODELS / "xgboost_pipeline.joblib"
LGBM_FILE = MODELS / "lightgbm_pipeline.joblib"
RF_FILE = MODELS / "random_forest_pipeline.joblib"
LR_FILE = MODELS / "logistic_regression_pipeline.joblib"
ENSEMBLE_FILE = MODELS / "stacking_ensemble_pipeline.joblib"
BEST_MODEL_FILE = MODELS / "best_model_pipeline.joblib"
META_FILE = MODELS / "metadata.json"

