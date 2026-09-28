import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BASE_FEATURES
from .features import DomainFeatureEngineer, ENGINEERED_NUMERIC

NUMERIC = [
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mort_acc", "delinq_2yrs", "inq_last_6mths", "credit_history_years"
]
ALL_NUMERIC = NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL = ["term", "home_ownership", "verification_status", "purpose"]

def make_preprocessor():
    """
    Constructs a full preprocessor pipeline:
    1. Feature Engineering step (DomainFeatureEngineer)
    2. ColumnTransformer (Numeric imputer + scaler, Categorical imputer + OneHotEncoder)
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    transformer = ColumnTransformer([
        ("num", numeric_pipe, ALL_NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=False)

    return Pipeline([
        ("domain_features", DomainFeatureEngineer()),
        ("column_transformer", transformer)
    ])

def clean_numeric_values(X):
    X = X.copy()
    for c in NUMERIC:
        if c in X.columns:
            X[c] = pd.to_numeric(
                X[c].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
                errors="coerce"
            )
    X["term"] = X["term"].astype(str).replace({"nan": np.nan})
    return X[BASE_FEATURES]
