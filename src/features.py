import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ENGINEERED_NUMERIC = [
    "installment_to_inc_ratio",
    "dti_installment_ratio",
    "revol_bal_to_inc",
    "credit_limit_est",
    "open_acc_ratio",
    "delinq_per_year",
    "inq_per_year",
    "utilization_x_dti",
    "log_annual_inc",
    "log_loan_amnt",
    "log_revol_bal",
    "composite_risk_index",
    "credit_risk_flag"
]

def _get_series(df: pd.DataFrame, col_name: str, default_val: float) -> pd.Series:
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce").fillna(default_val)
    return pd.Series(default_val, index=df.index)

class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to generate domain-specific financial features,
    non-linear interaction terms, and composite credit risk indices without data leakage.
    """
    def __init__(self, eps=1e-5):
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.copy()

        annual_inc = _get_series(X, "annual_inc", 60000.0)
        loan_amnt = _get_series(X, "loan_amnt", 15000.0)
        installment = _get_series(X, "installment", 350.0)
        int_rate = _get_series(X, "int_rate", 12.0)
        dti = _get_series(X, "dti", 16.0)
        revol_bal = _get_series(X, "revol_bal", 6000.0)
        revol_util = _get_series(X, "revol_util", 40.0)
        open_acc = _get_series(X, "open_acc", 9.0)
        total_acc = _get_series(X, "total_acc", 18.0)
        delinq_2yrs = _get_series(X, "delinq_2yrs", 0.0)
        inq_last_6mths = _get_series(X, "inq_last_6mths", 0.0)
        credit_history_years = _get_series(X, "credit_history_years", 9.0)

        monthly_inc = np.maximum(annual_inc / 12.0, self.eps)
        
        X["installment_to_inc_ratio"] = installment / monthly_inc
        X["dti_installment_ratio"] = dti * X["installment_to_inc_ratio"]
        X["revol_bal_to_inc"] = revol_bal / np.maximum(annual_inc, self.eps)
        X["credit_limit_est"] = revol_bal / np.maximum(revol_util / 100.0, self.eps)
        X["open_acc_ratio"] = open_acc / np.maximum(total_acc, self.eps)
        X["delinq_per_year"] = delinq_2yrs / np.maximum(credit_history_years, self.eps)
        X["inq_per_year"] = inq_last_6mths / np.maximum(credit_history_years, self.eps)
        X["utilization_x_dti"] = (revol_util / 100.0) * dti

        # Non-linear log transforms
        X["log_annual_inc"] = np.log1p(np.maximum(annual_inc, 0.0))
        X["log_loan_amnt"] = np.log1p(np.maximum(loan_amnt, 0.0))
        X["log_revol_bal"] = np.log1p(np.maximum(revol_bal, 0.0))

        # Composite Credit Risk Index
        composite = (
            (int_rate / 10.0) +
            (dti / 15.0) +
            (revol_util / 50.0) +
            (delinq_2yrs * 1.5) +
            (inq_last_6mths * 1.2) -
            (credit_history_years / 10.0)
        )
        X["composite_risk_index"] = composite
        X["credit_risk_flag"] = ((composite > 4.0) | (dti > 30.0) | (delinq_2yrs >= 2)).astype(float)

        return X

    def get_feature_names_out(self, input_features=None):
        return ENGINEERED_NUMERIC
