import re
from pathlib import Path
import numpy as np
import pandas as pd

from .config import BASE_FEATURES, RANDOM_STATE

TARGET_ALIASES = {
    "loan_status": "loan_status",
    "loan_statuses": "loan_status",
    "status": "loan_status",
}

ALIASES = {
    "loan_amnt": ["loan_amnt", "loan_amount"],
    "term": ["term", "loan_term"],
    "int_rate": ["int_rate", "interest_rate"],
    "installment": ["installment"],
    "annual_inc": ["annual_inc", "annual_income"],
    "dti": ["dti", "debt_to_income"],
    "home_ownership": ["home_ownership", "home_ownership_status"],
    "verification_status": ["verification_status", "income_verification"],
    "purpose": ["purpose", "loan_purpose"],
    "open_acc": ["open_acc", "open_accounts"],
    "pub_rec": ["pub_rec", "public_records"],
    "revol_bal": ["revol_bal", "revolving_balance"],
    "revol_util": ["revol_util", "revolving_utilization"],
    "total_acc": ["total_acc", "total_accounts"],
    "mort_acc": ["mort_acc", "mortgage_accounts"],
    "delinq_2yrs": ["delinq_2yrs", "delinquencies_2yrs"],
    "inq_last_6mths": ["inq_last_6mths", "inquiries_6m"],
    "credit_history_years": ["credit_history_years"],
}

def _norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def normalize_columns(df):
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    rename = {}
    for canonical, choices in ALIASES.items():
        for choice in choices:
            if _norm(choice) in df.columns:
                rename[_norm(choice)] = canonical
                break
    return df.rename(columns=rename)

def clean_target(df):
    df = df.copy()
    if "loan_status" not in df.columns:
        raise ValueError("No loan_status column found.")
    s = df["loan_status"].astype(str).str.strip().str.lower()
    positive = {
        "charged off", "default", "late (31-120 days)", "late (16-30 days)",
        "does not meet the credit policy. status:charged off"
    }
    negative = {
        "fully paid", "does not meet the credit policy. status:fully paid"
    }
    df["default"] = np.where(s.isin(positive), 1,
                             np.where(s.isin(negative), 0, np.nan))
    df = df.dropna(subset=["default"])
    df["default"] = df["default"].astype(int)
    return df

def prepare_frame(df):
    df = normalize_columns(df)
    if "loan_status" in df.columns:
        df = clean_target(df)
    for c in BASE_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    return df

def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    return prepare_frame(df)
