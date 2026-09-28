import pandas as pd
import pytest
from src.data import normalize_columns, clean_target, prepare_frame

def test_normalize_columns():
    df = pd.DataFrame({
        "Loan Amount": [10000],
        "Annual Income": [50000],
        "Debt To Income": [18.5]
    })
    df_norm = normalize_columns(df)
    assert "loan_amnt" in df_norm.columns
    assert "annual_inc" in df_norm.columns
    assert "dti" in df_norm.columns

def test_clean_target():
    df = pd.DataFrame({
        "loan_status": ["Fully Paid", "Charged Off", "Default", "Current"]
    })
    df_cleaned = clean_target(df)
    assert len(df_cleaned) == 3  # Current is excluded
    assert list(df_cleaned["default"]) == [0, 1, 1]
