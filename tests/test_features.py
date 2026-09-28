import numpy as np
import pandas as pd
import pytest
from src.features import DomainFeatureEngineer, ENGINEERED_NUMERIC

def test_domain_feature_engineer_transform():
    df = pd.DataFrame({
        "annual_inc": [60000.0, 120000.0],
        "loan_amnt": [10000.0, 20000.0],
        "installment": [300.0, 600.0],
        "dti": [15.0, 25.0],
        "revol_bal": [5000.0, 15000.0],
        "revol_util": [40.0, 80.0],
        "open_acc": [8, 12],
        "total_acc": [16, 24],
        "delinq_2yrs": [0, 2],
        "inq_last_6mths": [1, 3],
        "credit_history_years": [10.0, 15.0]
    })

    transformer = DomainFeatureEngineer()
    df_trans = transformer.fit_transform(df)

    for col in ENGINEERED_NUMERIC:
        assert col in df_trans.columns
        assert not df_trans[col].isnull().any()

    # Verify installment to income calculation
    # Monthly income for row 0 = 60000 / 12 = 5000. Installment = 300. Ratio = 300 / 5000 = 0.06
    assert np.isclose(df_trans.loc[0, "installment_to_inc_ratio"], 0.06)
    
    # Credit risk flag for row 1 should be 1.0 (since revol_util > 75 and delinq_2yrs > 1)
    assert df_trans.loc[1, "credit_risk_flag"] == 1.0
