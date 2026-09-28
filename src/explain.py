import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

from .config import MODELS, BASE_FEATURES, BEST_MODEL_FILE, MODEL_FILE
from .preprocess import clean_numeric_values

def evaluate_credit_capacity(data: dict, raw_prob: float) -> float:
    """
    Applies banking debt-capacity guardrails (Debt Service Ratio & Loan-to-Income limits).
    Prevents absurd unserviceable loans (e.g. $80 Billion loan on $95k salary) from bypassing risk thresholds.
    """
    try:
        loan_amnt = float(data.get("loan_amnt", 0.0))
        annual_inc = float(data.get("annual_inc", 0.0))
        installment = float(data.get("installment", 0.0))
        dti = float(data.get("dti", 0.0))
    except Exception:
        return raw_prob

    monthly_inc = max(annual_inc / 12.0, 1.0)
    loan_to_inc = loan_amnt / max(annual_inc, 1.0)
    debt_service_ratio = installment / monthly_inc

    adjusted_prob = raw_prob

    # Unserviceable Debt Capacity Penalties
    if debt_service_ratio > 0.45 or loan_to_inc > 4.5 or dti > 42.0:
        severity = max(debt_service_ratio / 0.45, loan_to_inc / 4.5, dti / 42.0)
        penalty = 0.40 + 0.55 * (1.0 - 1.0 / (1.0 + np.log1p(max(0.0, severity - 1.0))))
        adjusted_prob = float(np.clip(max(raw_prob, penalty), 0.0, 0.9999))

    return adjusted_prob

def get_risk_grade(default_prob: float) -> str:
    if default_prob < 0.05:
        return "A+"
    elif default_prob < 0.10:
        return "A"
    elif default_prob < 0.20:
        return "B"
    elif default_prob < 0.35:
        return "C"
    elif default_prob < 0.50:
        return "D"
    elif default_prob < 0.70:
        return "E"
    else:
        return "F"

def get_financial_decision(data: dict, default_prob: float = None, threshold: float = 0.35) -> str:
    if isinstance(data, (int, float)):
        # Backward compatibility if called as get_financial_decision(prob, threshold)
        if default_prob is not None and isinstance(default_prob, (int, float)):
            threshold = float(default_prob)
        default_prob = float(data)
        data = {}

    if default_prob is None:
        default_prob = 0.0

    loan_amnt = float(data.get("loan_amnt", 0.0))
    annual_inc = float(data.get("annual_inc", 0.0))
    installment = float(data.get("installment", 0.0))
    dti = float(data.get("dti", 0.0))

    monthly_inc = max(annual_inc / 12.0, 1.0)
    loan_to_inc = loan_amnt / max(annual_inc, 1.0) if annual_inc > 0 else (999.0 if loan_amnt > 0 else 0.0)
    debt_service_ratio = installment / monthly_inc if monthly_inc > 0 else (999.0 if installment > 0 else 0.0)

    # Automatic Hard Rejection Criteria (Banking Compliance Guardrails)
    if debt_service_ratio > 0.50 or loan_to_inc > 5.0 or dti > 45.0 or default_prob >= threshold:
        return "REJECT"
    elif default_prob >= (threshold * 0.6) or debt_service_ratio > 0.35 or loan_to_inc > 3.5:
        return "MANUAL_REVIEW"
    else:
        return "APPROVE"

def calculate_expected_loss(loan_amnt: float, default_prob: float, lgd: float = 0.6) -> float:
    """
    Expected Loss (EL) = Loan Amount * Probability of Default (PD) * Loss Given Default (LGD)
    Default LGD = 60% (Standard retail unsecured credit benchmark)
    """
    return round(float(loan_amnt * default_prob * lgd), 2)

def explain_row(row: dict, top_n: int = 8):
    """
    Computes local SHAP feature importance for an applicant.
    """
    model_path = BEST_MODEL_FILE if BEST_MODEL_FILE.exists() else MODEL_FILE
    if not model_path.exists():
        return []

    pipe = joblib.load(model_path)
    X = clean_numeric_values(pd.DataFrame([row], columns=BASE_FEATURES))
    
    pre = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]

    Xt = pre.transform(X)
    
    # Get feature names out of column_transformer
    col_trans = pre.named_steps["column_transformer"]
    try:
        feature_names = col_trans.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(Xt.shape[1])]

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(Xt)
        if isinstance(values, list):
            values = values[-1]
        vals = values[0]
    except Exception:
        # Fallback for Stacking or Linear Models
        try:
            explainer = shap.Explainer(model, Xt)
            values = explainer(Xt).values
            vals = values[0]
        except Exception:
            return []

    order = sorted(range(len(vals)), key=lambda i: abs(float(vals[i])), reverse=True)[:top_n]
    
    explanations = []
    for i in order:
        impact_val = float(vals[i])
        fname = str(feature_names[i])
        
        # Clean up feature name display
        clean_name = fname.replace("num__", "").replace("cat__", "").replace("domain_features__", "")
        
        explanations.append({
            "feature": clean_name,
            "raw_feature": fname,
            "impact": round(impact_val, 5),
            "direction": "Increases Risk" if impact_val > 0 else "Decreases Risk"
        })

    return explanations
