import io
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import BASE_FEATURES, BEST_MODEL_FILE, MODEL_FILE, MODELS, REPORTS
from src.explain import (
    calculate_expected_loss, evaluate_credit_capacity, explain_row, get_financial_decision, get_risk_grade
)
from src.monitoring import analyze_feature_drift
from src.preprocess import clean_numeric_values

app = FastAPI(
    title="Enterprise Loan Risk Intelligence & Default Prediction API",
    version="2.0.0",
    description="High-performance machine learning API for credit risk evaluation, expected loss quantification, SHAP explainability, and model drift monitoring."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
METADATA = None

class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., ge=100, description="Requested loan amount ($)")
    term: str = Field("36 months", description="Loan term (36 months or 60 months)")
    int_rate: float = Field(..., ge=0, description="Interest rate (%)")
    installment: float = Field(..., ge=0, description="Monthly installment payment ($)")
    annual_inc: float = Field(..., ge=0, description="Annual gross income ($)")
    dti: float = Field(..., ge=0, description="Debt-to-Income ratio (%)")
    home_ownership: str = Field("RENT", description="Home ownership status (RENT, MORTGAGE, OWN, OTHER)")
    verification_status: str = Field("Not Verified", description="Income verification status")
    purpose: str = Field("debt_consolidation", description="Loan purpose")
    open_acc: float = Field(8, ge=0, description="Number of open credit lines")
    pub_rec: float = Field(0, ge=0, description="Number of derogatory public records")
    revol_bal: float = Field(5000, ge=0, description="Total credit revolving balance ($)")
    revol_util: float = Field(40, ge=0, description="Revolving line utilization rate (%)")
    total_acc: float = Field(15, ge=0, description="Total number of credit lines")
    mort_acc: float = Field(0, ge=0, description="Number of mortgage accounts")
    delinq_2yrs: float = Field(0, ge=0, description="30+ days delinquencies in past 2 years")
    inq_last_6mths: float = Field(1, ge=0, description="Credit inquiries in last 6 months")
    credit_history_years: float = Field(8, ge=0, description="Length of credit history in years")

class BatchLoanApplications(BaseModel):
    applications: List[LoanApplication]

def load_model_and_meta():
    global MODEL, METADATA
    if MODEL is None:
        model_path = BEST_MODEL_FILE if BEST_MODEL_FILE.exists() else MODEL_FILE
        if not model_path.exists():
            raise FileNotFoundError("Model binary not found. Please train models first via: python -m src.train")
        MODEL = joblib.load(model_path)

    if METADATA is None:
        meta_path = MODELS / "metadata.json"
        if meta_path.exists():
            METADATA = json.loads(meta_path.read_text())
        else:
            METADATA = {"selected_model": "XGBoost", "default_threshold": 0.35}
    return MODEL, METADATA

@app.get("/")
def root():
    return {
        "title": "Enterprise Loan Risk Intelligence API",
        "version": "2.0.0",
        "docs_url": "/docs",
        "status": "operational"
    }

@app.get("/health")
def health():
    model_path = BEST_MODEL_FILE if BEST_MODEL_FILE.exists() else MODEL_FILE
    return {
        "status": "ok",
        "model_loaded": MODEL is not None or model_path.exists(),
        "model_path": str(model_path),
        "metadata_exists": (MODELS / "metadata.json").exists()
    }

@app.get("/model-info")
def model_info():
    _, meta = load_model_and_meta()
    return meta

@app.post("/predict")
def predict(application: LoanApplication, lgd: float = 0.60):
    """
    Single applicant loan risk inference endpoint.
    Returns default probability, risk grade, financial decision, expected loss, and SHAP top factors.
    """
    try:
        model, meta = load_model_and_meta()
        data = application.model_dump()
        X = clean_numeric_values(pd.DataFrame([data], columns=BASE_FEATURES))
        
        raw_prob = float(model.predict_proba(X)[0, 1])
        prob = evaluate_credit_capacity(data, raw_prob)
        repay_prob = 1.0 - prob
        
        threshold = meta.get("default_threshold", 0.35)
        decision = get_financial_decision(data, prob, threshold)
        grade = get_risk_grade(prob)
        expected_loss = calculate_expected_loss(data["loan_amnt"], prob, lgd)

        if prob < 0.15:
            risk_level = "VERY_LOW"
        elif prob < 0.28:
            risk_level = "LOW"
        elif prob < 0.45:
            risk_level = "MEDIUM"
        elif prob < 0.65:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        top_shap = explain_row(data, top_n=6)

        return {
            "default_probability": round(prob, 4),
            "repayment_probability": round(repay_prob, 4),
            "default_probability_percent": round(prob * 100, 2),
            "repayment_probability_percent": round(repay_prob * 100, 2),
            "risk_level": risk_level,
            "risk_grade": grade,
            "decision": decision,
            "expected_loss_dollars": expected_loss,
            "applied_threshold": threshold,
            "model": meta.get("selected_model", "stacking_ensemble"),
            "model_used": meta.get("selected_model", "stacking_ensemble"),
            "top_risk_factors": top_shap
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
def predict_batch(batch: BatchLoanApplications, lgd: float = 0.60):
    """
    Batch processing endpoint for multiple loan applications.
    Calculates individual risk metrics + aggregate portfolio statistics.
    """
    try:
        model, meta = load_model_and_meta()
        records = [app.model_dump() for app in batch.applications]
        if not records:
            return {"count": 0, "predictions": [], "portfolio_summary": {}}

        df_in = pd.DataFrame(records)
        X = clean_numeric_values(df_in[BASE_FEATURES])
        
        raw_probs = model.predict_proba(X)[:, 1]
        threshold = meta.get("default_threshold", 0.35)

        predictions = []
        total_loan_amount = 0.0
        total_expected_loss = 0.0
        approved_count = 0
        review_count = 0
        rejected_count = 0
        adjusted_probs = []

        for i, data in enumerate(records):
            p = evaluate_credit_capacity(data, float(raw_probs[i]))
            adjusted_probs.append(p)
            decision = get_financial_decision(data, p, threshold)
            grade = get_risk_grade(p)
            el = calculate_expected_loss(data["loan_amnt"], p, lgd)
            
            total_loan_amount += data["loan_amnt"]
            total_expected_loss += el
            
            if decision == "APPROVE":
                approved_count += 1
            elif decision == "MANUAL_REVIEW":
                review_count += 1
            else:
                rejected_count += 1

            predictions.append({
                "applicant_id": i + 1,
                "loan_amount": data["loan_amnt"],
                "default_probability": round(p, 4),
                "risk_grade": grade,
                "decision": decision,
                "expected_loss_dollars": el
            })

        return {
            "total_applicants": len(records),
            "predictions": predictions,
            "portfolio_summary": {
                "total_loan_requested_dollars": round(total_loan_amount, 2),
                "total_expected_loss_dollars": round(total_expected_loss, 2),
                "avg_default_probability_percent": round(float(np.mean(adjusted_probs)) * 100, 2),
                "approval_rate_percent": round((approved_count / len(records)) * 100, 2),
                "approved_count": approved_count,
                "manual_review_count": review_count,
                "rejected_count": rejected_count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch error: {str(e)}")

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    CSV File Upload Batch Inference Endpoint.
    Accepts raw CSV file with loan application headers and returns predictions dataframe JSON.
    """
    try:
        contents = await file.read()
        df_raw = pd.read_csv(io.BytesIO(contents))
        
        model, meta = load_model_and_meta()
        X = clean_numeric_values(df_raw)
        
        raw_probs = model.predict_proba(X)[:, 1]
        threshold = meta.get("default_threshold", 0.35)

        adjusted_probs = []
        decisions = []
        grades = []
        expected_losses = []

        for idx, row in df_raw.iterrows():
            row_dict = row.to_dict()
            p_adj = evaluate_credit_capacity(row_dict, float(raw_probs[idx]))
            adjusted_probs.append(p_adj)
            decisions.append(get_financial_decision(row_dict, p_adj, threshold))
            grades.append(get_risk_grade(p_adj))
            l_amt = float(row_dict.get("loan_amnt", 0.0))
            expected_losses.append(calculate_expected_loss(l_amt, p_adj, 0.60))

        df_raw["default_probability"] = np.round(adjusted_probs, 4)
        df_raw["default_probability_percent"] = np.round(np.array(adjusted_probs) * 100, 2)
        df_raw["risk_grade"] = grades
        df_raw["decision"] = decisions
        
        if "loan_amnt" in df_raw.columns:
            df_raw["expected_loss_dollars"] = expected_losses

        return {
            "file_name": file.filename,
            "rows_processed": len(df_raw),
            "results": df_raw.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV processing failed: {str(e)}")

@app.post("/explain")
def explain(application: LoanApplication, top_n: int = 8):
    """
    Returns full SHAP waterfall feature attributions for an application.
    """
    data = application.model_dump()
    explanations = explain_row(data, top_n=top_n)
    return {
        "applicant_data": data,
        "shap_feature_attributions": explanations
    }

@app.post("/monitoring/drift")
def monitoring_drift(batch: BatchLoanApplications):
    """
    Model Monitoring Endpoint: Calculates Population Stability Index (PSI) & KS test metrics
    comparing recent inference requests against reference baseline training data.
    """
    ref_csv_path = Path("data/raw/loan_data_demo.csv")
    if not ref_csv_path.exists():
        raise HTTPException(status_code=404, detail="Reference training baseline CSV not found.")

    df_ref = pd.read_csv(ref_csv_path)
    records = [a.model_dump() for a in batch.applications]
    df_cur = pd.DataFrame(records)

    num_feats = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "revol_bal", "revol_util"]
    drift_report = analyze_feature_drift(df_ref, df_cur, num_feats)
    return drift_report
