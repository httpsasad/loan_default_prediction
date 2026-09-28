from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "selected_model" in data

def test_predict_single():
    payload = {
        "loan_amnt": 15000,
        "term": "36 months",
        "int_rate": 12.5,
        "installment": 500.0,
        "annual_inc": 75000,
        "dti": 16.0,
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "open_acc": 9,
        "pub_rec": 0,
        "revol_bal": 8000,
        "revol_util": 45.0,
        "total_acc": 20,
        "mort_acc": 1,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "credit_history_years": 10
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "risk_grade" in data
    assert "decision" in data
    assert "expected_loss_dollars" in data
    assert "top_risk_factors" in data

def test_predict_batch():
    payload = {
        "applications": [
            {
                "loan_amnt": 10000,
                "term": "36 months",
                "int_rate": 8.5,
                "installment": 315.0,
                "annual_inc": 90000,
                "dti": 10.0,
                "home_ownership": "MORTGAGE",
                "verification_status": "Not Verified",
                "purpose": "credit_card",
                "open_acc": 10,
                "pub_rec": 0,
                "revol_bal": 4000,
                "revol_util": 20.0,
                "total_acc": 25,
                "mort_acc": 2,
                "delinq_2yrs": 0,
                "inq_last_6mths": 0,
                "credit_history_years": 12
            },
            {
                "loan_amnt": 35000,
                "term": "60 months",
                "int_rate": 22.0,
                "installment": 965.0,
                "annual_inc": 42000,
                "dti": 32.0,
                "home_ownership": "RENT",
                "verification_status": "Verified",
                "purpose": "debt_consolidation",
                "open_acc": 14,
                "pub_rec": 1,
                "revol_bal": 28000,
                "revol_util": 88.0,
                "total_acc": 18,
                "mort_acc": 0,
                "delinq_2yrs": 2,
                "inq_last_6mths": 4,
                "credit_history_years": 4
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_applicants"] == 2
    assert "portfolio_summary" in data
    assert data["portfolio_summary"]["approved_count"] >= 0

def test_explain():
    payload = {
        "loan_amnt": 12000,
        "term": "36 months",
        "int_rate": 11.5,
        "installment": 395.0,
        "annual_inc": 65000,
        "dti": 18.5,
        "home_ownership": "RENT",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "open_acc": 8,
        "pub_rec": 0,
        "revol_bal": 7200,
        "revol_util": 42.0,
        "total_acc": 18,
        "mort_acc": 1,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "credit_history_years": 8
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "shap_feature_attributions" in data

def test_predict_extreme_loan_amount():
    payload = {
        "loan_amnt": 8000000000000000000,
        "term": "36 months",
        "int_rate": 6.5,
        "installment": 245192023012080030,
        "annual_inc": 95000,
        "dti": 8.0,
        "home_ownership": "MORTGAGE",
        "verification_status": "Verified",
        "purpose": "credit_card",
        "open_acc": 10,
        "pub_rec": 0,
        "revol_bal": 3500,
        "revol_util": 15.0,
        "total_acc": 22,
        "mort_acc": 2,
        "delinq_2yrs": 0,
        "inq_last_6mths": 0,
        "credit_history_years": 14.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "REJECT"
    assert data["risk_grade"] == "F"
    assert data["risk_level"] == "CRITICAL"
    assert data["default_probability"] > 0.80

