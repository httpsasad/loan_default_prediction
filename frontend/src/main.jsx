import React, {useState, useEffect} from "react";
import {createRoot} from "react-dom/client";
import "./styles.css";

function calcInstallment(amount, rate, termStr) {
  const months = termStr === "60 months" ? 60 : 36;
  const r = (rate / 100) / 12;
  if (r === 0) return Math.round(amount / months);
  const pmt = (amount * r * Math.pow(1 + r, months)) / (Math.pow(1 + r, months) - 1);
  return Math.round(pmt);
}

const lowRiskProfile = {
  loan_amnt: 8000, term: "36 months", int_rate: 6.5, installment: 245,
  annual_inc: 95000, dti: 8.0, home_ownership: "MORTGAGE",
  verification_status: "Verified", purpose: "credit_card",
  open_acc: 10, pub_rec: 0, revol_bal: 3500, revol_util: 15.0,
  total_acc: 22, mort_acc: 2, delinq_2yrs: 0, inq_last_6mths: 0,
  credit_history_years: 14.0
};

const medRiskProfile = {
  loan_amnt: 18000, term: "36 months", int_rate: 14.5, installment: 619,
  annual_inc: 58000, dti: 22.0, home_ownership: "RENT",
  verification_status: "Source Verified", purpose: "debt_consolidation",
  open_acc: 9, pub_rec: 0, revol_bal: 12000, revol_util: 55.0,
  total_acc: 18, mort_acc: 0, delinq_2yrs: 0, inq_last_6mths: 1,
  credit_history_years: 7.5
};

const highRiskProfile = {
  loan_amnt: 45000, term: "60 months", int_rate: 28.5, installment: 1412,
  annual_inc: 32000, dti: 44.0, home_ownership: "RENT",
  verification_status: "Verified", purpose: "debt_consolidation",
  open_acc: 14, pub_rec: 2, revol_bal: 38000, revol_util: 92.0,
  total_acc: 16, mort_acc: 0, delinq_2yrs: 3, inq_last_6mths: 5,
  credit_history_years: 3.5
};

function App() {
  const [form, setForm] = useState(lowRiskProfile);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const updateField = (name, value) => {
    const numeric = ["loan_amnt","int_rate","installment","annual_inc","dti","open_acc","pub_rec","revol_bal","revol_util","total_acc","mort_acc","delinq_2yrs","inq_last_6mths","credit_history_years"];
    const val = numeric.includes(name) ? Number(value) : value;
    
    setForm(prev => {
      const next = {...prev, [name]: val};
      if (name === "loan_amnt" || name === "int_rate" || name === "term") {
        next.installment = calcInstallment(next.loan_amnt, next.int_rate, next.term);
      }
      return next;
    });
  };

  const handleChange = (e) => {
    updateField(e.target.name, e.target.value);
  };

  const applyPreset = (profile) => {
    setForm(profile);
    setResult(null);
  };

  async function predict(e) {
    if (e) e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(form)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Prediction failed");
      setResult(data);
    } catch(err) {
      setResult({error: err.message});
    } finally {
      setLoading(false);
    }
  }

  const fields = [
    ["loan_amnt", "Loan Amount ($)"],
    ["int_rate", "Interest Rate (%)"],
    ["installment", "Monthly Installment ($)"],
    ["annual_inc", "Annual Income ($)"],
    ["dti", "Debt-to-Income Ratio (%)"],
    ["revol_util", "Revolving Utilization (%)"],
    ["revol_bal", "Revolving Balance ($)"],
    ["open_acc", "Open Accounts"],
    ["total_acc", "Total Credit Accounts"],
    ["delinq_2yrs", "Delinquencies (2 yrs)"],
    ["inq_last_6mths", "Inquiries (Last 6 mo)"],
    ["pub_rec", "Public Records"],
    ["mort_acc", "Mortgage Accounts"],
    ["credit_history_years", "Credit History (years)"]
  ];

  return (
    <div className="app">
      <header>
        <div>
          <span className="eyebrow">ENTERPRISE ML CREDIT RISK PLATFORM</span>
          <h1>Loan Risk Intelligence</h1>
          <p>Multi-Model Stacking Ensemble & Financial Expected Loss Quantification</p>
        </div>
        <div className="badge">Best Model: 98.33% Acc</div>
      </header>

      <main>
        <section className="card">
          <h2>Applicant Profile</h2>
          
          <div className="preset-bar">
            <span style={{fontSize: "12px", fontWeight: 700, alignSelf: "center", color: "#64748b"}}>Quick Profiles:</span>
            <button type="button" className="preset-btn low-btn" onClick={() => applyPreset(lowRiskProfile)}>🟢 Low Risk Applicant</button>
            <button type="button" className="preset-btn med-btn" onClick={() => applyPreset(medRiskProfile)}>🟡 Moderate Risk Applicant</button>
            <button type="button" className="preset-btn high-btn" onClick={() => applyPreset(highRiskProfile)}>🔴 Extreme Risk Applicant</button>
          </div>

          <form onSubmit={predict}>
            <div className="grid">
              {fields.map(([name, label]) => (
                <label key={name}>{label}
                  <input name={name} type="number" step="any" value={form[name]} onChange={handleChange}/>
                </label>
              ))}
              <label>Loan Term
                <select name="term" value={form.term} onChange={handleChange}>
                  <option>36 months</option>
                  <option>60 months</option>
                </select>
              </label>
              <label>Home Ownership
                <select name="home_ownership" value={form.home_ownership} onChange={handleChange}>
                  <option>RENT</option>
                  <option>MORTGAGE</option>
                  <option>OWN</option>
                  <option>OTHER</option>
                </select>
              </label>
              <label>Income Verification
                <select name="verification_status" value={form.verification_status} onChange={handleChange}>
                  <option>Not Verified</option>
                  <option>Source Verified</option>
                  <option>Verified</option>
                </select>
              </label>
              <label>Loan Purpose
                <select name="purpose" value={form.purpose} onChange={handleChange}>
                  <option>debt_consolidation</option>
                  <option>credit_card</option>
                  <option>home_improvement</option>
                  <option>major_purchase</option>
                  <option>medical</option>
                  <option>other</option>
                </select>
              </label>
            </div>
            <button className="submit-btn" disabled={loading}>
              {loading ? "Evaluating Risk Engine..." : "Analyze Loan Risk & Expected Loss"}
            </button>
          </form>
        </section>

        <section className="card result">
          <h2>Risk Assessment & Financial Decision</h2>
          {!result && <div className="empty">Submit an applicant profile or select a quick profile above.</div>}
          {result?.error && <div className="error">{result.error}<br/>Make sure FastAPI server is running on http://127.0.0.1:8000</div>}
          {result && !result.error && (
            <>
              <div className="risk-header">
                <div className={"risk " + result.risk_level.toLowerCase()}>{result.risk_level} RISK</div>
                <div className="grade-badge" title="Credit Risk Grade">{result.risk_grade}</div>
              </div>

              <div className={"decision-banner " + result.decision.toLowerCase()}>
                RECOMMENDATION: {result.decision.replace("_", " ")}
              </div>

              <div className="metrics">
                <div><span>Default Probability</span><strong>{result.default_probability_percent}%</strong></div>
                <div><span>Expected Loss ($)</span><strong>${result.expected_loss_dollars?.toLocaleString()}</strong></div>
              </div>

              <div className="bar">
                <div style={{
                  width: `${result.default_probability_percent}%`,
                  background: result.default_probability > 0.4 ? "#dc2626" : (result.default_probability > 0.2 ? "#d97706" : "#16a34a")
                }}/>
              </div>

              <p className="model">Evaluated by: <strong>{result.model_used.toUpperCase()}</strong> (Applied Threshold: {result.applied_threshold})</p>

              {result.top_risk_factors && result.top_risk_factors.length > 0 && (
                <div className="shap-section">
                  <div className="shap-title">Top SHAP Feature Attributions:</div>
                  <div className="shap-chips">
                    {result.top_risk_factors.map((f, idx) => (
                      <span key={idx} className={"shap-chip " + (f.impact > 0 ? "increases" : "decreases")}>
                        {f.impact > 0 ? "▲" : "▼"} {f.feature} ({f.impact > 0 ? "+" : ""}{f.impact})
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="notice">Expected Loss = Loan Amount × Probability of Default × 60% Loss Given Default (LGD benchmark).</div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
