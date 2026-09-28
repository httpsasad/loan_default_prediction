from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 15000

annual_inc = np.clip(RNG.lognormal(mean=np.log(72000), sigma=.48, size=N), 20000, 350000)
loan_amnt = np.clip(RNG.normal(15000, 7000, N), 1000, 40000)
dti = np.clip(RNG.normal(16.5, 7.0, N), 0, 45)
int_rate = np.clip(6.0 + 0.00004 * loan_amnt + 0.40 * dti + RNG.normal(0, 1.5, N), 5.0, 30.0)
open_acc = np.clip(RNG.normal(10, 4, N), 1, 35)
total_acc = np.clip(open_acc + RNG.normal(10, 6, N), 2, 70)
revol_bal = np.clip(RNG.lognormal(np.log(7500), .70, N), 0, 90000)
revol_util = np.clip(100 * revol_bal / np.maximum(500, RNG.normal(19000, 6000, N)), 0, 140)
mort_acc = np.clip(RNG.poisson(1.5, N), 0, 8)
delinq = np.clip(RNG.poisson(.18, N), 0, 5)
inq = np.clip(RNG.poisson(0.7, N), 0, 8)
credit_years = np.clip(RNG.normal(11, 4, N), 1, 30)

term = RNG.choice(["36 months", "60 months"], N, p=[.74, .26])
home = RNG.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], N, p=[.42,.43,.12,.03])
verification = RNG.choice(["Not Verified", "Source Verified", "Verified"], N, p=[.36,.39,.25])
purpose = RNG.choice(
    ["debt_consolidation","credit_card","home_improvement","major_purchase","medical","other"],
    N, p=[.53,.19,.10,.08,.04,.06]
)

installment = loan_amnt * (int_rate/1200) / (1-(1+int_rate/1200)**(-np.where(term=="36 months",36,60)))

# High-fidelity realistic credit score simulation with non-linear thresholding
risk_score = (
    0.35 * (int_rate - 10.0) +
    0.28 * (dti - 15.0) +
    0.30 * (revol_util - 40.0) / 10.0 +
    1.20 * (delinq >= 1) +
    0.85 * (inq >= 2) +
    0.70 * (term == "60 months") +
    0.45 * (home == "RENT") -
    0.25 * (annual_inc / 30000.0) -
    0.15 * (credit_years / 5.0) +
    RNG.normal(0, 0.08, N)
)

# Precise default cutoff (approx 18-20% defaults)
prob = 1 / (1 + np.exp(-(risk_score - 3.5)))
default = (prob > 0.65).astype(int)

df = pd.DataFrame({
    "loan_status": np.where(default == 1, "Charged Off", "Fully Paid"),
    "loan_amnt": loan_amnt.round(2),
    "term": term,
    "int_rate": int_rate.round(2),
    "installment": installment.round(2),
    "annual_inc": annual_inc.round(2),
    "dti": dti.round(2),
    "home_ownership": home,
    "verification_status": verification,
    "purpose": purpose,
    "open_acc": open_acc.round(0),
    "pub_rec": RNG.poisson(.06, N),
    "revol_bal": revol_bal.round(0),
    "revol_util": revol_util.round(2),
    "total_acc": total_acc.round(0),
    "mort_acc": mort_acc,
    "delinq_2yrs": delinq,
    "inq_last_6mths": inq,
    "credit_history_years": credit_years.round(1),
})

# Minor realistic missingness
for col in ["dti", "mort_acc", "revol_util", "annual_inc"]:
    idx = RNG.choice(N, size=int(.01 * N), replace=False)
    df.loc[idx, col] = np.nan

out = Path("data/raw/loan_data_demo.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} rows to {out}")
print(df["loan_status"].value_counts(normalize=True))
