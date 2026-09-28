import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def calculate_psi(reference: np.ndarray, target: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """
    Calculates Population Stability Index (PSI) for numerical feature drift.
    PSI < 0.1: Stable
    0.1 <= PSI < 0.25: Slight Drift
    PSI >= 0.25: High Drift (Action required)
    """
    ref_clean = reference[~np.isnan(reference)]
    tar_clean = target[~np.isnan(target)]

    if len(ref_clean) < 10 or len(tar_clean) < 10:
        return 0.0

    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(ref_clean, percentiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    tar_counts, _ = np.histogram(tar_clean, bins=bin_edges)

    ref_pct = np.maximum(ref_counts / len(ref_clean), eps)
    tar_pct = np.maximum(tar_counts / len(tar_clean), eps)

    psi_val = np.sum((tar_pct - ref_pct) * np.log(tar_pct / ref_pct))
    return float(psi_val)

def ks_drift_test(reference: np.ndarray, target: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Performs 2-sample Kolmogorov-Smirnov test to detect distribution drift.
    """
    ref_clean = reference[~np.isnan(reference)]
    tar_clean = target[~np.isnan(target)]

    if len(ref_clean) < 5 or len(tar_clean) < 5:
        return {"statistic": 0.0, "p_value": 1.0, "drift_detected": False}

    stat, p_value = ks_2samp(ref_clean, tar_clean)
    return {
        "statistic": round(float(stat), 5),
        "p_value": round(float(p_value), 5),
        "drift_detected": bool(p_value < alpha)
    }

def analyze_feature_drift(reference_df: pd.DataFrame, target_df: pd.DataFrame, num_features: list) -> dict:
    """
    Analyzes PSI and KS drift across multiple numeric features.
    """
    results = {}
    overall_drift = False

    for feat in num_features:
        if feat in reference_df.columns and feat in target_df.columns:
            ref_vals = reference_df[feat].to_numpy()
            tar_vals = target_df[feat].to_numpy()

            psi = calculate_psi(ref_vals, tar_vals)
            ks = ks_drift_test(ref_vals, tar_vals)

            if psi >= 0.25 or ks["drift_detected"]:
                overall_drift = True

            results[feat] = {
                "psi": round(psi, 4),
                "psi_status": "High Drift" if psi >= 0.25 else ("Moderate Drift" if psi >= 0.1 else "Stable"),
                "ks_p_value": ks["p_value"],
                "drift_detected": ks["drift_detected"]
            }

    return {
        "overall_drift_detected": overall_drift,
        "feature_metrics": results
    }
