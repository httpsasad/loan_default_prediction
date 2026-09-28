import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .config import BASE_FEATURES, MODELS, REPORTS, RANDOM_STATE, BEST_MODEL_FILE
from .data import load_csv
from .preprocess import clean_numeric_values, make_preprocessor

def make_high_performance_models(scale_pos_weight):
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1500, C=1.5, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=16, min_samples_split=3, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "xgboost": XGBClassifier(
            n_estimators=600, max_depth=7, learning_rate=0.03,
            subsample=0.88, colsample_bytree=0.88, gamma=0.05,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=600, max_depth=8, num_leaves=63, learning_rate=0.03,
            subsample=0.88, colsample_bytree=0.88, min_child_samples=15,
            objective="binary", metric="binary_logloss", n_jobs=-1, random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight, verbose=-1
        )
    }

def find_optimal_accuracy_threshold(y_true, y_probs):
    """
    Finds the probability threshold that maximizes overall accuracy while preserving balanced F1 performance.
    """
    thresholds = np.linspace(0.15, 0.85, 141)
    best_acc = 0.0
    best_th = 0.50
    
    for th in thresholds:
        preds = (y_probs >= th).astype(int)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        # Combined objective: maximize Accuracy with weighted F1 penalization for degenerate predictions
        score = acc * 0.70 + f1 * 0.30
        if score > best_acc:
            best_acc = score
            best_th = th
            
    return round(float(best_th), 4)

def evaluate(name, model, X_test, y_test, threshold=0.50):
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= threshold).astype(int)
    
    metrics = {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "accuracy_percent": round(float(accuracy_score(y_test, pred)) * 100, 2),
        "precision_default": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall_default": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_default": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, p)), 4),
        "pr_auc": round(float(average_precision_score(y_test, p)), 4),
    }
    print(f"\n--- {name.upper()} (Threshold = {threshold:.4f}) ---")
    print(classification_report(y_test, pred, target_names=["repaid", "default"], zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    return metrics, p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    # 1. Regenerate high-fidelity dataset if script exists
    demo_script = Path("scripts/generate_demo_data.py")
    if demo_script.exists():
        import subprocess
        subprocess.run(["python", str(demo_script)], check=True)

    data_path = Path(args.data) if args.data else Path("data/raw/loan_data_demo.csv")
    df = load_csv(data_path)
    if "default" not in df:
        raise ValueError("Training data must contain loan_status column.")

    X = clean_numeric_values(df[BASE_FEATURES])
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    scale_pos_weight = max(1.0, float((y_train == 0).sum() / max(1, (y_train == 1).sum())))

    MODELS.mkdir(exist_ok=True, parents=True)
    REPORTS.mkdir(exist_ok=True, parents=True)

    base_estimators = make_high_performance_models(scale_pos_weight)
    
    results = []
    fitted_pipelines = {}
    probabilities_dict = {}

    # 2. Fit High Performance Models
    for name, estimator in base_estimators.items():
        pipe = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        
        train_probs = pipe.predict_proba(X_train)[:, 1]
        opt_th = find_optimal_accuracy_threshold(y_train, train_probs)
        metrics, probs = evaluate(name, pipe, X_test, y_test, threshold=opt_th)
        metrics["optimal_threshold"] = opt_th
        
        results.append(metrics)
        fitted_pipelines[name] = pipe
        probabilities_dict[name] = probs

        joblib.dump(pipe, MODELS / f"{name}_pipeline.joblib")

    # 3. Build Ultra Stacking Ensemble
    print("\n--- Training Ultra Stacking Meta-Ensemble ---")
    stacking_estimators = [
        ("xgb", base_estimators["xgboost"]),
        ("lgbm", base_estimators["lightgbm"]),
        ("rf", base_estimators["random_forest"]),
        ("lr", base_estimators["logistic_regression"])
    ]
    stacker = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(C=2.0, max_iter=1000),
        cv=5,
        n_jobs=-1
    )
    ensemble_pipe = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", stacker)
    ])
    ensemble_pipe.fit(X_train, y_train)
    
    train_ens_probs = ensemble_pipe.predict_proba(X_train)[:, 1]
    opt_th_ens = find_optimal_accuracy_threshold(y_train, train_ens_probs)
    ens_metrics, ens_probs = evaluate("stacking_ensemble", ensemble_pipe, X_test, y_test, threshold=opt_th_ens)
    ens_metrics["optimal_threshold"] = opt_th_ens
    
    results.append(ens_metrics)
    fitted_pipelines["stacking_ensemble"] = ensemble_pipe
    probabilities_dict["stacking_ensemble"] = ens_probs
    joblib.dump(ensemble_pipe, MODELS / "stacking_ensemble_pipeline.joblib")

    # 4. Model Selection & Persistence
    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    results_df.to_csv(REPORTS / "model_comparison.csv", index=False)
    print("\n================ MODEL COMPARISON ================")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    best_pipe = fitted_pipelines[best_model_name]
    best_threshold = results_df.iloc[0]["optimal_threshold"]
    best_accuracy = results_df.iloc[0]["accuracy_percent"]
    joblib.dump(best_pipe, BEST_MODEL_FILE)

    # 5. Generate Visualizations
    plt.figure(figsize=(9, 6))
    for mname, pvals in probabilities_dict.items():
        fpr, tpr, _ = roc_curve(y_test, pvals)
        auc_score = roc_auc_score(y_test, pvals)
        plt.plot(fpr, tpr, label=f"{mname} (AUC = {auc_score:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves — Best Model Accuracy: {best_accuracy:.2f}%", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS / "roc_curves_all_models.png", dpi=200)
    plt.close()

    # Confusion Matrix Plot
    best_probs = probabilities_dict[best_model_name]
    best_preds = (best_probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, best_preds)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Repaid", "Default"],
                yticklabels=["Repaid", "Default"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix — {best_model_name.upper()} ({best_accuracy:.2f}% Accuracy)")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix_best_model.png", dpi=200)
    plt.close()

    metadata = {
        "selected_model": best_model_name,
        "best_accuracy_percent": best_accuracy,
        "default_threshold": best_threshold,
        "features": BASE_FEATURES,
        "target": "default",
        "models": results,
    }
    (MODELS / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n[SUCCESS] Training completed successfully! Best Model: {best_model_name.upper()} | Accuracy: {best_accuracy:.2f}% (ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f})")

if __name__ == "__main__":
    main()

