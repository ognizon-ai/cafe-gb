# models/train_classifiers.py

"""
Classifier Training Module for CAFÉ-GB 

Trains classical ML classifiers using:
- Baseline (all features)
- CAFÉ-GB selected feature subsets

Saves:
- Performance metrics
- Trained models with explicit provenance tags
"""

import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# Classifier Factory
# ============================================================

def get_classifiers(seed):
    return {
        "LR": LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            random_state=seed,
        ),
        "RF": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=seed,
        ),
        "XGB": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
        "LGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
        ),
    }


# ============================================================
# Metrics
# ============================================================

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
    }


# ============================================================
# Training Pipeline
# ============================================================

def train_and_evaluate(
    train_df,
    test_df,
    feature_list,
    target_col,
    seed,
    dataset,
    output_path,
):
    """
    Train classifiers and save:
    - Performance metrics
    - Models with explicit tags (baseline / cafe_gb)
    """

    # -------------------------------
    # Feature handling
    # -------------------------------
    if feature_list is None:
        X_train = train_df.drop(columns=[target_col])
        X_test = test_df.drop(columns=[target_col])
        tag = "baseline"
    else:
        X_train = train_df[feature_list]
        X_test = test_df[feature_list]
        tag = "cafe_gb"

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    classifiers = get_classifiers(seed)
    results = []

    model_dir = Path(f"models/saved/{dataset}")
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, model in classifiers.items():
        # -------------------------------
        # Train
        # -------------------------------
        model.fit(X_train, y_train)

        # -------------------------------
        # Evaluate
        # -------------------------------
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics.update({
            "Classifier": name,
            "Seed": seed,
            "Num_Features": X_train.shape[1],
        })
        results.append(metrics)

        # -------------------------------
        # SAVE MODEL (FIXED)
        # -------------------------------
        model_path = model_dir / f"{name}_{tag}_seed{seed}.joblib"
        dump(model, model_path)

    # -------------------------------
    # SAVE RESULTS TABLE
    # -------------------------------
    results_df = pd.DataFrame(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_excel(output_path, index=False)

    return results_df
