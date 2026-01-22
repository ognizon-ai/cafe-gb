# stats/k_selection.py

"""
k-selection & Stability Analysis for CAFÉ-GB (Paper 1)

This module:
- Evaluates multiple k values
- Measures performance, stability, and coverage
- Produces table + plots for Section 5.2
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# ============================================================
# Utility: Jaccard similarity
# ============================================================

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


# ============================================================
# Load CAFÉ-GB features for all seeds
# ============================================================

def load_features_for_k(base_dir, seeds, k):
    """
    Load top-k features per seed.
    """
    features = {}
    for seed in seeds:
        path = Path(base_dir) / f"aggregated_importance_seed{seed}.parquet"
        df = pd.read_parquet(path)
        features[seed] = df.head(k)["feature"].tolist()
    return features


# ============================================================
# Stability computation
# ============================================================

def compute_stability(feature_sets):
    """
    Compute mean Jaccard similarity across all seed pairs.
    """
    pairs = combinations(feature_sets.values(), 2)
    scores = [jaccard(a, b) for a, b in pairs]
    return np.mean(scores)


# ============================================================
# Performance evaluation (single classifier, lightweight)
# ============================================================

def evaluate_accuracy(train_df, test_df, features, target_col, seed):
    """
    Use XGBoost only (fast & strong) for k-selection.
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


# ============================================================
# Main k-selection routine
# ============================================================

def run_k_selection(
    train_df,
    test_df,
    seeds,
    k_values,
    target_col,
    cafe_dir="fs/cafe_gb",
    out_table="results/tables/table3_k_selection.xlsx",
    out_fig_dir="results/figures",
):
    records = []

    for k in k_values:
        feature_sets = load_features_for_k(cafe_dir, seeds, k)

        # Stability
        stability = compute_stability(feature_sets)

        # Performance (mean accuracy across seeds)
        accs = []
        for seed in seeds:
            acc = evaluate_accuracy(
                train_df,
                test_df,
                feature_sets[seed],
                target_col,
                seed,
            )
            accs.append(acc)

        records.append({
            "k": k,
            "Accuracy_Mean": np.mean(accs),
            "Accuracy_Std": np.std(accs),
            "Jaccard_Stability": stability,
        })

    results = pd.DataFrame(records)
    Path(out_table).parent.mkdir(parents=True, exist_ok=True)
    results.to_excel(out_table, index=False)

    # ------------------------
    # Plots
    # ------------------------
    Path(out_fig_dir).mkdir(parents=True, exist_ok=True)

    # Accuracy vs k
    plt.figure()
    plt.errorbar(
        results["k"],
        results["Accuracy_Mean"],
        yerr=results["Accuracy_Std"],
        marker="o",
    )
    plt.xlabel("Number of Selected Features (k)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k")
    plt.tight_layout()
    plt.savefig(f"{out_fig_dir}/fig3_accuracy_vs_k.png")
    plt.close()

    # Stability vs k
    plt.figure()
    plt.plot(
        results["k"],
        results["Jaccard_Stability"],
        marker="o",
    )
    plt.xlabel("Number of Selected Features (k)")
    plt.ylabel("Jaccard Stability")
    plt.title("Stability vs k")
    plt.tight_layout()
    plt.savefig(f"{out_fig_dir}/fig4_stability_vs_k.png")
    plt.close()

    return results
