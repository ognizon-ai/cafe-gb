#!/usr/bin/env python3
"""
Statistical significance analysis for CAFÉ-GB (Paper 1)

Design principles:
- Uses SAME classifier across seeds
- Baseline vs CAFÉ-GB distinguished via Num_Features
- Paired Wilcoxon signed-rank test
- 95% bootstrap confidence intervals
- Supports --metric all
- Compatible with tables containing:
  Accuracy, Precision, Recall, F1, MCC, ROC_AUC, PR_AUC,
  Classifier, Seed, Num_Features
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--metric",
        default="all",
        help="Metric name (e.g., MCC) or 'all'"
    )
    p.add_argument(
        "--classifier",
        default="LGBM",
        help="Classifier to evaluate (default: LGBM)"
    )
    p.add_argument(
        "--baseline_k",
        type=int,
        default=None,
        help="Num_Features for baseline (auto-detect if omitted)"
    )
    p.add_argument(
        "--proposed_k",
        type=int,
        default=None,
        help="Num_Features for CAFÉ-GB (auto-detect if omitted)"
    )
    return p.parse_args()


# ============================================================
# UTILS
# ============================================================

def bootstrap_ci(values, n_boot=2000, alpha=0.05):
    values = np.asarray(values)
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return round(lo, 4), round(hi, 4)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    table_dir = Path(f"results/tables/{args.dataset}")
    baseline_files = sorted(table_dir.glob("performance_baseline_seed*.xlsx"))
    proposed_files = sorted(table_dir.glob("performance_cafe_gb_seed*.xlsx"))

    if not baseline_files or not proposed_files:
        raise FileNotFoundError(
            f"Expected both baseline and CAFÉ-GB result files in {table_dir}. "
            f"Found baseline={len(baseline_files)}, cafe_gb={len(proposed_files)}"
        )

    files = baseline_files + proposed_files


    # Load all seed tables
    df = pd.concat(
        [pd.read_excel(f) for f in files],
        ignore_index=True
    )

    required_cols = {"Classifier", "Seed", "Num_Features"}
    if not required_cols.issubset(df.columns):
        raise KeyError(
            f"Missing required columns. "
            f"Expected at least {required_cols}, "
            f"found {set(df.columns)}"
        )

    # Filter classifier
    df = df[df["Classifier"] == args.classifier]

    if df.empty:
        raise ValueError(
            f"No rows found for classifier '{args.classifier}'"
        )

    # Auto-detect baseline and proposed feature sizes
    ks = sorted(df["Num_Features"].unique())

    if len(ks) < 2:
        raise ValueError(
            "Statistical testing requires at least two feature sizes "
            "(baseline and CAFÉ-GB)."
        )

    baseline_k = args.baseline_k or max(ks)
    proposed_k = args.proposed_k or min(ks)

    if baseline_k == proposed_k:
        raise ValueError("Baseline and proposed k must be different")

    # Determine metrics
    if args.metric.lower() == "all":
        metrics = [
            c for c in df.columns
            if c not in ["Classifier", "Seed", "Num_Features"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        metrics = [args.metric]

    print(f"[INFO] Dataset     : {args.dataset}")
    print(f"[INFO] Classifier  : {args.classifier}")
    print(f"[INFO] Baseline k  : {baseline_k}")
    print(f"[INFO] Proposed k  : {proposed_k}")
    print(f"[INFO] Metrics     : {metrics}")

    results = []

    for metric in metrics:
        if metric not in df.columns:
            print(f"[WARN] Skipping missing metric: {metric}")
            continue

        base_df = df[df["Num_Features"] == baseline_k].sort_values("Seed")
        prop_df = df[df["Num_Features"] == proposed_k].sort_values("Seed")

        if not np.array_equal(
            base_df["Seed"].values,
            prop_df["Seed"].values
        ):
            raise ValueError(
                f"Seed mismatch for metric '{metric}'. "
                "Baseline and CAFÉ-GB must use identical seeds."
            )

        base = base_df[metric].values
        prop = prop_df[metric].values

        stat, pval = wilcoxon(prop, base)
        ci_lo, ci_hi = bootstrap_ci(prop)

        results.append({
            "Dataset": args.dataset,
            "Classifier": args.classifier,
            "Metric": metric,
            "Baseline_k": baseline_k,
            "Proposed_k": proposed_k,
            "Mean_Baseline": round(base.mean(), 4),
            "Mean_Proposed": round(prop.mean(), 4),
            "CI_95_Lower": ci_lo,
            "CI_95_Upper": ci_hi,
            "Wilcoxon_p": round(pval, 6),
        })

    out_df = pd.DataFrame(results)

    out_path = table_dir / "table_statistical_significance.xlsx"
    out_df.to_excel(out_path, index=False)

    print(f"[DONE] Statistical analysis saved to {out_path}")


if __name__ == "__main__":
    main()
