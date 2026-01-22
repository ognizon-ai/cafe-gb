#!/usr/bin/env python3 

""" 

Correlation & Redundancy Analysis for CAFÉ-GB (Paper 1) 

 

This script: 

- Loads CAFÉ-GB selected features 

- Computes inter-feature Pearson correlations 

- Quantifies redundancy using: 

    * mean absolute correlation 

    * maximum absolute correlation 

    * percentage of strongly correlated pairs (|rho| > threshold) 

- Exports feature names and correlation tables 

- Generates heatmaps for the paper 

""" 

 

import argparse 

from pathlib import Path 

import yaml 

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns 

 

 

# ============================================================ 

# ARGUMENTS 

# ============================================================ 

 

def parse_args(): 

    parser = argparse.ArgumentParser( 

        description="Correlation & Redundancy Analysis for CAFÉ-GB" 

    ) 

    parser.add_argument("--dataset", type=str, required=True) 

    parser.add_argument("--seed", type=int, default=42) 

    parser.add_argument("--k", type=int, default=100) 

    parser.add_argument( 

        "--corr-threshold", 

        type=float, 

        default=0.8, 

        help="Threshold for strong correlation (default: 0.8)", 

    ) 

    return parser.parse_args() 

 

 

# ============================================================ 

# DATA LOADING 

# ============================================================ 

 

def load_selected_features(dataset, seed, k): 

    path = Path( 

        f"fs/cafe_gb/{dataset}/aggregated_importance_seed{seed}.parquet" 

    ) 

 

    if not path.exists(): 

        raise FileNotFoundError(f"Missing CAFÉ-GB output: {path}") 

 

    df = pd.read_parquet(path) 

 

    if "feature" not in df.columns: 

        raise KeyError("Expected column 'feature' not found") 

 

    return df.head(k)["feature"].tolist() 

 

 

def load_training_data(dataset): 

    cfg_path = Path("config/data.yaml") 

    if not cfg_path.exists(): 

        raise FileNotFoundError("config/data.yaml not found") 

 

    with open(cfg_path, "r") as f: 

        data_cfg = yaml.safe_load(f) 

 

    train_path = Path(data_cfg["datasets"][dataset]["train"]) 

    if not train_path.exists(): 

        raise FileNotFoundError(f"Training file not found: {train_path}") 

 

    return pd.read_parquet(train_path) 

 

 

# ============================================================ 

# ANALYSIS 

# ============================================================ 

 

def compute_correlation_matrix(df): 

    return df.corr(method="pearson") 

 

 

def summarize_redundancy(corr_matrix, threshold=0.8): 

    upper = corr_matrix.where( 

        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool) 

    ) 

 

    abs_upper = upper.abs() 

 

    mean_abs_corr = abs_upper.mean().mean() 

    max_abs_corr = abs_upper.max().max() 

 

    total_pairs = abs_upper.count().sum() 

    high_corr_pairs = (abs_upper > threshold).sum().sum() 

    pct_high_corr = ( 

        100.0 * high_corr_pairs / total_pairs 

        if total_pairs > 0 

        else 0.0 

    ) 

 

    return mean_abs_corr, max_abs_corr, pct_high_corr 

 

 

def correlation_pairs(corr_matrix, threshold=0.8): 

    pairs = ( 

        corr_matrix 

        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) 

        .stack() 

        .reset_index() 

    ) 

    pairs.columns = ["feature_1", "feature_2", "correlation"] 

    pairs["abs_correlation"] = pairs["correlation"].abs() 

    pairs["is_highly_correlated"] = pairs["abs_correlation"] > threshold 

    return pairs.sort_values("abs_correlation", ascending=False) 

 

 

# ============================================================ 

# PLOTTING 

# ============================================================ 

 

def plot_heatmap(corr_matrix, dataset, k): 

    out_dir = Path(f"results/figures/{dataset}") 

    out_dir.mkdir(parents=True, exist_ok=True) 

 

    plt.figure(figsize=(10, 8)) 

    sns.heatmap( 

        corr_matrix, 

        cmap="coolwarm", 

        center=0, 

        square=True, 

        cbar_kws={"shrink": 0.75}, 

    ) 

    plt.title(f"Feature Correlation Heatmap (k={k})") 

    plt.tight_layout() 

    plt.savefig(out_dir / f"fig_corr_heatmap_k{k}.png", dpi=300) 

    plt.close() 

 

 

# ============================================================ 

# MAIN 

# ============================================================ 

 

def main(): 

    args = parse_args() 

 

    print(f"[INFO] Dataset: {args.dataset}") 

    print(f"[INFO] Seed: {args.seed}") 

    print(f"[INFO] k: {args.k}") 

    print(f"[INFO] Correlation threshold: {args.corr_threshold}") 

 

    # Load selected features 

    features = load_selected_features( 

        args.dataset, args.seed, args.k 

    ) 

 

    # Load training data 

    df = load_training_data(args.dataset) 

 

    # Keep only selected features 

    df = df[features] 

 

    # Remove constant features 

    df = df.loc[:, df.nunique() > 1] 

 

    assert df.columns.is_unique 

    assert df.isnull().sum().sum() == 0 

 

    print(f"[INFO] Features used for correlation: {df.shape[1]}") 

 

    out_dir = Path(f"results/tables/{args.dataset}") 

    out_dir.mkdir(parents=True, exist_ok=True) 

 

    # Save selected feature names 

    pd.DataFrame({"feature": df.columns}).to_csv( 

        out_dir / f"selected_features_k{args.k}.csv", 

        index=False, 

    ) 

 

    # Correlation matrix 

    corr_matrix = compute_correlation_matrix(df) 

 

    corr_matrix.to_csv( 

        out_dir / f"correlation_matrix_k{args.k}.csv" 

    ) 

 

    # Redundancy summary 

    mean_corr, max_corr, pct_high_corr = summarize_redundancy( 

        corr_matrix, threshold=args.corr_threshold 

    ) 

 

    summary = pd.DataFrame( 

        { 

            "Dataset": [args.dataset], 

            "k": [args.k], 

            "Mean_Abs_Correlation": [round(mean_corr, 4)], 

            "Max_Abs_Correlation": [round(max_corr, 4)], 

            f"Pct_|corr|>{args.corr_threshold}": [ 

                round(pct_high_corr, 2) 

            ], 

        } 

    ) 

 

    summary_path = out_dir / f"table_corr_redundancy_k{args.k}.xlsx" 

    summary.to_excel(summary_path, index=False) 

 

    # Pairwise correlations 

    pairs = correlation_pairs( 

        corr_matrix, threshold=args.corr_threshold 

    ) 

    pairs.to_csv( 

        out_dir / f"correlation_pairs_k{args.k}.csv", 

        index=False, 

    ) 

 

    # Top correlated pairs 

    pairs.head(20).to_excel( 

        out_dir / f"top_correlated_pairs_k{args.k}.xlsx", 

        index=False, 

    ) 

 

    # Heatmap 

    plot_heatmap(corr_matrix, args.dataset, args.k) 

 

    print(f"[DONE] Redundancy table saved to {summary_path}") 

    print( 

        f"[DONE] Mean |corr| = {mean_corr:.4f}, " 

        f"Max |corr| = {max_corr:.4f}, " 

        f"|corr|>{args.corr_threshold} = {pct_high_corr:.2f}%" 

    ) 

 

 

if __name__ == "__main__": 

    main() 