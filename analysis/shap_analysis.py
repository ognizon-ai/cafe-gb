#!/usr/bin/env python3 

""" 

SHAP explainability for CAFÉ-GB models. 

Supports tree-based models (RF, XGB, LGBM). 

 

Generates: 

- SHAP summary (beeswarm) plot 

- SHAP bar plot (global importance) 

- Mean |SHAP| values (CSV) 

- Directional SHAP contributions (CSV) 

- Top-20 SHAP features per seed (for stability analysis) 

- Optional SHAP dependence plot (top feature) 

""" 

 

import argparse 

from pathlib import Path 

import joblib 

import pandas as pd 

import shap 

import matplotlib.pyplot as plt 

import yaml 

import sys 

 

 

# ============================================================ 

# ARGUMENTS 

# ============================================================ 

 

def parse_args(): 

    p = argparse.ArgumentParser() 

    p.add_argument("--dataset", required=True) 

    p.add_argument("--seed", default=42, type=int) 

    p.add_argument("--model", default="LGBM") 

    p.add_argument("--k", default=100, type=int) 

    p.add_argument("--max_samples", default=2000, type=int) 

    p.add_argument("--dependence", action="store_true", 

                   help="Generate SHAP dependence plot for top feature") 

    return p.parse_args() 

 

 

# ============================================================ 

# MAIN 

# ============================================================ 

 

def main(): 

    args = parse_args() 

 

    # ------------------------------- 

    # Load CAFÉ-GB trained model 

    # ------------------------------- 

    model_path = Path( 

        f"models/saved/{args.dataset}/{args.model}_cafe_gb_seed{args.seed}.joblib" 

    ) 

    if not model_path.exists(): 

        sys.exit( 

            f"[ERROR] CAFÉ-GB model not found:\n  {model_path}\n" 

            f"Available models:\n  " 

            + "\n  ".join( 

                p.name for p in Path(f"models/saved/{args.dataset}").glob(f"{args.model}_*") 

            ) 

        ) 

 

    model = joblib.load(model_path) 

 

    # ------------------------------- 

    # Load CAFÉ-GB selected features 

    # ------------------------------- 

    agg_path = Path( 

        f"fs/cafe_gb/{args.dataset}/aggregated_importance_seed{args.seed}.parquet" 

    ) 

    if not agg_path.exists(): 

        sys.exit(f"[ERROR] Feature importance file not found: {agg_path}") 

 

    agg = pd.read_parquet(agg_path) 

    features = agg.head(args.k)["feature"].tolist() 

 

    # ------------------------------- 

    # Load dataset configuration 

    # ------------------------------- 

    with open("config/data.yaml", "r") as f: 

        data_cfg = yaml.safe_load(f) 

 

    if args.dataset not in data_cfg["datasets"]: 

        sys.exit(f"[ERROR] Dataset '{args.dataset}' not found in config/data.yaml") 

 

    train_path = data_cfg["datasets"][args.dataset]["train"] 

    train_df = pd.read_parquet(train_path) 

 

    # ------------------------------- 

    # Subsample data for SHAP 

    # ------------------------------- 

    X = train_df[features].sample( 

        n=min(args.max_samples, len(train_df)), 

        random_state=0 

    ) 

 

    # ------------------------------- 

    # SHAP computation 

    # ------------------------------- 

    explainer = shap.TreeExplainer(model) 

    shap_values = explainer.shap_values(X) 

 

    # Handle binary classification output 

    if isinstance(shap_values, list): 

        shap_values = shap_values[1] 

 

    # ------------------------------- 

    # Output directory 

    # ------------------------------- 

    out_dir = Path(f"results/figures/{args.dataset}") 

    out_dir.mkdir(parents=True, exist_ok=True) 

 

    # ============================================================ 

    # 1. SHAP Summary Plot (Beeswarm) 

    # ============================================================ 

    shap.summary_plot( 

        shap_values, 

        X, 

        max_display=20, 

        show=False 

    ) 

    plt.tight_layout() 

    plt.savefig( 

        out_dir / f"fig_shap_summary_{args.model}_cafe_gb_k{args.k}_seed{args.seed}.png", 

        dpi=300, 

        bbox_inches="tight" 

    ) 

    plt.close() 

 

    # ============================================================ 

    # 2. SHAP Bar Plot (Global Importance) 

    # ============================================================ 

    shap.summary_plot( 

        shap_values, 

        X, 

        plot_type="bar", 

        max_display=20, 

        show=False 

    ) 

    plt.tight_layout() 

    plt.savefig( 

        out_dir / f"fig_shap_bar_{args.model}_cafe_gb_k{args.k}_seed{args.seed}.png", 

        dpi=300, 

        bbox_inches="tight" 

    ) 

    plt.close() 

 

    # ============================================================ 

    # 3. Mean |SHAP| Values (Global Importance Table) 

    # ============================================================ 

    mean_abs_shap = ( 

        pd.DataFrame(shap_values, columns=X.columns) 

        .abs() 

        .mean() 

        .sort_values(ascending=False) 

    ) 

 

    mean_abs_shap.to_csv( 

        out_dir / f"shap_global_importance_{args.model}_k{args.k}_seed{args.seed}.csv" 

    ) 

 

    # ============================================================ 

    # 4. Directional SHAP Contribution 

    # ============================================================ 

    directional_shap = pd.DataFrame({ 

        "feature": X.columns, 

        "mean_shap": shap_values.mean(axis=0) 

    }).sort_values("mean_shap", key=abs, ascending=False) 

 

    directional_shap.to_csv( 

        out_dir / f"shap_directionality_{args.model}_k{args.k}_seed{args.seed}.csv", 

        index=False 

    ) 

 

    # ============================================================ 

    # 5. Save Top-20 SHAP Features (for Stability Analysis) 

    # ============================================================ 

    top20_features = mean_abs_shap.head(20).index.tolist() 

 

    pd.Series(top20_features).to_csv( 

        out_dir / f"shap_top20_features_{args.model}_seed{args.seed}.txt", 

        index=False, 

        header=False 

    ) 

 

    # ============================================================ 

    # 6. Optional SHAP Dependence Plot (Top Feature Only) 

    # ============================================================ 

    if args.dependence: 

        top_feature = mean_abs_shap.index[0] 

        shap.dependence_plot( 

            top_feature, 

            shap_values, 

            X, 

            show=False 

        ) 

        plt.tight_layout() 

        plt.savefig( 

            out_dir / f"fig_shap_dependence_{top_feature}_{args.model}_seed{args.seed}.png", 

            dpi=300, 

            bbox_inches="tight" 

        ) 

        plt.close() 

 

    print("[DONE] SHAP explainability analysis completed successfully") 

 

 

# ============================================================ 

# ENTRY POINT 

# ============================================================ 

 

if __name__ == "__main__": 

    main() 