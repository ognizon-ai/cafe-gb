#!/usr/bin/env python3
"""
CAFÉ-GB – Paper 1
Master Experiment Runner (Final, Baseline-Fixed)

Supports:
- CAFÉ-GB feature selection
- Baseline (all features) classification
- Multiple datasets
- Dataset-specific target labels
- Stage-wise execution
- Runtime & memory profiling
- Full logging & fault tolerance
"""

import sys
import argparse
import yaml
import logging
import traceback
from pathlib import Path
from datetime import datetime
import time
import psutil
import os

# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logger():
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"

    logger = logging.getLogger("CAFE-GB")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info("Logger initialized")
    logger.info(f"Log file: {log_file}")
    return logger


logger = setup_logger()


def log(msg, level="info"):
    getattr(logger, level)(msg)


# ============================================================
# UTILITIES
# ============================================================

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def exists(path):
    return Path(path).exists()


def safe_run(stage_name, func, *args):
    log(f"===== START STAGE: {stage_name} =====")
    try:
        func(*args)
        log(f"===== END STAGE: {stage_name} =====")
    except Exception as e:
        log(f"Stage failed: {stage_name}", "error")
        log(str(e), "error")
        logger.error(traceback.format_exc())
        log("Continuing pipeline execution", "warning")


# ============================================================
# PROFILING UTILITIES
# ============================================================

def start_profiling():
    process = psutil.Process(os.getpid())
    process.memory_info()
    return time.time(), process


def stop_profiling(start_time, process):
    runtime = time.time() - start_time
    memory_mb = process.memory_info().rss / (1024 ** 2)
    return runtime, memory_mb


def log_profile(stage, dataset, seed, runtime, memory):
    import pandas as pd

    out = Path("results/tables/runtime_memory.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "Stage": stage,
        "Dataset": dataset,
        "Seed": seed,
        "Runtime_Seconds": round(runtime, 2),
        "Memory_MB": round(memory, 2),
    }

    if out.exists():
        df = pd.read_csv(out)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(out, index=False)


# ============================================================
# STAGES
# ============================================================

def stage_cafe_gb(cfg, dataset, train_path, target_col, seed):
    out = f"fs/cafe_gb/{dataset}/aggregated_importance_seed{seed}.parquet"
    if exists(out):
        log(f"[{dataset}] CAFÉ-GB exists for seed {seed}. Skipping.")
        return

    log(f"[{dataset}] Running CAFÉ-GB (seed={seed})")
    start_time, process = start_profiling()

    import pandas as pd
    from fs.cafe_gb import (
        generate_overlapping_chunks,
        compute_chunk_importance,
        aggregate_importances,
    )

    df = pd.read_parquet(train_path)
    cafe_cfg = load_yaml("config/cafe_gb.yaml")

    chunk_importances = []
    for chunk in generate_overlapping_chunks(
        df,
        cafe_cfg["chunk_size"],
        cafe_cfg["overlap_ratio"],
    ):
        imp = compute_chunk_importance(
            chunk, target_col=target_col, random_state=seed
        )
        chunk_importances.append(imp)

    agg_df = aggregate_importances(chunk_importances)
    ensure_dir(f"fs/cafe_gb/{dataset}")
    agg_df.to_parquet(out)

    runtime, memory = stop_profiling(start_time, process)
    log_profile("CAFÉ-GB", dataset, seed, runtime, memory)

    log(f"[{dataset}] CAFÉ-GB done | Time={runtime:.2f}s | Mem={memory:.2f}MB")


def stage_k_selection(cfg, dataset, train_path, test_path, target_col):
    out = f"results/tables/{dataset}/table3_k_selection.xlsx"
    if exists(out):
        log(f"[{dataset}] k-selection exists. Skipping.")
        return

    log(f"[{dataset}] Running k-selection")

    import pandas as pd
    from stats.k_selection import run_k_selection

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    run_k_selection(
        train_df=train_df,
        test_df=test_df,
        seeds=cfg["seeds"],
        k_values=[50, 100, 200, 300],
        target_col=target_col,
        cafe_dir=f"fs/cafe_gb/{dataset}",
        out_table=out,
        out_fig_dir=f"results/figures/{dataset}",
    )


def stage_classification(cfg, dataset, train_path, test_path, target_col, seed, baseline):
    tag = "baseline" if baseline else "cafe_gb"
    out = f"results/tables/{dataset}/performance_{tag}_seed{seed}.xlsx"
    if exists(out):
        log(f"[{dataset}] Classification ({tag}) seed={seed} exists. Skipping.")
        return

    log(f"[{dataset}] Running classification ({tag}) seed={seed}")
    start_time, process = start_profiling()

    import pandas as pd
    from models.train_classifiers import train_and_evaluate

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if baseline:
        features = None
    else:
        agg_path = f"fs/cafe_gb/{dataset}/aggregated_importance_seed{seed}.parquet"
        agg_df = pd.read_parquet(agg_path)
        features = agg_df.head(cfg["k"])["feature"].tolist()

    ensure_dir(f"results/tables/{dataset}")

    train_and_evaluate(
        train_df=train_df,
        test_df=test_df,
        feature_list=features,   # None → all features
        target_col=target_col,
        seed=seed,
        dataset=dataset,
        output_path=out,
    )

    runtime, memory = stop_profiling(start_time, process)
    log_profile("Classification", dataset, seed, runtime, memory)

    log(f"[{dataset}] Classification ({tag}) done | Time={runtime:.2f}s | Mem={memory:.2f}MB")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser("CAFÉ-GB Master Runner")
    parser.add_argument("--stage", choices=["all", "cafe", "k", "classify"], default="all")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--baseline", action="store_true", help="Run baseline (all features)")
    args = parser.parse_args()

    log("Starting CAFÉ-GB experiment pipeline")

    cfg = {
        "seeds": load_yaml("config/seeds.yaml")["seeds"],
        "k": load_yaml("config/k.yaml")["k"],
    }

    data_cfg = load_yaml("config/data.yaml")

    for dataset, meta in data_cfg["datasets"].items():
        if args.dataset != "all" and dataset != args.dataset:
            continue

        train_path = meta["train"]
        test_path = meta["test"]
        target_col = meta["target_column"]

        log(f"=== DATASET: {dataset.upper()} | Target: {target_col} ===")

        if args.stage in ("all", "cafe"):
            for seed in cfg["seeds"]:
                safe_run(
                    f"CAFÉ-GB [{dataset}] seed={seed}",
                    stage_cafe_gb,
                    cfg, dataset, train_path, target_col, seed
                )

        if args.stage in ("all", "k"):
            safe_run(
                f"k-selection [{dataset}]",
                stage_k_selection,
                cfg, dataset, train_path, test_path, target_col
            )

        if args.stage in ("all", "classify"):
            for seed in cfg["seeds"]:
                safe_run(
                    f"Classification [{dataset}] seed={seed}",
                    stage_classification,
                    cfg, dataset, train_path, test_path, target_col, seed, args.baseline
                )

    log("CAFÉ-GB experiment pipeline finished")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.critical(traceback.format_exc())
        sys.exit(1)
