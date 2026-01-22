#!/usr/bin/env python3
"""
Aggregate classification results across seeds.
Produces mean ± std tables for the paper.

Compatible with:
- performance_baseline_seed*.xlsx
- performance_cafe_gb_seed*.xlsx

Outputs:
- table_classification_results.xlsx
- table_classification_results.tex
"""

import argparse
from pathlib import Path
import pandas as pd
import re


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate classification results across seeds"
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g., bodmas, cicandmal2020)",
    )
    return p.parse_args()


def extract_seed(path: Path) -> int:
    """
    Extract seed value from filename: *_seedXX.xlsx
    """
    m = re.search(r"seed(\d+)", path.stem)
    if m is None:
        raise ValueError(f"Cannot extract seed from filename: {path.name}")
    return int(m.group(1))


def load_with_method(files, method_name):
    """
    Load Excel files and attach Method and Seed metadata.
    """
    dfs = []
    for f in files:
        seed = extract_seed(f)
        df = pd.read_excel(f)

        if "Classifier" not in df.columns:
            raise ValueError(f"'Classifier' column missing in {f}")

        df["Seed"] = seed
        df["Method"] = method_name
        dfs.append(df)

    return dfs


def main():
    args = parse_args()
    table_dir = Path(f"results/tables/{args.dataset}")

    if not table_dir.exists():
        raise FileNotFoundError(f"Directory not found: {table_dir}")

    baseline_files = sorted(table_dir.glob("performance_baseline_seed*.xlsx"))
    cafe_files = sorted(table_dir.glob("performance_cafe_gb_seed*.xlsx"))

    if not baseline_files or not cafe_files:
        raise FileNotFoundError(
            f"Expected both baseline and CAFÉ-GB result files in {table_dir}\n"
            f"Found baseline={len(baseline_files)}, cafe_gb={len(cafe_files)}"
        )

    dfs = []
    dfs.extend(load_with_method(baseline_files, "Baseline"))
    dfs.extend(load_with_method(cafe_files, "CAFÉ-GB"))

    df = pd.concat(dfs, ignore_index=True)

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "ROC_AUC",
        "PR_AUC",
    ]

    missing = set(metrics) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected metric columns: {missing}")

    # Aggregate per Method × Classifier
    grouped = df.groupby(["Method", "Classifier"])[metrics]

    mean = grouped.mean()
    std = grouped.std(ddof=1)

    # Format mean ± std
    out = (
        mean.round(4).astype(str)
        + " $\\pm$ "
        + std.round(4).astype(str)
    )
    out.reset_index(inplace=True)

    # Enforce column order
    out = out[
        ["Method", "Classifier"]
        + metrics
    ]

    # Output paths
    out_xlsx = table_dir / "table_classification_results.xlsx"
    out_tex = table_dir / "table_classification_results.tex"

    out.to_excel(out_xlsx, index=False)

    out.to_latex(
        out_tex,
        index=False,
        escape=False,
        column_format="ll" + "c" * len(metrics),
    )

    print(f"[DONE] Aggregated results saved to:")
    print(f"  - {out_xlsx}")
    print(f"  - {out_tex}")


if __name__ == "__main__":
    main()
