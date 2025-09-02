
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_methods.py
------------------
Compare two methods (baseline vs method) using per-subject metrics and
perform a paired Wilcoxon test. Also produces distribution plots.

Inputs
------
- Two directories containing per-record predictions CSVs or a single combined predictions.csv each.
  Use eval_from_csv.py beforehand to produce per-subject metrics CSV for each method.
- Or, pass the per_subject_metrics.csv files directly.

Outputs
-------
- CSV: comparison_by_subject.csv (acc/κ/macro/weighted/F1 per class for both + deltas)
- Text: wilcoxon.txt (statistics and p-values for each metric)
- Plots: boxplots/violins for deltas

Usage
-----
python compare_methods.py \
  --baseline out_eval_baseline/per_subject_metrics.csv \
  --method   out_eval_tent/per_subject_metrics.csv \
  --out out_compare/baseline_vs_tent
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

METRICS = ["accuracy", "kappa", "macro_f1", "weighted_f1"]
CLASS_ORDER = ["W","N1","N2","N3","REM"]


def load_per_subject(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect a "subject" column
    if "subject" not in df.columns:
        # try common variants
        for c in ["Subject","SUBJECT","sid","record"]:
            if c in df.columns:
                df = df.rename(columns={c: "subject"})
                break
    return df


def paired_stats(bdf: pd.DataFrame, mdf: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(bdf, mdf, on="subject", suffixes=("_base","_meth"))
    rows = []
    for key in METRICS + [f"f1_{c}" for c in CLASS_ORDER if f"f1_{c}" in merged.columns]:
        x = merged[f"{key}_meth"].to_numpy()
        y = merged[f"{key}_base"].to_numpy()
        # compute deltas
        d = x - y
        stat, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided", mode="auto")
        rows.append({"metric": key, "mean_delta": float(np.mean(d)), "median_delta": float(np.median(d)), "wilcoxon_W": float(stat), "p_value": float(p)})
    return pd.DataFrame(rows), merged


def plot_deltas(merged: pd.DataFrame, out_dir: str):
    import seaborn as sns  # only for violin/box convenience; if unavailable, fallback
    os.makedirs(out_dir, exist_ok=True)
    # Prepare long-form for seaborn
    pairs = []
    for key in METRICS:
        delta = merged[f"{key}_meth"] - merged[f"{key}_base"]
        for v in delta:
            pairs.append({"metric": key, "delta": float(v)})
    df_long = pd.DataFrame(pairs)

    # Boxplot
    fig = plt.figure(figsize=(6,4))
    sns.boxplot(data=df_long, x="metric", y="delta")
    plt.title("Per-subject Δ (method - baseline)")
    fig.savefig(os.path.join(out_dir, "box_deltas.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Violin
    fig = plt.figure(figsize=(6,4))
    sns.violinplot(data=df_long, x="metric", y="delta", cut=0)
    plt.title("Per-subject Δ (method - baseline)")
    fig.savefig(os.path.join(out_dir, "violin_deltas.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="per_subject_metrics.csv for baseline")
    ap.add_argument("--method", required=True, help="per_subject_metrics.csv for method")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bdf = load_per_subject(args.baseline)
    mdf = load_per_subject(args.method)

    stats_df, merged = paired_stats(bdf, mdf)
    stats_df.to_csv(os.path.join(args.out, "wilcoxon.csv"), index=False)
    merged.to_csv(os.path.join(args.out, "comparison_by_subject.csv"), index=False)

    try:
        plot_deltas(merged, args.out)
    except Exception as e:
        # seaborn might be missing; fallback simple matplotlib scatter
        import matplotlib.pyplot as plt
        import numpy as np
        deltas = []
        labels = []
        for i, key in enumerate(METRICS):
            d = (merged[f"{key}_meth"] - merged[f"{key}_base"]).to_numpy()
            deltas.append(d)
            labels.append(key)
        fig = plt.figure(figsize=(6,4))
        for i, d in enumerate(deltas):
            x = np.full_like(d, i, dtype=float) + (np.random.rand(len(d))-0.5)*0.1
            plt.scatter(x, d, s=8)
        plt.xticks(range(len(labels)), labels)
        plt.title("Per-subject Δ (method - baseline)")
        fig.savefig(os.path.join(args.out, "scatter_deltas.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] Wrote {args.out}/wilcoxon.csv and plots.")

if __name__ == "__main__":
    main()
