
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_plots.py
-------------
Plot helper for:
- Confusion matrix (normalized)
- Metric vs hyperparameter curves (ablation sweeps)
- Per-subject distributions from per_subject_metrics.csv
"""
import argparse, os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def plot_metric_vs_param(manifest_json: str, metric_csv: str, metric: str, out_png: str):
    """Plot metric vs sweep parameter from run_ablation sweep manifest + eval CSV per run (optional)."""
    data = json.load(open(manifest_json))
    vals = []
    params = []
    for run_csv in data["runs"]:
        # Expect each run has out_eval/overall_metrics.csv sidecar (run via eval_from_csv.py)
        root = os.path.dirname(run_csv)
        eval_csv = os.path.join(root, "out_eval", "overall_metrics.csv")
        if not os.path.exists(eval_csv):
            continue
        df = pd.read_csv(eval_csv)
        if metric in df.columns:
            vals.append(float(df[metric].iloc[0]))
            params.append(os.path.basename(root))
    if not vals:
        return
    fig = plt.figure(figsize=(6,4))
    plt.plot(range(len(vals)), vals, marker="o")
    plt.xticks(range(len(params)), params, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} vs sweep")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def subj_distribution(per_subject_csv: str, out_png: str, metric: str = "accuracy"):
    df = pd.read_csv(per_subject_csv)
    fig = plt.figure(figsize=(6,4))
    plt.boxplot(df[metric].values)
    plt.title(f"Per-subject {metric} distribution")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="sweep_manifest.json from run_ablation.py")
    ap.add_argument("--per-subject", help="per_subject_metrics.csv to plot distribution")
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.manifest:
        plot_metric_vs_param(args.manifest, None, args.metric, os.path.join(args.out, f"sweep_{args.metric}.png"))
    if args.per_subject:
        subj_distribution(args.per_subject, os.path.join(args.out, f"subj_{args.metric}.png"), metric=args.metric)

if __name__ == "__main__":
    main()
