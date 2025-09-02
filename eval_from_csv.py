
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_from_csv.py
----------------
Flexible evaluator for sleep staging predictions in CSV(s).

Inputs
------
- One or more CSV files via --csv or --glob.
- CSVs should contain (or be mappable to) columns for:
    subject_id, y_true, y_pred
  Optional: probability columns for each class (e.g., proba_W, proba_N1, ...).

What it computes
----------------
- Overall: accuracy, Cohen's kappa, macro F1, weighted F1
- Per-class F1 (W, N1, N2, N3, REM) if present
- Per-subject metrics and paired comparisons (baseline vs method if --compare is used later)
- Confusion matrix (counts and normalized by true)
- (Optional) Calibration: ECE & reliability curve if probabilities are provided

Outputs
-------
- CSVs:
    out/overall_metrics.csv
    out/per_subject_metrics.csv
    out/per_class_f1.csv
- Figures:
    out/confusion_matrix.png
    out/reliability_curve.png (if probs available)
- LaTeX:
    out/overall_table.tex
    out/per_class_table.tex

Usage
-----
python eval_from_csv.py --glob "outputs/*.csv" --out out_eval

Notes
-----
- If your column names differ, pass explicit flags:
    --col-true stage_true --col-pred stage_pred --col-subject subject
- If your labels are numeric, provide a mapping file or rely on the script to label them as class_0..N.
"""

import argparse
import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
from collections import defaultdict


DEFAULT_CLASS_ORDER = ["W","N1","N2","N3","REM"]

CANDIDATE_COLS = {
    "subject": ["subject","subject_id","rec","record","file","record_id","pid","sid"],
    "true":    ["y_true","true","label_true","stage_true","target","label"],
    "pred":    ["y_pred","pred","label_pred","stage_pred","prediction"],
}


def infer_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # Also try case-insensitive match
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def find_prob_columns(df: pd.DataFrame, classes: List[str]) -> Dict[str,str]:
    # Try exact matches like proba_W, p_W, prob_W; also index-based like proba_0..
    mapping = {}
    lower_cols = {c.lower(): c for c in df.columns}
    # name-based
    for cls in classes:
        for prefix in ["proba_","prob_","p_"]:
            key = f"{prefix}{cls}".lower()
            if key in lower_cols:
                mapping[cls] = lower_cols[key]
                break
    # index-based fallback
    if not mapping and all(f"proba_{i}" in lower_cols for i in range(len(classes))):
        for i, cls in enumerate(classes):
            mapping[cls] = lower_cols[f"proba_{i}"]
    return mapping


def labels_to_ordered_classes(y: pd.Series) -> List[str]:
    # Map to known order if possible, else keep discovery order
    uniq = list(pd.unique(y.dropna()))
    uniq_str = [str(v) for v in uniq]
    # If they are subset of default order, sort by that
    if all(u in DEFAULT_CLASS_ORDER for u in uniq_str):
        order = [c for c in DEFAULT_CLASS_ORDER if c in uniq_str]
    else:
        order = uniq_str
    return order


def ensure_class_dtype(y: pd.Series, classes: List[str]) -> pd.Series:
    # Convert to categorical with fixed ordering
    return pd.Categorical(y.astype(str), categories=classes, ordered=True)


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], classes: List[str]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    # Handle labels mapping for kappa and F1
    y_true_idx = pd.Categorical(y_true, categories=classes, ordered=True).codes
    y_pred_idx = pd.Categorical(y_pred, categories=classes, ordered=True).codes
    kappa = cohen_kappa_score(y_true_idx, y_pred_idx, weights=None)
    macro = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    weighted = f1_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)
    per_class = f1_score(y_true_idx, y_pred_idx, average=None, labels=list(range(len(classes))), zero_division=0)
    out = {
        "accuracy": acc,
        "kappa": kappa,
        "macro_f1": macro,
        "weighted_f1": weighted,
    }
    for cls, f in zip(classes, per_class):
        out[f"f1_{cls}"] = f
    return out


def reliability_diagram(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15):
    # Multi-class ECE with max-prob
    confidences = proba.max(axis=1)
    preds = proba.argmax(axis=1)
    correct = (preds == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bins) - 1
    ece = 0.0
    xs, accs, confs, counts = [], [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            xs.append((bins[b]+bins[b+1])/2)
            accs.append(np.nan)
            confs.append(np.nan)
            counts.append(0)
            continue
        acc_b = correct[mask].mean()
        conf_b = confidences[mask].mean()
        w = mask.mean()
        ece += w * abs(acc_b - conf_b)
        xs.append((bins[b]+bins[b+1])/2)
        accs.append(acc_b)
        confs.append(conf_b)
        counts.append(int(mask.sum()))
    return np.array(xs), np.array(accs), np.array(confs), np.array(counts), float(ece)


def plot_confusion(cm: np.ndarray, classes: List[str], out_png: str):
    fig = plt.figure(figsize=(6,5))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (normalized by true)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def df_to_latex(df: pd.DataFrame) -> str:
    header = " & ".join(df.columns) + " \\\\"
    lines = [header, "\\hline"]
    for _, r in df.iterrows():
        row = " & ".join(str(r[c]) for c in df.columns) + " \\\\"
        lines.append(row)
    return "\\begin{tabular}{%s}\n\\hline\n%s\n\\hline\n\\end{tabular}\n" % (
        "l" + "r"*(len(df.columns)-1),
        "\n".join(lines)
    )


def main():
    ap = argparse.ArgumentParser()
    grp_in = ap.add_mutually_exclusive_group(required=True)
    grp_in.add_argument("--csv", nargs="+", help="One or more CSV files")
    grp_in.add_argument("--glob", help="Glob pattern for CSVs, e.g., 'outputs/*.csv'")
    ap.add_argument("--out", default="out_eval", help="Output directory")
    ap.add_argument("--col-subject", help="Subject column name override")
    ap.add_argument("--col-true", help="True label column name override")
    ap.add_argument("--col-pred", help="Pred label column name override")
    ap.add_argument("--classes", nargs="+", help="Explicit class order, e.g., W N1 N2 N3 REM")
    args = ap.parse_args()

    csvs = args.csv if args.csv else glob.glob(args.glob)
    assert csvs, "No CSV files matched."

    os.makedirs(args.out, exist_ok=True)

    frames = []
    for path in csvs:
        df = pd.read_csv(path)
        df["__src__"] = os.path.basename(path)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Columns
    col_subject = args.col_subject or infer_col(df, CANDIDATE_COLS["subject"])
    col_true    = args.col_true    or infer_col(df, CANDIDATE_COLS["true"])
    col_pred    = args.col_pred    or infer_col(df, CANDIDATE_COLS["pred"])
    if not all([col_subject, col_true, col_pred]):
        raise SystemExit(f"Could not infer columns. Found subject={col_subject}, true={col_true}, pred={col_pred}. Use overrides.")

    # Classes
    if args.classes:
        classes = args.classes
    else:
        classes = labels_to_ordered_classes(df[col_true])

    y_true = ensure_class_dtype(df[col_true], classes)
    y_pred = ensure_class_dtype(df[col_pred], classes)

    # Overall metrics
    overall = compute_metrics(y_true, y_pred, classes)
    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(os.path.join(args.out, "overall_metrics.csv"), index=False)

    # Per-class table
    per_class = {k:v for k,v in overall.items() if k.startswith("f1_")}
    per_class_df = pd.DataFrame([per_class])
    per_class_df.insert(0, "metric", ["F1_per_class"])
    per_class_df.to_csv(os.path.join(args.out, "per_class_f1.csv"), index=False)

    # Per-subject metrics
    rows = []
    for sid, g in df.groupby(col_subject):
        yt = ensure_class_dtype(g[col_true], classes)
        yp = ensure_class_dtype(g[col_pred], classes)
        m = compute_metrics(yt, yp, classes)
        m["subject"] = sid
        rows.append(m)
    per_subj_df = pd.DataFrame(rows)
    per_subj_df.to_csv(os.path.join(args.out, "per_subject_metrics.csv"), index=False)

    # Confusion matrix (normalized by true)
    y_true_idx = pd.Categorical(y_true, categories=classes, ordered=True).codes
    y_pred_idx = pd.Categorical(y_pred, categories=classes, ordered=True).codes
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(classes))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1.0)
    np.savetxt(os.path.join(args.out, "confusion_matrix_counts.csv"), cm, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(args.out, "confusion_matrix_normalized.csv"), cm_norm, fmt="%.6f", delimiter=",")
    plot_confusion(cm_norm, classes, os.path.join(args.out, "confusion_matrix.png"))

    # LaTeX
    overall_tex = df_to_latex(overall_df.round(4))
    with open(os.path.join(args.out, "overall_table.tex"), "w") as f:
        f.write(overall_tex)

    per_class_tex = df_to_latex(pd.DataFrame([ {c: per_class.get(f"f1_{c}", np.nan) for c in classes} ]).round(4))
    with open(os.path.join(args.out, "per_class_table.tex"), "w") as f:
        f.write(per_class_tex)

    # Optional reliability if probabilities exist
    prob_cols = find_prob_columns(df, classes)
    if prob_cols and len(prob_cols) == len(classes):
        proba = df[list(prob_cols.values())].to_numpy()
        xs, accs, confs, counts, ece = reliability_diagram(y_true_idx, proba)
        fig = plt.figure(figsize=(5,4))
        plt.plot([0,1],[0,1], linestyle="--")
        plt.plot(xs, accs, marker="o", label="Accuracy")
        plt.plot(xs, confs, marker="s", label="Confidence")
        plt.xlabel("Confidence bin")
        plt.ylabel("Value")
        plt.title(f"Reliability (ECE={ece:.3f})")
        plt.legend()
        fig.savefig(os.path.join(args.out, "reliability_curve.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"[OK] Wrote outputs in: {args.out}")
    print(f"Columns used: subject={col_subject}, true={col_true}, pred={col_pred}")
    print(f"Classes: {classes}")


if __name__ == "__main__":
    main()
