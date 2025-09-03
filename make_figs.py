#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NeurIPS-ready figure generator for Sleep-EDF TTA results.

- Auto-detects class labels (4- or 5-class) from confusion CSVs.
- Robustly parses per-class F1 (wide/long) and per-subject metrics (varied column names).
- Generates:
    * Row-normalized confusion matrices (baseline and side-by-side comparisons)
    * Stage-wise F1 grouped bars
    * Per-subject ΔAccuracy / Δκ vs baseline
    * Stage distribution histogram (from baseline predictions.csv)
    * Hypnogram overlay (GT + predictions) for one subject
- Optionally:
    * Entropy gate & EMA reset plot if you provide --tta_log
    * Latency & memory bars if you provide --profiling

Examples
--------
python make_figs.py \
  --method no_adapt out_eval/val \
  --method bn_only  runs/eval_bn_only_val_20250829-151512/out_eval \
  --method tent_160400 runs/eval_tent_val_20250829-160400/out_eval \
  --baseline no_adapt \
  --outdir figs/val
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Defaults / palettes ----
STAGES_DEFAULT = ["W", "N1", "N2", "N3", "REM"]  # used only as a preference/order
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


# ----------------- IO helpers -----------------
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not Path(path).exists():
        print(f"[WARN] Missing file: {path}", file=sys.stderr)
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)
        return None


def ensure_outdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# ----------------- Label detection & confusion loading -----------------
def _read_conf_df(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """Read a confusion CSV and return (df, row_labels). If the first column is labels, set it as index."""
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    if df.empty:
        return None, None

    first_col = df.columns[0]
    # Heuristic: if first column looks like a label column (strings or common names), use as index
    if df[first_col].dtype == object or str(first_col).lower() in ("", "unnamed: 0", "label", "true", "stage", "class"):
        try:
            df = df.set_index(first_col)
        except Exception:
            pass

    # Extract 1-level string labels if possible
    if df.index.nlevels == 1:
        labels = [str(x) for x in df.index.astype(str).tolist()]
    else:
        labels = None
    return df, labels


def _detect_labels(bundle: dict) -> List[str]:
    """Detect label set (order) for a method, preferring confusion counts -> normalized -> per-class F1 -> default."""
    df, idx = _read_conf_df(Path(bundle["dir"]) / "confusion_matrix_counts.csv")
    if idx:
        return idx
    df, idx = _read_conf_df(Path(bundle["dir"]) / "confusion_matrix_normalized.csv")
    if idx:
        return idx

    pc = bundle.get("per_class_f1")
    if pc is not None and not pc.empty:
        lower = {c.lower(): c for c in pc.columns}
        for key in ("class", "stage", "label"):
            if key in lower:
                labs = [str(x) for x in pc[lower[key]].astype(str).tolist()]
                if 3 <= len(labs) <= 6:
                    return labs
        # or index-as-labels
        if pc.index.name or pc.index.dtype == object:
            labs = [str(x) for x in pc.index.tolist()]
            if 3 <= len(labs) <= 6:
                return labs

    return STAGES_DEFAULT


def _reindex_square(df: Optional[pd.DataFrame], labels: List[str]) -> Optional[pd.DataFrame]:
    """Return confusion df reindexed to labels on both axes (fill missing with 0)."""
    if df is None:
        return None
    df2 = df.copy()

    # If columns are numeric 0..N-1 or otherwise not labels, try to coerce to match index
    if len(df2.columns) == len(df2.index):
        # If columns don't match index, set columns to index order (common export format)
        if list(map(str, df2.columns)) != list(map(str, df2.index)):
            df2.columns = list(df2.index)

    # Now reindex to requested labels
    df2 = df2.reindex(index=labels, columns=labels, fill_value=0.0)
    return df2


# ----------------- Method bundle loader -----------------
def load_method_bundle(name: str, dirpath: str) -> dict:
    b = {
        "name": name,
        "dir": dirpath,
        "per_subject": safe_read_csv(Path(dirpath) / "per_subject_metrics.csv"),
        "per_class_f1": safe_read_csv(Path(dirpath) / "per_class_f1.csv"),
        "overall": safe_read_csv(Path(dirpath) / "overall_metrics.csv"),
        "predictions": safe_read_csv(Path(dirpath) / "predictions.csv"),
        "conf_counts_raw": safe_read_csv(Path(dirpath) / "confusion_matrix_counts.csv"),
        "conf_norm_raw": safe_read_csv(Path(dirpath) / "confusion_matrix_normalized.csv"),
    }
    # label set per method
    b["labels"] = _detect_labels(b)
    # square, ordered confusions
    cc, _ = _read_conf_df(Path(dirpath) / "confusion_matrix_counts.csv")
    b["conf_counts"] = _reindex_square(cc, b["labels"])
    cn, _ = _read_conf_df(Path(dirpath) / "confusion_matrix_normalized.csv")
    b["conf_norm"] = _reindex_square(cn, b["labels"])
    return b


# ----------------- Plot: confusion matrices -----------------
def plot_confusion(ax, conf: pd.DataFrame, labels: List[str], title: str):
    mat = conf.values.astype(float)
    im = ax.imshow(mat, aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # annotate cells with values
    # if already normalized: numbers 0..1; if counts: also ok to annotate raw counts
    vmax = np.nanmax(mat) if mat.size else 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = f"{val:.2f}" if vmax <= 1.5 else f"{int(val)}"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    fontsize=8,
                    color=("white" if (vmax <= 1.5 and val > 0.6) else "black"))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def make_confusion_figs(baseline: dict, others: List[dict], outdir: str):
    bl = baseline["conf_counts"]
    if bl is None or bl.empty:
        print("[WARN] Cannot plot confusions (baseline counts missing).", file=sys.stderr)
        return

    # Row-normalize baseline
    bl_row = bl.div(bl.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Single baseline plot
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.3))
    plot_confusion(ax, bl_row, baseline["labels"], f"{baseline['name']} (row-normalized)")
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"confmat_{baseline['name']}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Side-by-side comparisons (align to common labels)
    for m in others:
        mc = m["conf_counts"]
        if mc is None or mc.empty:
            continue
        common = [lab for lab in baseline["labels"] if lab in m["labels"]]
        if len(common) < 2:
            # fall back to baseline labels
            common = baseline["labels"]

        bl_aligned = bl.loc[common, common]
        mc_aligned = mc.loc[common, common]

        bl_row = bl_aligned.div(bl_aligned.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        mc_row = mc_aligned.div(mc_aligned.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3))
        plot_confusion(axes[0], bl_row, common, f"{baseline['name']}")
        plot_confusion(axes[1], mc_row, common, f"{m['name']}")
        fig.suptitle("Row-normalized confusion matrices")
        fig.tight_layout()
        fig.savefig(Path(outdir) / f"confmat_{baseline['name']}_vs_{m['name']}.pdf", bbox_inches="tight")
        plt.close(fig)


# ----------------- Plot: stage-wise F1 bars -----------------
def _parse_per_class_f1(df: Optional[pd.DataFrame]) -> Optional[Tuple[List[str], List[float]]]:
    if df is None or df.empty:
        return None

    lower = {c.lower(): c for c in df.columns}

    # Long form: class/stage/label + f1 (or f1-score/f1_score)
    for key in ("class", "stage", "label"):
        if key in lower:
            f1col = lower.get("f1") or lower.get("f1-score") or lower.get("f1_score")
            if f1col:
                labs = [str(x) for x in df[lower[key]].astype(str).tolist()]
                vals = [float(x) for x in df[f1col].astype(float).tolist()]
                return labs, vals

    # Wide form: columns named exactly as stages
    cols = [c for c in df.columns if c.upper() in STAGES_DEFAULT]
    if len(cols) >= 3:
        labs = [c.upper() for c in cols]
        vals = [float(df[c].astype(float).mean()) for c in cols]
        return labs, vals

    # Index-as-labels + 'f1' column
    if "f1" in lower:
        labs = [str(x) for x in df.index.tolist()]
        if 3 <= len(labs) <= 6:
            vals = [float(x) for x in df[lower["f1"]].astype(float).tolist()]
            return labs, vals

    return None


def make_stage_f1_bars(bundles: List[dict], outdir: str):
    series = []
    names = []
    label_union = []
    for b in bundles:
        parsed = _parse_per_class_f1(b.get("per_class_f1"))
        if parsed is None:
            print(f"[WARN] per_class_f1.csv unexpected format for {b['name']}", file=sys.stderr)
            continue
        labs, vals = parsed
        idx = [l.upper() for l in labs]
        series.append(pd.Series(vals, index=idx, name=b["name"]))
        names.append(b["name"])
        label_union.extend(idx)

    if not series:
        print("[WARN] No per_class_f1 available; skipping F1 bars.", file=sys.stderr)
        return

    labels = [l for l in STAGES_DEFAULT if l in label_union] or sorted(set(label_union))
    mat = pd.concat(series, axis=1).reindex(labels).T  # methods x classes

    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels) + 2), 4))
    for i, name in enumerate(mat.index):
        ax.bar(x + i * width, mat.loc[name].values, width=width, label=name)
    ax.set_xticks(x + (len(series) - 1) * width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 score")
    ax.set_title("Stage-wise F1 by method")
    ax.legend(ncols=min(len(series), 3), fontsize=8)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "stage_f1_bars.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------- Plot: per-subject ΔAccuracy / Δκ -----------------
def _metric_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    cols = {c.lower(): c for c in df.columns}
    subj = cols.get("subject") or cols.get("subject_id") or cols.get("id") or cols.get("record") or cols.get("rec")
    acc = cols.get("accuracy") or cols.get("overall_accuracy") or cols.get("acc")
    kap = (cols.get("kappa") or cols.get("cohen_kappa") or cols.get("cohen's kappa")
           or cols.get("cohen_kappa_score") or cols.get("cohen’s kappa"))
    return subj, acc, kap


def make_delta_plots(baseline: dict, others: List[dict], outdir: str):
    bl = baseline["per_subject"]
    if bl is None or bl.empty:
        print("[WARN] Baseline per_subject_metrics.csv missing; skipping delta plots.", file=sys.stderr)
        return

    subj_col, bl_acc, bl_kap = _metric_cols(bl)
    if subj_col is None:
        print("[WARN] per_subject_metrics.csv must contain a subject column.", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    any_acc, any_kap = False, False

    for m in others:
        df = m["per_subject"]
        if df is None or df.empty:
            continue
        m_subj, m_acc, m_kap = _metric_cols(df)
        if m_subj is None:
            continue
        merged = bl.merge(df, left_on=subj_col, right_on=m_subj, suffixes=("_bl", "_m"))

        if bl_acc and m_acc and bl_acc in merged.columns and m_acc in merged.columns:
            dacc = merged[m_acc] - merged[bl_acc]
            axes[0].plot(np.arange(len(dacc)), dacc.values, marker="o", linestyle="", alpha=0.7, label=m["name"])
            any_acc = True
        if bl_kap and m_kap and bl_kap in merged.columns and m_kap in merged.columns:
            dkap = merged[m_kap] - merged[bl_kap]
            axes[1].plot(np.arange(len(dkap)), dkap.values, marker="o", linestyle="", alpha=0.7, label=m["name"])
            any_kap = True

    if any_acc:
        axes[0].axhline(0, color="gray", linewidth=1)
        axes[0].set_title("Per-subject ΔAccuracy vs baseline")
        axes[0].set_xlabel("Subject index")
        axes[0].set_ylabel("ΔAccuracy")
        axes[0].legend(fontsize=8)
    else:
        axes[0].set_visible(False)

    if any_kap:
        axes[1].axhline(0, color="gray", linewidth=1)
        axes[1].set_title("Per-subject Δκ vs baseline")
        axes[1].set_xlabel("Subject index")
        axes[1].set_ylabel("Δκ")
        axes[1].legend(fontsize=8)
    else:
        axes[1].set_visible(False)

    if any_acc or any_kap:
        fig.tight_layout()
        fig.savefig(Path(outdir) / f"delta_acc_kappa_vs_{baseline['name']}.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------- Plot: stage distribution histogram -----------------
def _detect_true_col(pred: pd.DataFrame) -> Optional[str]:
    candidates = ["y_true", "true", "label", "stage_true", "gt", "target", "truth"]
    lower = {c.lower(): c for c in pred.columns}
    for c in candidates:
        if c in lower:
            return lower[c]
    # Sometimes GT is named 'stage' when preds are separate
    if "stage" in lower and "y_pred" in lower:
        return lower["stage"]
    return None


def make_stage_histogram(bundle: dict, outdir: str):
    pred = bundle.get("predictions")
    if pred is None or pred.empty:
        print("[WARN] Missing predictions.csv; skipping stage histogram.", file=sys.stderr)
        return

    true_col = _detect_true_col(pred)
    if true_col is None:
        print("[WARN] predictions.csv needs a ground truth column (e.g., y_true).", file=sys.stderr)
        return

    y = pred[true_col].astype(str).str.strip().str.upper().replace({"N4": "N3"})
    labels = [l for l in STAGES_DEFAULT if l in y.unique().tolist()] or sorted(y.unique().tolist())
    counts = y.value_counts().reindex(labels).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(np.arange(len(labels)), counts.values)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Epoch count")
    ax.set_title("Stage distribution (ground truth)")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "stage_hist.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------- Plot: hypnogram overlay -----------------
def _detect_subject_col(df: pd.DataFrame) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for key in ("subject", "subject_id", "subj", "record", "rec", "file", "edfx_record"):
        if key in lower:
            return lower[key]
    return None


def _detect_pred_col(df: pd.DataFrame) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for key in ("y_pred", "pred", "prediction", "stage_pred", "pred_label"):
        if key in lower:
            return lower[key]
    return None


def _encode_series_to_order(series: pd.Series, order: List[str]) -> np.ndarray:
    s = series.astype(str).str.strip().str.upper().replace({"N4": "N3"})
    mapping = {lab: i for i, lab in enumerate(order)}
    return s.map(mapping).fillna(-1).astype(int).values


def choose_subject_for_hypnogram(baseline: dict, others: List[dict], prefer_subject: Optional[str] = None) -> Optional[str]:
    if prefer_subject is not None:
        return prefer_subject

    # Try "best gain in κ"
    try:
        bl = baseline["per_subject"]
        if bl is not None and not bl.empty:
            subj_col, bl_acc, bl_kap = _metric_cols(bl)
            if subj_col and bl_kap:
                best_subj, best_gain = None, -np.inf
                for m in others:
                    df = m["per_subject"]
                    if df is None or df.empty:
                        continue
                    m_subj, m_acc, m_kap = _metric_cols(df)
                    if not (m_subj and m_kap):
                        continue
                    merged = bl.merge(df, left_on=subj_col, right_on=m_subj, suffixes=("_bl", "_m"))
                    if bl_kap in merged.columns and m_kap in merged.columns:
                        merged["dK"] = merged[m_kap] - merged[bl_kap]
                        idx = merged["dK"].idxmax()
                        if pd.notna(idx):
                            if merged.loc[idx, "dK"] > best_gain:
                                best_gain = merged.loc[idx, "dK"]
                                best_subj = str(merged.loc[idx, subj_col])
                if best_subj is not None:
                    return best_subj
    except Exception:
        pass

    # Fallback: first subject in baseline predictions
    pred = baseline.get("predictions")
    if pred is not None and not pred.empty:
        subj_col = _detect_subject_col(pred)
        if subj_col and len(pred[subj_col]) > 0:
            return str(pred[subj_col].iloc[0])
    return None


def plot_hypnogram_overlay(bundles: List[dict], baseline: dict, outdir: str, subject: Optional[str] = None):
    base_pred = baseline.get("predictions")
    if base_pred is None or base_pred.empty:
        print("[WARN] Baseline predictions.csv missing; skipping hypnogram.", file=sys.stderr)
        return

    subj_col = _detect_subject_col(base_pred)
    if subj_col is None:
        print("[WARN] predictions.csv needs a subject column.", file=sys.stderr)
        return

    subject = choose_subject_for_hypnogram(baseline, [b for b in bundles if b is not baseline], prefer_subject=subject)
    if subject is None:
        print("[WARN] Could not select a subject for hypnogram.", file=sys.stderr)
        return

    # Filter baseline rows for this subject
    sdf = base_pred[base_pred[subj_col].astype(str) == str(subject)].copy()
    if sdf.empty:
        print(f"[WARN] Subject {subject} not found in baseline predictions.", file=sys.stderr)
        return

    true_col = _detect_true_col(sdf)
    if true_col is None:
        print("[WARN] predictions.csv needs a ground truth column (e.g., y_true).", file=sys.stderr)
        return

    # Determine stage order from GT present
    gt_vals = sdf[true_col].astype(str).str.upper().replace({"N4": "N3"})
    order = [l for l in STAGES_DEFAULT if l in gt_vals.unique().tolist()] or sorted(gt_vals.unique().tolist())

    T = len(sdf)
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    # Ground truth
    gt = _encode_series_to_order(sdf[true_col], order)
    ax.step(x, gt, where="post", linewidth=2, label="Ground truth")

    # Each method's predictions
    for b in bundles:
        p = b.get("predictions")
        if p is None or p.empty:
            continue
        s_col = _detect_subject_col(p)
        y_pred_col = _detect_pred_col(p)
        if s_col is None or y_pred_col is None:
            continue
        sdf_b = p[p[s_col].astype(str) == str(subject)]
        if sdf_b.empty:
            continue
        yp = _encode_series_to_order(sdf_b[y_pred_col], order)
        n = min(len(yp), T)
        ax.step(x[:n], yp[:n] + 0.02 * (np.random.rand(n) - 0.5), where="post",
                alpha=0.9, linewidth=1.4, label=b["name"])

    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Epoch (30 s each)")
    ax.set_title(f"Hypnogram overlay — subject {subject}")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(ncols=4, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"hypnogram_compare_subject_{subject}.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------- Optional plots: entropy/EMA & latency/memory -----------------
def maybe_plot_entropy_gate(log_path: str, outdir: str):
    if not Path(log_path).exists():
        return
    df = pd.read_csv(log_path)
    if df.empty or "batch_entropy" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["batch_entropy"].values, linewidth=1.2, label="Batch entropy")
    if "ema_entropy" in df.columns:
        ax.plot(df["ema_entropy"].values, linewidth=1.2, linestyle="--", label="EMA entropy")
    if "gated" in df.columns:
        gated = df["gated"].astype(int).values
        ax.fill_between(np.arange(len(gated)), 0, 1, where=gated > 0,
                        transform=ax.get_xaxis_transform(), alpha=0.15, label="Updates enabled")
    if "reset_event" in df.columns:
        resets = np.where(df["reset_event"].astype(int).values > 0)[0]
        for r in resets:
            ax.axvline(r, color="k", linewidth=1, alpha=0.4)
    ax.set_title("Entropy gate & EMA reset (example)")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Entropy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "entropy_gate_ema_reset.pdf", bbox_inches="tight")
    plt.close(fig)


def maybe_plot_latency_memory(prof_path: str, outdir: str):
    if not Path(prof_path).exists():
        return
    df = pd.read_csv(prof_path)
    if df.empty or not {"method", "latency_ms", "memory_mb"} <= set(df.columns):
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    d1 = df.sort_values("method")
    # latency
    axes[0].bar(np.arange(len(d1)), d1["latency_ms"].values)
    axes[0].set_xticks(np.arange(len(d1)))
    axes[0].set_xticklabels(d1["method"].values, rotation=20)
    axes[0].set_ylabel("ms/epoch")
    axes[0].set_title("Latency")
    # memory
    axes[1].bar(np.arange(len(d1)), d1["memory_mb"].values)
    axes[1].set_xticks(np.arange(len(d1)))
    axes[1].set_xticklabels(d1["method"].values, rotation=20)
    axes[1].set_ylabel("MB")
    axes[1].set_title("Peak memory")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "latency_memory.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Make NeurIPS-ready figures from out_eval/ results.")
    p.add_argument("--method", nargs=2, action="append", metavar=("NAME", "DIR"),
                   help="Method name and its out_eval directory (repeat for multiple).")
    p.add_argument("--baseline", type=str, required=True, help="Name of the baseline method to compare against.")
    p.add_argument("--outdir", type=str, default="figs", help="Output directory for figures (PDF).")
    p.add_argument("--subject", type=str, default=None, help="Subject ID to use for hypnogram overlay.")
    p.add_argument("--tta_log", type=str, default=None, help="Optional: CSV with batch_entropy/ema_entropy/gated/reset_event.")
    p.add_argument("--profiling", type=str, default=None, help="Optional: CSV with method,latency_ms,memory_mb.")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    if not args.method:
        print("Provide at least one --method NAME DIR", file=sys.stderr)
        sys.exit(1)

    bundles = []
    baseline = None
    for name, d in args.method:
        b = load_method_bundle(name, d)
        bundles.append(b)
        if name == args.baseline:
            baseline = b

    if baseline is None:
        print(f"[ERR] Baseline '{args.baseline}' not among methods.", file=sys.stderr)
        sys.exit(1)

    others = [b for b in bundles if b is not baseline]

    # 1) Confusion matrices (row-normalized)
    make_confusion_figs(baseline, others, args.outdir)

    # 2) Stage-wise F1 grouped bars
    make_stage_f1_bars(bundles, args.outdir)

    # 3) Per-subject delta plots vs baseline (ΔAccuracy, Δκ)
    make_delta_plots(baseline, others, args.outdir)

    # 4) Stage distribution histogram (from baseline ground truth)
    make_stage_histogram(baseline, args.outdir)

    # 5) Hypnogram overlay (GT + each method) for a chosen subject
    plot_hypnogram_overlay(bundles, baseline, args.outdir, subject=args.subject)

    # 6) Optional: entropy gate / EMA reset
    if args.tta_log:
        maybe_plot_entropy_gate(args.tta_log, args.outdir)

    # 7) Optional: latency & memory
    if args.profiling:
        maybe_plot_latency_memory(args.profiling, args.outdir)

    print(f"[OK] Saved figures to: {args.outdir}")


if __name__ == "__main__":
    main()
