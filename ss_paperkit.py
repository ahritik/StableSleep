
#!/usr/bin/env python3
# ============================================================================
# ss_paperkit_v2.py  —  One-stop paper assets generator (fixed & streamlined)
# ----------------------------------------------------------------------------
# Changes vs v1:
#   • Fixed LaTeX strings (escaped backslashes) to silence SyntaxWarnings.
#   • Fixed pipeline arrows (use matplotlib.patches.FancyArrow).
#   • Nicer errors when files are missing.
#   • Added `doctor` to inspect NPZ files and report shapes/keys.
# ============================================================================

import argparse, json, os, re, glob, sys
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle

from sklearn.metrics import (
    f1_score, confusion_matrix, cohen_kappa_score,
    roc_curve, precision_recall_curve
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

STAGES = ["W","N1","N2","N3","REM"]

# ------------------------------- I/O HELPERS -------------------------------

def must_exist(path: str, kind: str = "file"):
    if not os.path.exists(path):
        print(f"[error] {kind} not found: {path}", file=sys.stderr)
        raise SystemExit(2)

def load_npz(path: str) -> Dict[str, Any]:
    must_exist(path, "file")
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    # decode meta if JSON-ish
    if "meta" in out:
        meta = out["meta"]
        try:
            if isinstance(meta, (np.void, np.ndarray)):
                meta = meta.item()
            if isinstance(meta, (bytes, bytearray)):
                meta = json.loads(meta.decode("utf-8"))
            elif isinstance(meta, str):
                meta = json.loads(meta)
            out["meta"] = meta
        except Exception:
            out["meta"] = meta
    # subjects as strings for grouping
    if "subjects" in out:
        s = out["subjects"]
        if getattr(s, "dtype", None) is not None and s.dtype.kind in ("i","u","f"):
            out["subjects"] = s.astype(int).astype(str)
        else:
            out["subjects"] = s.astype(str)
    return out

def ensure_probs(d: Dict[str, Any]) -> Optional[np.ndarray]:
    if "y_proba" in d: return d["y_proba"]
    if "logits" in d:
        z = d["logits"]
        z = z - z.max(axis=1, keepdims=True)       # stable softmax
        return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return None

def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ------------------------------- METRICS CORE ------------------------------

def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))

def per_class_f1(y_true, y_pred, labels=None) -> np.ndarray:
    if labels is None: labels = np.unique(y_true)
    return f1_score(y_true, y_pred, average=None, labels=labels)

def kappa(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred))

def ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    inds = np.digitize(conf, bins) - 1
    val = 0.0; n = len(y_true)
    for m in range(n_bins):
        sel = inds == m
        if not np.any(sel): continue
        acc = np.mean((pred[sel]==y_true[sel]).astype(float))
        avg = np.mean(conf[sel])
        val += (sel.sum()/n) * abs(acc - avg)
    return float(val)

def group_by_subject(subjects: np.ndarray) -> Dict[str, np.ndarray]:
    g = {}
    for i, sid in enumerate(subjects):
        g.setdefault(sid, []).append(i)
    return {k: np.array(v, dtype=int) for k, v in g.items()}

def metrics_dict(d: Dict[str, Any], ece_bins: int = 15) -> Dict[str, Any]:
    y_true = d["y_true"].astype(int)
    y_pred = d["y_pred"].astype(int)
    labels = np.arange(len(STAGES))
    out: Dict[str, Any] = {}
    out["overall"] = {
        "acc": float(np.mean(y_true == y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
        "kappa": kappa(y_true, y_pred),
    }
    out["per_class_f1"] = dict(zip(STAGES, per_class_f1(y_true, y_pred, labels=labels).tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out["confusion_matrix"] = cm.tolist()
    proba = ensure_probs(d)
    if proba is not None:
        out["ece"] = ece(y_true, proba, ece_bins)
    if "subjects" in d:
        groups = group_by_subject(d["subjects"])
        mf = [macro_f1(y_true[idx], y_pred[idx]) for idx in groups.values()]
        out["subject_macro_f1_mean"] = float(np.mean(mf))
        out["subject_macro_f1_sd"] = float(np.std(mf, ddof=1) if len(mf)>1 else 0.0)
    if "latencies" in d:
        out["latency_ms_per_epoch_median"] = float(np.median(np.asarray(d["latencies"], float)) * 1000.0)
    return out

# ---------------------------------- FIGURES ---------------------------------

def fig_confusion(cm: np.ndarray, outpath: str):
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cmn, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(STAGES))); ax.set_yticks(range(len(STAGES)))
    ax.set_xticklabels(STAGES, rotation=45, ha="right"); ax.set_yticklabels(STAGES)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalized Confusion")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

def fig_reliability(y_true: np.ndarray, proba: np.ndarray, n_bins: int, outpath: str):
    pred = proba.argmax(axis=1); conf = proba.max(axis=1)
    bins = np.linspace(0.0,1.0,n_bins+1); centers = 0.5*(bins[:-1]+bins[1:])
    accs, confs, fracs = [], [], []
    for m in range(n_bins):
        sel = (conf >= bins[m]) & ((conf < bins[m+1]) if m < n_bins-1 else (conf <= bins[m+1]))
        if not np.any(sel): accs.append(np.nan); confs.append(np.nan); fracs.append(0.0); continue
        accs.append(np.mean((pred[sel]==y_true[sel]).astype(float)))
        confs.append(np.mean(conf[sel])); fracs.append(np.mean(sel.astype(float)))
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot([0,1],[0,1], linestyle="--"); ax.plot(confs, accs, marker="o")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title("Reliability")
    ax2 = ax.twinx(); ax2.bar(centers, fracs, width=1.0/n_bins, alpha=0.3); ax2.set_ylabel("Fraction in bin")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

def fig_subject_delta(subs: np.ndarray, y_true: np.ndarray, ya: np.ndarray, yb: np.ndarray, outpath: str):
    groups = group_by_subject(subs)
    pairs = []
    for sid, idx in groups.items():
        pairs.append((sid, macro_f1(y_true[idx], yb[idx]) - macro_f1(y_true[idx], ya[idx])))
    pairs.sort(key=lambda x: x[0])
    lab = [p[0] for p in pairs]; val = [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(8,3.5))
    ax.bar(np.arange(len(val)), val); ax.set_xticks(np.arange(len(val))); ax.set_xticklabels(lab, rotation=90)
    ax.set_ylabel("Δ Macro-F1 (B − A)"); ax.set_title("Subject-wise ΔF1")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

def fig_traces(entropy: np.ndarray, gate: Optional[np.ndarray], resets: Optional[np.ndarray], outpath: str):
    t = np.arange(len(entropy))
    fig, ax = plt.subplots(figsize=(9,3.5))
    ax.plot(t, entropy, label="Entropy")
    if gate is not None:
        g = gate.astype(bool); ax.scatter(t[g], entropy[g], marker="x", label="Gate skip")
    if resets is not None:
        r = resets.astype(bool); y0,y1 = float(np.nanmin(entropy)), float(np.nanmax(entropy))
        for i in np.where(r)[0]: ax.vlines(i, y0, y1, linestyles="dotted")
        ax.plot([], [], linestyle="dotted", label="EMA reset")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Entropy"); ax.legend(); ax.set_title("Entropy / events")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

def fig_latency(lat_s: np.ndarray, label: str, outpath: str):
    ms = np.asarray(lat_s, float) * 1000.0
    fig, ax = plt.subplots(figsize=(4,3.2))
    ax.bar([0], [np.median(ms)]); ax.set_xticks([0]); ax.set_xticklabels([label])
    ax.set_ylabel("Median latency (ms/epoch)"); ax.set_title("Latency")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout(); fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

# ------------------------------- SUBCOMMANDS -------------------------------

def cmd_perrun(args):
    d = load_npz(args.input)
    os.makedirs(os.path.join(args.outdir,"figs"), exist_ok=True)

    # Metrics JSON
    m = metrics_dict(d, ece_bins=args.ece_bins)
    save_json(m, os.path.join(args.outdir, "metrics.json"))

    # Confusion
    cm = np.asarray(m["confusion_matrix"])
    fig_confusion(cm, os.path.join(args.outdir,"figs","confusion.pdf"))

    # Reliability / ECE
    proba = ensure_probs(d)
    if proba is not None:
        fig_reliability(d["y_true"].astype(int), proba, args.ece_bins, os.path.join(args.outdir,"figs","reliability.pdf"))

    # Traces
    if "entropy" in d:
        fig_traces(d["entropy"].astype(float), d.get("gate_skips"), d.get("ema_resets"), os.path.join(args.outdir,"figs","traces.pdf"))

    # Latency
    if "latencies" in d:
        label = str(d.get("meta",{}).get("batch_size","run"))
        fig_latency(d["latencies"], label, os.path.join(args.outdir,"figs","latency.pdf"))

    # Optional: embeddings or proba as proxy (TSNE)
    if "embeddings" in d or proba is not None:
        X = d.get("embeddings", proba)
        if X is not None:
            Z = X
            if Z.shape[1] > 2:
                if Z.shape[1] > 32: Z = PCA(n_components=32, random_state=0).fit_transform(Z)
                Z = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(Z)
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(Z[:,0], Z[:,1], s=6, c=d["y_true"].astype(int), cmap="tab10")
            ax.set_title("Embedding projection")
            os.makedirs(os.path.join(args.outdir,"figs"), exist_ok=True)
            fig.tight_layout(); fig.savefig(os.path.join(args.outdir,"figs","embeddings.pdf"), bbox_inches="tight"); plt.close(fig)

    # Optional: ROC/PR per stage
    if proba is not None:
        outd = os.path.join(args.outdir,"figs"); os.makedirs(outd, exist_ok=True)
        K = proba.shape[1]; y = d["y_true"].astype(int)
        # ROC
        fig, ax = plt.subplots(figsize=(6,4))
        for k in range(K):
            yk = (y==k).astype(int); fpr,tpr,_ = roc_curve(yk, proba[:,k])
            ax.plot(fpr, tpr, label=STAGES[k])
        ax.plot([0,1],[0,1], linestyle="--"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC by stage"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outd,"roc.pdf"), bbox_inches="tight"); plt.close(fig)
        # PR
        from sklearn.metrics import precision_recall_curve
        fig, ax = plt.subplots(figsize=(6,4))
        for k in range(K):
            yk = (y==k).astype(int); P,R,_ = precision_recall_curve(yk, proba[:,k])
            ax.plot(R, P, label=STAGES[k])
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR by stage"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outd,"pr.pdf"), bbox_inches="tight"); plt.close(fig)

    print(f"[perrun] wrote metrics + figures under {args.outdir}")

def cmd_compare(args):
    A = load_npz(args.source); B = load_npz(args.tent); C = load_npz(args.ours)
    outd = os.path.join(args.outdir,"figs"); os.makedirs(outd, exist_ok=True)
    labels = np.arange(len(STAGES))

    # Confusions: fig2/fig3/fig4
    for D, name in [(A,"fig2_confusion_source.pdf"), (B,"fig3_confusion_tent.pdf"), (C,"fig4_confusion_ours.pdf")]:
        cm = confusion_matrix(D["y_true"].astype(int), D["y_pred"].astype(int), labels=labels)
        fig_confusion(cm, os.path.join(outd, name))

    # Reliability (3-run) if all have proba/logits
    Ap, Bp, Cp = ensure_probs(A), ensure_probs(B), ensure_probs(C)
    if not any(p is None for p in (Ap,Bp,Cp)):
        def bin_stats(y, p, n_bins):
            pred = p.argmax(1); conf = p.max(1)
            bins = np.linspace(0,1,n_bins+1); accs, confs = [], []
            for m in range(n_bins):
                sel = (conf>=bins[m]) & ((conf<bins[m+1]) if m<n_bins-1 else (conf<=bins[m+1]))
                if not np.any(sel): accs.append(np.nan); confs.append(np.nan); continue
                accs.append(np.mean((pred[sel]==y[sel]).astype(float))); confs.append(np.mean(conf[sel]))
            return np.array(confs), np.array(accs)
        yA,yB,yC = A["y_true"].astype(int), B["y_true"].astype(int), C["y_true"].astype(int)
        Acx,Aax = bin_stats(yA,Ap,args.ece_bins); Bcx,Bax = bin_stats(yB,Bp,args.ece_bins); Ccx,Cax = bin_stats(yC,Cp,args.ece_bins)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.plot([0,1],[0,1], linestyle="--")
        ax.plot(Acx, Aax, marker="o", label="Source-only")
        ax.plot(Bcx, Bax, marker="s", label="Tent")
        ax.plot(Ccx, Cax, marker="^", label="Tent+Rails")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.set_title("Reliability (3 runs)"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outd,"fig_reliability_three.pdf"), bbox_inches="tight"); plt.close(fig)

    # Subject-wise ΔF1: ours - source
    if A["subjects"].shape[0] == C["subjects"].shape[0]:
        fig_subject_delta(A["subjects"], A["y_true"].astype(int), A["y_pred"].astype(int), C["y_pred"].astype(int), os.path.join(outd,"fig6_subjectwise_delta.pdf"))

    print(f"[compare] wrote figs under {outd}")

def extract_param(path: str, key: str) -> Optional[float]:
    m = re.search(rf"{key}\s*[:=_-]\s*([0-9.]+)", os.path.basename(path))
    if m: return float(m.group(1))
    try:
        meta = load_npz(path).get("meta",{})
        if key in meta: return float(meta[key])
    except Exception:
        pass
    return None

def cmd_ablation(args):
    files = sorted(glob.glob(args.glob))
    xs, mf1s, eces = [], [], []
    for f in files:
        x = extract_param(f, args.key)
        if x is None:
            print(f"[warn] cannot parse {args.key} from {f}")
            continue
        d = load_npz(f); m = metrics_dict(d, ece_bins=args.ece_bins)
        xs.append(x); mf1s.append(m["overall"]["macro_f1"]); eces.append(m.get("ece", np.nan))
    if not xs:
        print("[ablation] nothing to plot — check --glob or filenames/meta"); return
    xs = np.array(xs); mf1s = np.array(mf1s); eces = np.array(eces)
    ord = np.argsort(xs); xs,mf1s,eces = xs[ord], mf1s[ord], eces[ord]
    fig, ax = plt.subplots(figsize=(5,3.5))
    ax.plot(xs, mf1s, marker="o"); ax.set_xlabel(args.key); ax.set_ylabel("Macro-F1"); ax.set_title(f"Ablation over {args.key}")
    if not np.all(np.isnan(eces)):
        ax2 = ax.twinx(); ax2.plot(xs, eces, marker="s"); ax2.set_ylabel("ECE")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, bbox_inches="tight"); plt.close(fig)
    print(f"[ablation] saved {args.out}")

def fmt(x: Optional[float], d=3) -> str:
    return "--" if x is None else f"{x:.{d}f}"

def cmd_tables(args):
    os.makedirs(args.outdir, exist_ok=True)
    # main + perstage require source/tent/ours
    def R(p):
        must_exist(p, "file")
        with open(p, "r") as f: return json.load(f)

    if args.source and args.tent and args.ours:
        S,T,O = R(args.source), R(args.tent), R(args.ours)
        # Main table
        lines = []
        lines += ["\\begin{table}[t]","\\centering","\\caption{Sleep staging on Sleep-EDF Expanded (by-subject test).}","\\label{tab:main}","\\begin{tabular}{lcccc}","\\toprule","Method & Acc & Macro-F1 & $\\kappa$ & Latency (ms/epoch)\\\\","\\midrule"]
        for name,M in [("Source-only",S),("Tent (BN only)",T),("Tent + Safety rails (ours)",O)]:
            acc = M["overall"]["acc"]; mf1 = M["overall"]["macro_f1"]; kap = M["overall"]["kappa"]; lat = M.get("latency_ms_per_epoch_median", None)
            lines.append(f"{name} & {fmt(acc)} & {fmt(mf1)} & {fmt(kap)} & {fmt(lat)} \\\\")
        lines += ["\\bottomrule","\\end{tabular}","\\end{table}"]
        open(os.path.join(args.outdir,"table_main.tex"),"w").write("\n".join(lines))
        # Per-stage F1
        lines = []
        lines += ["\\begin{table}[t]","\\centering","\\caption{Per-stage F1 (W, N1, N2, N3, REM).}","\\label{tab:perstage}","\\begin{tabular}{lccccc}","\\toprule","Method & W & N1 & N2 & N3 & REM\\\\","\\midrule"]
        for name,M in [("Source-only",S),("Tent",T),("Ours",O)]:
            pf = M["per_class_f1"]; lines.append(f"{name} & {fmt(pf['W'])} & {fmt(pf['N1'])} & {fmt(pf['N2'])} & {fmt(pf['N3'])} & {fmt(pf['REM'])} \\\\")
        lines += ["\\bottomrule","\\end{tabular}","\\end{table}"]
        open(os.path.join(args.outdir,"table_perstage.tex"),"w").write("\n".join(lines))
        print(f"[tables] wrote table_main.tex and table_perstage.tex in {args.outdir}")
    # optional ablation table rows
    if args.ablate:
        entries = []
        for pair in args.ablate:
            if "=" not in pair: raise SystemExit(f"--ablate must be Label=path, got: {pair}")
            label,path = pair.split("=",1); must_exist(path, "file")
            with open(path,"r") as f: M = json.load(f)
            entries.append((label, M["overall"]["acc"], M["overall"]["macro_f1"], M["overall"]["kappa"]))
        lines = []
        lines += ["\\begin{table}[t]","\\centering","\\caption{Ablation: components and rails.}","\\label{tab:ablation}","\\begin{tabular}{lccc}","\\toprule","Model & Acc & Macro-F1 & $\\kappa$\\\\","\\midrule"]
        for label,acc,mf1,kap in entries:
            lines.append(f"{label} & {fmt(acc)} & {fmt(mf1)} & {fmt(kap)} \\\\")
        lines += ["\\bottomrule","\\end{tabular}","\\end{table}"]
        open(os.path.join(args.outdir,"table_ablation.tex"),"w").write("\n".join(lines))
        print(f"[tables] wrote table_ablation.tex in {args.outdir}")

def cmd_pipeline(args):
    fig, ax = plt.subplots(figsize=(9,2.2))
    def box(x,y,w,h,txt):
        r = Rectangle((x,y), w, h, fill=False, linewidth=1.5); ax.add_patch(r); ax.text(x+w/2, y+h/2, txt, ha="center", va="center")
    def arrow(x0,y0,x1,y1):
        arr = FancyArrow(x0,y0,x1-x0,y1-y0,width=0.002,length_includes_head=True,head_width=0.05); ax.add_patch(arr)
    box(0.05,0.4,0.20,0.3,"Source Training\n(CNN + SE + Focal)"); arrow(0.25,0.55,0.35,0.55)
    box(0.35,0.4,0.20,0.3,"Test Stream\n(EEG epochs)"); arrow(0.55,0.55,0.65,0.55)
    box(0.65,0.55,0.25,0.25,"Tent (Entropy)\nBN-only updates")
    box(0.65,0.15,0.25,0.25,"Safety Rails\nGate (pause)\nEMA Reset (recover)"); arrow(0.77,0.40,0.77,0.55)
    arrow(0.90,0.55,0.98,0.55); ax.text(0.93,0.75,"Predictions\n+ Logs", ha="center")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off"); os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, bbox_inches="tight"); plt.close(fig)
    print(f"[pipeline] saved {args.out}")

def cmd_doctor(args):
    paths = sorted(glob.glob(args.glob, recursive=True))
    if not paths:
        print(f"[doctor] no files matched: {args.glob}")
        return
    report = {}
    for p in paths:
        try:
            d = load_npz(p)
            info = {k: (list(d[k].shape) if hasattr(d[k], "shape") else "scalar") for k in d.keys()}
            info["_has_proba"] = bool("y_proba" in d or "logits" in d)
            report[p] = info
        except SystemExit:
            raise
        except Exception as e:
            report[p] = {"error": str(e)}
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "doctor_report.json")
    save_json(report, out)
    print(f"[doctor] wrote {out}")
    # Also print a quick summary
    for p, info in report.items():
        if "error" in info:
            print(f" - {p}: ERROR {info['error']}")
        else:
            keys = ", ".join(sorted([k for k in info.keys() if not k.startswith("_")]))
            print(f" - {p}: keys= [{keys}]  has_proba={info['_has_proba']}")

# ---------------------------------- CLI ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="One-stop assets generator for the paper (v2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("perrun", help="Metrics + figures for one NPZ run")
    p.add_argument("--input", required=True); p.add_argument("--outdir", required=True); p.add_argument("--ece_bins", type=int, default=15)
    p.set_defaults(func=cmd_perrun)

    c = sub.add_parser("compare", help="3-run comparisons (confusions, reliability, ΔF1)")
    c.add_argument("--source", required=True); c.add_argument("--tent", required=True); c.add_argument("--ours", required=True)
    c.add_argument("--outdir", required=True); c.add_argument("--ece_bins", type=int, default=15)
    c.set_defaults(func=cmd_compare)

    a = sub.add_parser("ablation", help="Ablation plot over one hyperparameter")
    a.add_argument("--glob", required=True); a.add_argument("--key", required=True); a.add_argument("--out", required=True); a.add_argument("--ece_bins", type=int, default=15)
    a.set_defaults(func=cmd_ablation)

    t = sub.add_parser("tables", help="Emit LaTeX tables from metrics.json")
    t.add_argument("--source", type=str); t.add_argument("--tent", type=str); t.add_argument("--ours", type=str)
    t.add_argument("--ablate", action="append", help="Label=path/to/metrics.json", default=[])
    t.add_argument("--outdir", required=True)
    t.set_defaults(func=cmd_tables)

    q = sub.add_parser("pipeline", help="Pipeline diagram placeholder (Fig. 1)")
    q.add_argument("--out", required=True); q.set_defaults(func=cmd_pipeline)

    d = sub.add_parser("doctor", help="Scan NPZ files and summarize keys/shapes")
    d.add_argument("--glob", default="runs/**/*.npz")
    d.add_argument("--outdir", default="out/doctor")
    d.set_defaults(func=cmd_doctor)

    args = ap.parse_args()
    os.makedirs(getattr(args, "outdir", "."), exist_ok=True)
    args.func(args)

if __name__ == "__main__":
    main()
