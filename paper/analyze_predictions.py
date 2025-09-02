# paper/analyze_predictions.py
"""
Aggregate per-record prediction CSVs -> paper-ready metrics & figures.

Inputs:
  - outputs/<split>/*.csv  (from predict_npz.py; columns: epoch,pred,conf,true)
  - config.yaml (for num_classes & class names)
  - data/processed_npy/manifest.json (for listing records, optional)

Outputs in paper/:
  - tables/<split>_summary.csv, <split>_perclass.csv, <split>_perrecord.csv
  - figs/<split>_confmat.png, <split>_reliability.png, <split>_stage_hist.png
  - figs/hypnograms/<split>_<recid>.png  (a few examples)
  - report_<split>.md (Markdown summary)
"""
import os, csv, glob, json, yaml, math
import numpy as np

# ---------- metrics ----------
def per_class_f1(y_true, y_pred, n_classes=5):
    f1 = {}
    for c in range(n_classes):
        tp = np.sum((y_true==c) & (y_pred==c))
        fp = np.sum((y_true!=c) & (y_pred==c))
        fn = np.sum((y_true==c) & (y_pred!=c))
        p  = tp / (tp+fp) if (tp+fp)>0 else 0.0
        r  = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1[c] = (2*p*r/(p+r)) if (p+r)>0 else 0.0
    return f1

def macro_f1(y_true, y_pred, n_classes=5):
    f1s = per_class_f1(y_true, y_pred, n_classes)
    return float(np.mean([f1s[c] for c in range(n_classes)]))

def cohen_kappa(y_true, y_pred, n_classes=5):
    C = np.zeros((n_classes, n_classes), dtype=np.float64)
    for t,p in zip(y_true, y_pred): C[t,p] += 1
    n = C.sum()
    if n == 0: return 0.0
    po = np.trace(C) / n
    pe = (C.sum(0)*C.sum(1)).sum() / (n*n)
    den = (1 - pe)
    return float((po - pe) / den) if den>0 else 0.0

def accuracy(y_true, y_pred): 
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0

def balanced_accuracy(y_true, y_pred, n_classes=5):
    vals = []
    for c in range(n_classes):
        m = (y_true == c)
        if m.any(): vals.append(float((y_pred[m]==c).mean()))
    return float(np.mean(vals)) if vals else 0.0

def weighted_f1(y_true, y_pred, n_classes=5):
    f1s = per_class_f1(y_true, y_pred, n_classes)
    counts = np.bincount(y_true, minlength=n_classes)
    d = counts.sum()
    if d == 0: return 0.0
    return float(sum(f1s[c]*counts[c] for c in range(n_classes)) / d)

def mcc_safe(y_true, y_pred, n_classes=5):
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t,p in zip(y_true, y_pred): C[t,p] += 1
    Cf = C.astype(np.float64)
    t_sum = Cf.sum(1); p_sum = Cf.sum(0); n = Cf.sum(); tr = np.trace(Cf)
    num = tr * n - np.dot(p_sum, t_sum)
    den_sq = (n**2 - np.dot(p_sum,p_sum)) * (n**2 - np.dot(t_sum,t_sum))
    den_sq = max(den_sq, 0.0)
    return float(num / (np.sqrt(den_sq) + 1e-12)) if den_sq>0 else 0.0

def ece(y_true, proba_max, y_pred, n_bins=15):
    if proba_max is None or len(y_true)==0: return 0.0
    conf = proba_max
    preds= y_pred
    bins = np.linspace(0,1,n_bins+1)
    e = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i+1])
        if m.any():
            acc = float((preds[m]==y_true[m]).mean())
            e += abs(acc - float(conf[m].mean())) * float(m.mean())
    return float(e)

def confusion_matrix(y_true, y_pred, n_classes=5):
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t,p in zip(y_true, y_pred): C[t,p] += 1
    return C

# ---------- plotting (best-effort; skips if matplotlib missing) ----------
def maybe_plot_confmat(C, class_names, path):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5,4.5), dpi=120)
        im = ax.imshow(C, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=np.arange(C.shape[1]), yticks=np.arange(C.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               xlabel="Predicted", ylabel="True")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                ax.text(j, i, str(int(C[i,j])), ha="center", va="center",
                        color="white" if C[i,j] > C.max()/2. else "black", fontsize=8)
        fig.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path); plt.close(fig)
    except Exception as e:
        print("[plot] skip confmat:", e)

def maybe_plot_reliability(y_true, y_pred, conf, path, n_bins=15):
    try:
        import matplotlib.pyplot as plt
        bins = np.linspace(0,1,n_bins+1)
        xs, accs, confs, ws = [], [], [], []
        for i in range(n_bins):
            m = (conf>=bins[i]) & (conf<bins[i+1])
            if m.any():
                xs.append((bins[i]+bins[i+1])/2)
                accs.append(float((y_pred[m]==y_true[m]).mean()))
                confs.append(float(conf[m].mean()))
                ws.append(float(m.mean()))
        fig, ax = plt.subplots(figsize=(5,4), dpi=120)
        ax.plot([0,1],[0,1], '--', lw=1, color='gray')
        ax.plot(xs, accs, marker='o', label='Accuracy')
        ax.plot(xs, confs, marker='s', label='Confidence')
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy / Confidence")
        ax.set_title("Reliability"); ax.legend(); fig.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True); fig.savefig(path); plt.close(fig)
    except Exception as e:
        print("[plot] skip reliability:", e)

def maybe_plot_stage_hist(y_true, y_pred, class_names, path):
    try:
        import matplotlib.pyplot as plt
        n = len(class_names)
        true_counts = np.bincount(y_true, minlength=n)
        pred_counts = np.bincount(y_pred, minlength=n)
        true_pct = true_counts / max(1, true_counts.sum()) * 100
        pred_pct = pred_counts / max(1, pred_counts.sum()) * 100
        x = np.arange(n); w=0.35
        fig, ax = plt.subplots(figsize=(6,3.5), dpi=120)
        ax.bar(x - w/2, true_pct, width=w, label="True")
        ax.bar(x + w/2, pred_pct, width=w, label="Pred")
        ax.set_xticks(x); ax.set_xticklabels(class_names)
        ax.set_ylabel("% of epochs"); ax.set_title("Stage distribution")
        ax.legend(); fig.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True); fig.savefig(path); plt.close(fig)
    except Exception as e:
        print("[plot] skip stage hist:", e)

def maybe_plot_hypnogram(y_true, y_pred, class_names, path, title=""):
    try:
        import matplotlib.pyplot as plt
        t = np.arange(len(y_true))
        fig, ax = plt.subplots(figsize=(10,2.5), dpi=120)
        ax.step(t, y_true, where="post", label="True", linewidth=1.5)
        ax.step(t, y_pred, where="post", label="Pred", linewidth=1.0)
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
        ax.set_xlabel("Epoch"); ax.set_title(title or "Hypnogram")
        ax.legend(loc="upper right")
        fig.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True); fig.savefig(path); plt.close(fig)
    except Exception as e:
        print("[plot] skip hypnogram:", e)

# ---------- IO helpers ----------
def read_csv_preds(path):
    # epoch,pred,conf,(true)
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    pred = arr["pred"].astype(np.int64)
    conf = arr["conf"].astype(np.float32)
    true = arr["true"].astype(np.int64) if "true" in arr.dtype.names else None
    return true, pred, conf

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    ap.add_argument("--split", type=str, required=True, choices=["val","test"])
    ap.add_argument("--pred_dir", type=str, default="outputs")
    ap.add_argument("--out_dir", type=str, default="paper")
    args = ap.parse_args()

    cfg  = yaml.safe_load(open(args.cfg))
    ncls = int(cfg["model"]["num_classes"])
    class_names = ['W','N1','N2','N3','REM'][:ncls]

    csv_files = sorted(glob.glob(os.path.join(args.pred_dir, args.split, "rec_*.csv")))
    if not csv_files:
        raise SystemExit(f"No CSVs found under {os.path.join(args.pred_dir,args.split)}")

    # aggregate
    all_true, all_pred, all_conf = [], [], []
    perrec = []  # (rec_id, macroF1, kappa, acc)
    for f in csv_files:
        rec_id = os.path.splitext(os.path.basename(f))[0]  # rec_XXXX
        y_true, y_pred, conf = read_csv_preds(f)
        if y_true is None:
            # fall back: load npz to get truth
            npz = os.path.join("data/processed_npy", rec_id + ".npz")
            y_true = np.load(npz)["y"].astype(np.int64)
        # metrics per record
        mf1 = macro_f1(y_true, y_pred, n_classes=ncls)
        kap = cohen_kappa(y_true, y_pred, n_classes=ncls)
        acc = accuracy(y_true, y_pred)
        perrec.append((rec_id, mf1, kap, acc, len(y_true)))
        all_true.append(y_true); all_pred.append(y_pred); all_conf.append(conf)

    y_true = np.concatenate(all_true); y_pred = np.concatenate(all_pred); conf = np.concatenate(all_conf)

    # global metrics
    mF1  = macro_f1(y_true, y_pred, n_classes=ncls)
    kap  = cohen_kappa(y_true, y_pred, n_classes=ncls)
    acc  = accuracy(y_true, y_pred)
    bacc = balanced_accuracy(y_true, y_pred, n_classes=ncls)
    wf1  = weighted_f1(y_true, y_pred, n_classes=ncls)
    mcc  = mcc_safe(y_true, y_pred, n_classes=ncls)
    ecev = ece(y_true, conf, y_pred, n_bins=15)
    f1s  = per_class_f1(y_true, y_pred, n_classes=ncls)
    C    = confusion_matrix(y_true, y_pred, n_classes=ncls)

    # ----- save tables -----
    tab_dir = os.path.join(args.out_dir, "tables"); os.makedirs(tab_dir, exist_ok=True)
    # summary
    with open(os.path.join(tab_dir, f"{args.split}_summary.csv"), "w", newline="") as f:
        w=csv.writer(f); w.writerow(["macroF1","kappa","accuracy","balanced_accuracy","weighted_F1","MCC","ECE"])
        w.writerow([f"{mF1:.4f}", f"{kap:.4f}", f"{acc:.4f}", f"{bacc:.4f}", f"{wf1:.4f}", f"{mcc:.4f}", f"{ecev:.4f}"])
    # per-class
    with open(os.path.join(tab_dir, f"{args.split}_perclass.csv"), "w", newline="") as f:
        w=csv.writer(f); w.writerow(["class","F1","count"])
        counts = np.bincount(y_true, minlength=ncls)
        for i, nm in enumerate(class_names):
            w.writerow([nm, f"{f1s.get(i,0.0):.4f}", int(counts[i])])
    # per-record
    with open(os.path.join(tab_dir, f"{args.split}_perrecord.csv"), "w", newline="") as f:
        w=csv.writer(f); w.writerow(["record","macroF1","kappa","accuracy","n_epochs"])
        for rec_id, mf1, kap, acc, n in sorted(perrec, key=lambda r: r[1], reverse=True):
            w.writerow([rec_id, f"{mf1:.4f}", f"{kap:.4f}", f"{acc:.4f}", n])

    # ----- figures -----
    fig_dir = os.path.join(args.out_dir, "figs"); os.makedirs(fig_dir, exist_ok=True)
    maybe_plot_confmat(C, class_names, os.path.join(fig_dir, f"{args.split}_confmat.png"))
    maybe_plot_reliability(y_true, y_pred, conf, os.path.join(fig_dir, f"{args.split}_reliability.png"))
    maybe_plot_stage_hist(y_true, y_pred, class_names, os.path.join(fig_dir, f"{args.split}_stage_hist.png"))

    # hypnogram examples: best, median, worst by per-record macroF1
    try:
        import matplotlib.pyplot as plt  # ensure available
        hyp_dir = os.path.join(fig_dir, "hypnograms"); os.makedirs(hyp_dir, exist_ok=True)
        # pick ids
        by_mf1 = sorted(perrec, key=lambda r: r[1])
        picks = [by_mf1[0][0], by_mf1[len(by_mf1)//2][0], by_mf1[-1][0]]
        for rec_id in picks:
            csv_path = os.path.join(args.pred_dir, args.split, rec_id + ".csv")
            y_true, y_pred, _ = read_csv_preds(csv_path)
            maybe_plot_hypnogram(y_true, y_pred, class_names, os.path.join(hyp_dir, f"{args.split}_{rec_id}.png"),
                                 title=f"Hypnogram {rec_id}")
    except Exception as e:
        print("[plot] skip hypnograms:", e)

    # ----- tiny markdown report -----
    rep = os.path.join(args.out_dir, f"report_{args.split}.md")
    with open(rep, "w") as f:
        f.write(f"# {args.split.upper()} Results\n\n")
        f.write(f"**macroF1** {mF1:.3f}  |  **κ** {kap:.3f}  |  **acc** {acc:.3f}  |  **bAcc** {bacc:.3f}  |  **wF1** {wf1:.3f}  |  **MCC** {mcc:.3f}  |  **ECE** {ecev:.3f}\n\n")
        f.write(f"Confusion matrix: `paper/figs/{args.split}_confmat.png`\n\n")
        f.write(f"Reliability: `paper/figs/{args.split}_reliability.png`\n\n")
        f.write(f"Stage distribution: `paper/figs/{args.split}_stage_hist.png`\n\n")
        f.write(f"Per-class F1 table: `paper/tables/{args.split}_perclass.csv`\n\n")
        f.write(f"Per-record metrics: `paper/tables/{args.split}_perrecord.csv`\n\n")
    print("Wrote tables & figs to:", args.out_dir)

if __name__ == "__main__":
    main()
