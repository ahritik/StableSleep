# paper/figs_plus.py
import os, glob, csv, json, math, argparse
import numpy as np

CLASS_NAMES = ['W','N1','N2','N3','REM']

# ---------------- IO ----------------
def load_split(pred_dir, split):
    csvs = sorted(glob.glob(os.path.join(pred_dir, split, "rec_*.csv")))
    if not csvs:
        raise SystemExit(f"No CSVs under {pred_dir}/{split}")
    items = []
    for cpath in csvs:
        base = os.path.splitext(cpath)[0]
        ppath = base + "_probs.npz"
        arr   = np.genfromtxt(cpath, delimiter=",", names=True, dtype=None, encoding=None)
        yhat  = arr["pred"].astype(np.int64)
        conf  = arr["conf"].astype(np.float32)
        y     = arr["true"].astype(np.int64) if "true" in arr.dtype.names else None
        probs = None
        if os.path.exists(ppath):
            z = np.load(ppath)
            probs = z["probs"].astype(np.float32)
            if y is None and "true" in z.files and z["true"].ndim==1 and z["true"].size==yhat.size and z["true"][0]>=0:
                y = z["true"].astype(np.int64)
        if y is None:
            shard = os.path.join("data/processed_npy", os.path.basename(base) + ".npz")
            y = np.load(shard)["y"].astype(np.int64)
        items.append(dict(id=os.path.basename(base), y=y, yhat=yhat, conf=conf, probs=probs))
    return items

def ensure_dir(p): os.makedirs(os.path.dirname(p), exist_ok=True)

# ------------- core metrics -------------
def confusion(y, yhat, K):
    C = np.zeros((K,K), dtype=np.int64)
    for t,p in zip(y,yhat): C[t,p]+=1
    return C

def per_class_f1(y,yhat,K):
    f1={}
    for c in range(K):
        tp = np.sum((y==c)&(yhat==c))
        fp = np.sum((y!=c)&(yhat==c))
        fn = np.sum((y==c)&(yhat!=c))
        p  = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1[c] = (2*p*r/(p+r)) if (p+r)>0 else 0.0
    return f1

def macro_f1(y,yhat,K): return float(np.mean([per_class_f1(y,yhat,K)[c] for c in range(K)]))
def kappa(y,yhat,K):
    C=confusion(y,yhat,K).astype(np.float64)
    n=C.sum()
    po=np.trace(C)/n if n>0 else 0.0
    pe=(C.sum(0)*C.sum(1)).sum()/(n*n) if n>0 else 0.0
    return float((po-pe)/(1-pe)) if (1-pe)>0 else 0.0

# ------------- sleep metrics (clinical) -------------
def hypno_to_metrics(y, epoch_sec=30, K=5):
    """Returns dict of clinical metrics derived from stage labels."""
    y = np.asarray(y)
    n = len(y)
    mins = epoch_sec/60.0
    time_in_bed_min = n * mins
    sleep_mask = y != 0  # non-W
    # SOL: first non-W index
    idx = np.where(sleep_mask)[0]
    sol_min = (idx[0] * mins) if idx.size>0 else np.nan
    # TST: total non-W time
    tst_min = float(sleep_mask.sum() * mins)
    se = tst_min / time_in_bed_min if time_in_bed_min>0 else np.nan
    # WASO: wake after sleep onset
    waso_min = float(((~sleep_mask) & (np.arange(n) > (idx[0] if idx.size>0 else n))).sum() * mins) if idx.size>0 else np.nan
    # REM latency
    rem_idx = np.where(y==4)[0]
    rem_lat_min = ((rem_idx[0] - idx[0]) * mins) if (idx.size>0 and rem_idx.size>0 and rem_idx[0]>=idx[0]) else np.nan
    # stage percentages
    counts = np.bincount(y, minlength=K)
    pct = counts / max(1, counts.sum()) * 100.0
    return dict(TST=tst_min, SE=se, SOL=sol_min, WASO=waso_min, REMlat=rem_lat_min, pct=pct)

def bland_altman(a, b):
    diff = b - a
    mean = (a + b) / 2.0
    mu = float(diff.mean()); sd = float(diff.std(ddof=1))
    lo, hi = mu - 1.96*sd, mu + 1.96*sd
    return mean, diff, (mu, lo, hi)

# ------------- plots (best-effort; skip if missing deps) -------------
def plot_confusions(C, names, out_png):
    try:
        import matplotlib.pyplot as plt
        Cn = C / np.maximum(1, C.sum(axis=1, keepdims=True))
        fig, axs = plt.subplots(1,2, figsize=(10,4), dpi=120)
        for ax, M, ttl in ((axs[0], C, "Counts"), (axs[1], Cn, "Row-normalized")):
            im=ax.imshow(M, cmap="Blues")
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
            ax.set_xlabel("Pred"); ax.set_ylabel("True"); ax.set_title(ttl)
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j,i,(str(int(M[i,j])) if ttl=="Counts" else f"{M[i,j]:.2f}"),
                            ha="center", va="center",
                            color="white" if M[i,j] > M.max()/2 else "black", fontsize=7)
        fig.tight_layout(); ensure_dir(out_png); fig.savefig(out_png); plt.close(fig)
    except Exception as e: print("[confusions] skip:", e)

def plot_roc_pr(y, probs, names, out_png):
    try:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, label_binarize
        import matplotlib.pyplot as plt
        K=len(names); Y=label_binarize(y, classes=list(range(K)))
        fig, axs = plt.subplots(1,2, figsize=(11,4), dpi=120)
        macro_auc=[]; macro_ap=[]
        # ROC
        for c in range(K):
            fpr,tpr,_=roc_curve(Y[:,c], probs[:,c]); auc_c=auc(fpr,tpr); macro_auc.append(auc_c)
            axs[0].plot(fpr,tpr,label=f"{names[c]} ({auc_c:.2f})")
        axs[0].plot([0,1],[0,1],'--',lw=1,color='gray'); axs[0].set_title(f"ROC (macro {np.mean(macro_auc):.2f})")
        axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR"); axs[0].legend(fontsize=8, ncols=K)
        # PR
        for c in range(K):
            pr,rc,_=precision_recall_curve(Y[:,c], probs[:,c])
            ap=average_precision_score(Y[:,c], probs[:,c]); macro_ap.append(ap)
            axs[1].plot(rc,pr,label=f"{names[c]} ({ap:.2f})")
        axs[1].set_title(f"PR (macro AP {np.mean(macro_ap):.2f})")
        axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].legend(fontsize=8, ncols=K)
        fig.tight_layout(); ensure_dir(out_png); fig.savefig(out_png); plt.close(fig)
    except Exception as e: print("[roc_pr] skip (needs scikit-learn):", e)

def plot_calibration(y, probs, names, out_png, n_bins=15):
    try:
        import matplotlib.pyplot as plt
        conf = probs.max(axis=1); pred = probs.argmax(axis=1)
        bins=np.linspace(0,1,n_bins+1)
        xs,accs,confs=[],[],[]
        for i in range(n_bins):
            m=(conf>=bins[i])&(conf<bins[i+1])
            if m.any():
                xs.append((bins[i]+bins[i+1])/2)
                accs.append(float((pred[m]==y[m]).mean()))
                confs.append(float(conf[m].mean()))
        fig,axs=plt.subplots(1,2, figsize=(10,4), dpi=120)
        axs[0].plot([0,1],[0,1],'--',color='gray'); axs[0].plot(xs,accs,marker='o'); axs[0].plot(xs,confs,marker='s')
        axs[0].set_title("Reliability (overall)"); axs[0].set_xlabel("Confidence"); axs[0].set_ylabel("Acc/Conf")
        # per-class
        for c in range(len(names)):
            pc=probs[:,c]; xs,accs=[],[]
            for i in range(n_bins):
                m=(pc>=bins[i])&(pc<bins[i+1])&(y==c)
                if m.any(): xs.append((bins[i]+bins[i+1])/2); accs.append(float((y[m]==c).mean()))
            axs[1].plot(xs,accs,marker='o',label=names[c])
        axs[1].plot([0,1],[0,1],'--',color='gray'); axs[1].legend(fontsize=8, ncols=2); axs[1].set_title("Per-class")
        fig.tight_layout(); ensure_dir(out_png); fig.savefig(out_png); plt.close(fig)
    except Exception as e: print("[calibration] skip:", e)

def plot_conf_hists(y, probs, names, out_png):
    try:
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2, figsize=(10,3.5), dpi=120)
        axs[0].hist(probs.max(axis=1), bins=20, edgecolor='k'); axs[0].set_title("Max-confidence")
        pred=probs.argmax(axis=1)
        for c in range(len(names)):
            axs[1].hist(probs[pred==c, c], bins=20, alpha=0.6, label=names[c])
        axs[1].legend(fontsize=8, ncols=2); axs[1].set_title("Confidence by predicted class")
        fig.tight_layout(); ensure_dir(out_png); fig.savefig(out_png); plt.close(fig)
    except Exception as e: print("[conf_hists] skip:", e)

def transitions(y,K):
    T=np.zeros((K,K), dtype=np.int64)
    for a,b in zip(y[:-1], y[1:]): T[a,b]+=1
    return T

def plot_transitions(Tt, Tp, names, out_png):
    try:
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,3, figsize=(13,4), dpi=120)
        for ax,M,title in ((axs[0],Tt,"True"), (axs[1],Tp,"Pred"), (axs[2],Tp-Tt,"Δ")):
            im=ax.imshow(M, cmap=("Blues" if title!="Δ" else "RdBu_r"))
            ax.figure.colorbar(im, ax=ax, fraction=0.046,pad=0.04)
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
            ax.set_xlabel("To"); ax.set_ylabel("From"); ax.set_title(title)
        fig.tight_layout(); ensure_dir(out_png); fig.savefig(out_png); plt.close(fig)
    except Exception as e: print("[transitions] skip:", e)

def plot_per_subject(perrec, out_png):
    try:
        import matplotlib.pyplot as plt
        mf1=[r["macroF1"] for r in perrec]; kap=[r["kappa"] for r in perrec]
        fig,axs=plt.subplots(1,2, figsize=(10,4), dpi=120)
        axs[0].boxplot(mf1, labels=["macroF1"]); axs[0].set_title("Per-subject macroF1")
        axs[1].scatter(mf1, kap, s=12, alpha=0.8); axs[1].set_xlabel("macroF1"); axs[1].set_ylabel("κ"); axs[1].set_title("Per-subject scatter")
        fig.tight_layout(); ensure_dir(out_png); fig.savefig(out_png); plt.close(fig)
    except Exception as e: print("[per_subject] skip:", e)

def plot_quartiles(recs, K, names, out_png):
    try:
        import matplotlib.pyplot as plt
        qf=[]
        for r in recs:
            N=len(r["y"]); q=np.linspace(0,N,5,dtype=int)
            per=[]
            for i in range(4):
                s,e=q[i],q[i+1]
                f=per_class_f1(r["y"][s:e], r["yhat"][s:e], K)
                per.append([f[c] for c in range(K)])
            qf.append(np.array(per))
        qf=np.stack(qf).mean(axis=0)
        fig,ax=plt.subplots(figsize=(7,3.5), dpi=120)
        for c in range(K):
            ax.plot([1,2,3,4], qf[:,c], marker='o', label=names[c])
        ax.set_xticks([1,2,3,4]); ax.set_xlabel("Night quartile"); ax.set_ylabel("F1"); ax.legend(fontsize=8, ncols=3)
        fig.tight_layout(); ensure_dir(out_png); fig.savefig(out_png); plt.close(fig)
    except Exception as e: print("[quartiles] skip:", e)

def plot_hypnogram(y,yhat,probs,names,out_png,title):
    try:
        import matplotlib.pyplot as plt
        t=np.arange(len(y))
        fig,ax=plt.subplots(figsize=(11,2.6), dpi=120)
        ax.step(t,y, where="post", label="True", lw=1.6)
        ax.step(t,yhat, where="post", label="Pred", lw=1.0)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.set_xlabel("Epoch"); ax.set_title(title)
        ax.legend(loc="upper right")
        if probs is not None:
            conf=probs.max(axis=1); ax2=ax.twinx(); ax2.plot(t,conf, alpha=0.35); ax2.set_ylim(0,1); ax2.set_ylabel("Conf")
        fig.tight_layout(); ensure_dir(out_png); fig.savefig(out_png); plt.close(fig)
    except Exception as e: print("[hypnogram] skip:", e)

def plot_sleep_metrics_agreement(perrec, names, out_dir, epoch_sec=30, K=5):
    try:
        import matplotlib.pyplot as plt
        # collect per-record pairs
        mets=["TST","SE","SOL","WASO","REMlat"]; labs={"TST":"TST (min)","SE":"Sleep efficiency","SOL":"Sleep onset latency (min)","WASO":"WASO (min)","REMlat":"REM latency (min)"}
        pairs={m:([],[]) for m in mets}
        pct_true=[]; pct_pred=[]
        for r in perrec:
            mt = hypno_to_metrics(r["y"], epoch_sec, K)
            mp = hypno_to_metrics(r["yhat"], epoch_sec, K)
            for m in mets:
                if not (np.isnan(mt[m]) or np.isnan(mp[m])):
                    pairs[m][0].append(mt[m]); pairs[m][1].append(mp[m])
            pct_true.append(mt["pct"]); pct_pred.append(mp["pct"])
        pct_true=np.array(pct_true); pct_pred=np.array(pct_pred)
        # scatter+BA for each metric
        for m in mets:
            a=np.array(pairs[m][0]); b=np.array(pairs[m][1])
            if a.size<3: continue
            mean,diff,(mu,lo,hi)=bland_altman(a,b)
            fig,axs=plt.subplots(1,2, figsize=(11,4), dpi=120)
            axs[0].scatter(a,b,s=12,alpha=0.8); lim=[min(a.min(),b.min()), max(a.max(),b.max())]; axs[0].plot(lim,lim,'--',color='gray'); axs[0].set_xlim(lim); axs[0].set_ylim(lim)
            axs[0].set_xlabel("True"); axs[0].set_ylabel("Pred"); axs[0].set_title(labs[m])
            axs[1].scatter(mean,diff,s=12,alpha=0.8); axs[1].axhline(mu,color='k'); axs[1].axhline(lo,color='r',ls='--'); axs[1].axhline(hi,color='r',ls='--')
            axs[1].set_xlabel("Mean"); axs[1].set_ylabel("Pred−True"); axs[1].set_title(f"Bland–Altman {m} (μ={mu:.2f}, LoA=[{lo:.2f},{hi:.2f}])")
            fig.tight_layout(); p=os.path.join(out_dir, f"metrics_{m}.png"); ensure_dir(p); fig.savefig(p); plt.close(fig)
        # stage percentage scatter
        for i,nm in enumerate(names):
            a=pct_true[:,i]; b=pct_pred[:,i]
            fig,ax=plt.subplots(figsize=(4.2,4.2), dpi=120)
            lim=[min(a.min(),b.min()), max(a.max(),b.max())]; ax.scatter(a,b,s=14,alpha=0.8); ax.plot(lim,lim,'--',color='gray'); ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_xlabel(f"True % {nm}"); ax.set_ylabel(f"Pred % {nm}"); ax.set_title(f"Stage % agreement: {nm}")
            p=os.path.join(out_dir, f"stagepct_{nm}.png"); fig.tight_layout(); ensure_dir(p); fig.savefig(p); plt.close(fig)
    except Exception as e: print("[sleep_metrics] skip:", e)

def plot_coverage_accuracy(y, probs, names, out_png):
    try:
        import matplotlib.pyplot as plt
        conf=probs.max(axis=1); pred=probs.argmax(axis=1)
        ths=np.linspace(0.0,1.0,41); cov=[]; acc=[]; kap=[]
        for t in ths:
            m = conf>=t
            cov.append(float(m.mean()))
            if m.any():
                acc.append(float((pred[m]==y[m]).mean()))
                kap.append(kappa(y[m], pred[m], len(names)))
            else:
                acc.append(np.nan); kap.append(np.nan)
        fig,ax=plt.subplots(figsize=(6,3.5), dpi=120)
        ax.plot(cov, acc, marker='o', label="accuracy")
        ax.plot(cov, kap, marker='s', label="kappa")
        ax.set_xlabel("Coverage (fraction of epochs kept)"); ax.set_ylabel("Score"); ax.set_title("Selective prediction")
        ax.legend(); fig.tight_layout(); ensure_dir(out_png); fig.savefig(out_png); plt.close(fig)
    except Exception as e: print("[coverage] skip:", e)

def plot_error_vs_conf(y, probs, names, out_png):
    try:
        import matplotlib.pyplot as plt
        conf=probs.max(axis=1); pred=probs.argmax(axis=1); err=(pred!=y).astype(np.float32)
        bins=np.linspace(0,1,11); xs=[]; ers=[]
        for i in range(10):
            m=(conf>=bins[i])&(conf<bins[i+1])
            if m.any():
                xs.append((bins[i]+bins[i+1])/2)
                ers.append(float(err[m].mean()))
        fig,ax=plt.subplots(figsize=(5,3.3), dpi=120); ax.plot(xs, ers, marker='o'); ax.set_xlabel("Confidence bin"); ax.set_ylabel("Error rate"); ax.set_title("Error vs confidence")
        fig.tight_layout(); ensure_dir(out_png); fig.savefig(out_png); plt.close(fig)
    except Exception as e: print("[err_conf] skip:", e)

def bootstrap_ci(perrec, B=200, K=5):
    rng=np.random.default_rng(1337)
    mf1s=[]; kaps=[]
    idx=np.arange(len(perrec))
    for _ in range(B):
        take=rng.choice(idx, size=len(idx), replace=True)
        y  = np.concatenate([perrec[i]["y"] for i in take])
        yh = np.concatenate([perrec[i]["yhat"] for i in take])
        mf1s.append(macro_f1(y,yh,K)); kaps.append(kappa(y,yh,K))
    return np.percentile(mf1s,[2.5,97.5]), np.percentile(kaps,[2.5,97.5])

# ---------------- main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, default="outputs")
    ap.add_argument("--split",    type=str, required=True, choices=["val","test"])
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--epoch_sec", type=int, default=30)
    ap.add_argument("--out_dir",  type=str, default="paper/figs_plus")
    args=ap.parse_args()

    K=args.num_classes; names=CLASS_NAMES[:K]
    recs=load_split(args.pred_dir, args.split)

    y_all = np.concatenate([r["y"] for r in recs])
    yh_all= np.concatenate([r["yhat"] for r in recs])
    probs_all = np.concatenate([r["probs"] for r in recs]) if all(r["probs"] is not None for r in recs) else None

    # Confusions
    C = confusion(y_all, yh_all, K)
    plot_confusions(C, names, os.path.join(args.out_dir, args.split, "confusions.png"))

    # ROC/PR, calibration, confidence hists
    if probs_all is not None:
        plot_roc_pr(y_all, probs_all, names, os.path.join(args.out_dir, args.split, "roc_pr.png"))
        plot_calibration(y_all, probs_all, names, os.path.join(args.out_dir, args.split, "calibration.png"))
        plot_conf_hists(y_all, probs_all, names, os.path.join(args.out_dir, args.split, "confidence_hists.png"))

    # Transitions + stage distribution
    Tt = transitions(y_all, K); Tp = transitions(yh_all, K)
    plot_transitions(Tt, Tp, names, os.path.join(args.out_dir, args.split, "transitions.png"))
    try:
        import matplotlib.pyplot as plt
        tc=np.bincount(y_all, minlength=K); pc=np.bincount(yh_all, minlength=K)
        tp=tc/tc.sum()*100; pp=pc/pc.sum()*100
        fig,ax=plt.subplots(figsize=(6,3.3), dpi=120); x=np.arange(K); w=0.35
        ax.bar(x-w/2,tp,width=w,label="True"); ax.bar(x+w/2,pp,width=w,label="Pred")
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("% epochs"); ax.legend(); ax.set_title("Stage distribution")
        p=os.path.join(args.out_dir, args.split, "stage_dist.png"); ensure_dir(p); fig.tight_layout(); fig.savefig(p); plt.close(fig)
    except Exception as e: print("[stage_dist] skip:", e)

    # Per-subject spread & quartiles
    perrec=[dict(id=r["id"], y=r["y"], yhat=r["yhat"], macroF1=macro_f1(r["y"], r["yhat"], K), kappa=kappa(r["y"], r["yhat"], K)) for r in recs]
    plot_per_subject(perrec, os.path.join(args.out_dir, args.split, "per_subject.png"))
    plot_quartiles(recs, K, names, os.path.join(args.out_dir, args.split, "f1_by_quartile.png"))

    # Hypnogram examples
    by = sorted(perrec, key=lambda d: d["macroF1"])
    picks = [by[0]["id"], by[len(by)//2]["id"], by[-1]["id"]]
    for pid in picks:
        r = next(rr for rr in recs if rr["id"]==pid)
        plot_hypnogram(r["y"], r["yhat"], r["probs"], names, os.path.join(args.out_dir, args.split, f"hypnogram_{pid}.png"), f"Hypnogram {pid}")

    # Sleep metrics agreement (clinical)
    plot_sleep_metrics_agreement(perrec, names, os.path.join(args.out_dir, args.split, "sleep_metrics"), epoch_sec=args.epoch_sec, K=K)

    # Selective prediction & error-vs-confidence
    if probs_all is not None:
        plot_coverage_accuracy(y_all, probs_all, names, os.path.join(args.out_dir, args.split, "selective_prediction.png"))
        plot_error_vs_conf(y_all, probs_all, names, os.path.join(args.out_dir, args.split, "error_vs_conf.png"))

    # Bootstrap CIs for macroF1 / κ
    ci_f1, ci_k = bootstrap_ci(perrec, B=400, K=K)
    with open(os.path.join(args.out_dir, args.split, "bootstrap_ci.txt"), "w") as f:
        f.write(f"macroF1 95% CI: [{ci_f1[0]:.3f}, {ci_f1[1]:.3f}]\n")
        f.write(f"kappa   95% CI: [{ci_k[0]:.3f}, {ci_k[1]:.3f}]\n")

    print("Wrote figures to", os.path.join(args.out_dir, args.split))

if __name__ == "__main__":
    main()
