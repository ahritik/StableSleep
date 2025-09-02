# eval_sleep.py
"""
Evaluate checkpoint (none / bn_only / tent) with richer metrics.
Supports CNN (single-epoch) and TSN/TCN (sequence).
"""
import argparse, yaml, torch, numpy as np, os
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from data.sleepedf_mne import build_loaders, build_sequence_loaders
from utils.metrics import summarize, macro_f1, kappa, per_class_f1
from utils.common import get_device, load_checkpoint
from tta_tent import tent_adapt

# ---- extra metrics ----
def accuracy(y_true, y_pred): return float((y_true == y_pred).mean()) if len(y_true) else 0.0
def balanced_accuracy(y_true, y_pred, n_classes=5):
    ba, k = 0.0, 0
    for c in range(n_classes):
        m = (y_true == c)
        if m.any(): ba += float((y_pred[m] == c).mean()); k += 1
    return ba / max(k, 1)
def weighted_f1(y_true, y_pred, n_classes=5):
    f1s = per_class_f1(y_true, y_pred)
    counts = np.bincount(y_true, minlength=n_classes)
    d = counts.sum()
    if d == 0: return 0.0
    return float(sum(f1s.get(c,0.0)*counts[c] for c in range(n_classes)) / d)
def mcc_safe(y_true, y_pred, n_classes=5):
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t,p in zip(y_true, y_pred): C[t,p] += 1
    Cf = C.astype(np.float64)
    t_sum = Cf.sum(1); p_sum = Cf.sum(0); n = Cf.sum(); tr = np.trace(Cf)
    num = tr * n - np.dot(p_sum, t_sum)
    den_sq = (n**2 - np.dot(p_sum,p_sum)) * (n**2 - np.dot(t_sum,t_sum))
    den_sq = max(den_sq, 0.0); den = np.sqrt(den_sq) + 1e-12
    return float(num/den) if den_sq>0 else 0.0
def ece(y_true, proba, n_bins=15):
    if proba is None or len(y_true)==0: return 0.0
    conf = proba.max(axis=1); preds = proba.argmax(axis=1)
    bins = np.linspace(0,1,n_bins+1); e=0.0
    for i in range(n_bins):
        m = (conf>=bins[i]) & (conf<bins[i+1])
        if m.any():
            acc = float((preds[m]==y_true[m]).mean())
            e += abs(acc - float(conf[m].mean())) * float(m.mean())
    return float(e)
def confusion_matrix(y_true, y_pred, n_classes=5):
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t,p in zip(y_true, y_pred): C[t,p] += 1
    return C
def log_confmat_figure(writer, tag, C, class_names, global_step):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5,4), dpi=120)
        im = ax.imshow(C, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=np.arange(C.shape[1]), yticks=np.arange(C.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               ylabel='True', xlabel='Pred')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = C.max()/2. if C.max()>0 else 0.5
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                ax.text(j, i, format(C[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if C[i, j] > thresh else "black", fontsize=8)
        fig.tight_layout()
        writer.add_figure(tag, fig, global_step=global_step)
        plt.close(fig)
    except Exception:
        pass

@torch.no_grad()
def eval_no_tta(model, loader, device, is_seq=False):
    model.eval()
    y_pred, y_true, y_prob = [], [], []
    for x, y in loader:
        x = x.to(device).float()
        if is_seq and x.dim() == 3:
            x = x.unsqueeze(1)  # (B,1,C,T)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        prob = F.softmax(logits, dim=1)
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_true.append(y.numpy())
        y_prob.append(prob.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)

def _has_bn(model):
    return any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in model.modules())

def eval_bn_only(model, loader, device, is_seq=False):
    if not _has_bn(model):
        print("[bn_only] No BatchNorm layers found — falling back to none.")
        return eval_no_tta(model, loader, device, is_seq=is_seq)
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()
    with torch.no_grad():
        for x, _ in loader:  # update BN running stats
            x = x.to(device).float()
            if is_seq and x.dim() == 3:
                x = x.unsqueeze(1)
            _ = model(x)
    return eval_no_tta(model.eval(), loader, device, is_seq=is_seq)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    ap.add_argument("--tta", type=str, default="none", choices=["none","bn_only","tent"])
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--split", type=str, default="test", choices=["val","test"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--logdir", type=str, default="runs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    device = get_device()
    bs = args.batch_size or cfg["eval"]["batch_size"]
    class_names = ['W','N1','N2','N3','REM']
    name = cfg["model"]["name"].lower()
    is_seq = name in ("tsn", "tcn")

    # ---- data & model ----
    if is_seq:
        loader, _ = build_sequence_loaders(cfg["data"]["processed_root"], args.split, bs,
                                           context_len=int(cfg["model"]["context_len"]),
                                           augment_cfg=None, balanced=False)
        if name == "tsn":
            from models.tsn_sleep import TinySleepNet
            model = TinySleepNet(in_channels=cfg["model"]["in_channels"],
                                 num_classes=cfg["model"]["num_classes"],
                                 base=cfg["model"]["base_channels"],
                                 dropout=cfg["model"]["dropout"],
                                 use_bilstm=cfg["model"].get("use_bilstm", True),
                                 lstm_hidden=cfg["model"].get("lstm_hidden", 128),
                                 lstm_layers=cfg["model"].get("lstm_layers", 1)).to(device)
        else:  # tcn
            from models.tcn_sleep import SleepTCN
            model = SleepTCN(in_channels=cfg["model"]["in_channels"],
                             num_classes=cfg["model"]["num_classes"],
                             base=cfg["model"]["base_channels"],
                             dropout=cfg["model"]["dropout"],
                             tcn_channels=cfg["model"].get("tcn_channels", 128),
                             tcn_layers=cfg["model"].get("tcn_layers", 6)).to(device)
    else:
        loader, _ = build_loaders(cfg["data"]["processed_root"], args.split, bs, augment_cfg=None, balanced=False)
        from models.cnn_sleep import SleepCNN
        model = SleepCNN(in_channels=cfg["model"]["in_channels"],
                         num_classes=cfg["model"]["num_classes"],
                         base=cfg["model"]["base_channels"],
                         dropout=cfg["model"]["dropout"],
                         attn=cfg["model"].get("attn", True)).to(device)

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    # ---- TTA modes ----
    if args.tta == "none":
        y_true, y_pred, y_prob = eval_no_tta(model, loader, device, is_seq=is_seq)
    elif args.tta == "bn_only":
        y_true, y_pred, y_prob = eval_bn_only(model, loader, device, is_seq=is_seq)
    else:
        y_true, y_pred = tent_adapt(model, loader, device, cfg)
        y_prob = None

    # ---- metrics ----
    print(summarize(y_true, y_pred))
    mF1  = macro_f1(y_true, y_pred)
    kap  = kappa(y_true, y_pred)
    f1s  = per_class_f1(y_true, y_pred)
    acc  = accuracy(y_true, y_pred)
    bacc = balanced_accuracy(y_true, y_pred, n_classes=cfg["model"]["num_classes"])
    wf1  = weighted_f1(y_true, y_pred, n_classes=cfg["model"]["num_classes"])
    mcc_v= mcc_safe(y_true, y_pred, n_classes=cfg["model"]["num_classes"])
    ece_v= ece(y_true, y_prob, n_bins=15)
    print(f"Extras | acc {acc:.4f} | bAcc {bacc:.4f} | wF1 {wf1:.4f} | MCC {mcc_v:.4f} | ECE {ece_v:.4f}")

    # ---- TensorBoard one-off ----
    run_name = f"eval_{args.tta}_{args.split}_{datetime.now():%Y%m%d-%H%M%S}"
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))
    writer.add_scalar("macroF1", mF1, 0); writer.add_scalar("kappa", kap, 0)
    writer.add_scalar("accuracy", acc, 0); writer.add_scalar("balanced_accuracy", bacc, 0)
    writer.add_scalar("weighted_F1", wf1, 0); writer.add_scalar("MCC", mcc_v, 0); writer.add_scalar("ECE", ece_v, 0)
    for i, nm in enumerate(class_names[:cfg["model"]["num_classes"]]):
        writer.add_scalar(f"F1_{nm}", f1s.get(i, 0.0), 0)
    C = confusion_matrix(y_true, y_pred, n_classes=cfg["model"]["num_classes"])
    log_confmat_figure(writer, "confusion_matrix", C, class_names[:cfg["model"]["num_classes"]], 0)
    writer.close()

if __name__ == "__main__":
    main()
