# train_source.py
"""
Source training with tqdm progress bars, TensorBoard logging, and checkpoints.

Features
- Works with both "tsn" (TinySleepNet-style sequence model) and "cnn" (single-epoch CNN).
- Logs (TensorBoard): train loss, LR, val macroF1, kappa, per-class F1.
- Checkpoints:
    * best.pt        — best validation macroF1 so far
    * last.pt        — final checkpoint at exit
    * epoch_XXX_*.pt — optional every-N-epoch snapshots (use --save-every)
- Resume training: pass --resume checkpoints/whatever.pt

CLI (common)
    --cfg config.yaml
    --logdir runs
    --ckpt-dir checkpoints
    --save-every 0          # 0 disables per-epoch snapshots; >0 saves every N epochs
    --keep-topk 3           # max # of per-epoch snapshots to keep (best/last are never deleted)
    --resume checkpoints/best.pt  # resume from a checkpoint (model/opt/sched/epoch if present)
    --no-tqdm               # disable progress bars
"""

import os
import argparse
from datetime import datetime

import yaml
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from contextlib import nullcontext


from utils.common import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,           # used for --resume
    WarmupCosine,
    FocalLoss,
)
from utils.metrics import macro_f1, kappa, per_class_f1


# ----------------------------- Helpers -----------------------------

def get_loss(cfg, class_weights=None):
    """Select loss (focal or cross-entropy) with optional class weights and label smoothing."""
    if cfg["train"]["loss"] == "focal":
        return FocalLoss(
            gamma=cfg["train"].get("gamma", 1.5),
            weight=class_weights,
            label_smoothing=cfg["train"].get("label_smoothing", 0.0),
        )
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg["train"].get("label_smoothing", 0.0),
    )


def make_class_weights(y, num_classes, device):
    """Inverse-frequency class weights (normalized to mean≈1)."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    if counts.sum() == 0:
        w = np.ones(num_classes, dtype=np.float32)
    else:
        inv = 1.0 / (counts + 1e-6)
        w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32, device=device), counts


def build_dataloaders(cfg):
    """Dispatch to single-epoch or sequence loaders based on model.name, with safe defaults."""
    processed = cfg["data"]["processed_root"]
    bs_train  = cfg["train"]["batch_size"]
    bs_val    = cfg["eval"]["batch_size"]
    augment   = cfg["train"].get("augment", {"enabled": False})
    balanced  = cfg["train"].get("balanced_sampler", True)

    name = cfg["model"]["name"].lower()
    if name in ("tsn", "tcn"):
        from data.sleepedf_mne import build_sequence_loaders
        L = int(cfg["model"]["context_len"])
        train_dl, y_train = build_sequence_loaders(processed, "train", bs_train, context_len=L,
                                                   augment_cfg=augment, balanced=balanced)
        val_dl,   y_val   = build_sequence_loaders(processed, "val",   bs_val,   context_len=L,
                                                   augment_cfg={"enabled": False}, balanced=False)
    else:
        from data.sleepedf_mne import build_loaders
        train_dl, y_train = build_loaders(processed, "train", bs_train,
                                          augment_cfg=augment, balanced=balanced)
        val_dl,   y_val   = build_loaders(processed, "val",   bs_val,
                                          augment_cfg={"enabled": False}, balanced=False)
    return train_dl, val_dl, y_train, y_val



def build_model(cfg, device):
    name = cfg["model"]["name"].lower()
    if name == "tsn":
        from models.tsn_sleep import TinySleepNet
        model = TinySleepNet(...).to(device)
    elif name == "tcn":                      # <-- add this
        from models.tcn_sleep import SleepTCN
        model = SleepTCN(
            in_channels=cfg["model"]["in_channels"],
            num_classes=cfg["model"]["num_classes"],
            base=cfg["model"]["base_channels"],
            dropout=cfg["model"]["dropout"],
            tcn_channels=cfg["model"].get("tcn_channels", 128),
            tcn_layers=cfg["model"].get("tcn_layers", 6),
        ).to(device)
    else:
        from models.cnn_sleep import SleepCNN
        model = SleepCNN(...).to(device)
    return model



def bias_init_from_priors(model, counts, num_classes, device):
    """Initialize classifier bias to log-priors for faster convergence."""
    total = counts.sum() + 1e-6
    probs = (counts + 1e-6) / total
    # tsn & cnn both expose a Linear head called 'classifier'
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        with torch.no_grad():
            model.classifier.bias.copy_(torch.log(torch.tensor(probs, dtype=torch.float32, device=device)))


def save_epoch_checkpoint(ckpt_dir, epoch, mF1, model, opt, sched, cfg, keep_topk=3):
    """Save an 'every N epochs' snapshot and keep only the most recent 'keep_topk' of them."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}_macroF1_{mF1:.4f}.pt")
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": getattr(sched, "state_dict", lambda: {})(),
        "best_macro": mF1,
        "cfg": cfg,
    }
    save_checkpoint(payload, path)

    # prune old epoch_* snapshots (keep best.pt/last.pt)
    snaps = sorted(
        [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pt")],
        key=lambda p: os.path.getmtime(p),
    )
    if keep_topk > 0 and len(snaps) > keep_topk:
        for p in snaps[:-keep_topk]:
            try:
                os.remove(p)
            except Exception:
                pass


def maybe_resume(resume_path, model, opt, sched, device):
    """Load checkpoint if provided; return start_epoch and best_macro."""
    if not resume_path:
        return 0, -1.0
    ckpt = load_checkpoint(resume_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    if "opt" in ckpt and opt is not None:
        try:
            opt.load_state_dict(ckpt["opt"])
        except Exception:
            pass
    if "sched" in ckpt and sched is not None:
        try:
            if hasattr(sched, "load_state_dict"):
                sched.load_state_dict(ckpt["sched"])
        except Exception:
            pass
    start_epoch = int(ckpt.get("epoch", 0))
    best_macro = float(ckpt.get("best_macro", -1.0))
    print(f"Resumed from {resume_path} at epoch {start_epoch} (best_macro={best_macro:.4f})")
    return start_epoch, best_macro


# ----------------------------- Train / Eval -----------------------------

def train_one_epoch(model, dl, opt, loss_fn, device, use_seq, grad_accum=1, use_tqdm=True):
    model.train()
    total = 0.0
    opt.zero_grad(set_to_none=True)
    iterator = tqdm(dl, desc="Train", dynamic_ncols=True) if use_tqdm else dl

    for step, (x, y) in enumerate(iterator):
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()

        # Fallback: if a 3D batch slips in for a sequence model, make it (B,1,C,T)
        if use_seq and x.dim() == 3:
            x = x.unsqueeze(1)

        if use_seq:
            logits_center, _ = model(x)           # (B, num_classes)
            loss = loss_fn(logits_center, y) / grad_accum
        else:
            logits = model(x)                    # (B, num_classes)
            loss = loss_fn(logits, y) / grad_accum

        loss.backward()
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        total += float(loss.item()) * grad_accum
        if use_tqdm:
            iterator.set_postfix(avg_loss=f"{(total/(step+1)):.4f}")

    if use_tqdm and hasattr(iterator, "close"):
        iterator.close()
    return total / max(1, len(dl))


@torch.no_grad()
@torch.no_grad()
def evaluate(model, dl, device, use_seq, use_tqdm=True, num_classes=5):
    import torch.nn.functional as F
    model.eval()
    preds, trues, probas = [], [], []
    iterator = tqdm(dl, desc="Eval", dynamic_ncols=True) if use_tqdm else dl

    for x, y in iterator:
        x = x.to(device).float()
        if use_seq and x.dim() == 3:
            x = x.unsqueeze(1)

        if use_seq:
            logits_center, _ = model(x)
            prob = F.softmax(logits_center, dim=1)
            pred = logits_center.argmax(1)
        else:
            logits = model(x)
            prob = F.softmax(logits, dim=1)
            pred = logits.argmax(1)

        preds.append(pred.cpu().numpy())
        trues.append(y.numpy())
        probas.append(prob.cpu().numpy())

    if use_tqdm and hasattr(iterator, "close"):
        iterator.close()

    import numpy as np
    y_pred = np.concatenate(preds) if preds else np.zeros((0,), dtype=np.int64)
    y_true = np.concatenate(trues) if trues else np.zeros((0,), dtype=np.int64)
    y_prob = np.concatenate(probas) if probas else None
    return y_true, y_pred, y_prob



# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml", help="Path to YAML config")
    ap.add_argument("--logdir", type=str, default="runs", help="TensorBoard root log directory")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    ap.add_argument("--save-every", type=int, default=0, help="Save snapshot every N epochs (0 disables)")
    ap.add_argument("--keep-topk", type=int, default=3, help="Keep at most K epoch_* snapshots (best/last are preserved)")
    ap.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    ap.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")
    args = ap.parse_args()

    # ---- config & setup ----
    cfg = yaml.safe_load(open(args.cfg))
    set_seed(cfg.get("seed", 1337))
    device = get_device()
    model_name = cfg["model"]["name"].lower()
    use_seq = model_name in ("tsn", "tcn")  # sequence models return (logits_center, logits_all)

    # ---- data ----
    train_dl, val_dl, y_train, y_val = build_dataloaders(cfg)
    num_classes = cfg["model"]["num_classes"]

    # ---- class weights (optional) ----
    class_weights = None
    if cfg["train"].get("class_weighted_loss", False) and len(y_train) > 0:
        class_weights, train_counts = make_class_weights(y_train, num_classes, device)
    else:
        train_counts = np.bincount(y_train, minlength=num_classes) if len(y_train) > 0 else np.zeros(num_classes)

    # ---- model ----
    model = build_model(cfg, device)
    if cfg["train"].get("prior_bias_init", True) and len(y_train) > 0:
        bias_init_from_priors(model, train_counts, num_classes, device)

    # ---- optim, sched, loss ----
    opt = optim.AdamW(model.parameters(),
                      lr=cfg["train"]["lr"],
                      weight_decay=cfg["train"]["weight_decay"])
    sched = WarmupCosine(opt,
                         warmup_epochs=cfg["train"]["warmup_epochs"],
                         max_epochs=cfg["train"]["epochs"])
    loss_fn = get_loss(cfg, class_weights=class_weights)

    # ---- resume (optional) ----
    start_epoch, best_macro = maybe_resume(args.resume, model, opt, sched, device)

    # ---- TensorBoard ----
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    run_name = f"{cfg['model']['name']}_{datetime.now():%Y%m%d-%H%M%S}"
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, run_name))
    writer.add_text("config", f"```\n{yaml.safe_dump(cfg)}\n```")
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    for i, n in enumerate(class_names[:num_classes]):
        writer.add_scalar(f"counts/train_{n}", float(train_counts[i]), start_epoch)

    patience = cfg["train"]["early_stop_patience"]

    # ---- training loop ----
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        tr_loss = train_one_epoch(
            model, train_dl, opt, loss_fn, device,
            use_seq=use_seq,
            grad_accum=cfg["train"]["grad_accum"],
            use_tqdm=not args.no_tqdm,
        )

        # step scheduler
        try:
            sched.step()
        except TypeError:
            sched.step(epoch + 1)

        lr = opt.param_groups[0]["lr"]

        # ---- validation (with probs for ECE) ----
        y_true, y_pred, y_prob = evaluate(
            model, val_dl, device,
            use_seq=use_seq,
            use_tqdm=not args.no_tqdm,
            num_classes=num_classes,
        )

        # core & extra metrics
        mF1  = macro_f1(y_true, y_pred)
        kap  = kappa(y_true, y_pred)
        f1s  = per_class_f1(y_true, y_pred)
        acc  = float((y_true == y_pred).mean()) if len(y_true) else 0.0
        # balanced accuracy
        ba, k = 0.0, 0
        for c in range(num_classes):
            m = (y_true == c)
            if m.any():
                ba += float((y_pred[m] == c).mean()); k += 1
        bacc = ba / max(k, 1)
        # weighted F1
        counts_val = np.bincount(y_true, minlength=num_classes) if len(y_true) > 0 else np.zeros(num_classes, dtype=int)
        denom = counts_val.sum() if len(y_true) > 0 else 0
        wf1 = float(sum(f1s.get(c, 0.0) * counts_val[c] for c in range(num_classes)) / denom) if denom else 0.0
        # MCC
        C = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred): C[t, p] += 1
        t_sum = C.sum(1); p_sum = C.sum(0); n = C.sum()
        trc = np.trace(C)
        num = trc * n - (p_sum * t_sum).sum()
        den = np.sqrt((n**2 - (p_sum**2).sum()) * (n**2 - (t_sum**2).sum()))
        mcc_v = float(num / den) if den > 0 else 0.0
        # ECE
        def _ece(y, proba, n_bins=15):
            if proba is None or len(y) == 0: return 0.0
            conf = proba.max(axis=1); preds = proba.argmax(axis=1)
            bins = np.linspace(0, 1, n_bins + 1); e = 0.0
            for i in range(n_bins):
                m = (conf >= bins[i]) & (conf < bins[i+1])
                if m.any():
                    acc_bin = float((preds[m] == y[m]).mean())
                    e += abs(acc_bin - float(conf[m].mean())) * float(m.mean())
            return float(e)
        ece_v = _ece(y_true, y_prob, n_bins=15)

        per = " | ".join([f"{class_names[i]}:F1={f1s.get(i,0):.2f},n={int(counts_val[i])}" for i in range(num_classes)])
        print(f"Epoch {epoch+1:03d} | train_loss {tr_loss:.4f} | val_macroF1 {mF1:.4f} | val_kappa {kap:.4f} | "
              f"acc {acc:.4f} | bAcc {bacc:.4f} | wF1 {wf1:.4f} | MCC {mcc_v:.4f} | ECE {ece_v:.4f}")
        print("  per-class:", per)

        # ---- TensorBoard logging ----
        writer.add_scalar("lr", lr, epoch+1)
        writer.add_scalar("loss/train", tr_loss, epoch+1)
        writer.add_scalar("val/macroF1", mF1, epoch+1)
        writer.add_scalar("val/kappa", kap, epoch+1)
        writer.add_scalar("val/accuracy", acc, epoch+1)
        writer.add_scalar("val/balanced_accuracy", bacc, epoch+1)
        writer.add_scalar("val/weighted_F1", wf1, epoch+1)
        writer.add_scalar("val/MCC", mcc_v, epoch+1)
        writer.add_scalar("val/ECE", ece_v, epoch+1)
        for i, nlab in enumerate(class_names[:num_classes]):
            writer.add_scalar(f"val/F1_{nlab}", f1s.get(i, 0.0), epoch+1)

        # Confusion matrix figure (best-effort)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
            im = ax.imshow(C, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set(xticks=np.arange(C.shape[1]), yticks=np.arange(C.shape[0]),
                   xticklabels=class_names[:num_classes], yticklabels=class_names[:num_classes],
                   ylabel='True', xlabel='Pred')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            thresh = C.max() / 2. if C.max() > 0 else 0.5
            for i in range(C.shape[0]):
                for j in range(C.shape[1]):
                    ax.text(j, i, format(C[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if C[i, j] > thresh else "black", fontsize=8)
            fig.tight_layout()
            writer.add_figure("val/confusion_matrix", fig, global_step=epoch+1)
            plt.close(fig)
        except Exception:
            pass

        # ---- checkpoints ----
        if mF1 > best_macro:
            best_macro = mF1
            save_checkpoint(
                {"epoch": epoch+1, "model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": getattr(sched, "state_dict", lambda: {})(), "best_macro": best_macro, "cfg": cfg},
                os.path.join(args.ckpt_dir, "best.pt"),
            )
            print("  ↳ saved new best to checkpoints/best.pt")
            patience = cfg["train"]["early_stop_patience"]
        else:
            patience -= 1

        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
            save_epoch_checkpoint(args.ckpt_dir, epoch+1, mF1, model, opt, sched, cfg, keep_topk=args.keep_topk)

        if patience <= 0:
            print("Early stopping.")
            break

    # ---- save last & close ----
    save_checkpoint(
        {"epoch": epoch+1, "model": model.state_dict(), "opt": opt.state_dict(),
         "sched": getattr(sched, "state_dict", lambda: {})(), "best_macro": best_macro, "cfg": cfg},
        os.path.join(args.ckpt_dir, "last.pt"),
    )
    writer.close()


if __name__ == "__main__":
    main()
