
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation_sleepedf.py
------------------------
Streaming TENT (and BN-only) runner with robust hooks for Sleep-EDF style NPZ shards.

Highlights
- Robust model loader for models.cnn_sleep.SleepCNN with common checkpoints.
- Robust NPZ loader that autodetects data/label keys and shapes (N, T) or (N, C, T).
- Subject-ordered, one-pass streaming with per-subject resets.
- Writes predictions.csv + stats.json for evaluation with eval_from_csv.py.

Assumptions
- NPZ directory like:  data/processed_npy/{split}/rec_XXXX.npz
- Labels are ints 0..4 (W,N1,N2,N3,REM); strings are mapped if present.

Usage
-----
# BN-only (val)
python run_ablation_sleepedf.py --ckpt artifacts/final_model.pt \
  --data-dir data/processed_npy --split val --method bn_only --bn-stats \
  --out runs_ablate/bn_only_val

# TENT (val) with rails
python run_ablation_sleepedf.py --ckpt artifacts/final_model.pt \
  --data-dir data/processed_npy --split val --method tent --lr 5e-4 --bn-stats --dropout-off \
  --entropy-gate 0.8 --ema-decay 0.99 --reset-interval 200 \
  --out runs_ablate/tent_minrails
"""
from __future__ import annotations

import argparse, json, os, glob, re, time, random
from typing import Dict, Iterable, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn

from streaming_tent import StreamingAdapter, freeze_all_but_bn_affine, disable_dropout

CLASSES = ["W","N1","N2","N3","REM"]
CLASS_TO_INT = {c:i for i,c in enumerate(CLASSES)}


def set_seed(s=42):
    import numpy as np, torch, random, os
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    os.environ["PYTHONHASHSEED"]=str(s)


# ------------------------------
# Robust model loader
# ------------------------------
def load_model(ckpt_path: str, in_channels: int = 1, num_classes: int = 5) -> nn.Module:
    """
    Tries to import SleepCNN from models.cnn_sleep or cnn_sleep.
    Falls back to a generic nn.Module error if import fails.
    """
    try:
        from models.cnn_sleep import SleepCNN  # project layout
    except Exception:
        try:
            from models.cnn_sleep import SleepCNN  # flat layout
        except Exception as e:
            raise ImportError("Could not import SleepCNN from models.cnn_sleep or cnn_sleep. "
                              "Edit load_model() to use your model class.") from e

    # Instantiate with minimal, common signature
    try:
        m = SleepCNN(in_channels=in_channels, num_classes=num_classes)
    except TypeError:
        # Try alternate param names
        try:
            m = SleepCNN(in_channels=in_channels, n_classes=num_classes)
        except Exception as e:
            raise RuntimeError("Instantiate SleepCNN(in_channels=?, num_classes=?) failed. "
                               "Adjust the constructor args to your model.") from e

    # Load checkpoint flexibly
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("model","state_dict","model_state","weights"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    m.load_state_dict(sd, strict=False)
    return m


# ------------------------------
# Robust NPZ loader
# ------------------------------
def _find_array_key(d: Dict[str, np.ndarray], preferred: List[str]) -> Optional[str]:
    for k in preferred:
        if k in d:
            return k
    # fallback: first array-like
    for k,v in d.items():
        if isinstance(v, np.ndarray):
            return k
    return None


def load_npz_record(npz_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
        x: (N, C, T) float32
        y: (N,) int (or None if not found)
    """
    z = np.load(npz_path, allow_pickle=True)
    keys = list(z.keys())

    x_key = _find_array_key(z, ["x","X","data","signal","eeg","inputs","samples"])
    y_key = _find_array_key(z, ["y","Y","label","labels","stage","stages","targets","target"])

    if x_key is None:
        raise ValueError(f"No data array found in {npz_path} (keys={keys})")
    x = z[x_key]
    if x.ndim == 2:
        # (N, T) -> (N, 1, T)
        x = x[:, None, :]
    elif x.ndim == 3:
        pass  # (N, C, T)
    else:
        raise ValueError(f"Unexpected x shape {x.shape} in {npz_path}")

    y = None
    if y_key is not None:
        y_raw = z[y_key]
        if y_raw.dtype.kind in "OUS":  # strings/objects
            y = np.array([CLASS_TO_INT.get(str(s), 0) for s in y_raw], dtype=np.int64)
        else:
            y = y_raw.astype(np.int64).reshape(-1)

    # clean up
    z.close()
    # standardize dtype
    x = x.astype(np.float32, copy=False)
    return x, y


def make_stream_loader(data_dir: str, split: str, batch_size: int = 128) -> Iterable[Tuple[torch.Tensor, Optional[torch.Tensor], Dict]]:
    """
    Iterates subject files in lexicographic order, yielding (x, y, meta) with meta['subject'] and meta['index'].
    """
    split_dir = os.path.join(data_dir, split)
    files = sorted(glob.glob(os.path.join(split_dir, "rec_*.npz")))
    if not files:
        # Also support flat dir without split subfolder
        files = sorted(glob.glob(os.path.join(data_dir, "rec_*.npz")))
    assert files, f"No rec_*.npz found in {split_dir} or {data_dir}"

    for npz in files:
        subj = os.path.splitext(os.path.basename(npz))[0]  # rec_XXXX
        x, y = load_npz_record(npz)
        N = x.shape[0]
        for i in range(0, N, batch_size):
            xb = torch.from_numpy(x[i:i+batch_size])
            yb = torch.from_numpy(y[i:i+batch_size]) if y is not None else None
            meta = {"subject": subj, "index": list(range(i, min(i+batch_size, N)))}
            yield xb, yb, meta


# ------------------------------
# CSV writing
# ------------------------------
def save_predictions_csv(out_dir: str, rows: List[List], classes: List[str]) -> str:
    import csv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictions.csv")
    header = ["subject","index","y_true","y_pred"] + [f"proba_{c}" for c in classes] + ["entropy","adapted"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


# ------------------------------
# One run
# ------------------------------
def one_run(args) -> str:
    device = args.device
    model = load_model(args.ckpt, in_channels=1, num_classes=len(CLASSES))
    if args.method == "bn_only":
        freeze_all_but_bn_affine(model)
        disable_dropout(model)
        adapter = StreamingAdapter(
            model, num_classes=len(CLASSES), lr=0.0,
            update_bn_stats=args.bn_stats, entropy_gate=None,
            ema_decay=None, reset_interval=None, device=device
        )
    elif args.method == "tent":
        freeze_all_but_bn_affine(model)
        if args.dropout_off:
            disable_dropout(model)
        adapter = StreamingAdapter(
            model, num_classes=len(CLASSES), lr=args.lr,
            update_bn_stats=args.bn_stats, entropy_gate=args.entropy_gate,
            ema_decay=args.ema_decay, reset_interval=args.reset_interval,
            device=device, grad_clip=args.grad_clip
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    rows = []
    total = 0
    t_start = time.time()
    for x, y, meta in make_stream_loader(args.data_dir, args.split, args.batch_size):
        info = adapter.observe((x, y, meta))
        preds = info["preds"]
        probs = info["probs"]
        ent = info["entropy_mean"]
        adapted = info["adapted"]
        subj = meta["subject"]
        idxs = meta["index"]
        ys = y.numpy().tolist() if y is not None else [None]*len(preds)
        for k in range(len(preds)):
            prob_row = probs[k].tolist()
            rows.append([subj, int(idxs[k]), ys[k], int(preds[k]), *map(float, prob_row), float(ent), int(adapted)])
        total += len(preds)

    stats = adapter.finalize().__dict__
    stats["coverage"] = float(stats["adapted_batches"]) / max(1, stats["total_batches"])
    stats["skip_rate"] = float(stats["skipped_batches"]) / max(1, stats["total_batches"])
    stats["duration_sec"] = time.time() - t_start
    stats["examples"] = total

    os.makedirs(args.out, exist_ok=True)
    csv_path = save_predictions_csv(args.out, rows, CLASSES)
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return csv_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True, help="Directory containing {split}/rec_*.npz or rec_*.npz")
    p.add_argument("--split", choices=["val","test"], default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--method", choices=["bn_only","tent"], required=True)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--bn-stats", action="store_true")
    p.add_argument("--dropout-off", action="store_true")
    p.add_argument("--entropy-gate", type=float, default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--reset-interval", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    csv_path = one_run(args)
    print(f"[OK] Wrote predictions: {csv_path}")
    print(f"[NOTE] Now evaluate: python eval_from_csv.py --csv {csv_path} --out {os.path.join(args.out,'out_eval')}")

if __name__ == "__main__":
    main()
