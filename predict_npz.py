# predict_npz.py  (safe windows + writable tensors)
import argparse, os, csv, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

def device_auto():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def median_smooth(labels, k=5):
    if k <= 1: return labels
    pad = k // 2
    x = np.pad(labels, (pad, pad), mode="edge")
    out = np.empty_like(labels)
    for i in range(labels.shape[0]):
        out[i] = int(np.median(x[i:i+k]))
    return out

def make_windows_safe(X, L):
    """X: (N,C,T) -> (N,L,C,T) with edge padding; simple & robust."""
    N, C, T = X.shape
    pad = L // 2
    Xpad = np.pad(X, ((pad, pad), (0, 0), (0, 0)), mode="edge")  # (N+2p, C, T)
    # stack slices: for each center i (0..N-1), take window [i, i+L)
    win_list = [Xpad[i:i+L] for i in range(N)]                    # each (L,C,T)
    Xw = np.stack(win_list, axis=0)                               # (N,L,C,T)
    return Xw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    ap.add_argument("--ckpt", type=str, default="artifacts/final_model.pt")
    ap.add_argument("--npz", type=str, required=True, help="Processed shard path, e.g. data/processed_npy/rec_0114.npz")
    ap.add_argument("--out", type=str, default="predictions.csv")
    ap.add_argument("--smooth", type=str, default="none", choices=["none","median5","median9"])
    ap.add_argument("--save-probs", action="store_true",
                help="Also save per-epoch class probabilities to a companion NPZ")
    ap.add_argument("--probs-out", type=str, default="",
                help="Optional path for probs NPZ (default: alongside CSV with _probs.npz suffix)")

    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    name = cfg["model"]["name"].lower()
    num_classes = int(cfg["model"]["num_classes"])
    device = device_auto()

    # --- load shard
    arr = np.load(args.npz)
    X = arr["X"].astype(np.float32)  # (N,C,T) or (N,T,C)
    y = arr["y"].astype(np.int64) if "y" in arr else None

    # --- ensure (N,C,T)
    if X.ndim != 3:
        raise SystemExit(f"Expected X to be 3D (N,C,T), got shape {X.shape}")
    N, A, B = X.shape
    # Heuristic: T should be big (~3000), C small (<=8). If (N,T,C), swap.
    if A > B and B <= 8:
        # looks like (N,T,C) -> (N,C,T)
        X = np.transpose(X, (0, 2, 1))
        N, A, B = X.shape
    C, T = A, B
    if T < 32:
        raise SystemExit(f"Unrealistic T={T}. Check channel/time axes for {args.npz}")

    # --- build model
    if name in ("tsn","tcn"):
        if name == "tcn":
            from models.tcn_sleep import SleepTCN
            model = SleepTCN(
                in_channels=cfg["model"]["in_channels"],
                num_classes=num_classes,
                base=cfg["model"]["base_channels"],
                dropout=cfg["model"]["dropout"],
                tcn_channels=cfg["model"].get("tcn_channels", 128),
                tcn_layers=cfg["model"].get("tcn_layers", 6),
            ).to(device)
        else:
            from models.tsn_sleep import TinySleepNet
            model = TinySleepNet(
                in_channels=cfg["model"]["in_channels"],
                num_classes=num_classes,
                base=cfg["model"]["base_channels"],
                dropout=cfg["model"]["dropout"],
                use_bilstm=cfg["model"].get("use_bilstm", True),
                lstm_hidden=cfg["model"].get("lstm_hidden", 128),
                lstm_layers=cfg["model"].get("lstm_layers", 1),
            ).to(device)
    else:
        from models.cnn_sleep import SleepCNN
        model = SleepCNN(
            in_channels=cfg["model"]["in_channels"],
            num_classes=num_classes,
            base=cfg["model"]["base_channels"],
            dropout=cfg["model"]["dropout"],
            attn=cfg["model"].get("attn", True),
        ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # --- build loader
    if name in ("tsn","tcn"):
        L = int(cfg["model"]["context_len"])
        Xw = make_windows_safe(X, L)                                # (N,L,C,T)
        # make contiguous & writable
        Xw = np.ascontiguousarray(Xw, dtype=np.float32)
        ds = TensorDataset(torch.tensor(Xw, dtype=torch.float32), torch.arange(N))
    else:
        X = np.ascontiguousarray(X, dtype=np.float32)
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.arange(N))

    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    # --- predict
    preds = np.empty((N,), dtype=np.int64)
    confs = np.empty((N,), dtype=np.float32)
    with torch.no_grad():
        for xb, idx in dl:
            xb = xb.to(device).float()
            if name in ("tsn","tcn") and xb.dim() == 4:
                out = model(xb)                  # (logits_center, logits_all) or logits
                logits = out[0] if isinstance(out, tuple) else out
            else:
                logits = model(xb)
            prob = F.softmax(logits, dim=1)
            pmax, yhat = torch.max(prob, dim=1)
            ii = idx.numpy()
            preds[ii] = yhat.cpu().numpy()
            confs[ii] = pmax.cpu().numpy()

    # collect probs for optional saving
    all_probs = np.empty((N, num_classes), dtype=np.float32)
    with torch.no_grad():
        for xb, idx in dl:
            xb = xb.to(device).float()
            if name in ("tsn","tcn") and xb.dim() == 4:
                out = model(xb)
                logits = out[0] if isinstance(out, tuple) else out
            else:
                logits = model(xb)
            prob = F.softmax(logits, dim=1).cpu().numpy()
            ii = idx.numpy()
            all_probs[ii] = prob
            pmax = prob.max(axis=1)
            yhat = prob.argmax(axis=1)
            preds[ii] = yhat
            confs[ii] = pmax

    # optional smoothing
    if args.smooth.startswith("median"):
        k = int(args.smooth.replace("median",""))
        preds = median_smooth(preds, k=k)

    # --- save per-class probabilities if asked ---
    if args.save_probs:
        probs_path = args.probs_out or os.path.splitext(args.out)[0] + "_probs.npz"
        os.makedirs(os.path.dirname(probs_path) or ".", exist_ok=True)
        np.savez_compressed(probs_path, probs=all_probs, true=(y if y is not None else -np.ones(N, dtype=np.int64)))
        print(f"Saved per-epoch probabilities to {probs_path}")


    # --- write CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        header = ["epoch","pred","conf"]
        if y is not None: header += ["true"]
        w.writerow(header)
        for i in range(N):
            row = [i, int(preds[i]), float(confs[i])]
            if y is not None: row += [int(y[i])]
            w.writerow(row)
    print(f"Wrote {args.out} with {N} rows.")

if __name__ == "__main__":
    main()
