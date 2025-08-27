from __future__ import annotations
import os, time, yaml, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from utils.common import set_seed, get_device
from utils.metrics import macro_f1, kappa, per_subject_scores
from models.cnn_sleep import SleepCNN
from data.sleepedf_mne import NPZDataset
from tta_tent import TTARailsConfig, TTALearner, BNOnly

class TorchDataset(Dataset):
    def __init__(self, npz_dataset):
        self.ds = npz_dataset
        self.n = len(self.ds)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        X, y, subs = self.ds.get_batch(np.array([idx]))
        return torch.from_numpy(X[0]).float(), int(y[0]), int(subs[0])

def corrupt(x, mode="none", level=0.2):
    if mode == "none":
        return x
    if mode == "emg":
        # add high-freq noise
        noise = torch.randn_like(x) * level
        return x + noise
    if mode == "line":
        B, C, T = x.shape
        t = torch.arange(T, device=x.device).float().unsqueeze(0).unsqueeze(0)
        line = 0.2 * torch.sin(2*3.14159*50*t/T)  # synthetic 50Hz-ish
        return x + level * line
    if mode == "dropout":
        mask = (torch.rand_like(x) > level).float()
        return x * mask
    return x

@torch.no_grad()
def evaluate_static(model, loader, device):
    model.eval()
    ys, ps, subs = [], [], []
    for X, y, s in loader:
        X = X.to(device)
        logits = model(X)
        pred = logits.argmax(dim=-1).cpu().numpy()
        ys.append(y.numpy()); ps.append(pred); subs.append(s.numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps); subjects = np.concatenate(subs)
    return y_true, y_pred, subjects

def evaluate_bn_only(model, loader, device, robust, robust_level, batch_size=8):
    bn = BNOnly(model, device)
    ys, ps, subs = [], [], []
    # streaming mini-batches
    bufX, bufY, bufS = [], [], []
    t_start = time.time()
    for X, y, s in loader:
        bufX.append(X); bufY.append(y); bufS.append(s)
        if len(bufX) == batch_size:
            Xb = torch.cat(bufX, dim=0)
            Xb = corrupt(Xb.to(device), robust, robust_level)
            bn.refresh(Xb)  # update BN stats only
            logits = bn.forward(Xb)
            pred = logits.argmax(dim=-1).cpu().numpy()
            ys.append(torch.cat(bufY).numpy())
            ps.append(pred)
            subs.append(torch.cat(bufS).numpy())
            bufX, bufY, bufS = [], [], []
    if bufX:
        Xb = torch.cat(bufX, dim=0)
        Xb = corrupt(Xb.to(device), robust, robust_level)
        bn.refresh(Xb)
        logits = bn.forward(Xb)
        pred = logits.argmax(dim=-1).cpu().numpy()
        ys.append(torch.cat(bufY).numpy())
        ps.append(pred)
        subs.append(torch.cat(bufS).numpy())
    latency = (time.time() - t_start) / max(1, sum(len(p) for p in ps))
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps); subjects = np.concatenate(subs)
    return y_true, y_pred, subjects, latency

def evaluate_tent(model, loader, device, tta_cfg: TTARailsConfig, robust, robust_level, batch_size=8):
    learner = TTALearner(model, tta_cfg, device)
    ys, ps, subs = [], [], []
    t_start = time.time()
    bufX, bufY, bufS = [], [], []
    for X, y, s in loader:
        bufX.append(X); bufY.append(y); bufS.append(s)
        if len(bufX) == batch_size:
            Xb = torch.cat(bufX, dim=0)
            Xb = corrupt(Xb.to(device), robust, robust_level)
            logits, H = learner.adapt_step(Xb)
            pred = logits.argmax(dim=-1).cpu().numpy()
            ys.append(torch.cat(bufY).numpy())
            ps.append(pred)
            subs.append(torch.cat(bufS).numpy())
            bufX, bufY, bufS = [], [], []
    if bufX:
        Xb = torch.cat(bufX, dim=0)
        Xb = corrupt(Xb.to(device), robust, robust_level)
        logits, H = learner.adapt_step(Xb)
        pred = logits.argmax(dim=-1).cpu().numpy()
        ys.append(torch.cat(bufY).numpy())
        ps.append(pred)
        subs.append(torch.cat(bufS).numpy())
    latency = (time.time() - t_start) / max(1, sum(len(p) for p in ps))
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps); subjects = np.concatenate(subs)
    return y_true, y_pred, subjects, latency

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    ap.add_argument("--tta", type=str, default="tent", choices=["none","bn_only","tent"])
    ap.add_argument("--robust", type=str, default="none", choices=["none","emg","line","dropout"])
    ap.add_argument("--robust_level", type=float, default=None)
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 1337))
    device = get_device(cfg.get("device", "auto"))

    if args.robust_level is None:
        args.robust_level = cfg["eval"]["robust_level"]

    proc = cfg["data"]["processed_root"]
    te = NPZDataset(os.path.join(proc, cfg["data"]["splits"]["test"]))
    C = np.load(te.files[0])["X"].shape[1]
    n_classes = cfg["model"]["n_classes"]

    model = SleepCNN(in_ch=C, n_classes=n_classes, hidden=cfg["model"]["hidden"],
                     attn_hidden=cfg["model"]["attn_hidden"], dropout=cfg["model"]["dropout"]).to(device)
    model.load_state_dict(torch.load("source_model.pt", map_location=device))

    # create a simple sequential index loader (per-epoch stream)
    class SeqDS(Dataset):
        def __init__(self, ds): self.ds, self.n = ds, len(ds)
        def __len__(self): return self.n
        def __getitem__(self, idx):
            X,y,s = self.ds.get_batch(np.array([idx]))
            return torch.from_numpy(X[0]).float(), int(y[0]), int(s[0])
    loader = DataLoader(SeqDS(te), batch_size=1, shuffle=False)

    robust = args.robust if args.robust != "none" else cfg["eval"]["robust"]
    robust_level = args.robust_level

    if args.tta == "none":
        y_true, y_pred, subjects = evaluate_static(model, loader, device)
        latency = 0.0
    elif args.tta == "bn_only":
        y_true, y_pred, subjects, latency = evaluate_bn_only(model, loader, device, robust, robust_level, batch_size=cfg["tta"]["batch_size"])
    else:
        tta_cfg = TTARailsConfig(
            entropy_gate=cfg["tta"]["entropy_gate"],
            ema=cfg["tta"]["ema"],
            reset_every=cfg["tta"]["reset_every"],
            grad_clip=cfg["tta"]["grad_clip"],
            lr=cfg["tta"]["lr"],
        )
        y_true, y_pred, subjects, latency = evaluate_tent(model, loader, device, tta_cfg, robust, robust_level, batch_size=cfg["tta"]["batch_size"])

    overall = dict(macro_f1=float(macro_f1(y_true, y_pred)), kappa=float(kappa(y_true, y_pred)))
    per_sub = per_subject_scores(y_true, y_pred, subjects)

    print("Overall:", overall)
    print(f"Median per-epoch latency (s): {latency:.6f}")
    # Save results for figures/tables
    os.makedirs("figs", exist_ok=True)
    np.savez("figs/results_eval.npz", y_true=y_true, y_pred=y_pred, subjects=subjects, overall=list(overall.items()))

if __name__ == "__main__":
    main()
