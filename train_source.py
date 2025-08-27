from __future__ import annotations
import os, time, yaml, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.common import set_seed, get_device
from utils.metrics import macro_f1, kappa
from models.cnn_sleep import SleepCNN
from data.sleepedf_mne import NPZDataset

class TorchDataset(Dataset):
    def __init__(self, npz_dataset):
        self.ds = npz_dataset
        self.n = len(self.ds)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        X, y, subs = self.ds.get_batch(np.array([idx]))
        return torch.from_numpy(X[0]).float(), int(y[0])

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    losses = []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return np.mean(losses)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys, ps = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        pred = logits.argmax(dim=-1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    return dict(macro_f1=macro_f1(y_true, y_pred), kappa=kappa(y_true, y_pred))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 1337))
    device = get_device(cfg.get("device", "auto"))

    # data
    proc = cfg["data"]["processed_root"]
    tr = NPZDataset(os.path.join(proc, cfg["data"]["splits"]["train"]))
    va = NPZDataset(os.path.join(proc, cfg["data"]["splits"]["val"]))

    C = np.load(tr.files[0])["X"].shape[1]
    n_classes = cfg["model"]["n_classes"]

    train_loader = DataLoader(TorchDataset(tr), batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    val_loader   = DataLoader(TorchDataset(va), batch_size=cfg["eval"]["batch_size"], shuffle=False)

    model = SleepCNN(in_ch=C, n_classes=n_classes, hidden=cfg["model"]["hidden"],
                     attn_hidden=cfg["model"]["attn_hidden"], dropout=cfg["model"]["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()

    best_k = -1e9
    patience = cfg["train"]["early_stop_patience"]
    bad = 0
    for epoch in range(cfg["train"]["epochs"]):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_scores = eval_model(model, val_loader, device)
        print(f"Epoch {epoch+1:02d} | loss={tr_loss:.4f} | val κ={val_scores['kappa']:.4f} F1={val_scores['macro_f1']:.4f}")
        if val_scores["kappa"] > best_k:
            best_k = val_scores["kappa"]
            torch.save(model.state_dict(), "source_model.pt")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

if __name__ == "__main__":
    main()
