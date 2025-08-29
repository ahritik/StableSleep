# eval_sleep.py
import argparse, yaml, torch, numpy as np
from torch import nn
from utils.common import get_device, load_checkpoint
from utils.metrics import summarize
from tta_tent import tent_adapt

def eval_no_tta(model, loader, device, is_tsn=False):
    model.eval(); y_pred, y_true = [], []
    for x, y in loader:
        x = x.to(device).float()
        if is_tsn:
            logits_center, _ = model(x)
            pred = logits_center.argmax(1).cpu().numpy()
        else:
            logits = model(x)
            pred = logits.argmax(1).cpu().numpy()
        y_pred.append(pred); y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)

def eval_bn_only(model, loader, device, is_tsn=False):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device).float()
            if is_tsn: _ = model(x)[0]
            else: _ = model(x)
    return eval_no_tta(model.eval(), loader, device, is_tsn=is_tsn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    ap.add_argument("--tta", type=str, default="none", choices=["none","bn_only","tent"])
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--split", type=str, default="test", choices=["val","test"])
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg)); device = get_device()
    bs = args.batch_size or cfg["eval"]["batch_size"]

    is_tsn = cfg["model"]["name"].lower() == "tsn"
    if is_tsn:
        from data.sleepedf_mne import build_sequence_loaders
        loader, _ = build_sequence_loaders(cfg["data"]["processed_root"], args.split, bs,
                                           context_len=int(cfg["model"]["context_len"]), augment_cfg=None, balanced=False)
        from models.tsn_sleep import TinySleepNet
        model = TinySleepNet(in_channels=cfg["model"]["in_channels"],
                             num_classes=cfg["model"]["num_classes"],
                             base=cfg["model"]["base_channels"],
                             dropout=cfg["model"]["dropout"],
                             use_bilstm=cfg["model"].get("use_bilstm", True),
                             lstm_hidden=cfg["model"].get("lstm_hidden", 128),
                             lstm_layers=cfg["model"].get("lstm_layers", 1)).to(device)
    else:
        from data.sleepedf_mne import build_loaders
        loader, _ = build_loaders(cfg["data"]["processed_root"], args.split, bs, augment_cfg=None, balanced=False)
        from models.cnn_sleep import SleepCNN
        model = SleepCNN(in_channels=cfg["model"]["in_channels"],
                         num_classes=cfg["model"]["num_classes"],
                         base=cfg["model"]["base_channels"],
                         dropout=cfg["model"]["dropout"],
                         attn=cfg["model"].get("attn", True)).to(device)

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    if args.tta == "none":
        y_true, y_pred = eval_no_tta(model, loader, device, is_tsn=is_tsn)
    elif args.tta == "bn_only":
        y_true, y_pred = eval_bn_only(model, loader, device, is_tsn=is_tsn)
    else:
        # Tent adapts BN; for tsn, it uses logits_center internally
        y_true, y_pred = tent_adapt(model, loader, device, cfg)

    print(summarize(y_true, y_pred))

if __name__ == "__main__":
    main()
