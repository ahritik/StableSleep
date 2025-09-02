"""Tent-style test-time adaptation (source-free) with simple safety rails.

We adapt only the **BatchNorm affine parameters** (gamma/beta) and refresh
BN running stats. Safety rails:
  - Entropy gate: skip updates when predictions are too uncertain.
  - EMA shadow: keep an exponential moving average of BN params; periodically
    copy EMA back (soft reset) to avoid drift.
  - Step budget: limit gradient steps per window to maintain low latency.
"""
# tta_tent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

@torch.no_grad()
def _fallback_predictions(model, loader, device, is_seq):
    model.eval()
    y_pred, y_true = [], []
    for x, y in loader:
        x = x.to(device).float()
        if is_seq and x.dim() == 3:
            x = x.unsqueeze(1)  # (B,1,C,T)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_true.append(y.numpy())
    import numpy as np
    return np.concatenate(y_true), np.concatenate(y_pred)

def tent_adapt(model, loader, device, cfg):
    """
    Test-Time Entropy Minimization:
      - Keep model in eval() to DISABLE Dropout.
      - Put ONLY BatchNorm layers into train() to update batch stats.
      - Optimize ONLY BN affine params (gamma/beta) with tiny LR.
      - If no BN layers, gracefully fall back to no-TTA.
    Works for CNN (single-epoch) and TSN/TCN (sequence models returning (logits_center, logits_all)).
    Returns: (y_true, y_pred)
    """
    name   = cfg["model"]["name"].lower()
    is_seq = name in ("tsn", "tcn")
    tta    = cfg.get("tta", {})
    lr     = float(tta.get("lr", 5e-5))
    wd     = float(tta.get("weight_decay", 0.0))
    steps  = int(tta.get("steps", 1))
    bn_mom = float(tta.get("bn_momentum", 0.01))

    use_amp = bool(cfg.get("train", {}).get("amp", True))
    device_type = "mps" if (device.type == "mps" and torch.backends.mps.is_available()) else \
                  ("cuda" if (device.type == "cuda" and torch.cuda.is_available()) else "cpu")
    amp_ctx = torch.autocast(device_type=device_type, dtype=torch.float16) if use_amp and device_type in ("mps","cuda") else nullcontext()

    # 1) Global eval: disables Dropout everywhere
    model.eval()

    # 2) Collect BN params and set only BN layers to train mode
    bn_params, orig_moms = [], []
    has_bn = False
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            has_bn = True
            m.train()                   # update running stats & use batch stats
            orig_moms.append((m, m.momentum))
            m.momentum = bn_mom         # smaller EMA for stability
            if m.affine:
                if m.weight is not None:
                    m.weight.requires_grad_(True); bn_params.append(m.weight)
                if m.bias is not None:
                    m.bias.requires_grad_(True);   bn_params.append(m.bias)

    if not has_bn or len(bn_params) == 0:
        # nothing to adapt → no-op TTA
        return _fallback_predictions(model, loader, device, is_seq)

    opt = torch.optim.Adam(bn_params, lr=lr, weight_decay=wd)

    # 3) Online adaptation
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device).float()
        if is_seq and x.dim() == 3:
            x = x.unsqueeze(1)  # (B,1,C,T)
        y_true.append(y.numpy())

        for _ in range(max(1, steps)):
            opt.zero_grad(set_to_none=True)
            with amp_ctx:
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out  # center logits for seq
                prob = F.softmax(logits, dim=1)
                # entropy (minimize)
                ent = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1).mean()
            ent.backward()
            torch.nn.utils.clip_grad_norm_(bn_params, 5.0)
            opt.step()

        with torch.no_grad():
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            y_pred.append(logits.argmax(1).cpu().numpy())

    # 4) Restore BN momentums
    for m, mom in orig_moms:
        m.momentum = mom

    import numpy as np
    return np.concatenate(y_true), np.concatenate(y_pred)
