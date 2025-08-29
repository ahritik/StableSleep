"""Tent-style test-time adaptation (source-free) with simple safety rails.

We adapt only the **BatchNorm affine parameters** (gamma/beta) and refresh
BN running stats. Safety rails:
  - Entropy gate: skip updates when predictions are too uncertain.
  - EMA shadow: keep an exponential moving average of BN params; periodically
    copy EMA back (soft reset) to avoid drift.
  - Step budget: limit gradient steps per window to maintain low latency.
"""
import numpy as np, torch
from torch import nn, optim

def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy over softmax probabilities per sample (B,)."""
    p = torch.softmax(logits, dim=1) + 1e-8
    return -(p * torch.log(p)).sum(dim=1)

def _bn_affine_params(m):
    """Yield BN-affine parameters (weight/bias) only."""
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm1d):
            if mod.weight is not None:
                yield mod.weight
            if mod.bias is not None:
                yield mod.bias

class BNEMAShadow:
    """Track an EMA of BN-affine params and provide a soft reset."""
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.params = [p.detach().clone() for p in _bn_affine_params(model)]
    @torch.no_grad()
    def update(self, model):
        for buf, p in zip(self.params, _bn_affine_params(model)):
            buf.mul_(self.decay).add_(p.detach(), alpha=(1-self.decay))
    @torch.no_grad()
    def copy_to(self, model):
        for buf, p in zip(self.params, _bn_affine_params(model)):
            p.copy_(buf)

def tent_adapt(model: nn.Module, loader, device, cfg):
    """Run Tent over a whole split and return (y_true, y_pred)."""
    model.train()
    # Train BN layers so running stats refresh; other layers unaffected
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.train()

    # Freeze everything except BN-affine parameters
    for p in model.parameters():
        p.requires_grad_(False)
    bn_params = list(_bn_affine_params(model))
    for p in bn_params:
        p.requires_grad_(True)

    opt = optim.Adam(bn_params, lr=cfg["tta"]["lr"], weight_decay=cfg["tta"]["weight_decay"])
    ema = BNEMAShadow(model, decay=cfg["tta"]["ema_decay"])

    y_pred_all, y_true_all = [], []
    window_steps = 0

    for x, y in loader:
        x = x.to(device).float()
        y_true_all.append(y.numpy())

        logits = model(x)
        if isinstance(logits_out, tuple):   # tsn: (logits_center, logits_all)
            logits = logits_out[0]
        else:
            logits = logits_out
        ent = prediction_entropy(logits)  # (B,)

        # Safety rail: only adapt when reasonably confident
        if ent.mean().item() <= cfg["tta"]["entropy_gate"]:
            loss = ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bn_params, 5.0)
            opt.step()
            ema.update(model)
            window_steps += 1

            # Budget: periodically pull back toward EMA to avoid drift
            if window_steps >= cfg["tta"]["max_steps_per_window"]:
                ema.copy_to(model)
                window_steps = 0

        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return y_true, y_pred
