from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_entropy(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * (p.clamp_min(1e-8).log())).sum(dim=-1)

def collect_bn_affine_params(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.affine:
                params += [m.weight, m.bias]
    return [p for p in params if p is not None]

def collect_bn_modules(model: nn.Module) -> List[nn.Module]:
    return [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]

@dataclass
class TTARailsConfig:
    entropy_gate: float = 1.6
    ema: float = 0.99
    reset_every: int = 200
    grad_clip: float = 1.0
    lr: float = 1e-3

class TTALearner:
    """Tent-style TTA on BN affine params + BN stats refresh, with simple rails."""
    def __init__(self, model: nn.Module, cfg: TTARailsConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        # freeze all but BN affine
        for p in self.model.parameters():
            p.requires_grad = False

        self.bn_params = collect_bn_affine_params(self.model)
        for p in self.bn_params:
            p.requires_grad = True

        self.opt = torch.optim.SGD(self.bn_params, lr=cfg.lr)
        self.step_count = 0

        # EMA state for BN affine + running stats
        self.ema_state = self._copy_bn_state()

    def _copy_bn_state(self):
        snap = {}
        for i, m in enumerate(collect_bn_modules(self.model)):
            snap[i] = dict(
                weight=m.weight.detach().clone() if m.affine else None,
                bias=m.bias.detach().clone() if m.affine else None,
                running_mean=m.running_mean.detach().clone(),
                running_var=m.running_var.detach().clone(),
                num_batches_tracked=m.num_batches_tracked.detach().clone(),
            )
        return snap

    def _load_bn_state(self, snap):
        for i, m in enumerate(collect_bn_modules(self.model)):
            s = snap[i]
            if m.affine:
                m.weight.data.copy_(s["weight"])
                m.bias.data.copy_(s["bias"])
            m.running_mean.data.copy_(s["running_mean"])
            m.running_var.data.copy_(s["running_var"])
            m.num_batches_tracked.data.copy_(s["num_batches_tracked"])

    @torch.no_grad()
    def _ema_update(self):
        for i, m in enumerate(collect_bn_modules(self.model)):
            s = self.ema_state[i]
            if m.affine:
                s["weight"].mul_(self.cfg.ema).add_(m.weight.detach(), alpha=1-self.cfg.ema)
                s["bias"].mul_(self.cfg.ema).add_(m.bias.detach(),   alpha=1-self.cfg.ema)
            s["running_mean"].mul_(self.cfg.ema).add_(m.running_mean.detach(), alpha=1-self.cfg.ema)
            s["running_var"].mul_(self.cfg.ema).add_(m.running_var.detach(),   alpha=1-self.cfg.ema)
            s["num_batches_tracked"].mul_(self.cfg.ema).add_(m.num_batches_tracked.detach(), alpha=1-self.cfg.ema)

    def adapt_step(self, x):
        """One streaming step: optionally update BN stats + one gradient step on entropy."""
        self.model.train()  # enable BN stats
        x = x.to(self.device)
        self.opt.zero_grad(set_to_none=True)
        logits = self.model(x)
        H = softmax_entropy(logits)  # (B,)
        if H.mean().item() <= self.cfg.entropy_gate:
            loss = H.mean()
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.bn_params, self.cfg.grad_clip)
            self.opt.step()
        # update EMA from current BN state
        self._ema_update()
        self.step_count += 1
        # periodic reset to EMA
        if self.cfg.reset_every and (self.step_count % self.cfg.reset_every == 0):
            self._load_bn_state(self.ema_state)
        return logits.detach(), H.detach()

    @torch.no_grad()
    def forward(self, x):
        self.model.eval()  # inference (no BN stats update)
        return self.model(x.to(self.device))

class BNOnly:
    """BN-only recalibration: update running stats without gradient steps."""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def refresh(self, x):
        self.model.train()  # update BN stats
        _ = self.model(x.to(self.device))

    @torch.no_grad()
    def forward(self, x):
        self.model.eval()
        return self.model(x.to(self.device))
