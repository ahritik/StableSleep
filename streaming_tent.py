
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streaming_tent.py
-----------------
Source-free Test-Time Adaptation (TENT) with stability rails for streaming EEG sleep staging.

Key features
------------
- BN-only adaptation (affine params), optional BN running stats update.
- Dropout kept OFF during adaptation.
- Entropy gate: skip updates on high-uncertainty batches.
- Periodic EMA reset towards source weights.
- Per-subject resets for streaming realism.
- Timing, coverage, skip rate, reset counts, divergence flags.
- Works with arbitrary PyTorch models producing logits.

Expected data loader
--------------------
An iterable that yields (x, y, meta) with shapes:
- x: (B, C, T) or (B, C, T, ...) tensor
- y: (B,) tensor of true labels (optional; can be None for unlabeled test)
- meta: dict with keys: 'subject' (str or int), 'index' (int) per-epoch

Integration
-----------
You must provide a model (nn.Module) and a factory/loader that yields per-subject ordered batches.
Use the provided helpers `freeze_all_but_bn_affine` and `disable_dropout` beforehand.

Outputs
-------
- Per-step metrics (optional)
- Aggregate stats in a dict
- (Optional) CSV writer hook for predictions per subject

Usage sketch
------------
from streaming_tent import (
    freeze_all_but_bn_affine, disable_dropout, tent_step, StreamingAdapter
)

model = load_model(...)
freeze_all_but_bn_affine(model)
disable_dropout(model)

adapter = StreamingAdapter(
    model, num_classes=5, lr=1e-3, update_bn_stats=True,
    entropy_gate=0.2, ema_decay=0.98, reset_interval=200, device="cuda"
)
for batch in loader:
    adapter.observe(batch)
stats = adapter.finalize()
print(stats)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def is_dropout(m: nn.Module) -> bool:
    name = m.__class__.__name__.lower()
    return name.startswith("dropout")


def is_batchnorm(m: nn.Module) -> bool:
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))


def disable_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if is_dropout(m):
            m.eval()


def freeze_all_but_bn_affine(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad_(False)
    # Allow BN affine only
    for m in model.modules():
        if is_batchnorm(m):
            if getattr(m, 'weight', None) is not None:
                m.weight.requires_grad_(True)
            if getattr(m, 'bias', None) is not None:
                m.bias.requires_grad_(True)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # logits: (B, C)
    p = F.softmax(logits, dim=-1)
    ent = -(p * (p + 1e-8).log()).sum(dim=1)
    return ent, p


@dataclass
class AdapterStats:
    total_batches: int = 0
    adapted_batches: int = 0
    skipped_batches: int = 0
    resets: int = 0
    divergence_events: int = 0
    total_examples: int = 0
    total_time_sec: float = 0.0
    per_batch_ms: float = 0.0
    notes: str = ""
    by_subject: Dict[str, Dict[str, float]] = field(default_factory=dict)


class StreamingAdapter:
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        update_bn_stats: bool = True,
        entropy_gate: Optional[float] = None,
        ema_decay: Optional[float] = 0.99,
        reset_interval: Optional[int] = None,
        device: str = "cpu",
        grad_clip: Optional[float] = None,
    ) -> None:
        """
        Args:
            update_bn_stats: if True, keep BN in train mode (running stats update)
            entropy_gate: if set (e.g., 0.2~1.5), skip adaptation when batch mean entropy > gate
            ema_decay: if set (0<d<1), EMA reset towards source weights every `reset_interval` steps
            reset_interval: steps between EMA resets; if None and ema_decay set, defaults to 200
        """
        self.model = model.to(device)
        self.model.train() if update_bn_stats else self.model.eval()
        disable_dropout(self.model)
        self.device = device
        self.entropy_gate = entropy_gate
        self.ema_decay = ema_decay
        self.reset_interval = reset_interval if reset_interval is not None else (200 if ema_decay is not None else None)
        self.grad_clip = grad_clip

        # Opt on BN affine only
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=lr)

        # Keep a copy of source weights for EMA resets
        self.source_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        self.stats = AdapterStats()
        self.curr_subject = None
        self.num_classes = num_classes
        self._step_idx = 0

    def _maybe_subject_reset(self, subject_id: str):
        if self.curr_subject is None:
            self.curr_subject = subject_id
            return
        if subject_id != self.curr_subject:
            self.curr_subject = subject_id
            # Soft reset: EMA towards source on subject boundary
            if self.ema_decay is not None:
                self._ema_reset(force=True)

    def _ema_reset(self, force: bool = False):
        self.stats.resets += 1
        with torch.no_grad():
            cur = self.model.state_dict()
            for k in cur:
                src = self.source_state[k]
                if cur[k].dtype.is_floating_point:
                    if force:
                        cur[k].copy_(src)
                    else:
                        cur[k].lerp_(src, 1 - self.ema_decay)
            self.model.load_state_dict(cur)

    def _should_skip(self, ent_mean: float) -> bool:
        return (self.entropy_gate is not None) and (ent_mean > self.entropy_gate)

    def observe(self, batch: Tuple[torch.Tensor, Optional[torch.Tensor], Dict]) -> Dict:
        """Process one batch: possible adaptation + return predictions and info.
        batch = (x, y, meta)
        """
        x, y, meta = batch
        t0 = time.time()
        self._maybe_subject_reset(str(meta.get("subject", "unknown")))

        self.model.train()  # ensure BN updates if enabled
        disable_dropout(self.model)
        x = x.to(self.device)
        logits = self.model(x)
        ent, p = entropy_from_logits(logits)
        ent_mean = float(ent.mean().detach().cpu())

        do_adapt = True
        if self._should_skip(ent_mean):
            do_adapt = False

        loss = ent.mean()
        if do_adapt and torch.isfinite(loss):
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self.stats.adapted_batches += 1
        else:
            self.stats.skipped_batches += 1

        preds = p.argmax(dim=1).detach().cpu()
        info = {
            "preds": preds.numpy(),
            "probs": p.detach().cpu().numpy(),
            "entropy_mean": ent_mean,
            "loss": float(loss.detach().cpu()),
            "subject": meta.get("subject", "unknown"),
            "index": meta.get("index", None),
            "adapted": int(do_adapt),
        }

        self.stats.total_batches += 1
        self.stats.total_examples += x.shape[0]
        if self.reset_interval is not None and (self._step_idx + 1) % self.reset_interval == 0:
            self._ema_reset(force=False)

        self._step_idx += 1
        self.stats.total_time_sec += (time.time() - t0)
        return info

    def finalize(self) -> AdapterStats:
        if self.stats.total_batches:
            self.stats.per_batch_ms = 1000.0 * self.stats.total_time_sec / self.stats.total_batches
        return self.stats
