
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corruption_suite.py
-------------------
Apply common distribution shifts to streaming EEG data for robustness evaluation.

Corruptions
-----------
- amplitude(x; s): multiply by scalar s
- additive_gaussian(x; sigma): N(0, sigma) additive noise
- powerline_hum(x; freq=50/60, snr_db): add sinusoid
- eog_leakage(x; alpha): mix with a low-freq drift component
- emg_noise(x; density): add sparse high-freq bursts
- resample_mismatch(x; factor): linear resample by factor
- band_limited_noise(x; f_lo, f_hi, snr_db)

Usage
-----
from corruption_suite import apply_corruption

x_corrupt = apply_corruption(x, fs=100, kind="powerline", level=0.2, params={"freq":60})
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Optional

def amplitude(x: np.ndarray, level: float, fs: int, **kw) -> np.ndarray:
    return x * (1.0 + level)

def additive_gaussian(x: np.ndarray, level: float, fs: int, **kw) -> np.ndarray:
    sigma = level * np.std(x, axis=-1, keepdims=True)
    return x + np.random.randn(*x.shape) * sigma

def powerline_hum(x: np.ndarray, level: float, fs: int, freq: int = 50, **kw) -> np.ndarray:
    t = np.arange(x.shape[-1]) / fs
    hum = np.sin(2*np.pi*freq*t)[None, :]
    # Scale to achieve rough SNR using level
    return x + level * hum * np.std(x, axis=-1, keepdims=True)

def eog_leakage(x: np.ndarray, level: float, fs: int, **kw) -> np.ndarray:
    # Low-frequency drift component
    t = np.arange(x.shape[-1]) / fs
    drift = np.sin(2*np.pi*0.3*t)[None, :]
    return x + level * drift * np.std(x, axis=-1, keepdims=True)

def emg_noise(x: np.ndarray, level: float, fs: int, **kw) -> np.ndarray:
    # Sparse high-frequency bursts
    y = x.copy()
    n_bursts = max(1, int(level * 5))
    for _ in range(n_bursts):
        start = np.random.randint(0, x.shape[-1] - fs//2)
        length = fs // 10
        burst = np.random.randn(length)[None, :]
        y[..., start:start+length] += burst * np.std(x, axis=-1, keepdims=True)
    return y

def resample_mismatch(x: np.ndarray, level: float, fs: int, **kw) -> np.ndarray:
    # naive linear resample by factor (1+level)
    factor = 1.0 + level
    n = x.shape[-1]
    idx = np.linspace(0, n-1, int(n*factor))
    y = np.interp(np.arange(n), np.arange(int(n*factor)), np.take(x, np.clip(idx.astype(int), 0, n-1), axis=-1))
    return y

def band_limited_noise(x: np.ndarray, level: float, fs: int, f_lo: float = 20.0, f_hi: float = 40.0, **kw) -> np.ndarray:
    n = x.shape[-1]
    noise = np.random.randn(*x.shape)
    # Simple spectral mask via FFT
    X = np.fft.rfft(noise, axis=-1)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    X *= mask[None, :]
    band = np.fft.irfft(X, n, axis=-1)
    band *= level * np.std(x, axis=-1, keepdims=True)
    return x + band

CORRUPTIONS = {
    "amplitude": amplitude,
    "gaussian": additive_gaussian,
    "powerline": powerline_hum,
    "eog": eog_leakage,
    "emg": emg_noise,
    "resample": resample_mismatch,
    "band_noise": band_limited_noise,
}

def apply_corruption(x: np.ndarray, fs: int, kind: str, level: float, params: Optional[Dict] = None) -> np.ndarray:
    fn = CORRUPTIONS.get(kind)
    if fn is None:
        raise ValueError(f"Unknown corruption kind: {kind}")
    params = params or {}
    return fn(x, level=level, fs=fs, **params)
