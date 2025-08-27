"""
Sleep-EDF helpers:
- Preferred: use preprocessed per-subject .npz files:
    X: (N, C, T), y: (N,), subjects: (N,)
- Optionally adapt "export_from_mne" to build these from raw EDF.
"""
from __future__ import annotations
import os, glob, json, argparse
from typing import List, Tuple, Dict
import numpy as np
import mne

STAGE_MAP_AASM = {
    "W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4,
}

def _clean_labels(labels: np.ndarray) -> np.ndarray:
    # Map various encodings to AASM 5-class
    mapping = {
        "Sleep stage W": "W",
        "Sleep stage 1": "N1",
        "Sleep stage 2": "N2",
        "Sleep stage 3": "N3",
        "Sleep stage 4": "N3",      # combine 3/4 to N3
        "Sleep stage R": "R",
        "Sleep stage ?": None,
        "Movement time": None,
    }
    clean = []
    for lab in labels:
        lab = mapping.get(str(lab), lab)
        if lab is None:
            clean.append(None)
        elif lab in STAGE_MAP_AASM:
            clean.append(STAGE_MAP_AASM[lab])
        else:
            clean.append(None)
    return np.array(clean, dtype=object)

class NPZDataset:
    def __init__(self, root_dir: str):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz found in {root_dir}")
        self._sizes = []
        for f in self.files:
            with np.load(f) as d:
                self._sizes.append(len(d["y"]))
        self.length = sum(self._sizes)

    def __len__(self): return self.length

    def _locate(self, idx: int) -> Tuple[str, int]:
        # map global idx -> file, local idx
        cum = 0
        for f, sz in zip(self.files, self._sizes):
            if idx < cum + sz:
                return f, idx - cum
            cum += sz
        raise IndexError

    def get_batch(self, idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Returns X:(B,C,T), y:(B,), subjects:(B,)
        parts = {}
        for i in idxs:
            f, li = self._locate(int(i))
            if f not in parts: parts[f] = []
            parts[f].append(li)
        Xs, ys, subs = [], [], []
        for f, lis in parts.items():
            with np.load(f) as d:
                Xs.append(d["X"][lis])
                ys.append(d["y"][lis])
                subs.append(d["subjects"][lis])
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        subjects = np.concatenate(subs, axis=0)
        return X, y, subjects

def export_from_mne(raw_edf: str, ann_edf: str, channels: List[str], fs: int, epoch_sec: int) -> Tuple[np.ndarray, np.ndarray]:
    """Best-effort example using MNE to build X,y from one PSG/annotations pair.

    Returns X: (N, C, T), y: (N,)
    """
    raw = mne.io.read_raw_edf(raw_edf, preload=True, verbose=False)
    ann = mne.read_annotations(ann_edf, verbose=False)
    raw.set_annotations(ann, emit_warning=False)
    # pick channels
    raw.pick(channels)
    # filter & resample (tweak as needed)
    raw.notch_filter(freqs=[50, 60], verbose=False)
    raw.filter(l_freq=0.3, h_freq=45.0, verbose=False)
    raw.resample(fs, npad="auto", verbose=False)
    # build 30s epochs
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # fixed-length epochs aligned to annotations is non-trivial; here we just slice
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_sec, preload=True, overlap=0.0, verbose=False)
    X = epochs.get_data()  # (N, C, T)
    # derive labels per epoch using majority label in window (placeholder logic)
    onsets = epochs.events[:, 0] / raw.info["sfreq"]
    labels = []
    for k in range(len(epochs)):
        t0 = onsets[k]
        t1 = t0 + epoch_sec
        # find annotation segments in [t0, t1)
        labs = []
        for a in ann:
            if a["onset"] < t1 and (a["onset"] + a["duration"]) > t0:
                labs.append(a["description"])
        labels.append(labs[0] if labs else "Sleep stage W")
    y = _clean_labels(np.array(labels))
    # drop unlabeled epochs
    keep = np.array([lab is not None for lab in y])
    return X[keep], y[keep].astype(int)

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=str, required=False, help="Root of Sleep-EDF raw files")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--channels", type=str, default="Fpz-Cz")
    ap.add_argument("--resample_hz", type=int, default=100)
    ap.add_argument("--epoch_sec", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    # Placeholder: user should adapt file discovery to their local tree.
    # Here we only demonstrate the API on a single pair if provided.
    print("This is a reference CLI. Please adapt to your Sleep-EDF file layout.")
    print("Expected output: *.npz with arrays X:(N,C,T), y:(N,), subjects:(N,)")

if __name__ == "__main__":
    cli()
