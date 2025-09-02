# data/sleepedf_mne.py
"""
Sleep-EDF preprocessing using MNE.

- Pairs PSG (*.edf) with Hypnogram (*.edf) inside the same folder.
- Hypnograms are EDF+ **annotations** → read with mne.read_annotations().
- Attach annotations to PSG via raw.set_annotations(ann) so times align (handles orig_time).
- Expand R&K stages (W,R,1,2,3,4,M,?) → AASM indices:
    W=0, N1=1, N2=2, N3=3 (merge 3/4), REM=4; M and ? are ignored.
- Resample EEG to fs, slice into epoch_sec windows aligned to stage onsets.
- Write per-recording NPZ shards (X:(N,C,T), y:(N,)) + manifest.json.
- "repack" subcommand copies/symlinks shards to a new folder.

Also provides:
- build_loaders(): single-epoch DataLoader
- build_sequence_loaders(): sequence window DataLoader (length L, label = center epoch)
"""
import os, json, glob, argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def _loader_common_kwargs():
    """Choose DataLoader kwargs based on backend; disable pin_memory on MPS."""
    is_mps = torch.backends.mps.is_available()
    workers = max(2, int(os.environ.get("NUM_WORKERS", "8")))
    # Default: pin on CUDA/CPU; OFF on MPS. Allow manual override via PIN_MEMORY.
    pin_env = os.environ.get("PIN_MEMORY")
    if pin_env is not None:
        pin = bool(int(pin_env))
    else:
        pin = False if is_mps else True
    return dict(
        num_workers=workers,
        persistent_workers=True if workers > 0 else False,
        prefetch_factor=2 if workers > 0 else None,
        pin_memory=pin,
    )

# ---------------- Label mapping (R&K → AASM) ----------------
def _stage_from_annot(desc: str):
    """Map R&K labels (W,R,1,2,3,4,M,?) to AASM {W=0,N1=1,N2=2,N3=3,REM=4}.
       Movement (M) and unknown (?) are ignored (return None)."""
    d = str(desc).strip().upper()
    d = d.replace("SLEEP STAGE ", "").replace("SLEEP_STAGE_", "")
    d = d.replace("STAGE ", "").replace("STAGE_", "")
    if d in {"M", "?", ""}:
        return None
    if d in {"W"}:
        return 0
    if d in {"1", "N1"}:
        return 1
    if d in {"2", "N2"}:
        return 2
    if d in {"3", "4", "N3"}:  # merge 4 → N3
        return 3
    if d in {"R", "REM"}:
        return 4
    return None

# ---------------- Pair PSG with Hypnogram ----------------
def find_pairs(root: str) -> List[Tuple[str, str]]:
    """Return (psg_path, hypnogram_path) pairs based on folder co-location."""
    root = Path(root)
    psgs = sorted([str(p) for p in root.rglob("*-PSG.edf")])
    pairs = []
    for psg in psgs:
        folder = os.path.dirname(psg)
        hyp = sorted(glob.glob(os.path.join(folder, "*Hypnogram*.edf")))
        if hyp:
            pairs.append((psg, hyp[0]))
    # de-dup by basename
    out, seen = [], set()
    for a, b in pairs:
        key = (os.path.basename(a), os.path.basename(b))
        if key not in seen:
            seen.add(key); out.append((a, b))
    return out

# ---------------- Core: PSG + annotations → epochs ----------------
def epoch_data(raw_sig: mne.io.BaseRaw, ann: mne.Annotations,
               channel_names: List[str], fs: int, epoch_sec: int, zscore: bool=True):
    """Return X:(N,C,T) float32 and y:(N,) int64 for one recording."""
    # Resample PSG
    sig = raw_sig.copy().load_data()
    if fs is not None:
        sig.resample(fs)

    # Tolerant channel selection
    def _norm(s: str) -> str:
        return s.replace("EEG", "").replace("eeg", "").replace("-", "").replace(" ", "").lower()

    # Try exact include first (older MNE: no raise_if_missing arg)
    picks = mne.pick_channels(sig.ch_names, include=channel_names)
    if len(picks) == 0:
        ch_norm = [_norm(c) for c in sig.ch_names]
        want = {_norm(cn) for cn in channel_names} | {_norm("EEG " + cn) for cn in channel_names}
        picks = [i for i, n in enumerate(ch_norm) if n in want]
    if len(picks) == 0:
        raise RuntimeError(f"Channels {channel_names} not found in {sig.ch_names}")

    sig.pick(picks)
    X = sig.get_data()  # (C, T_all)
    sfreq = float(sig.info["sfreq"])
    T = int(round(epoch_sec * sfreq))
    tmax = X.shape[1] / sfreq

    # Attach annotations to PSG timebase (handles orig_time offset)
    sig.set_annotations(ann, emit_warning=False)
    ann = sig.annotations

    # Collect valid sleep-stage intervals
    events = []
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        st = _stage_from_annot(desc)
        if st is None:
            continue
        events.append((float(onset), float(duration or 0.0), int(st)))
    if not events:
        raise RuntimeError("No valid sleep stage annotations parsed.")

    events.sort(key=lambda x: x[0])

    # Expand into epoch-aligned labels; for short/zero durations, infer from next onset (or 30s at end)
    onsets_sec, labels = [], []
    for i, (onset, dur, st) in enumerate(events):
        if onset >= tmax:
            continue
        # Treat dur < 0.9*epoch as a marker → extend to next onset (or 30s at end)
        if dur <= 0 or dur < (epoch_sec * 0.9):
            next_onset = events[i+1][0] if i+1 < len(events) else onset + epoch_sec
            dur_eff = max(epoch_sec, min(next_onset - onset, tmax - onset))
        else:
            dur_eff = min(dur, tmax - onset)
        n_ep = int(np.floor((dur_eff + 1e-6) / epoch_sec))
        for k in range(n_ep):
            t = onset + k * epoch_sec
            if t + epoch_sec <= tmax:
                onsets_sec.append(t)
                labels.append(st)

    if not labels:
        raise RuntimeError("No valid sleep epochs after expansion.")

    # Slice PSG
    starts = [int(round(t * sfreq)) for t in onsets_sec]
    valid = [s for s in starts if s + T <= X.shape[1]]
    if not valid:
        raise RuntimeError("No valid epochs within signal length after alignment.")
    starts = np.asarray(valid, dtype=np.int64)
    epochs = np.stack([X[:, s:s+T] for s in starts], axis=0)  # (N,C,T)

    y = np.asarray(labels, dtype=np.int64)[:len(epochs)]

    if zscore:
        mu = epochs.mean(axis=(0, 2), keepdims=True)
        sd = epochs.std(axis=(0, 2), keepdims=True) + 1e-6
        epochs = (epochs - mu) / sd

    return epochs.astype(np.float32), y

# ---------------- Dataset driver ----------------
def process_sleepedf(root: str, out: str, channels: List[str], fs: int, epoch: int,
                     zscore: bool, train_frac: float, val_frac: float, verbose=False):
    """Process entire dataset and write shards + manifest."""
    pairs = find_pairs(root)
    if verbose:
        print(f"Found {len(pairs)} PSG/Hypnogram pairs.")
    os.makedirs(out, exist_ok=True)
    shards = []

    for i, (psg, hyp) in enumerate(tqdm(pairs, desc="Transform", dynamic_ncols=True)):
        try:
            raw_sig = mne.io.read_raw_edf(psg, preload=False, verbose="ERROR")
            ann = mne.read_annotations(hyp)  # EDF+ annotations
            X, y = epoch_data(raw_sig, ann, channels, fs, epoch, zscore)
            sid = f"rec_{i:04d}"
            np.savez_compressed(os.path.join(out, f"{sid}.npz"), X=X, y=y, subject=sid)
            shards.append({"id": sid, "path": f"{sid}.npz", "n": int(len(y))})
            if verbose:
                print(f"[{i+1}/{len(pairs)}] {os.path.basename(psg)} -> {sid}: {X.shape}, labels {np.bincount(y, minlength=5)}")
        except Exception as e:
            print(f"Skip {psg}: {e}")

    # Subject-level split by recording (proxy)
    rng = np.random.default_rng(1337)
    idx = rng.permutation(len(shards))
    n_train = int(len(idx) * train_frac)
    n_val = int(len(idx) * val_frac)
    tr, va, te = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
    manifest = {"splits": {"train": [shards[i] for i in tr],
                           "val":   [shards[i] for i in va],
                           "test":  [shards[i] for i in te]},
                "channels": channels, "fs": fs, "epoch_sec": epoch}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        print(f"Saved manifest with {len(tr)} train, {len(va)} val, {len(te)} test subjects at {out}/manifest.json")

def repack_to_npy(src: str, dst: str):
    """Copy/symlink NPZ shards to new root and rewrite manifest paths."""
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "manifest.json")) as f:
        mani = json.load(f)
    new_mani = {"splits": {"train": [], "val": [], "test": []},
                "channels": mani["channels"], "fs": mani["fs"], "epoch_sec": mani["epoch_sec"]}
    for split in ["train", "val", "test"]:
        for rec in mani["splits"][split]:
            src_path = os.path.join(src, rec["path"])
            dst_path = os.path.join(dst, rec["path"])
            try:
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                os.link(src_path, dst_path)  # hardlink when possible
            except Exception:
                import shutil as _sh
                _sh.copy2(src_path, dst_path)
            new_mani["splits"][split].append(rec)
    with open(os.path.join(dst, "manifest.json"), "w") as f:
        json.dump(new_mani, f, indent=2)

# ---------------- Single-epoch DataLoader ----------------
def build_loaders(processed_root: str, split: str, batch_size: int, augment_cfg=None, balanced=False):
    """Single-epoch DataLoader. Returns (dl, y_all)."""
    import os, json
    import numpy as np
    from torch.utils.data import DataLoader
    from utils.common import SleepDataset, make_balanced_sampler

    # load manifest and concatenate shards
    with open(os.path.join(processed_root, "manifest.json")) as f:
        mani = json.load(f)
    recs = mani["splits"][split]
    Xs, ys = [], []
    for rec in recs:
        npz = np.load(os.path.join(processed_root, rec["path"]))
        Xs.append(npz["X"]); ys.append(npz["y"])
    if Xs:
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
    else:
        C = len(mani["channels"]); T = mani["fs"] * mani["epoch_sec"]
        X = np.zeros((0, C, T), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)

    ds = SleepDataset(X, y, augment=augment_cfg if split == "train" else None)

    # fast DataLoader settings
    workers    = max(2, int(os.environ.get("NUM_WORKERS", "8")))
    pin        = True
    persistent = True
    prefetch   = 2

    kw = _loader_common_kwargs()

    if balanced and split == "train" and len(y) > 0:
        sampler = make_balanced_sampler(y)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            **kw,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            **kw,
        )

    return dl, y

# ---------------- Sequence (context window) DataLoader ----------------
class SequenceSleepDataset(Dataset):
    """
    Builds windows of length L across each recording, label = center epoch.
    Augmentations (if any) are applied per-epoch inside the window.
    """
    def __init__(self, manifest, root, split, L, augment=None):
        self.root = root
        self.L = int(L)
        assert self.L % 2 == 1, "context_len must be odd (center epoch well-defined)"
        self.half = self.L // 2
        self.augment = augment or {}
        self.recs = manifest["splits"][split]
        self.items = []   # (rec_idx, start_idx)
        self.Xs, self.ys = [], []
        for r in self.recs:
            npz = np.load(os.path.join(root, r["path"]))
            X = npz["X"]  # (N, C, T)
            y = npz["y"]  # (N,)
            N = len(y)
            if N >= self.L:
                self.Xs.append(X)
                self.ys.append(y)
                rec_idx = len(self.Xs) - 1
                for s in range(0, N - self.L + 1):
                    self.items.append((rec_idx, s))
        # labels for sampler/statistics (center epoch label)
        self.labels = np.array([self.ys[ri][s + self.half] for (ri, s) in self.items], dtype=np.int64)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from utils.common import SleepDataset  # reuse single-epoch augment
        ri, s = self.items[i]
        X = self.Xs[ri][s:s+self.L]          # (L, C, T)
        y = int(self.ys[ri][s + self.half])  # center label
        if self.augment.get("enabled", False):
            X = X.copy()
            for t in range(self.L):
                X[t] = SleepDataset.apply_aug(X[t], self.augment)
        return torch.from_numpy(X), torch.tensor(y)

def build_sequence_loaders(processed_root: str, split: str, batch_size: int,
                           context_len: int, augment_cfg=None, balanced=False):
    """Sequence window DataLoader (L consecutive epochs, label = center epoch). Returns (dl, labels)."""
    import os, json
    from torch.utils.data import DataLoader
    from utils.common import make_balanced_sampler

    with open(os.path.join(processed_root, "manifest.json")) as f:
        mani = json.load(f)

    ds = SequenceSleepDataset(
        manifest=mani,
        root=processed_root,
        split=split,
        L=int(context_len),
        augment=augment_cfg if split == "train" else None,
    )

    workers    = max(2, int(os.environ.get("NUM_WORKERS", "8")))
    pin        = True
    persistent = True
    prefetch   = 2

    kw = _loader_common_kwargs()

    if balanced and split == "train" and len(ds.labels) > 0:
        sampler = make_balanced_sampler(ds.labels)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            **kw,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            **kw,
        )

    return dl, ds.labels

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    ap.add_argument("--root", type=str, help="Path to sleep-edfx raw folder")
    ap.add_argument("--out", type=str, default="data/processed")
    ap.add_argument("--channels", type=str, default="EEG Fpz-Cz",
                    help='Comma-separated, e.g., "EEG Fpz-Cz,EEG Pz-Oz"')
    ap.add_argument("--fs", type=int, default=100)
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--verbose", action="store_true")

    ap_rep = sub.add_parser("repack")
    ap_rep.add_argument("--src", type=str, required=True)
    ap_rep.add_argument("--dst", type=str, required=True)

    args = ap.parse_args()
    if args.cmd == "repack":
        repack_to_npy(args.src, args.dst); return

    if not args.root:
        raise SystemExit("--root is required (path to sleep-edfx)")
    chans = [c.strip() for c in args.channels.split(",") if c.strip()]
    process_sleepedf(args.root, args.out, chans, args.fs, args.epoch, args.zscore,
                     args.train_frac, args.val_frac, verbose=args.verbose)

if __name__ == "__main__":
    main()
