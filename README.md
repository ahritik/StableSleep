# Stabilize-While-You-Sleep: Source-Free TTA for Sleep-EDF

This project trains a **compact CNN** sleep-stage classifier and evaluates **source-free
test-time adaptation (TTA)** using **Tent** (entropy minimization) with simple **safety rails**.

**Why TTA?** Subject/recording shift makes static models brittle. In deployment we have
**no labels** and often cannot keep source data. Tent adapts **only BatchNorm parameters**
online using unlabeled streams—fast, privacy-friendly, and effective.

---

## Folder layout
```
tta_sleepedf/
  data/ sleepedf_mne.py        # MNE-based EDF → epochs + subject splits (+ repack)
  models/ cnn_sleep.py         # Compact CNN (+ tiny Temporal-SE), keeps BN for TTA
  utils/ metrics.py, common.py # Metrics, losses, scheduler, dataset, samplers
  train_source.py              # Imbalance-aware source training (prior-bias init)
  tta_tent.py                  # Tent loop with entropy gate, EMA resets, budget
  eval_sleep.py                # none / BN-only / Tent evaluation
  eda.py                       # Stage distribution + sample epoch plots
  figs/  paper/
  config.yaml                  # All hyperparameters and paths
  requirements.txt
  run_sleep_pipeline.sh        # End-to-end script
```

---

## Data format (raw → processed)

### 1) Raw input (must be in a folder named **`sleep-edfx`**)

Expected structure (examples; exact filenames may vary):
```
sleep-edfx/
  ├── sleep-cassette/
  │   ├── SC4001E0-PSG.edf
  │   ├── SC4001EC-Hypnogram.edf
  │   ├── SC4002E0-PSG.edf
  │   └── SC4002EC-Hypnogram.edf
  └── sleep-telemetry/
      ├── ST7011J0-PSG.edf
      ├── ST7011JV-Hypnogram.edf
      └── ...
```
**Notes**
- Hypnogram files are detected by the substring `Hypnogram` in the filename (case-insensitive).
- We start with EEG channel **`Fpz-Cz`** by default. You can pass multiple channels via
  `--channels "Fpz-Cz,Pz-Oz"` during transform.
- Default preprocessing: resample to **100 Hz**, epoch to **30 s**, per-recording **z-score** per channel.

### 2) Processed shards (NPZ) + manifest

After running the transform step, you will see:
```
data/processed/
  ├── rec_0000.npz         # one recording → many epochs
  ├── rec_0001.npz
  ├── ...
  └── manifest.json        # split info + meta
```
Then we "repack" (copy/symlink + fresh manifest) to:
```
data/processed_npy/
  ├── rec_0000.npz
  ├── rec_0001.npz
  ├── ...
  └── manifest.json
```

Each **`rec_xxxx.npz`** contains:
- **`X`**: `float32` array with shape **(N, C, T)**  
  - `N` = number of 30 s epochs in the recording  
  - `C` = number of selected channels (default 1)  
  - `T` = `fs * epoch_sec` samples (default `100 * 30 = 3000`)
- **`y`**: `int64` array with shape **(N, )** (sleep stage per epoch)
- **`subject`**: string ID (e.g., `"rec_0000"`)

**Label mapping (AASM)**:
```
0: W    (Wake)
1: N1
2: N2
3: N3   (N3 + N4 merged)
4: REM
```
Unknown/movement epochs are ignored during transform.

**`manifest.json` schema** (example):
```json
{
  "channels": ["Fpz-Cz"],
  "fs": 100,
  "epoch_sec": 30,
  "splits": {
    "train": [
      {"id": "rec_0000", "path": "rec_0000.npz", "n": 950},
      {"id": "rec_0001", "path": "rec_0001.npz", "n": 1020}
    ],
    "val": [
      {"id": "rec_0008", "path": "rec_0008.npz", "n": 880}
    ],
    "test": [
      {"id": "rec_0012", "path": "rec_0012.npz", "n": 910}
    ]
  }
}
```
- Splits are **by recording** (proxy for subject-level split) using the `train_frac`/`val_frac` settings.
- You can change channels, sample rate, epoch length, and z-score behavior via CLI or `config.yaml`.

**Python snippet to read a shard**
```python
import numpy as np, json
npz = np.load("data/processed_npy/rec_0000.npz")
X, y = npz["X"], npz["y"]          # X: (N,C,T) float32, y: (N,) int64
fs = json.load(open("data/processed_npy/manifest.json"))["fs"]
print(X.shape, y.shape, fs)        # e.g., (950, 1, 3000) (950,) 100
```

---

## Quickstart

```bash
# 0) Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# 1) Transform raw Sleep-EDF (adjust RAW to your 'sleep-edfx' path)
export RAW="/absolute/path/to/sleep-edfx"
python -m tta_sleepedf.data.sleepedf_mne --root "$RAW" --out data/processed --channels "Fpz-Cz" --fs 100 --epoch 30 --zscore --train_frac 0.6 --val_frac 0.2 --verbose

# 2) Repack to fast folder (symlink/copy + fresh manifest)
python -m tta_sleepedf.data.sleepedf_mne repack --src data/processed --dst data/processed_npy

# 3) Train source model (imbalance-aware)
python -m tta_sleepedf.train_source --cfg config.yaml

# 4) Evaluate (none / BN-only / Tent)
python -m tta_sleepedf.eval_sleep --cfg config.yaml --tta none
python -m tta_sleepedf.eval_sleep --cfg config.yaml --tta bn_only
python -m tta_sleepedf.eval_sleep --cfg config.yaml --tta tent

# 5) EDA
python -m tta_sleepedf.eda --root data/processed_npy
```

**One-liner end-to-end**
```bash
chmod +x run_sleep_pipeline.sh
RAW="$HOME/data/sleep-edfx" ./run_sleep_pipeline.sh
```

---

## Notes & tips
- To add a **second EEG channel** (e.g., `Pz-Oz`), either change the CLI to `--channels "Fpz-Cz,Pz-Oz"`
  during transform, or edit `data.channels` in `config.yaml` and re-run steps 1–2.
- If early training seems unstable, reduce time masking / jitter in `config.yaml`:
  ```yaml
  train:
    augment:
      time_mask_prob: 0.1
      jitter_std: 0.005
  ```
- For a CE vs Focal comparison, set `train.loss: ce` in `config.yaml` and re-train.

---

## EDA outputs
Running `python -m tta_sleepedf.eda --root data/processed_npy` produces:
- `figs/stage_distribution_counts.png`
- `figs/stage_distribution_fraction.png`
- `figs/example_epoch_0.png` … `example_epoch_4.png`

These help verify class imbalance, stage presence, and basic signal quality.

---

## Clinical note
This code is **research-grade**. Any clinical use requires additional validation,
governance, and regulatory review.
