# StableSleep: Source-Free Test-Time Adaptation for Sleep-EDF

StableSleep delivers a **streaming, source-free test-time adaptation (TTA)** recipe for sleep staging on **MNE Sleep-EDF (Expanded)**.

- **Method**: Tent-style entropy minimization on BN γ/β + BN stats refresh.  
- **Safety rails**: entropy gate + EMA reset.  
- **Baselines**: No-adapt, BN-only, Tent (ours), optional T3A.  
- **Metrics**: Macro-F1, Cohen’s κ (overall + per-subject), latency per epoch.

> ⚠️ Note: This skeleton prefers a simple **npz-based dataset** produced by a preprocessing script. You can adapt the provided `data/sleepedf_mne.py` to export per-subject `.npz` files from raw Sleep-EDF using MNE.

## Layout
```
tta_sleepedf/
  data/ sleepedf_mne.py           # Preprocessing helpers & Dataset
  models/ cnn_sleep.py            # Compact CNN (+ tiny temporal attention)
  utils/ metrics.py, common.py    # Metrics & helpers (seed, device)
  train_source.py                 # Train source model on train-subjects
  tta_tent.py                     # TTA loop (Tent + rails)
  eval_sleep.py                   # Evaluate baselines & TTA
  figs/  paper/                   # Outputs for the paper
  config.yaml                     # Hyperparameters
  requirements.txt
```
## Quickstart

1) **Create per-subject npz files** (or adapt loader to raw EDF):
```
# Expected structure per split dir:
#   data/processed/{train,val,test}/sub-XXXX_{Fpz-Cz}.npz  -> X: (N, C, T), y: (N,)
python -m data.sleepedf_mne \  --raw_root /path/to/sleep-edf \  --out_root ./data/processed \  --channels Fpz-Cz \  --resample_hz 100
```

2) **Train source model**:
```
python train_source.py --cfg config.yaml
```

3) **Evaluate** (No-adapt / BN-only / Tent + rails):
```
python eval_sleep.py --cfg config.yaml --tta tent
```

## Notes
- Subject-wise 60/20/20 split (seeded) is expected; adjust in `config.yaml`.
- If latency is high, use **BN-only** with small batches; report Tent numbers offline.
- For robustness tests, see `eval_sleep.py --robust [emg|line|dropout]`.
