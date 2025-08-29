#!/usr/bin/env bash
# End-to-end: transform → repack → train → eval (none/BN-only/Tent) → EDA
set -euo pipefail
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT_DIR"

# RAW should point to the *raw* Sleep-EDF folder named 'sleep-edfx'
: "${RAW:=${ROOT_DIR}/data/sleep-edfx}"

# You may override these on the command line, e.g. CHANNELS="Fpz-Cz,Pz-Oz" ./run_sleep_pipeline.sh
CHANNELS="${CHANNELS:-Fpz-Cz}"
FS="${FS:-100}"
EPOCH_SEC="${EPOCH_SEC:-30}"

echo "RAW=$RAW"
echo "CHANNELS=$CHANNELS FS=$FS EPOCH=$EPOCH_SEC"

# Start from a clean slate to avoid stale shards/checkpoints
rm -rf data/processed data/processed_npy checkpoints figs
mkdir -p data

# 1) Transform EDF → per-recording NPZ shards + manifest.json
python -m data.sleepedf_mne   --root "$RAW" --out data/processed   --channels "$CHANNELS" --fs "$FS" --epoch "$EPOCH_SEC"   --zscore --train_frac 0.6 --val_frac 0.2 --verbose

# 2) Repack to a faster folder (symlinks/copies + fresh manifest)
python -m data.sleepedf_mne repack --src data/processed --dst data/processed_npy

# 3) Train the source model (reads config.yaml)
python -m train_source --cfg config.yaml

# 4) Evaluate three modes
python -m eval_sleep --cfg config.yaml --tta none
python -m eval_sleep --cfg config.yaml --tta bn_only
python -m eval_sleep --cfg config.yaml --tta tent

# 5) EDA figures (stage distribution, example epochs)
python -m eda --root data/processed_npy

echo "Done."
