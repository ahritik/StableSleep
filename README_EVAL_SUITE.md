
# Sleep Stabilize — Evaluation & Ablation Suite

This folder contains ready-to-run utilities to produce a submission‑grade evaluation.
All scripts are pure Python; install extras as needed:
```bash
pip install tensorboard scipy pandas numpy matplotlib
```

## 1) Summarize existing TensorBoard runs
```bash
python tb_events_tools.py summarize --root runs --glob "eval_*" --dump-json \
  --out runs/summary_eval.csv
```

## 2) Evaluate predictions (val & test)
Assumes you have CSVs like `outputs/val/rec_*.csv`, `outputs/test/rec_*.csv`
with columns inferable as `subject, y_true, y_pred`. If different, pass overrides.

```bash
# VAL
python eval_from_csv.py --glob "outputs/val/*.csv" --out out_eval/val
# TEST
python eval_from_csv.py --glob "outputs/test/*.csv" --out out_eval/test
```

Outputs include overall metrics, per-class F1, per-subject metrics, confusion matrix,
and LaTeX tables (`overall_table.tex`, `per_class_table.tex`).

## 3) Compare methods with per-subject stats
Run the evaluator for each method’s predictions into separate folders, then compare:
```bash
python compare_methods.py \
  --baseline out_eval_baseline/per_subject_metrics.csv \
  --method   out_eval_tent/per_subject_metrics.csv \
  --out out_compare/baseline_vs_tent
```

This writes `wilcoxon.csv` and box/violin plots of per-subject deltas.

## 4) TENT adaptation & ablations (streaming realism)
Implement two hooks in `run_ablation.py`:
- `load_model(checkpoint_path)` to return your trained model
- `make_stream_loader(split, batch_size)` to yield `(x, y, meta)` in **subject order**

Then run:

**BN-only (no optimization):**
```bash
python run_ablation.py --ckpt artifacts/final_model.pt \
  --split val --method bn_only --bn-stats \
  --out runs_ablate/bn_only_val
```

**TENT with rails:**
```bash
python run_ablation.py --ckpt artifacts/final_model.pt \
  --split val --method tent --lr 1e-3 --bn-stats --dropout-off \
  --entropy-gate 0.6 --ema-decay 0.98 --reset-interval 200 \
  --out runs_ablate/tent_tau0p6_R200
```

**Sweeps (τ)**
```bash
python run_ablation.py --ckpt artifacts/final_model.pt \
  --split val --method tent --bn-stats --dropout-off \
  --sweep tau 0.2 0.4 0.6 0.8 1.0 \
  --out runs_ablate/sweep_tau
```

For each run, evaluate predictions and collect curves:
```bash
for d in runs_ablate/*; do
  test -d "$d" || continue
  python eval_from_csv.py --csv "$d/predictions.csv" --out "$d/out_eval"
done
python eval_plots.py --manifest runs_ablate/sweep_tau/sweep_manifest.json \
  --metric accuracy --out runs_ablate/plots
```

## 5) Robustness to shift
Use `corruption_suite.py` to generate corrupted streams (amplitude, powerline, EOG, EMG, resample, etc.).
Integrate into your `make_stream_loader` to apply a corruption per subject or per batch, varying `level`.
Plot accuracy/κ vs severity with and without TENT.

## 6) Reproducibility
- Log: seeds, splits, checkpoint, hyperparams in each `stats.json`.
- Fix seeds: `torch`, `numpy`, `random`, `PYTHONHASHSEED`.
- Repeat with ≥3 seeds and report mean±std.

### Minimal seed boilerplate
```python
import random, numpy as np, torch, os
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    os.environ["PYTHONHASHSEED"]=str(s)
```

---

If you want, I can wire `run_ablation.py` to your exact dataloader/model once you paste those functions.
