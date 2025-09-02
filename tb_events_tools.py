
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tb_events_tools.py
------------------
Utilities for working with TensorBoard event files:
1) Convert a single events file (or the newest in a directory) to JSON with all scalar tags.
2) Summarize multiple eval/val runs under a root directory into a CSV of best metrics.

Dependencies:
    pip install tensorboard

Usage:
    # 1) Convert one run's events file to JSON
    python tb_events_tools.py to-json \
        runs/eval_tent_val_20250829-160400/events.out.tfevents.1756508640.Hritiks-MacBook-Pro.local.5469.0 \
        --out runs/eval_tent_val_20250829-160400/metrics.json

    # Or point to a directory; it will pick the newest events file inside
    python tb_events_tools.py to-json runs/eval_tent_val_20250829-160400

    # 2) Summarize all eval runs under ./runs into a CSV
    python tb_events_tools.py summarize \
        --root runs \
        --glob "eval_*" \
        --out runs/summary_eval.csv \
        --dump-json  # also emits <run>/metrics.json for each run

Notes:
- The summarizer looks for common validation tag patterns like:
  "val/acc", "eval/accuracy", "validation/f1", "val_loss", etc.
- For each metric, it picks the tag(s) that match aliases and contain 'val'/'eval' if available.
- For accuracy/κ/F1/AUC it reports the MAX across steps; for loss it reports the MIN.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Import TensorBoard event loader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


VAL_KEYS = ("val", "valid", "validation", "eval", "evaluation")
ALIASES = {
    "acc":   ("acc", "accuracy", "balanced_acc", "bal_acc"),
    "kappa": ("kappa", "cohen_kappa"),
    "f1":    ("f1", "f1_score", "macro_f1", "weighted_f1", "micro_f1"),
    "auc":   ("auc", "auroc", "roc_auc"),
    "loss":  ("loss", "val_loss"),
}


@dataclass
class ScalarPoint:
    step: int
    wall_time: float
    value: float


def _newest_events_file(path: str) -> Optional[str]:
    """Return the newest events file if path is a dir; otherwise return path."""
    if os.path.isdir(path):
        cands = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.startswith("events.out.tfevents")
        ]
        if not cands:
            return None
        return max(cands, key=os.path.getmtime)
    return path if os.path.exists(path) else None


def _load_scalars(event_path: str) -> Dict[str, List[ScalarPoint]]:
    """Load all scalar tags from a TensorBoard events file."""
    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()
    out: Dict[str, List[ScalarPoint]] = {}
    for tag in ea.Tags().get("scalars", []):
        evs = ea.Scalars(tag)
        out[tag] = [ScalarPoint(step=e.step, wall_time=e.wall_time, value=float(e.value)) for e in evs]
    return out


def _serialize_payload(event_path: str, scalars: Dict[str, List[ScalarPoint]]) -> Dict:
    tags = sorted(scalars.keys())
    summary = {}
    for tag, arr in scalars.items():
        if not arr:
            continue
        last = arr[-1]
        mx = max(arr, key=lambda x: x.value)
        mn = min(arr, key=lambda x: x.value)
        summary[tag] = {
            "last": {"step": last.step, "wall_time": last.wall_time, "value": last.value},
            "max":  {"step": mx.step,   "wall_time": mx.wall_time,   "value": mx.value},
            "min":  {"step": mn.step,   "wall_time": mn.wall_time,   "value": mn.value},
        }
    return {
        "run_file": os.path.abspath(event_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "num_tags": len(tags),
        "tags": tags,
        "summary": summary,
        "scalars": {
            tag: [{"step": p.step, "wall_time": p.wall_time, "value": p.value} for p in arr]
            for tag, arr in scalars.items()
        },
    }


def cmd_to_json(args: argparse.Namespace) -> int:
    event_file = _newest_events_file(args.event_path)
    if not event_file:
        print(f"[ERROR] No events file found at: {args.event_path}", file=sys.stderr)
        return 2
    scalars = _load_scalars(event_file)
    payload = _serialize_payload(event_file, scalars)
    out_path = args.out or (event_file + ".json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Wrote {out_path} with {payload['num_tags']} scalar tags.")
    return 0


def _tokenize_tag(tag: str) -> List[str]:
    s = tag.lower()
    # Split on common separators
    parts = re.split(r"[\/\.\:\_\-\s]+", s)
    # Also keep the whole lowercase string for substring checks
    return list(filter(None, parts)) + [s]


def _contains_any(text_tokens: List[str], needles: Tuple[str, ...]) -> bool:
    for t in text_tokens:
        for n in needles:
            if n in t:
                return True
    return False


def _candidate_tags_for_metric(tags: List[str], metric_aliases: Tuple[str, ...], require_val: bool = True) -> List[str]:
    """Return tags likely to represent the metric, optionally requiring val/eval markers."""
    cands = []
    for tag in tags:
        toks = _tokenize_tag(tag)
        has_metric = _contains_any(toks, metric_aliases)
        if not has_metric:
            continue
        has_val = _contains_any(toks, VAL_KEYS)
        if require_val and not has_val:
            continue
        cands.append(tag)
    # If none with val/eval and require_val, relax
    if require_val and not cands:
        for tag in tags:
            toks = _tokenize_tag(tag)
            if _contains_any(toks, metric_aliases):
                cands.append(tag)
    return cands


def _best_point(arr: List[ScalarPoint], mode: str) -> Tuple[ScalarPoint, int]:
    """Return (best_point, idx) given mode 'max' or 'min'."""
    if not arr:
        raise ValueError("Empty scalar array")
    if mode == "max":
        idx = max(range(len(arr)), key=lambda i: arr[i].value)
    elif mode == "min":
        idx = min(range(len(arr)), key=lambda i: arr[i].value)
    else:
        raise ValueError("mode must be 'max' or 'min'")
    return arr[idx], idx


def _pick_best_across_tags(scalars: Dict[str, List[ScalarPoint]], cands: List[str], mode: str) -> Tuple[Optional[str], Optional[ScalarPoint]]:
    """Among candidate tags, pick the one with the global best value across steps."""
    best_tag = None
    best_pt = None
    for tag in cands:
        arr = scalars.get(tag, [])
        if not arr:
            continue
        pt, _ = _best_point(arr, mode)
        if best_pt is None:
            best_tag, best_pt = tag, pt
        else:
            if (mode == "max" and pt.value > best_pt.value) or (mode == "min" and pt.value < best_pt.value):
                best_tag, best_pt = tag, pt
    return best_tag, best_pt


def _event_file_for_run(run_dir: str) -> Optional[str]:
    cands = [
        os.path.join(run_dir, f) for f in os.listdir(run_dir)
        if f.startswith("events.out.tfevents")
    ]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def _summarize_run(run_dir: str, dump_json: bool = False) -> Dict[str, Optional[float]]:
    """Summarize one run directory by extracting best metrics."""
    event_file = _event_file_for_run(run_dir)
    if not event_file:
        return {
            "run": os.path.basename(run_dir),
            "path": run_dir,
            "event_file": None,
            "num_tags": 0,
            "best_acc": None,
            "best_kappa": None,
            "best_f1": None,
            "best_auc": None,
            "best_loss": None,
        }

    scalars = _load_scalars(event_file)
    tags = list(scalars.keys())

    # Optionally dump a per-run JSON
    if dump_json:
        payload = _serialize_payload(event_file, scalars)
        out_json = os.path.join(run_dir, "metrics.json")
        try:
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write {out_json}: {e}", file=sys.stderr)

    # For each metric: gather candidate tags and pick global best
    res = {
        "run": os.path.basename(run_dir),
        "path": run_dir,
        "event_file": os.path.relpath(event_file),
        "num_tags": len(tags),
    }

    metric_modes = {
        "acc": "max",
        "kappa": "max",
        "f1": "max",
        "auc": "max",
        "loss": "min",
    }

    for metric_key, aliases in ALIASES.items():
        cands = _candidate_tags_for_metric(tags, aliases, require_val=True)
        mode = metric_modes.get(metric_key, "max")
        best_tag, best_pt = _pick_best_across_tags(scalars, cands, mode)
        res[f"tag_{metric_key}"] = best_tag
        res[f"best_{metric_key}"] = round(best_pt.value, 6) if best_pt else None

    return res


def cmd_summarize(args: argparse.Namespace) -> int:
    root = args.root
    pattern = args.glob or "*"
    # Build list of run directories
    search_glob = os.path.join(root, pattern)
    run_dirs = [p for p in glob.glob(search_glob) if os.path.isdir(p)]
    if not run_dirs:
        print(f"[ERROR] No run directories matched: {search_glob}", file=sys.stderr)
        return 2

    rows = []
    for run_dir in sorted(run_dirs):
        row = _summarize_run(run_dir, dump_json=args.dump_json)
        rows.append(row)

    # Determine output CSV columns
    cols = [
        "run", "path", "event_file", "num_tags",
        "tag_acc", "best_acc",
        "tag_kappa", "best_kappa",
        "tag_f1", "best_f1",
        "tag_auc", "best_auc",
        "tag_loss", "best_loss",
    ]

    out_csv = args.out or os.path.join(root, "summary_eval.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"[OK] Wrote summary CSV: {out_csv} ({len(rows)} runs)")
    return 0


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TensorBoard events utilities: to-json & summarize.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # to-json
    p_json = sub.add_parser("to-json", help="Convert a TensorBoard events file (or dir) to JSON.")
    p_json.add_argument("event_path", help="Path to events.out.tfevents.* file OR a directory containing one.")
    p_json.add_argument("--out", help="Output JSON file path (default: <event_file>.json)")
    p_json.set_defaults(func=cmd_to_json)

    # summarize
    p_sum = sub.add_parser("summarize", help="Summarize multiple eval runs under a root folder into a CSV.")
    p_sum.add_argument("--root", default="runs", help="Root directory that contains run subfolders (default: runs)")
    p_sum.add_argument("--glob", default="eval_*", help="Glob for selecting run folders under --root (default: eval_*)")
    p_sum.add_argument("--out", help="Output CSV path (default: <root>/summary_eval.csv)")
    p_sum.add_argument("--dump-json", action="store_true", help="Also write <run>/metrics.json for each run")
    p_sum.set_defaults(func=cmd_summarize)

    return p


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    args = make_argparser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
