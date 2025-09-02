
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pack_outputs_to_predictions.py
------------------------------
Merge per-record CSVs like outputs/{split}/rec_XXXX.csv into a single predictions.csv
that downstream evaluators can consume.

Input CSV schema (as in your example):
    epoch,pred,conf,true

We add a 'subject' column from the filename (e.g., rec_0004.csv -> subject=0004)
and rename columns to the evaluator-friendly names:
    subject, y_true, y_pred, conf

Optionally map integer labels to class names (W, N1, N2, N3, REM).

Usage
-----
# default mapping 0..4 -> W N1 N2 N3 REM
python pack_outputs_to_predictions.py --dir outputs/test --out out_eval/test/predictions.csv

# override mapping explicitly
python pack_outputs_to_predictions.py --dir outputs/val --out out_eval/val/predictions.csv \
    --map 0:W 1:N1 2:N2 3:N3 4:REM
"""
import argparse, os, glob, re
import pandas as pd

def parse_map(pairs):
    mapping = {}
    for p in pairs or []:
        k, v = p.split(":", 1)
        mapping[int(k)] = v
    if not mapping:
        # default Sleep-EDF mapping
        mapping = {0:"W", 1:"N1", 2:"N2", 3:"N3", 4:"REM"}
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Folder containing rec_*.csv files")
    ap.add_argument("--out", required=True, help="Output predictions.csv path")
    ap.add_argument("--map", nargs="+", help="Optional int->class pairs like 0:W 1:N1 ...")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.dir, "rec_*.csv")))
    assert files, f"No files matched in {args.dir}"
    label_map = parse_map(args.map)

    rows = []
    for fp in files:
        rec = re.search(r"rec_(\d+)\.csv$", fp)
        subject = rec.group(1) if rec else os.path.basename(fp).replace(".csv","")
        df = pd.read_csv(fp)
        # normalize column names
        df = df.rename(columns={"pred":"y_pred","true":"y_true","conf":"pred_conf"})
        df["subject"] = subject
        rows.append(df[["subject","epoch","y_true","y_pred","pred_conf"]])
    out = pd.concat(rows, ignore_index=True)

    # Optional map: convert integer labels to class names (strings)
    def map_label(x):
        try:
            xi = int(x)
            return label_map.get(xi, str(x))
        except Exception:
            return str(x)
    out["y_true"] = out["y_true"].apply(map_label)
    out["y_pred"] = out["y_pred"].apply(map_label)

    out.to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out} with {len(out)} rows across {len(files)} subjects.")

if __name__ == "__main__":
    main()
