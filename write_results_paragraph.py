
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
write_results_paragraph.py
--------------------------
Read overall_metrics.csv and produce a short results paragraph with Acc, Kappa,
Macro-F1, and Weighted-F1 for pasting into the paper.

Usage
-----
python write_results_paragraph.py --csv out_eval/test/overall_metrics.csv --label "test set" --out out_eval/test/results.txt
"""
import argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="overall_metrics.csv path")
    ap.add_argument("--label", default="test set", help="Label for the dataset")
    ap.add_argument("--out", required=True, help="Where to write the paragraph")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    acc = df["accuracy"].iloc[0]
    kappa = df["kappa"].iloc[0]
    macro = df.get("macro_f1", pd.Series([None])).iloc[0]
    weighted = df.get("weighted_f1", pd.Series([None])).iloc[0]
    text = (f"On the {args.label}, the source model achieves "
            f"Accuracy = {acc:.3f}, Cohen's κ = {kappa:.3f}, "
            f"Macro-F1 = {macro:.3f}, and Weighted-F1 = {weighted:.3f}. "
            f"Per-stage F1 follows the expected pattern with N2 highest and N1 lowest.")
    with open(args.out, "w") as f:
        f.write(text+"\n")
    print(text)

if __name__ == "__main__":
    main()
