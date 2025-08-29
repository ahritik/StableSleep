"""Metrics and summaries for sleep staging."""
from typing import Dict
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix

STAGE_NAMES = ["W","N1","N2","N3","REM"]

def per_class_f1(y_true, y_pred) -> Dict[int, float]:
    """F1 for each class index (0..4)."""
    f1s = f1_score(y_true, y_pred, average=None, labels=[0,1,2,3,4])
    return {i: float(v) for i, v in enumerate(f1s)}

def macro_f1(y_true, y_pred) -> float:
    """Unweighted mean of per-class F1 (treats rare classes equally)."""
    return float(f1_score(y_true, y_pred, average="macro", labels=[0,1,2,3,4]))

def kappa(y_true, y_pred) -> float:
    """Cohen's κ (agreement beyond chance)."""
    return float(cohen_kappa_score(y_true, y_pred))

def summarize(y_true, y_pred) -> str:
    """Convenient text block for logs: macroF1/κ + per-class breakdown."""
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    f1s = per_class_f1(y_true, y_pred)
    out = []
    out.append(f"macroF1 {macro_f1(y_true, y_pred):.4f} | kappa {kappa(y_true, y_pred):.4f}")
    counts = cm.sum(axis=1)
    for i, name in enumerate(STAGE_NAMES):
        out.append(f"  {name}:F1={f1s.get(i,0):.2f},n={int(counts[i])}")
    return "\n".join(out)
