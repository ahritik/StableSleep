from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score

def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro")

def kappa(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred)

def per_subject_scores(y_true, y_pred, subjects):
    # subjects: same length as y_true, indicates subject id per epoch
    out = {}
    for s in np.unique(subjects):
        idx = subjects == s
        out[str(s)] = dict(
            macro_f1=macro_f1(y_true[idx], y_pred[idx]),
            kappa=kappa(y_true[idx], y_pred[idx])
        )
    return out
