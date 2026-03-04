from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, matthews_corrcoef
)

def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": float(acc), "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1)}

def spread_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray):
    out = {}
    # AUC needs both classes present
    try:
        out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc_roc"] = None
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out.update({"precision": float(p), "recall": float(r), "f1": float(f1)})
    try:
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        out["mcc"] = None
    return out
