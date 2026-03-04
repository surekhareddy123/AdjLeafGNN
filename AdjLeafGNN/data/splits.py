from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold

def make_stratified_folds(labels, n_splits: int = 5, seed: int = 42):
    labels = np.asarray(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        yield fold, train_idx.tolist(), val_idx.tolist()
