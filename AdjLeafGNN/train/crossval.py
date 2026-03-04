from __future__ import annotations
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.splits import make_stratified_folds
from utils.logging import Logger
from utils.metrics import cls_metrics, spread_metrics
from train.trainer import Trainer

def run_crossval(dataset, build_model_fn, build_optim_fn, build_sched_fn, loss_fn, device, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    labels = dataset.targets
    class_names = dataset.info.classes

    fold_summaries = []
    for fold, train_idx, val_idx in make_stratified_folds(labels, n_splits=5, seed=int(cfg["seed"])):
        fold_dir = os.path.join(out_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        logger = Logger(fold_dir)
        logger.log(f"Fold {fold}: train={len(train_idx)} val={len(val_idx)}")

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

        train_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
                                  num_workers=int(cfg["num_workers"]), pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
                                num_workers=int(cfg["num_workers"]), pin_memory=True)

        model = build_model_fn(num_classes=len(class_names)).to(device)
        optimizer = build_optim_fn(model)
        scheduler = build_sched_fn(optimizer)

        trainer = Trainer(model, loss_fn, optimizer, scheduler, device, logger, cfg, class_names)

        fit_out = trainer.fit(train_loader, val_loader, fold_dir)
        metrics = trainer.evaluate(val_loader)

        summary = {"fold": fold, "fit": fit_out, "metrics": metrics}
        logger.save_json(summary, "metrics.json")
        fold_summaries.append(summary)

    # aggregate
    cls_acc = [s["metrics"]["cls"]["acc"] for s in fold_summaries]
    cls_f1 = [s["metrics"]["cls"]["f1_macro"] for s in fold_summaries]
    spread_auc = [s["metrics"]["spread"]["auc_roc"] for s in fold_summaries if s["metrics"]["spread"]["auc_roc"] is not None]
    spread_mcc = [s["metrics"]["spread"]["mcc"] for s in fold_summaries if s["metrics"]["spread"]["mcc"] is not None]

    agg = {
        "cls_acc_mean": float(np.mean(cls_acc)),
        "cls_acc_std": float(np.std(cls_acc)),
        "cls_f1_mean": float(np.mean(cls_f1)),
        "cls_f1_std": float(np.std(cls_f1)),
        "spread_auc_mean": float(np.mean(spread_auc)) if spread_auc else None,
        "spread_auc_std": float(np.std(spread_auc)) if spread_auc else None,
        "spread_mcc_mean": float(np.mean(spread_mcc)) if spread_mcc else None,
        "spread_mcc_std": float(np.std(spread_mcc)) if spread_mcc else None,
    }
    with open(os.path.join(out_dir, "crossval_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"folds": fold_summaries, "aggregate": agg}, f, indent=2)

    return agg
