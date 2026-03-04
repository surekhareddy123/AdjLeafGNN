from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils.metrics import cls_metrics, spread_metrics
from data.spread_labels import infer_infected_mask, spread_labels_neighbor_any, spread_labels_distance_threshold

class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, device, logger, cfg, class_names):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.cfg = cfg
        self.class_names = class_names

    def _make_spread_labels(self, batch, forward_out):
        # batch: dict of tensors
        y_cls = batch["y_cls"].detach().cpu().numpy()
        classes = self.class_names
        infected_mask = infer_infected_mask(y_cls, classes, policy=self.cfg["multitask"]["infected_class_policy"])
        mode = self.cfg["multitask"]["spread_label_mode"]
        if mode == "neighbor_any":
            adj = forward_out["adj"].detach().cpu().numpy()
            y_sp = spread_labels_neighbor_any(adj=adj, infected_mask=infected_mask)
        elif mode == "distance_threshold":
            dist = forward_out["dist"].detach().cpu().numpy()
            # set diagonal to inf to avoid trivial self-match
            np.fill_diagonal(dist, np.inf)
            delta = float(self.cfg["multitask"]["spread_delta"])
            y_sp = spread_labels_distance_threshold(dist_mat=dist, infected_mask=infected_mask, delta=delta)
        else:
            raise ValueError(f"Unknown spread_label_mode: {mode}")
        return torch.tensor(y_sp, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_y = []
        all_pred = []
        all_sp_true = []
        all_sp_prob = []
        all_sp_pred = []
        for batch in loader:
            images = batch["image"].to(self.device)
            y_cls = batch["y_cls"].to(self.device)
            out = self.model(images)
            y_sp = self._make_spread_labels(batch, out)

            logits = out["logits_cls"]
            pred = torch.argmax(logits, dim=1)

            sp_prob = torch.sigmoid(out["logits_spread"])
            sp_pred = (sp_prob >= 0.5).long()

            all_y.append(y_cls.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
            all_sp_true.append(y_sp.detach().cpu().numpy())
            all_sp_prob.append(sp_prob.detach().cpu().numpy())
            all_sp_pred.append(sp_pred.detach().cpu().numpy())

        y = np.concatenate(all_y)
        yhat = np.concatenate(all_pred)
        sp_y = np.concatenate(all_sp_true)
        sp_prob = np.concatenate(all_sp_prob)
        sp_hat = np.concatenate(all_sp_pred)

        return {
            "cls": cls_metrics(y, yhat),
            "spread": spread_metrics(sp_y, sp_prob, sp_hat)
        }

    def fit(self, train_loader, val_loader, out_dir):
        best_val_loss = float("inf")
        best_path = os.path.join(out_dir, "best.pt")
        last_path = os.path.join(out_dir, "last.pt")

        patience = int(self.cfg["train"]["early_stopping_patience"])
        bad_epochs = 0

        for epoch in range(1, int(self.cfg["train"]["epochs"]) + 1):
            self.model.train()
            losses = []
            pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
            for batch in pbar:
                images = batch["image"].to(self.device)
                y_cls = batch["y_cls"].to(self.device)

                out = self.model(images)
                y_spread = self._make_spread_labels(batch, out)

                loss, parts = self.loss_fn(out["logits_cls"], y_cls, out["logits_spread"], y_spread)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                losses.append(parts["loss"])
                pbar.set_postfix(loss=float(np.mean(losses)))

            # Validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(self.device)
                    y_cls = batch["y_cls"].to(self.device)
                    out = self.model(images)
                    y_spread = self._make_spread_labels(batch, out)
                    loss, parts = self.loss_fn(out["logits_cls"], y_cls, out["logits_spread"], y_spread)
                    val_losses.append(parts["loss"])
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

            # Scheduler
            if self.scheduler is not None:
                kind = self.cfg["train"]["scheduler"]
                if kind == "plateau":
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Metrics (lightweight)
            metrics = self.evaluate(val_loader)
            self.logger.log(f"Epoch {epoch}: train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f} "
                            f"val_acc={metrics['cls']['acc']:.4f} val_f1={metrics['cls']['f1_macro']:.4f} "
                            f"spread_auc={metrics['spread'].get('auc_roc')} spread_mcc={metrics['spread'].get('mcc')}")

            # Checkpoint
            torch.save({"model": self.model.state_dict(), "cfg": self.cfg}, last_path)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                bad_epochs = 0
                torch.save({"model": self.model.state_dict(), "cfg": self.cfg}, best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    self.logger.log(f"Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
                    break

        return {"best_val_loss": best_val_loss, "best_ckpt": best_path, "last_ckpt": last_path}
