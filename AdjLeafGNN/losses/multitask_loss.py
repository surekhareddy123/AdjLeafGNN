from __future__ import annotations
import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_cls: float = 1.0, lambda_spread: float = 0.5):
        super().__init__()
        self.lambda_cls = float(lambda_cls)
        self.lambda_spread = float(lambda_spread)
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits_cls: torch.Tensor, y_cls: torch.Tensor, logits_spread: torch.Tensor, y_spread: torch.Tensor):
        loss_cls = self.ce(logits_cls, y_cls)
        loss_spread = self.bce(logits_spread, y_spread.float())
        loss = self.lambda_cls * loss_cls + self.lambda_spread * loss_spread
        return loss, {"loss": float(loss.item()), "loss_cls": float(loss_cls.item()), "loss_spread": float(loss_spread.item())}
