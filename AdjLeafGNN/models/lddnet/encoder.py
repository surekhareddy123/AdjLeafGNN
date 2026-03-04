from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SimpleBackbone
from .aspp import ASPP
from .csam import CSAM

class LDDEncoder(nn.Module):
    """LDDNet-style encoder: backbone -> ASPP -> CSAM -> feature map + GAP embedding."""
    def __init__(self, base_ch: int = 64, aspp_dilations=(1,6,12,18), dropout: float = 0.3, emb_dim: int = 256):
        super().__init__()
        self.backbone = SimpleBackbone(base_ch=base_ch)
        in_ch = base_ch * 4
        self.aspp = ASPP(in_ch=in_ch, out_ch=emb_dim, dilations=aspp_dilations)
        self.csam = CSAM(ch=emb_dim)
        self.dropout = nn.Dropout(p=dropout)

    @property
    def embedding_dim(self) -> int:
        return self.aspp.project[0].out_channels  # emb_dim

    def forward(self, x: torch.Tensor):
        fmap = self.backbone(x)          # (B, C, H, W)
        fmap = self.aspp(fmap)           # (B, emb, H, W)
        fmap = self.csam(fmap)           # attention
        fmap = self.dropout(fmap)
        emb = F.adaptive_avg_pool2d(fmap, 1).flatten(1)  # GAP (B, emb)
        return fmap, emb
