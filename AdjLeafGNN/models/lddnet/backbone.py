from __future__ import annotations
import torch
import torch.nn as nn

def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class SimpleBackbone(nn.Module):
    """Lightweight CNN backbone to produce feature maps."""
    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn_relu(3, base_ch, 3, 2, 1),
            conv_bn_relu(base_ch, base_ch, 3, 1, 1),
        )
        self.stage1 = nn.Sequential(
            conv_bn_relu(base_ch, base_ch*2, 3, 2, 1),
            conv_bn_relu(base_ch*2, base_ch*2, 3, 1, 1),
        )
        self.stage2 = nn.Sequential(
            conv_bn_relu(base_ch*2, base_ch*4, 3, 2, 1),
            conv_bn_relu(base_ch*4, base_ch*4, 3, 1, 1),
        )

    @property
    def out_channels(self):
        return None  # not used

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x
