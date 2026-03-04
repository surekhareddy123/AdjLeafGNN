from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

@dataclass
class DatasetInfo:
    classes: List[str]
    class_to_idx: dict

class LeafImageDataset(Dataset):
    """Thin wrapper around ImageFolder exposing targets and paths."""
    def __init__(self, root: str, transform=None):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")
        self.ds = ImageFolder(root=root, transform=transform)
        self.transform = transform

    @property
    def targets(self) -> List[int]:
        return list(self.ds.targets)

    @property
    def paths(self) -> List[str]:
        return [p for (p, _) in self.ds.samples]

    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(classes=self.ds.classes, class_to_idx=self.ds.class_to_idx)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, y_cls = self.ds[idx]
        path, _ = self.ds.samples[idx]
        return {"image": img, "y_cls": int(y_cls), "path": path}
