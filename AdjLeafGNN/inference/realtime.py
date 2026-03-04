from __future__ import annotations
import torch
from PIL import Image

from data.transforms import build_transforms

@torch.no_grad()
def predict_single(model, image_path: str, device, img_size: int, class_names: list[str]):
    model.eval()
    tfm = build_transforms(img_size=img_size, train=False)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    out = model(x)
    logits = out["logits_cls"][0]
    probs = torch.softmax(logits, dim=0)
    pred_idx = int(torch.argmax(probs).item())

    spread_prob = float(torch.sigmoid(out["logits_spread"][0]).item())

    return {
        "pred_class": class_names[pred_idx],
        "pred_class_idx": pred_idx,
        "class_probs": {class_names[i]: float(probs[i].item()) for i in range(len(class_names))},
        "spread_prob": spread_prob,
    }
