from __future__ import annotations
import os
import argparse
import yaml
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from data.datasets import LeafImageDataset
from data.transforms import build_transforms
from losses.multitask_loss import MultiTaskLoss
from models.adjleafgnn import AdjLeafGNN
from train.crossval import run_crossval
from utils.seed import set_seed
from utils.device import resolve_device
from utils.logging import Logger
from inference.realtime import predict_single

def load_cfg(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_model_from_cfg(cfg, num_classes: int):
    return AdjLeafGNN(
        num_classes=num_classes,
        base_ch=int(cfg["model"]["encoder_channels"]),
        aspp_dilations=tuple(cfg["model"]["aspp_dilations"]),
        dropout=float(cfg["train"]["dropout"]),
        gnn_type=str(cfg["model"]["gnn_type"]),
        gnn_hidden=int(cfg["model"]["gnn_hidden"]),
        gat_heads=int(cfg["model"]["gat_heads"]),
        k_nn=int(cfg["graph"]["k_nn"]),
        self_loops=bool(cfg["graph"]["use_self_loops"]),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval", "infer"], required=True)
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--data_root", default=None, help="ImageFolder root. Required for train/eval.")
    ap.add_argument("--out", default="runs/exp", help="Output directory for training.")
    ap.add_argument("--ckpt", default=None, help="Checkpoint path for eval/infer.")
    ap.add_argument("--image", default=None, help="Image path for infer.")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(int(cfg["seed"]))
    device = resolve_device(str(cfg["device"]))

    if args.mode in ("train", "eval") and not args.data_root:
        raise ValueError("--data_root is required for train/eval")

    if args.mode == "train":
        tfm_train = build_transforms(img_size=int(cfg["img_size"]), train=True)
        dataset = LeafImageDataset(root=args.data_root, transform=tfm_train)

        def build_model_fn(num_classes: int):
            return build_model_from_cfg(cfg, num_classes=num_classes)

        def build_optim_fn(model):
            return Adam(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))

        def build_sched_fn(optim):
            kind = cfg["train"]["scheduler"]
            if kind == "none":
                return None
            if kind == "step":
                return StepLR(optim, step_size=int(cfg["train"]["step_size"]), gamma=float(cfg["train"]["gamma"]))
            if kind == "plateau":
                return ReduceLROnPlateau(optim, mode="min", patience=5, factor=0.5)
            return None

        loss_fn = MultiTaskLoss(lambda_cls=float(cfg["multitask"]["lambda_cls"]),
                                lambda_spread=float(cfg["multitask"]["lambda_spread"]))

        os.makedirs(args.out, exist_ok=True)
        logger = Logger(args.out)
        logger.log(f"Device: {device}")
        logger.log(f"Classes: {dataset.info.classes}")

        agg = run_crossval(dataset, build_model_fn, build_optim_fn, build_sched_fn, loss_fn, device, cfg, args.out)
        logger.save_json(agg, "aggregate.json")
        logger.log(f"Cross-val aggregate: {agg}")

    elif args.mode == "eval":
        if not args.ckpt:
            raise ValueError("--ckpt required for eval")
        tfm = build_transforms(img_size=int(cfg["img_size"]), train=False)
        dataset = LeafImageDataset(root=args.data_root, transform=tfm)
        class_names = dataset.info.classes

        model = build_model_from_cfg(cfg, num_classes=len(class_names)).to(device)
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"], strict=True)
        model.eval()

        # quick evaluation over full dataset (not fold-specific)
        from torch.utils.data import DataLoader
        from train.trainer import Trainer
        loss_fn = MultiTaskLoss(lambda_cls=float(cfg["multitask"]["lambda_cls"]),
                                lambda_spread=float(cfg["multitask"]["lambda_spread"]))
        opt = Adam(model.parameters(), lr=1e-4)
        tr = Trainer(model, loss_fn, opt, None, device, Logger(os.path.dirname(args.ckpt)), cfg, class_names)
        loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=int(cfg["num_workers"]))
        metrics = tr.evaluate(loader)
        print(metrics)

    else:  # infer
        if not args.ckpt or not args.image:
            raise ValueError("--ckpt and --image required for infer")

        # need class names: require data_root OR store in ckpt in future
        if not args.data_root:
            raise ValueError("--data_root is required for infer to load class names")
        tfm = build_transforms(img_size=int(cfg["img_size"]), train=False)
        dataset = LeafImageDataset(root=args.data_root, transform=tfm)
        class_names = dataset.info.classes

        model = build_model_from_cfg(cfg, num_classes=len(class_names)).to(device)
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"], strict=True)

        pred = predict_single(model, args.image, device, int(cfg["img_size"]), class_names)
        print(pred)

if __name__ == "__main__":
    main()
