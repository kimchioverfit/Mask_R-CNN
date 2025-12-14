# %%
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchvision
import matplotlib.pyplot as plt

import utils
from engine import train_one_epoch, evaluate
from mrcnn_dataset import BallsDataset, balls_get_transform
from scramble import scramble_text


@dataclass
class TrainConfig:
    seed: int = 9102

    data_root: str = "data/compressed_ball_2025203_voc"
    num_classes: int = 2 # background + ball

    # split
    test_size: int = 30

    # dataloader
    batch_size: int = 2
    num_workers: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 3

    # model
    backbone_name: str = "resnet101"  # 'resnet50', 'resnet101', etc.
    backbone_weights: str = "IMAGENET1K_V2"  # for resnet101
    detections_per_img: int = 100
    rpn_nms_thresh: float = 0.5

    # optim
    lr: float = 1e-4
    weight_decay: float = 0.0
    num_epochs: int = 1000
    step_size: Optional[int] = None  # default: num_epochs//10
    gamma: float = 0.5

    # saving
    model_dir: str = "./model"
    save_prefix: str = "model_101v2.1_lmtmcmlm"
    peek_epochs: int = 100
    best_loss_init: float = 0.38
    save_flag_file: str = "save.model.txt"

    # debug
    show_first_n: int = 0  # e.g., 5면 처음 5장 띄움


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # 필요하면 켜기
    # torch.backends.cudnn.deterministic = True # 성능/재현성 트레이드오프
    # torch.backends.cudnn.benchmark = False # 성능/재현성 트레이드오프


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def build_datasets(cfg: TrainConfig):
    ds_train = BallsDataset(cfg.data_root, balls_get_transform(train=True), seed=cfg.seed)
    ds_test = BallsDataset(cfg.data_root, balls_get_transform(train=False), seed=cfg.seed)

    g = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(len(ds_train), generator=g).tolist()

    train_idx = indices[:-cfg.test_size] if cfg.test_size > 0 else indices
    test_idx = indices[-cfg.test_size:] if cfg.test_size > 0 else []

    ds_train = torch.utils.data.Subset(ds_train, train_idx)
    ds_test = torch.utils.data.Subset(ds_test, test_idx) if test_idx else ds_test
    return ds_train, ds_test


def build_loaders(cfg: TrainConfig, ds_train, ds_test):
    train_loader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        collate_fn=utils.collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        collate_fn=utils.collate_fn,
    )
    return train_loader, test_loader


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models import ResNet101_Weights

    if cfg.backbone_name.lower() == "resnet101":
        weights = getattr(ResNet101_Weights, cfg.backbone_weights)
    else:
        # 필요하면 다른 backbone weights 매핑 추가
        weights = None

    backbone = resnet_fpn_backbone(backbone_name=cfg.backbone_name, weights=weights)

    # 기본 anchor_generator 그대로 쓰고 싶으면 제거 가능.
    # 아래는 MaskRCNN 생성에 필수는 아니지만, 명시하면 구조가 분명해짐. # TODO
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    model = MaskRCNN(
        backbone=backbone,
        num_classes=cfg.num_classes,
        rpn_anchor_generator=anchor_generator,
    )

    model.roi_heads.detections_per_img = cfg.detections_per_img
    model.rpn.nms_thresh = cfg.rpn_nms_thresh
    return model


def attach_provenance(model: torch.nn.Module, src_py: Path, dataset_py: Path) -> None:
    model.pro = {}
    with open(src_py, "r", encoding="utf-8", errors="ignore") as f:
        model.pro["src"] = scramble_text(f.read())
    with open(dataset_py, "r", encoding="utf-8", errors="ignore") as f:
        model.pro["dta"] = scramble_text(f.read())


def maybe_show_samples(cfg: TrainConfig, ds_train) -> None:
    n = int(cfg.show_first_n)
    if n <= 0:
        return
    for i in range(min(n, len(ds_train))):
        img, tgt = ds_train[i]
        plt.figure()
        plt.title(tgt.get("name", f"sample_{i}"))
        plt.imshow(img.numpy().transpose(1, 2, 0))
        plt.axis("off")
        plt.show()


def should_save(cfg: TrainConfig, epoch: int, loss_val: float, best_loss: float) -> Tuple[bool, float]:
    is_best = loss_val < best_loss
    if is_best:
        return True, loss_val

    # save flag file
    flag_exists = Path(cfg.save_flag_file).exists()
    if flag_exists:
        return True, best_loss

    # periodic peek
    if cfg.peek_epochs > 0 and (epoch + 1) % cfg.peek_epochs == 0:
        return True, best_loss

    return False, best_loss


def save_model(cfg: TrainConfig, model: torch.nn.Module, epoch: int, loss_val: float, loss_train: float) -> Path:
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    out = Path(cfg.model_dir) / f"{cfg.save_prefix}_val_{epoch+1}_{loss_val:.4f}_{loss_train:.4f}.pth"
    torch.save(model, out)
    return out



def main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = get_device()

    ds_train, ds_test = build_datasets(cfg)
    train_loader, test_loader = build_loaders(cfg, ds_train, ds_test)

    maybe_show_samples(cfg, ds_train)

    model = build_model(cfg).to(device)

    # optimizer / scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    step_size = cfg.step_size if cfg.step_size is not None else max(1, cfg.num_epochs // 10)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=cfg.gamma)

    try: # __file__이 없는 환경 주피터 노트북 예외 처리
        src_py = Path(__file__) 
        dataset_py = Path("mrcnn_dataset.py")
        if src_py.exists() and dataset_py.exists():
            attach_provenance(model, src_py, dataset_py)
    except NameError:
        pass

    best_loss = float(cfg.best_loss_init)
    last_metric_logger = None
    last_metric_logger_val = None

    for epoch in range(cfg.num_epochs):
        metric_logger = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=50
        )
        lr_scheduler.step()

        metric_logger_val = evaluate(model, test_loader, device)
        loss_val = float(metric_logger_val.loss.global_avg)
        loss_train = float(metric_logger.loss.global_avg)

        do_save, best_loss = should_save(cfg, epoch, loss_val, best_loss)
        if do_save:
            out = save_model(cfg, model, epoch, loss_val, loss_train)
            print(f"[SAVE] epoch={epoch+1} val_loss={loss_val:.4f} train_loss={loss_train:.4f} -> {out}")

        last_metric_logger = metric_logger
        last_metric_logger_val = metric_logger_val

    if last_metric_logger is not None and last_metric_logger_val is not None:
        loss_val = float(last_metric_logger_val.loss.global_avg)
        loss_train = float(last_metric_logger.loss.global_avg)

        Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
        out = Path(cfg.model_dir) / f"{cfg.save_prefix}_val_last_{cfg.num_epochs}_{loss_val:.4f}_{loss_train:.4f}.pth"
        torch.save(model, out)
        print(f"[SAVE-LAST] -> {out}")


if __name__ == "__main__":
    cfg = TrainConfig(
        data_root="data/compressed_ball_2025203_voc",
        show_first_n=5,   # 체크용 5장 

    )
    main(cfg)
# %%
