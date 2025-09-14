# train_maskrcnn.py
import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

from dataset_labelme import LabelMeInstanceDataset, collate_fn


def get_model(num_classes: int = 2, pretrained: bool = True):
    """
    num_classes = 2  (background, object[1])
    """
    if pretrained:
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
        # box 헤더 교체
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        # mask 헤더 교체
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
    else:
        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)

    return model


def filter_empty_targets(images, targets):
    """
    학습 시 최소 한 개의 positive instance(labels==1)가 있는 샘플만 사용.
    """
    keep_imgs, keep_tgts = [], []
    for img, tgt in zip(images, targets):
        if "labels" in tgt and (tgt["labels"] == 1).any():
            keep_imgs.append(img)
            keep_tgts.append(tgt)
    return keep_imgs, keep_tgts


def train(
    train_root: str,
    val_root: str = None,
    epochs: int = 20,
    batch_size: int = 2,
    lr: float = 0.005,
    num_workers: int = 4,
    out_dir: str = "./outputs",
    pretrained: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(out_dir, exist_ok=True)

    ds_train = LabelMeInstanceDataset(train_root, drop_empty=True)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    ds_val = LabelMeInstanceDataset(val_root, drop_empty=False) if val_root else None
    dl_val = (
        DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        if ds_val
        else None
    )

    model = get_model(num_classes=2, pretrained=pretrained).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for images, targets in dl_train:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 비어있는(positive 없음) 배치 필터링
            images, targets = filter_empty_targets(images, targets)
            if len(images) == 0:
                continue

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()

        lr_scheduler.step()
        mean_loss = running / max(1, len(dl_train))
        print(f"[Epoch {epoch:02d}] train_loss={mean_loss:.4f}")

        if dl_val is not None:
            model.eval()
            total_preds = 0
            total_imgs = 0
            with torch.no_grad():
                for images, _targets in dl_val:
                    images = [img.to(device) for img in images]
                    outputs = model(images)
                    total_preds += sum(len(o["boxes"]) for o in outputs)
                    total_imgs += len(outputs)
            print(f"          val_pred_objects_per_image≈{total_preds/max(1,total_imgs):.2f}")

        ckpt = Path(out_dir) / f"maskrcnn_binary_{epoch:02d}.pth"
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt)
        print(f"Saved: {ckpt}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True, help="root with images/ and annotations/")
    parser.add_argument("--val_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        train_root=args.train_root,
        val_root=args.val_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
        pretrained=args.pretrained,
        device=device,
    )
