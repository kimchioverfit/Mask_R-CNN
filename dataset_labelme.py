# dataset_labelme.py
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class LabelMeInstanceDataset(Dataset):
    """
    디렉토리 구조(예):
      root/
        images/
          0001.jpg, ...
        annotations/
          0001.json, ...
    모든 객체는 label과 무관하게 클래스=1 로 처리 (배경=0).
    """

    def __init__(
        self,
        root_dir: str,
        img_dir: str = "images",
        ann_dir: str = "annotations",
        drop_empty: bool = True,
    ):
        self.root = Path(root_dir)
        self.img_dir = self.root / img_dir
        self.ann_dir = self.root / ann_dir
        self.json_files = sorted(self.ann_dir.glob("*.json"))
        if len(self.json_files) == 0:
            raise RuntimeError(f"No JSON files found in: {self.ann_dir}")

        self.drop_empty = drop_empty

    def __len__(self):
        return len(self.json_files)

    def _find_image_path_for_json(self, ann_path: Path, ann: dict) -> Path:
        image_path = ann.get("imagePath")
        if image_path:
            p = self.img_dir / image_path
            if p.exists():
                return p
        stem = ann_path.stem
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            p = self.img_dir / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No image found for {ann_path.name}")

    def _rasterize_polygon(self, size_hw: Tuple[int, int], polygon: List[Tuple[float, float]]) -> np.ndarray:
        h, w = size_hw
        mask_img = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
        return np.array(mask_img, dtype=np.uint8)

    def _mask_to_box(self, m: np.ndarray) -> Optional[List[float]]:
        ys, xs = np.where(m > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        return [float(x1), float(y1), float(x2), float(y2)]

    def __getitem__(self, idx: int):
        ann_path = self.json_files[idx]
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        img_path = self._find_image_path_for_json(ann_path, ann)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        shapes = ann.get("shapes", [])
        instance_masks = []
        boxes = []
        labels = []

        for sh in shapes:
            pts = sh.get("points", [])
            if len(pts) < 3:
                continue
            poly = [(float(x), float(y)) for x, y in pts]
            m = self._rasterize_polygon((h, w), poly)
            box = self._mask_to_box(m)
            if box is None:
                continue
            instance_masks.append(m)
            boxes.append(box)
            labels.append(1)  # 단일 클래스

        # 비어있는 샘플 처리
        if len(instance_masks) == 0 and self.drop_empty:
            # 빈 샘플은 건너뛰고 다음 인덱스를 반환
            # (DataLoader가 다시 호출하도록 IndexError 대신 재귀)
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

        # 빈 샘플을 유지하려면(검증용 등) zero-length 텐서로 넘길 수도 있음
        if len(instance_masks) == 0:
            image = F.to_tensor(img)
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, h, w), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
            return image, target

        masks = np.stack(instance_masks, axis=0)  # (N, H, W)
        image = F.to_tensor(img)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([idx]),
        }
        # area / iscrowd
        b = target["boxes"]
        target["area"] = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        target["iscrowd"] = torch.zeros((b.shape[0],), dtype=torch.int64)

        return image, target


def collate_fn(batch):
    # list of images, list of targets
    return tuple(zip(*batch))
