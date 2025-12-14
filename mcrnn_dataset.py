# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.ops import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


@dataclass(frozen=True)
class SamplePaths:
    img: Path
    mask: Path
    name: str  # image filename


def _pair_by_stem(images_dir: Path, masks_dir: Path) -> List[SamplePaths]:
    img_map = {p.stem: p for p in images_dir.iterdir() if p.is_file()}
    mask_map = {p.stem: p for p in masks_dir.iterdir() if p.is_file()}

    common = sorted(set(img_map.keys()) & set(mask_map.keys()))
    if not common:
        raise RuntimeError(f"No matched pairs by stem between:\n- {images_dir}\n- {masks_dir}")

    pairs: List[SamplePaths] = []
    for stem in common:
        pairs.append(SamplePaths(img=img_map[stem], mask=mask_map[stem], name=img_map[stem].name))
    return pairs


def _build_target(
    img: tv_tensors.Image,
    mask: torch.Tensor,
    image_id: int,
) -> Dict:
    if mask.ndim == 3:
        # read_image는 (C,H,W). 보통 single channel PNG라면 C=1
        mask2d = mask[0]
    else:
        mask2d = mask

    obj_ids = torch.unique(mask2d)
    obj_ids = obj_ids[obj_ids != 0] # background 제거(0 가정)
    num_objs = int(obj_ids.numel())

    canvas_size = F.get_size(img) # (H,W)

    if num_objs == 0:
        empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
        empty_masks = torch.zeros((0, canvas_size[0], canvas_size[1]), dtype=torch.uint8)
        target = {
            "boxes": tv_tensors.BoundingBoxes(empty_boxes, format="XYXY", canvas_size=canvas_size),
            "masks": tv_tensors.Mask(empty_masks),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
        }
        return target

    masks = (mask2d == obj_ids[:, None, None]).to(dtype=torch.uint8) # (N,H,W)
    boxes = masks_to_boxes(masks).to(dtype=torch.float32) # (N,4)
    labels = torch.ones((num_objs,), dtype=torch.int64)
    iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    target = {
        "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=canvas_size),
        "masks": tv_tensors.Mask(masks),
        "labels": labels,
        "image_id": torch.tensor(image_id, dtype=torch.int64),
        "area": area,
        "iscrowd": iscrowd,
    }
    return target


class BallsDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transforms=None,
        seed: int = 9102,
        rotate90_prob: float = 0.5,
    ):
        self.root = Path(root)
        self.transforms = transforms

        self.images_dir = self.root / "JPEGImages"
        self.masks_dir = self.root / "SegmentationObject"
        self.samples = _pair_by_stem(self.images_dir, self.masks_dir)

        # 재현성용: Dataset 내부 RNG (멀티워커면 worker_init_fn까지 같이 쓰는 게 베스트)
        self._g = torch.Generator()
        self._g.manual_seed(seed)
        self.rotate90_prob = float(rotate90_prob)

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_rotate90(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotate90_prob <= 0:
            return img, mask
        do = torch.rand((), generator=self._g).item() < self.rotate90_prob
        if do:
            img = torch.rot90(img, k=1, dims=(1, 2))
            mask = torch.rot90(mask, k=1, dims=(1, 2))
        return img, mask

    def __getitem__(self, idx: int):
        sp = self.samples[idx]

        img = read_image(str(sp.img))     # (C,H,W), uint8
        mask = read_image(str(sp.mask))   # (C,H,W), uint8

        # (기존 코드와 동일한) 간단 rotate90 증강을 dataset 단계에서 유지
        # -> 원하면 이걸 transforms 쪽으로 완전히 옮기는 것도 가능
        img, mask = self._maybe_rotate90(img, mask)

        img_tv = tv_tensors.Image(img)
        target = _build_target(img_tv, mask, image_id=idx)

        # 디버깅/추적용 메타
        target["path"] = [str(sp.img), str(sp.mask)]
        target["name"] = sp.name

        if self.transforms is not None:
            img_tv, target = self.transforms(img_tv, target)

        return img_tv, target

#%% 
from torchvision.transforms import v2 as T

def balls_get_transform(train: bool):
    tfms = []
    if train:
        tfms += [
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
        ]
    tfms += [
        T.ToDtype(torch.float32, scale=True),
        T.ToPureTensor(),
    ]
    return T.Compose(tfms)


BALLS_COLORS: List[Tuple[int, int, int]] = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
    (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (255, 255, 255),
    (255, 99, 71), (64, 224, 208), (176, 224, 230), (255, 140, 0),
    (154, 205, 50), (255, 20, 147), (123, 104, 238), (255, 160, 122),
    (50, 205, 50), (186, 85, 211), (222, 184, 135), (100, 149, 237),
    (255, 69, 0), (107, 142, 35), (255, 182, 193), (72, 61, 139),
    (85, 107, 47), (255, 228, 225), (138, 43, 226), (32, 178, 170),
    (255, 127, 80), (46, 139, 87), (205, 133, 63), (210, 105, 30),
    (128, 0, 128), (102, 205, 170), (199, 21, 133),
]
BALLS_COLORS = BALLS_COLORS * 20
# %%
