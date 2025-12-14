from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torch import Tensor, nn
from torchvision import ops
from torchvision.transforms import (
    InterpolationMode,
    functional as F,
    transforms as T,
)


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped = kps[:, flip_inds]
    flipped[..., 0] = width - flipped[..., 0]

    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped[..., 2] == 0
    flipped[inds] = 0
    return flipped


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)

            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]

                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)

                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    target["keypoints"] = _flip_coco_person_keypoints(keypoints, width)

        return image, target


class PILToTensor(nn.Module):
    def forward(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ToDtype(nn.Module):
    def __init__(self, dtype: torch.dtype, scale: bool = False) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def forward(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if not self.scale:
            return image.to(dtype=self.dtype), target

        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class RandomIoUCrop(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        # Similar to SSD config
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(
                    f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions."
                )
            if image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_h, orig_w = F.get_dimensions(image)

        while True:
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]

            # >= 1 means leave as-is
            if min_jaccard_overlap >= 1.0:
                return image, target

            for _ in range(self.trials):
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])

                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h

                if left == right or top == bottom:
                    continue

                # check centers inside crop
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                inside = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not inside.any():
                    continue

                boxes = target["boxes"][inside]
                crop_box = torch.tensor(
                    [[left, top, right, bottom]],
                    dtype=boxes.dtype,
                    device=boxes.device,
                )
                ious = torchvision.ops.boxes.box_iou(boxes, crop_box)
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and crop
                target["boxes"] = boxes
                target["labels"] = target["labels"][inside]

                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)

                image = F.crop(image, top, left, new_h, new_w)
                return image, target


class RandomZoomOut(nn.Module):
    def __init__(
        self,
        fill: Optional[List[float]] = None,
        side_range: Tuple[float, float] = (1.0, 4.0),
        p: float = 0.5,
    ):
        super().__init__()
        self.fill = [0.0, 0.0, 0.0] if fill is None else fill
        self.side_range = side_range
        self.p = p

        if side_range[0] < 1.0 or side_range[0] > side_range[1]:
            raise ValueError(f"Invalid canvas side range provided {side_range}.")

    @torch.jit.unused
    def _get_fill_value(self, is_pil):
        # type: (bool) -> int
        return tuple(int(x) for x in self.fill) if is_pil else 0

    def forward(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(
                    f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions."
                )
            if image.ndimension() == 2:
                image = image.unsqueeze(0)

        if torch.rand(1) >= self.p:
            return image, target

        _, orig_h, orig_w = F.get_dimensions(image)

        r = self.side_range[0] + torch.rand(1) * (self.side_range[1] - self.side_range[0])
        canvas_w = int(orig_w * r)
        canvas_h = int(orig_h * r)

        r = torch.rand(2)
        left = int((canvas_w - orig_w) * r[0])
        top = int((canvas_h - orig_h) * r[1])
        right = canvas_w - (left + orig_w)
        bottom = canvas_h - (top + orig_h)

        if torch.jit.is_scripting():
            fill = 0
        else:
            fill = self._get_fill_value(F._is_pil_image(image))

        image = F.pad(image, [left, top, right, bottom], fill=fill)

        if isinstance(image, torch.Tensor):
            # pad fill is integer-only; overwrite with float fill for tensors
            v = torch.tensor(self.fill, device=image.device, dtype=image.dtype).view(-1, 1, 1)
            image[..., :top, :] = v
            image[..., (top + orig_h) :, :] = v
            image[..., :, :left] = v
            image[..., :, (left + orig_w) :] = v

        if target is not None:
            target["boxes"][:, 0::2] += left
            target["boxes"][:, 1::2] += top

        return image, target


class RandomPhotometricDistort(nn.Module):
    def __init__(
        self,
        contrast: Tuple[float, float] = (0.5, 1.5),
        saturation: Tuple[float, float] = (0.5, 1.5),
        hue: Tuple[float, float] = (-0.05, 0.05),
        brightness: Tuple[float, float] = (0.875, 1.125),
        p: float = 0.5,
    ):
        super().__init__()
        self._brightness = T.ColorJitter(brightness=brightness)
        self._contrast = T.ColorJitter(contrast=contrast)
        self._hue = T.ColorJitter(hue=hue)
        self._saturation = T.ColorJitter(saturation=saturation)
        self.p = p

    def forward(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(
                    f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions."
                )
            if image.ndimension() == 2:
                image = image.unsqueeze(0)

        r = torch.rand(7)

        if r[0] < self.p:
            image = self._brightness(image)

        contrast_before = r[1] < 0.5
        if contrast_before and r[2] < self.p:
            image = self._contrast(image)

        if r[3] < self.p:
            image = self._saturation(image)

        if r[4] < self.p:
            image = self._hue(image)

        if (not contrast_before) and r[5] < self.p:
            image = self._contrast(image)

        if r[6] < self.p:
            channels, _, _ = F.get_dimensions(image)
            perm = torch.randperm(channels)

            is_pil = F._is_pil_image(image)
            if is_pil:
                image = F.pil_to_tensor(image)
                image = F.convert_image_dtype(image)

            image = image[..., perm, :, :]

            if is_pil:
                image = F.to_pil_image(image)

        return image, target


class ScaleJitter(nn.Module):
    """
    Randomly resizes the image and its bounding boxes within a specified scale range.

    Implements Scale Jitter augmentation from:
    "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation"
    https://arxiv.org/abs/2012.07177
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.1, 2.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias=True,
    ):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(
                    f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions."
                )
            if image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_h, orig_w = F.get_dimensions(image)

        scale = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        r = min(self.target_size[1] / orig_h, self.target_size[0] / orig_w) * scale

        new_w = int(orig_w * r)
        new_h = int(orig_h * r)

        image = F.resize(
            image,
            [new_h, new_w],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )

        if target is not None:
            target["boxes"][:, 0::2] *= new_w / orig_w
            target["boxes"][:, 1::2] *= new_h / orig_h

            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"],
                    [new_h, new_w],
                    interpolation=InterpolationMode.NEAREST,
                    antialias=self.antialias,
                )

        return image, target


class FixedSizeCrop(nn.Module):
    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        size = tuple(T._setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))
        self.crop_height = size[0]
        self.crop_width = size[1]
        self.fill = fill
        self.padding_mode = padding_mode

    def _pad(self, img, target, padding):
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        elif len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        else:
            pad_left, pad_top, pad_right, pad_bottom = padding

        padding = [pad_left, pad_top, pad_right, pad_bottom]
        img = F.pad(img, padding, self.fill, self.padding_mode)

        if target is not None:
            target["boxes"][:, 0::2] += pad_left
            target["boxes"][:, 1::2] += pad_top
            if "masks" in target:
                target["masks"] = F.pad(target["masks"], padding, 0, "constant")

        return img, target

    def _crop(self, img, target, top, left, height, width):
        img = F.crop(img, top, left, height, width)

        if target is not None:
            boxes = target["boxes"]
            boxes[:, 0::2] -= left
            boxes[:, 1::2] -= top
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)

            is_valid = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])

            target["boxes"] = boxes[is_valid]
            target["labels"] = target["labels"][is_valid]

            if "masks" in target:
                target["masks"] = F.crop(target["masks"][is_valid], top, left, height, width)

        return img, target

    def forward(self, img, target=None):
        _, height, width = F.get_dimensions(img)

        new_h = min(height, self.crop_height)
        new_w = min(width, self.crop_width)

        if new_h != height or new_w != width:
            offset_h = max(height - self.crop_height, 0)
            offset_w = max(width - self.crop_width, 0)

            r = torch.rand(1)
            top = int(offset_h * r)
            left = int(offset_w * r)

            img, target = self._crop(img, target, top, left, new_h, new_w)

        pad_bottom = max(self.crop_height - new_h, 0)
        pad_right = max(self.crop_width - new_w, 0)

        if pad_bottom != 0 or pad_right != 0:
            img, target = self._pad(img, target, [0, 0, pad_right, pad_bottom])

        return img, target


class RandomShortestSize(nn.Module):
    def __init__(
        self,
        min_size: Union[List[int], Tuple[int], int],
        max_size: int,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.min_size = [min_size] if isinstance(min_size, int) else list(min_size)
        self.max_size = max_size
        self.interpolation = interpolation

    def forward(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        _, orig_h, orig_w = F.get_dimensions(image)

        min_size = self.min_size[torch.randint(len(self.min_size), (1,)).item()]
        r = min(
            min_size / min(orig_h, orig_w),
            self.max_size / max(orig_h, orig_w),
        )

        new_w = int(orig_w * r)
        new_h = int(orig_h * r)

        image = F.resize(image, [new_h, new_w], interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_w / orig_w
            target["boxes"][:, 1::2] *= new_h / orig_h

            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"],
                    [new_h, new_w],
                    interpolation=InterpolationMode.NEAREST,
                )

        return image, target


def _copy_paste(
    image: torch.Tensor,
    target: Dict[str, Tensor],
    paste_image: torch.Tensor,
    paste_target: Dict[str, Tensor],
    blending: bool = True,
    resize_interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
) -> Tuple[torch.Tensor, Dict[str, Tensor]]:
    num_masks = len(paste_target["masks"])
    if num_masks < 1:
        # Degenerate case can happen (e.g., with LSJ). Just return original.
        return image, target

    # Random paste target selection (explicit torch.long for scripting)
    random_selection = torch.randint(0, num_masks, (num_masks,), device=paste_image.device)
    random_selection = torch.unique(random_selection).to(torch.long)

    paste_masks = paste_target["masks"][random_selection]
    paste_boxes = paste_target["boxes"][random_selection]
    paste_labels = paste_target["labels"][random_selection]

    masks = target["masks"]

    # Resize paste data if sizes differ
    size1 = image.shape[-2:]
    size2 = paste_image.shape[-2:]
    if size1 != size2:
        paste_image = F.resize(paste_image, size1, interpolation=resize_interpolation)
        paste_masks = F.resize(paste_masks, size1, interpolation=F.InterpolationMode.NEAREST)

        ratios = torch.tensor(
            (size1[1] / size2[1], size1[0] / size2[0]),
            device=paste_boxes.device,
        )
        paste_boxes = paste_boxes.view(-1, 2, 2).mul(ratios).view(paste_boxes.shape)

    paste_alpha_mask = paste_masks.sum(dim=0) > 0

    if blending:
        paste_alpha_mask = F.gaussian_blur(
            paste_alpha_mask.unsqueeze(0),
            kernel_size=(5, 5),
            sigma=[2.0],
        )

    # Copy-paste images
    image = (image * (~paste_alpha_mask)) + (paste_image * paste_alpha_mask)

    # Copy-paste masks
    masks = masks * (~paste_alpha_mask)
    non_all_zero = masks.sum((-1, -2)) > 0
    masks = masks[non_all_zero]

    # Shallow copy of target
    out_target = {k: v for k, v in target.items()}

    out_target["masks"] = torch.cat([masks, paste_masks])

    # Copy-paste boxes and labels
    boxes = ops.masks_to_boxes(masks)
    out_target["boxes"] = torch.cat([boxes, paste_boxes])

    labels = target["labels"][non_all_zero]
    out_target["labels"] = torch.cat([labels, paste_labels])

    # Optional keys: area and iscrowd
    if "area" in target:
        out_target["area"] = out_target["masks"].sum((-1, -2)).to(torch.float32)

    if "iscrowd" in target and "iscrowd" in paste_target:
        if len(target["iscrowd"]) == len(non_all_zero):
            iscrowd = target["iscrowd"][non_all_zero]
            paste_iscrowd = paste_target["iscrowd"][random_selection]
            out_target["iscrowd"] = torch.cat([iscrowd, paste_iscrowd])

    # Remove degenerate boxes
    boxes = out_target["boxes"]
    degenerate = boxes[:, 2:] <= boxes[:, :2]
    if degenerate.any():
        valid = ~degenerate.any(dim=1)

        out_target["boxes"] = boxes[valid]
        out_target["masks"] = out_target["masks"][valid]
        out_target["labels"] = out_target["labels"][valid]

        if "area" in out_target:
            out_target["area"] = out_target["area"][valid]

        if "iscrowd" in out_target and len(out_target["iscrowd"]) == len(valid):
            out_target["iscrowd"] = out_target["iscrowd"][valid]

    return image, out_target


class SimpleCopyPaste(nn.Module):
    def __init__(
        self,
        blending: bool = True,
        resize_interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.resize_interpolation = resize_interpolation
        self.blending = blending

    def forward(
        self,
        images: List[torch.Tensor],
        targets: List[Dict[str, Tensor]],
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Tensor]]]:
        torch._assert(
            isinstance(images, (list, tuple)) and all(isinstance(v, torch.Tensor) for v in images),
            "images should be a list of tensors",
        )
        torch._assert(
            isinstance(targets, (list, tuple)) and len(images) == len(targets),
            "targets should be a list of the same size as images",
        )

        for target in targets:
            for k in ["masks", "boxes", "labels"]:
                torch._assert(k in target, f"Key {k} should be present in targets")
                torch._assert(isinstance(target[k], torch.Tensor), f"Value for the key {k} should be a tensor")

        # Paste targets as shifted list: [t2, t3, ..., tN, t1]
        images_rolled = images[-1:] + images[:-1]
        targets_rolled = targets[-1:] + targets[:-1]

        out_images: List[torch.Tensor] = []
        out_targets: List[Dict[str, Tensor]] = []

        for image, target, paste_image, paste_target in zip(images, targets, images_rolled, targets_rolled):
            out_img, out_tgt = _copy_paste(
                image,
                target,
                paste_image,
                paste_target,
                blending=self.blending,
                resize_interpolation=self.resize_interpolation,
            )
            out_images.append(out_img)
            out_targets.append(out_tgt)

        return out_images, out_targets

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"blending={self.blending}, "
            f"resize_interpolation={self.resize_interpolation})"
        )
