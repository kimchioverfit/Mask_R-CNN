# infer_maskrcnn.py
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision.transforms import functional as F

from train_maskrcnn import get_model  # 동일한 헤더 구조를 위해 재사용


@torch.no_grad()
def run_inference(ckpt_path: str, image_path: str, score_thresh: float = 0.5, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(num_classes=2, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    img_pil = Image.open(image_path).convert("RGB")
    x = F.to_tensor(img_pil).to(device)
    out = model([x])[0]

    keep = out["scores"] >= score_thresh
    result = {
        "boxes": out["boxes"][keep].detach().cpu(),
        "labels": out["labels"][keep].detach().cpu(),
        "scores": out["scores"][keep].detach().cpu(),
        "masks": out["masks"][keep].detach().cpu(),  # (N,1,H,W), sigmoid output
    }
    return img_pil, result


def visualize_and_save(img_pil: Image.Image, result: dict, save_path: str, mask_thresh: float = 0.5):
    img = np.array(img_pil)[:, :, ::-1]  # RGB->BGR
    h, w = img.shape[:2]

    masks = (result["masks"].numpy() > mask_thresh).astype(np.uint8)
    masks = masks[:, 0, :, :] if masks.ndim == 4 else masks

    for i in range(masks.shape[0]):
        m = masks[i]
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

        ys, xs = np.where(m > 0)
        if len(xs) > 0:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to .pth (saved by train)")
    parser.add_argument("--image", type=str, required=True, help="image to run inference on")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument("--out_path", type=str, default="./outputs/viz_result.jpg")
    args = parser.parse_args()

    img, res = run_inference(args.ckpt, args.image, score_thresh=args.score_thresh)
    visualize_and_save(img, res, args.out_path, mask_thresh=args.mask_thresh)
    print(f"Saved: {args.out_path}")
