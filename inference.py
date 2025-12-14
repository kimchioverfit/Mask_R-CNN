# Hide details of inference process.

import cv2
import numpy as np
import torch
from PIL import Image

from mcrnn_dataset import balls_get_transform
# from get_roi import getROI # Disabled temporarily
# from logger import logger # Disabled temporarily


eval_transform = balls_get_transform()


def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image)
    height, width, _ = image_np.shape

    tensor_image = (
        torch.tensor(image_np)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )
    return tensor_image, height, width


def transform_image(tensor_image, device):
    x = eval_transform(tensor_image)[:3, ...].to(device)
    return x


def get_filtered_masks(model, x, mask_thresh, box_thresh):
    pred = model([x])[0]
    masks = (pred["masks"] > mask_thresh).squeeze(1)
    filtered_masks = masks[pred["scores"] > box_thresh]
    return filtered_masks


def find_section(gap, center, roi):
    # NOTE: hardcoded grid size
    m = 6  # need to remove someday
    n = 9

    i = int((center[0] - roi["leftmost"][0]) // gap[0])
    j = int((center[1] - roi["topmost"][1]) // gap[1])

    return (i + 1, n - j)


def draw_circles_and_labels(
    image_rgb,
    filtered_masks,
    roi,
    pixel_per_μm,
    offset,
    img_path,
    width,
    height,
    gap,
    diameter_range,
):
    temp_list = []
    lower, upper = diameter_range

    for mask in filtered_masks:
        mask_np = mask.cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)

            if round(radius, 2) <= 5:
                continue

            # ROI boundary check
            if not (
                (x - radius) >= roi["leftmost"][0]
                and (x + radius) <= roi["rightmost"][0]
                and (y - radius) >= roi["topmost"][1]
                and (y + radius) <= roi["bottommost"][1]
            ):
                continue

            diameter = (radius) / (pixel_per_μm * 2) + offset

            # diameter range filter
            if not (lower <= diameter <= upper):
                continue

            logger.info(f"diameter: {diameter}")

            text = f"{diameter:.2f}"
            ax = int(x) + int(radius)
            ay = int(y) + int(radius)

            text_w, text_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            ay = max(text_h + 10, min(ay, height - 10))
            ax = max(10, min(ax, width - text_w - 10))
            adjusted_position = (ax, ay)

            center = (int(x), int(y))
            section = find_section(gap, center, roi)

            font_color_orange = (0, 165, 255)

            cv2.circle(image_rgb, center, int(radius + offset), (255, 0, 0), 2)

            # outline text (black)
            cv2.putText(
                image_rgb,
                text,
                adjusted_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            # fill text (orange)
            cv2.putText(
                image_rgb,
                text,
                adjusted_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                font_color_orange,
                1,
                cv2.LINE_AA,
            )

            # section text (optional)
            # section_text_position = (ax, ay + 20)
            # cv2.putText(image_rgb, str(section), section_text_position,
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            temp_list.append([img_path.name, section, center, diameter])

    # draw count
    count_text = f"Count: {len(temp_list)}"
    text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = width - text_size[0] - 20
    text_y = height - 20

    cv2.putText(
        image_rgb,
        count_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_rgb,
        count_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return temp_list


def save_results(image_rgb, img_path, diameter_dict):
    dir_name = "ai_results"
    target_dir = img_path.parent / dir_name
    target_dir.mkdir(parents=True, exist_ok=True)

    results_path = img_path.parent / dir_name / img_path.stem
    logger.info(f"ai_results_path: {results_path}.png")
    logger.info(f"Total_count: {len(diameter_dict.get(str(img_path), []))}")

    file_path = f"{results_path}_ai.png"

    ai_img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ai_img_pil = Image.fromarray(ai_img_bgr)
    ai_img_pil.save(file_path)


def inference(cls, img_path, pixel_per_μm):
    logger.info("inference start")
    diameter_dict = {}

    with torch.no_grad():
        tensor_image, height, width = preprocess_image(img_path)

        x = transform_image(tensor_image, cls.device)
        filtered_masks = get_filtered_masks(cls.model, x, cls.mask_thresh, cls.box_thresh)

        # normalize to byte for visualization
        tensor_vis = (255.0 * (tensor_image - tensor_image.min()) / (tensor_image.max() - tensor_image.min())).byte()
        tensor_vis = tensor_vis[:3, ...]

        image_rgb = cv2.cvtColor(
            tensor_vis.permute(1, 2, 0).cpu().numpy(),
            cv2.COLOR_BGR2RGB,
        )

        image_rgb, roi, gap = getROI(image_rgb)

        diameter_range = [
            float(cls.lower_bound_input.text()),
            float(cls.upper_bound_input.text()),
        ]

        temp_list = draw_circles_and_labels(
            image_rgb=image_rgb,
            filtered_masks=filtered_masks,
            roi=roi,
            pixel_per_μm=pixel_per_μm,
            offset=0,
            img_path=img_path,
            width=width,
            height=height,
            gap=gap,
            diameter_range=diameter_range,
        )

        diameter_dict[str(img_path)] = temp_list
        save_results(image_rgb, img_path, diameter_dict)

    logger.info("inference end")
    return diameter_dict, roi