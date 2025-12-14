import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn

import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    print_freq,
    scaler=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_factor,
            total_iters=warmup_iters,
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = [img.to(device) for img in images]
        targets = [
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
            for t in targets
        ]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module

    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")

    return iou_types


@torch.inference_mode()
def evaluate_coco(model, data_loader, device):
    cpu_device = torch.device("cpu")

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = [img.to(device) for img in images]

        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in out.items()} for out in outputs]
        model_time = time.time() - model_time

        res = {t["image_id"]: o for t, o in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    print("Averaged stats:", metric_logger)

    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator


@torch.inference_mode()
def evaluate(model, data_loader, device):
    # NOTE: original code uses model.train() to compute loss in eval mode
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = [img.to(device) for img in images]
        targets = [
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
            for t in targets
        ]

        model_time = time.time()
        loss_dict = model(images, targets)
        model_time = time.time() - model_time

        loss_dict_reduced = loss_dict
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(model_time=model_time)

    return metric_logger
