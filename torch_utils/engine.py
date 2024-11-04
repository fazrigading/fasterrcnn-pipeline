import math
import sys
import time
import datetime
from tqdm import tqdm

import torch
import torchvision.models.detection.mask_rcnn
from torch_utils import utils
from torch_utils.coco_eval import CocoEvaluator
from torch_utils.coco_utils import get_coco_api_from_dataset
from utils.general import save_validation_results
import numpy as np

def train_one_epoch(
    model, 
    optimizer, 
    data_loader, 
    device, 
    epoch, 
    train_loss_hist,
    print_freq, 
    scaler=None,
    scheduler=None
):
    model.train()
    header = f"Epoch: [{epoch}]"

    # Lists to store batch losses
    batch_loss_list = []
    batch_loss_cls_list = []
    batch_loss_box_reg_list = []
    batch_loss_objectness_list = []
    batch_loss_rpn_list = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    start_time = time.time()
    end = time.time()
    iter_time = utils.SmoothedValue(fmt="{avg:.4f}")
    data_time = utils.SmoothedValue(fmt="{avg:.4f}")
    MB = 1024.0 * 1024.0

    # Use tqdm to create a progress bar with additional logging
    progress_bar = tqdm(data_loader, desc=header, leave=False)
    for i, (images, targets) in enumerate(progress_bar):
        data_time.update(time.time() - end)
        
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Reduce losses for logging
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update lists with loss values
        batch_loss_list.append(loss_value)
        batch_loss_cls_list.append(loss_dict_reduced['loss_classifier'].detach().cpu())
        batch_loss_box_reg_list.append(loss_dict_reduced['loss_box_reg'].detach().cpu())
        batch_loss_objectness_list.append(loss_dict_reduced['loss_objectness'].detach().cpu())
        batch_loss_rpn_list.append(loss_dict_reduced['loss_rpn_box_reg'].detach().cpu())
        train_loss_hist.send(loss_value)

        if scheduler is not None:
            scheduler.step(epoch + (i / len(data_loader)))

        # Update iteration time
        iter_time.update(time.time() - end)
        end = time.time()

        # Calculate ETA
        eta_seconds = iter_time.global_avg * (len(data_loader) - i - 1)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # Update progress bar with custom metrics
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / MB
            progress_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                lr=optimizer.param_groups[0]["lr"],
                time=iter_time.avg,
                data=data_time.avg,
                eta=eta_string,
                max_mem=f"{max_mem:.0f} MB"
            )
        else:
            progress_bar.set_postfix(
                loss=f"{loss_value:.4f}",
                lr=optimizer.param_groups[0]["lr"],
                time=iter_time.avg,
                data=data_time.avg,
                eta=eta_string
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"{header} Total time: {total_time_str} ({total_time / len(data_loader):.4f} s/it)")

    return (
        batch_loss_list, 
        batch_loss_cls_list, 
        batch_loss_box_reg_list, 
        batch_loss_objectness_list, 
        batch_loss_rpn_list
    )
