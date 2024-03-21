# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def BvSB(output):
    if isinstance(output, torch.Tensor):
        # print(f"output.shape: {output.shape}")  # output.shape: torch.Size([batch_size, num_class])
        output = output.tolist()
        uncertainty = []
        for e in output:
            e.sort(reverse=True)
            uncertainty.append(e[0] - e[1])
        # assert len(uncertainty) == len(output)
        return uncertainty
    else:
        print("[Warning] `output` is not a Tensor.")


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def eval_trnData(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    cnt = 1
    uncertainty_list = []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            # print(f"output: {output}")  # output: tensor([[ 0.7285,  1.0293,  1.5693, -0.3376,  0.2382,  0.2976,  2.3223, -0.6074, 1.8105, -0.2380]], device='cuda:0', dtype=torch.float16)
            # print(f"target: {target}")  # target: tensor([6], device='cuda:0')
            # print(f"loss: {loss}")      # loss: 1.1005859375

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        uncertainty = BvSB(output)
        # print(f"length of uncertainty: {len(uncertainty)}")
        uncertainty_list.extend(uncertainty)
        # print(f"[Eval Trn Data] [Batch.{cnt}({output.shape[0]})] [Acc1: {acc1.item()}] [Acc5: {acc5.item()}]")  # [Eval Trn Data] [Batch.2(768)] [Acc1: 81.38021087646484] [Acc5: 96.09375]
        cnt = cnt + 1
        if cnt == 0:  # 5 for observe
            print(f"length of uncertainty_list: {len(uncertainty_list)}")
            assert len(uncertainty_list) == (cnt-1) * output.shape[0]
            sys.exit()
    print(f"len of uncertainty_list: {len(uncertainty_list)}")          # len of uncertainty_list: 24960
    print(f"len of data_loader.dataset: {len(data_loader.dataset)}")    # len of data_loader.dataset: 50000
    assert len(uncertainty_list) == len(data_loader.dataset)            # eval_oneByOne.sh, --nproc_per_node=1 

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, uncertainty_list

