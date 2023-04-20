# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
from typing import Iterable

import mae_st.util.lr_sched as lr_sched
import mae_st.util.misc as misc
import torch
from iopath.common.file_io import g_pathmgr as pathmgr


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
    fp32=False,
    config=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "cpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cpu_mem_all", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "gpu_mem", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "mask_ratio", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = config['accum_iter']

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    metric_dict = train_wrapper(config, metric_logger, data_loader, header, print_freq, accum_iter, epoch,
                  optimizer, lr_sched, device, args, fp32, model, loss_scaler, log_writer)
    return metric_dict


def train_wrapper(config, metric_logger, data_loader, header, print_freq, accum_iter, epoch,
                  optimizer, lr_sched, device, args, fp32, model, loss_scaler, log_writer):
    if config['distributed_dataset_backend'] == 'pytorch':
        for data_iter_step, (samples, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(
                    optimizer, data_iter_step / len(data_loader) + epoch, args, config,
                )
            samples = samples.to(device, non_blocking=True)
            metric_dict = __train_one_epoch(samples, fp32, model, config, accum_iter, args, loss_scaler, optimizer,
                      data_iter_step, metric_logger, data_loader, log_writer, epoch)
    elif config['distributed_dataset_backend'] == 'monai':
        for data_iter_step, samples in enumerate(data_loader, 0):
            if data_iter_step % accum_iter == 0:
                lr_sched.adjust_learning_rate(
                    optimizer, data_iter_step / len(data_loader) + epoch, args, config,
                )
            samples = samples['inputs']
            samples = samples.permute(0, 1, 4, 2, 3)
            metric_dict = __train_one_epoch(samples, fp32, model, config, accum_iter, args, loss_scaler, optimizer,
                      data_iter_step, metric_logger, data_loader, log_writer, epoch)
    else:
        raise ValueError('distributed_dataset_backend must be either pytorch or monai')
    return metric_dict

def __train_one_epoch(samples, fp32, model, config, accum_iter, args, loss_scaler, optimizer,
                      data_iter_step, metric_logger, data_loader, log_writer, epoch):
    
    if len(samples.shape) == 6:
        b, r, c, t, h, w = samples.shape
        samples = samples.reshape(b * r, c, t, h, w)

    with torch.cuda.amp.autocast(enabled=not fp32):
        loss, _, _ = model(
            samples,
            mask_ratio=config['mask_ratio'],
        )

    loss_value = loss.item()

    if not math.isfinite(loss_value):
        for _ in range(args.num_checkpoint_del):
            try:
                path = misc.get_last_checkpoint(args)
                pathmgr.rm(path)
                print(f"remove checkpoint {path}")
            except Exception as _:
                pass
        raise Exception("Loss is {}, stopping training".format(loss_value))

    loss /= accum_iter
    loss_scaler(
        loss,
        optimizer,
        parameters=model.parameters(),
        update_grad=(data_iter_step + 1) % accum_iter == 0,
        clip_grad=config['clip_grad'],
    )

    if (data_iter_step + 1) % accum_iter == 0:
        optimizer.zero_grad()

    torch.cuda.synchronize()

    metric_logger.update(loss=loss_value)
    metric_logger.update(cpu_mem=misc.cpu_mem_usage()[0])
    metric_logger.update(cpu_mem_all=misc.cpu_mem_usage()[1])
    metric_logger.update(gpu_mem=misc.gpu_mem_usage())
    metric_logger.update(mask_ratio=config['mask_ratio'])

    lr = optimizer.param_groups[0]["lr"]
    metric_logger.update(lr=lr)

    loss_value_reduce = misc.all_reduce_mean(loss_value)
    if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        """We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        epoch_1000x = int(
            (data_iter_step / len(data_loader) + epoch) * 1000 * config['repeat_aug']
        )
        log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
        log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
