# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import time

import mae_st.util.env

import mae_st.util.misc as misc

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
from iopath.common.file_io import g_pathmgr as pathmgr
from mae_st import models_mae
from mae_st.engine_pretrain import train_one_epoch
from mae_st.util.kinetics import Kinetics
from mae_st.util.cag import CAGDataset
from mae_st.util.cag_dataset import CAGDATASET
from mae_st.util.misc import NativeScalerWithGradNormCount as NativeScaler
from tensorboard.compat.tensorflow_stub.io.gfile import register_filesystem
from torch.utils.tensorboard import SummaryWriter
import yaml
import sys
from monai.data import (
    list_data_collate, pad_list_data_collate,
    ThreadDataLoader)
from monai.data import DistributedWeightedRandomSampler, DistributedSampler, Dataset

sys.path.append("/home/alatar/miacag")
from miacag.dataloader.Classification._3D.dataloader_monai_classification_3D import \
                    train_monai_classification_loader, val_monai_classification_loader

def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)

    # Model parameters

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--config_path",
        default="",
        help="path to yaml config path",
    )
        
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--local_rank", type=int) #chr edit
    parser.add_argument("--local-rank", type=int) #chr edit
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--no_env", action="store_true")

    # Video related configs
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--decoder_num_heads", default=16, type=int)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--num_checkpoint_del", default=20, type=int)
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument(
        "--trunc_init",
        action="store_true",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
    )
    parser.set_defaults(fp32=True)
    parser.add_argument(
        "--jitter_scales_relative",
        default=[0.5, 1.0],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--jitter_aspect_relative",
        default=[0.75, 1.3333],
        type=float,
        nargs="+",
    )
    parser.add_argument(
        "--beta",
        default=None,
        type=float,
        nargs="+",
    )
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)
    return parser

def read_yaml(config_path):
    # read yaml file:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
        
def main(args):
    print('starting')
    misc.init_distributed_mode(args)
    config = read_yaml(args.config_path)
    config['loaders']['mode'] = 'training'
    config['cpu'] = "False"
    config['use_DDP'] = 'True'
    config['num_workers'] = args.num_workers

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    
    if config['distributed_dataset_backend'] == 'monai':
        cag_dataset = CAGDataset(config=config)
        dataset_train = train_monai_classification_loader(
            cag_dataset.df_train,
            config)
        dataset_train = dataset_train()
    elif config['distributed_dataset_backend'] == 'pytorch':
        dataset_train = CAGDATASET(
            config=config,
            mode="pretrain",
            sampling_rate=config['sampling_rate'],
            num_frames=config['loaders']['Crop_depth'],
            train_jitter_scales=(256, 320),
            repeat_aug=config['repeat_aug'],
            jitter_aspect_relative=args.jitter_aspect_relative,
            jitter_scales_relative=args.jitter_scales_relative,
        )
    else:
        raise ValueError('Unknown distributed dataset backend: {}'.format(config['distributed_dataset_backend']))
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if config["distributed_dataset_backend"] == "monai":
            sampler_train = DistributedSampler(
                    dataset=dataset_train,
                    even_divisible=True,
                    shuffle=True)
        elif config["distributed_dataset_backend"] == "pytorch":
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            raise ValueError("Unknown distributed dataset backend: {}".format(config["distributed_dataset_backend"]))
        print("Sampler_train = %s" % str(sampler_train))
    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and config['log_dir'] is not None:
        try:
            pathmgr.mkdirs(config['log_dir'])
        except Exception as _:
            pass
        log_writer = SummaryWriter(log_dir=config['log_dir'])
    else:
        log_writer = None
    
    # chr edit
    if config["distributed_dataset_backend"] == "monai":
        data_loader_train = ThreadDataLoader(
                dataset_train,
                sampler=sampler_train,
                batch_size=config['loaders']['batchSize'],
                shuffle=False,
                num_workers=0, #config['num_workers'],
                collate_fn=pad_list_data_collate,
                pin_memory=False,) #True if config['cpu'] == "False" else False,)
    elif config["distributed_dataset_backend"] == "pytorch":
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=config['loaders']['batchSize'],
            num_workers=args.num_workers,
            pin_memory=config['pin_mem'],
            drop_last=True,
        )
    else:
        raise ValueError("Unknown distributed dataset backend: %s" % config["distributed_dataset_backend"])
    
    # define the model
    model = models_mae.__dict__[config['model_pretrain']](
        **vars(args),
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = config['loaders']['batchSize']* config['accum_iter']* misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = config['blr'] * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % config['accum_iter'])
    print("effective batch size: %d" % eff_batch_size)

    if config['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            # find_unused_parameters=True,
        )
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(
        model_without_ddp,
        args.weight_decay,
        bias_wd=args.bias_wd,
    )
    if args.beta is None:
        beta = (0.9, 0.95)
    else:
        beta = args.beta
    optimizer = torch.optim._multi_tensor.AdamW(
        param_groups,
        lr=args.lr,
        betas=beta,
    )
    loss_scaler = NativeScaler(fp32=args.fp32)

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    checkpoint_path = ""
    print(f"Start training for {config['epochs']} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, config['epochs']):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            fp32=args.fp32,
            config=config,
        )
        if args.output_dir and (
            epoch % args.checkpoint_period == 0 or epoch + 1 == config['epochs']
        ):
            if config['distributed_dataset_backend'] == "monai":
                if epoch % config['frequence_save_checkpoint'] == 0:
                    checkpoint_path = misc.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                    )
            elif config['distributed_dataset_backend'] == "pytorch":
                checkpoint_path = misc.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                    )
            else:
                raise ValueError("Unknown distributed dataset backend: %s" % config["distributed_dataset_backend"])

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with pathmgr.open(
                f"{args.output_dir}/log.txt",
                "a",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    print(torch.cuda.memory_allocated())
    return [checkpoint_path]


def launch_one_thread(
    local_rank,
    shard_rank,
    num_gpus_per_node,
    num_shards,
    init_method,
    output_path,
    opts,
    stats_queue,
):
    print(opts)
    args = get_args_parser()
    args = args.parse_args(opts)
    args.rank = shard_rank * num_gpus_per_node + local_rank
    args.world_size = num_shards * num_gpus_per_node
    args.gpu = local_rank
    args.dist_url = init_method
    args.output_dir = output_path
    output = main(args)
    stats_queue.put(output)
