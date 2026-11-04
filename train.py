"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import warnings
warnings.filterwarnings("ignore", message="The pynvml package is deprecated")
warnings.filterwarnings("ignore", message=".*DtypeTensor.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *
import torch.distributed as dist

def bind_device_to_local_rank():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if dist.is_initialized():
            print(f"[DDP] rank={dist.get_rank()} bound to cuda:{local_rank}", flush=True)
    else:
        # 单卡/非 torchrun 情况
        print("[DDP] LOCAL_RANK not set; running single GPU.")

def ddp_sanity_check():
    rank = dist.get_rank()
    world = dist.get_world_size()
    dev  = torch.cuda.current_device()
    t = torch.tensor([dev], device="cuda")
    buf = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(buf, t)
    devs = [int(x.item()) for x in buf]
    # 用方案A的 sys.__stdout__ 或方案B的文件写
    import sys
    sys.__stdout__.write(f"[DDP-CHECK] rank={rank}/{world} devices={devs}\n")
    sys.__stdout__.flush()

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    # replace some settings in the used config
    parser.add_argument("--replace_cfg", nargs="+", help="replace some settings in the used config", default=None)
    parser.add_argument("--job_id", default=None, help="job id")
    # python train.py --cfg-path configs/cont_train.yaml \
    # --replace-cfg run_cfg.seed=1 run_cfg.local_rank=0 --job-id 1

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "myrunner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    args = parse_args()
    
    torch.multiprocessing.set_start_method("spawn")
    job_id = now() if args.job_id is None else args.job_id

    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)
    bind_device_to_local_rank()
    ddp_sanity_check()
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    # with torch.cuda.amp.autocast(dtype=torch.float32):
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    runner.train()


if __name__ == "__main__":
    main()
