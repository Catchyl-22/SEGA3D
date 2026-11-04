"""Derived from LAVIS. Copyright (c) 2022, salesforce.com, inc. SPDX-License-Identifier: BSD-3-Clause."""
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.tasks.sega3d_seg_task import SEGA3DSegTask

from lavis.sega3d_registry import register_components

register_components("tasks")

def setup_task(cfg):
    task_name = cfg.run_cfg.task
    task_cls = registry.get_task_class(task_name)
    if task_cls is None:
        raise ValueError(f"Unknown task: {task_name}")
    return task_cls.setup_task(cfg=cfg)

__all__ = ["BaseTask", "SEGA3DSegTask", "setup_task"]
