"""Derived from LAVIS. Copyright (c) 2022, salesforce.com, inc. SPDX-License-Identifier: BSD-3-Clause."""
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.scanrefer_dataset import SEGA3DScanReferDataset

class SEGA3DScanReferBuilder(BaseDatasetBuilder):
    train_dataset_cls = SEGA3DScanReferDataset
    eval_dataset_cls = SEGA3DScanReferDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/sega3d_scanrefer/defaults.yaml"}

from lavis.datasets.datasets.reasoning_scannet_dataset import SEGA3DReasoningDataset as SEGA3DScanNetDataset

class SEGA3DScanNetBuilder(BaseDatasetBuilder):
    train_dataset_cls = SEGA3DScanNetDataset
    eval_dataset_cls = SEGA3DScanNetDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/sega3d_reasoning_scannet/defaults.yaml"}

from lavis.datasets.datasets.reasoning_matterport_dataset import SEGA3DReasoningDataset as SEGA3DMatterportDataset

class SEGA3DMatterportBuilder(BaseDatasetBuilder):
    train_dataset_cls = SEGA3DMatterportDataset
    eval_dataset_cls = SEGA3DMatterportDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/sega3d_reasoning_matterport/defaults.yaml"}
