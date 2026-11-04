"""Derived from LAVIS. Copyright (c) 2022, salesforce.com, inc. SPDX-License-Identifier: BSD-3-Clause."""
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.blip_processors import BlipQuestionProcessor

def load_processor(name, cfg=None):
    return registry.get_processor_class(name).from_config(cfg)

__all__ = ["BaseProcessor", "BlipQuestionProcessor", "load_processor"]
