"""Derived from LAVIS. Copyright (c) 2022, salesforce.com, inc. SPDX-License-Identifier: BSD-3-Clause."""
import re
from omegaconf import OmegaConf
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
@registry.register_processor('blip_question')
class BlipQuestionProcessor(BaseProcessor):

    def __init__(self, max_words=50):
        self.max_words = max_words

    def __call__(self, question):
        return self.pre_question(question)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        max_words = cfg.get('max_words', 50)
        return cls(max_words=max_words)

    def pre_question(self, question):
        question = re.sub('([.!\\"()*#:;~])', '', question.lower())
        question = question.rstrip(' ')
        question_words = question.split(' ')
        if len(question_words) > self.max_words:
            question = ' '.join(question_words[:self.max_words])
        return question
