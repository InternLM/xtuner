# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path
from typing import Optional, Union

import torch.distributed as dist
from mmengine import print_log
from mmengine._strategy import DeepSpeedStrategy
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import FlexibleRunner

from xtuner.registry import BUILDER
from xtuner.utils import get_origin_state_dict

DATA_BATCH = Optional[Union[dict, tuple, list]]


class HFCheckpointHook(Hook):

    priority = 95  # lower than CheckpointHook in MMEngine

    def __init__(self, out_dir: Optional[Union[str, Path]] = None) -> None:
        self.out_dir = out_dir

    @staticmethod
    def _use_shard_moe(llm):
        config = llm.config
        moe_implementation = getattr(config, 'moe_implementation', 'origin')
        return moe_implementation == 'shard'

    def after_run(self, runner) -> None:
        assert isinstance(runner,
                          FlexibleRunner), 'Runner should be `FlexibleRunner`'
        assert isinstance(
            runner.strategy,
            DeepSpeedStrategy), 'Strategy should be `DeepSpeedStrategy`'

        if self.out_dir is None:
            self.out_dir = osp.join(runner.work_dir, 'hf_model')

        wrapped_model = runner.strategy.model
        if wrapped_model.zero_optimization_partition_weights():
            assert wrapped_model.zero_gather_16bit_weights_on_model_save(), \
                ('Please set `gather_16bit_weights_on_model_save=True` '
                 'in your DeepSpeed config.')
            state_dict = wrapped_model._zero3_consolidated_16bit_state_dict()
        else:
            state_dict = wrapped_model.module_state_dict(
                exclude_frozen_parameters=runner.strategy.
                exclude_frozen_parameters)

        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        llm = model.llm
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            # keys in state_dict are prefixed with 'llm.'
            keys = list(state_dict.keys())
            for k in keys:
                val = state_dict.pop(k)
                state_dict[k[4:]] = val

            if self._use_shard_moe(llm):
                print_log('recover the origin state_dict from merged one ...')
                state_dict = get_origin_state_dict(state_dict, llm)

            print_log(f'Saving LLM to {self.out_dir}')
            llm.save_pretrained(self.out_dir, state_dict=state_dict)

            print_log(f'Saving LLM tokenizer to {self.out_dir}')
            tokenizer = BUILDER.build(runner.cfg.tokenizer)
            tokenizer.save_pretrained(self.out_dir)
