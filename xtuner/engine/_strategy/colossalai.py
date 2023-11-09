# Copyright (c) OpenMMLab. All rights reserved.
import imp
from mmengine._strategy import ColossalAIStrategy as MMEngineColossalAIStrategy
from mmengine._strategy.colossalai import ColossalAIOptimWrapper
from mmengine.model import BaseDataPreprocessor
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
import copy
from mmengine.registry import DATASETS
import torch.nn as nn
from mmengine.optim import OptimWrapper
from mmengine.device import get_device
from mmengine.registry import FUNCTIONS
from functools import partial
from xtuner.model import SupervisedFinetune
from colossalai.shardformer.policies.auto_policy import get_autopolicy
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim import BaseOptimWrapper, OptimWrapper, _ParamScheduler
from mmengine.registry.root import MODEL_WRAPPERS, OPTIM_WRAPPERS, OPTIMIZERS
from contextlib import nullcontext


class ColossalAIStrategy(MMEngineColossalAIStrategy):
    
    def prepare(
        self,
        model: Union[nn.Module, dict],
        *,
        optim_wrapper: Union[BaseOptimWrapper, dict, None] = None,
        param_scheduler: Union[_ParamScheduler, Dict, List, None] = None,
        compile: Union[dict, bool] = False,
        dispatch_kwargs: Optional[dict] = None,
    ):
        """Prepare model and some components.

        Args:
            model (:obj:`torch.nn.Module` or dict): The model to be run. It
                can be a dict used for build a model.

        Keyword Args:
            optim_wrapper (BaseOptimWrapper or dict, optional): Computing the
                gradient of model parameters and updating them.
                Defaults to None.
                See :meth:`build_optim_wrapper` for examples.
            param_scheduler (_ParamScheduler or dict or list, optional):
                Parameter scheduler for updating optimizer parameters. If
                specified, :attr:`optim_wrapper` should also be specified.
                Defaults to None.
                See :meth:`build_param_scheduler` for examples.
            compile (dict, optional): Config to compile model.
                Defaults to False. Requires PyTorch>=2.0.
            dispatch_kwargs (dict, optional): Kwargs to be passed to other
                methods of Strategy. Defaults to None.
                If ``accumulative_counts`` is set in ``optim_wrapper``, you
                need to provide ``max_iters`` in ``dispatch_kwargs``.
        """
        from colossalai.lazy import LazyInitContext
        from colossalai.utils import get_current_device
        from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, TorchFSDPPlugin, LowLevelZeroPlugin
        if self._prepared:
            return self._prepared_components()
        if dispatch_kwargs is not None:
            self.dispatch_kwargs.update(dispatch_kwargs)

        init_ctx = (
            LazyInitContext(default_device=get_current_device())
            if isinstance(self.booster.plugin, (GeminiPlugin, HybridParallelPlugin))
            else nullcontext()
        )
        print(1)
        pretrained_model_name_or_path = model.llm.pretrained_model_name_or_path
        # with init_ctx:
        model = self.build_model(model)
        print(2)

        # optim_wrapper is required by booster
        if optim_wrapper is not None and isinstance(optim_wrapper, dict):
            optim_wrapper.setdefault('type', 'ColossalAIOptimWrapper')
            # optim_wrapper.setdefault('booster', self.booster)
            optim_wrapper_type = OPTIM_WRAPPERS.get(optim_wrapper['type'])
            if optim_wrapper_type is None:
                raise ValueError(f'Failed to find {optim_wrapper["type"]} in '
                                 '`OPTIM_WRAPPERS`.')
            if 'clip_grad' in optim_wrapper:
                raise ValueError('`Please configure `clip_grad` in `plugin`')
            if not issubclass(optim_wrapper_type, ColossalAIOptimWrapper):
                raise ValueError(
                    'The type of `optim_wrapper` must be '
                    '`ColossalAIOptimWrapper` (or subclass), but got '
                    f'{optim_wrapper_type}')
            optim_wrapper = self.build_optim_wrapper(optim_wrapper, model)
            optim_wrapper.booster = self.booster

        print(3)
        if optim_wrapper is not None:
            self.model, self.optim_wrapper = self._wrap(
                model, optim_wrapper)  # type: ignore
        else:
            self.model = self._wrap(model)  # type: ignore
        # TODO: Check whether `compile` is compatible with colossalai.

        print(4)
        if param_scheduler is not None:
            self.param_schedulers = self.build_param_scheduler(
                param_scheduler, optim_wrapper)  # type: ignore

        if optim_wrapper is not None:
            self._scale_lr()
            accumulative_counts = getattr(self.optim_wrapper,
                                          '_accumulative_counts', 1)
            if accumulative_counts > 1:
                if 'max_iters' not in self.dispatch_kwargs:
                    raise ValueError(
                        '"max_iters" must be specified because '
                        '"accumulative_counts" was set as '
                        f'{accumulative_counts} which is greater than 1.')

                self.optim_wrapper.initialize_count_status(  # type: ignore
                    self.model, 0, self.dispatch_kwargs['max_iters'])
        
        print(5)
        self.booster.load_model(self.model.model_wrapper, pretrained_model_name_or_path)
        print(6)

        self._prepared = True
        return self._prepared_components()