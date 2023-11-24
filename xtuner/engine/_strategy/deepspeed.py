# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch.nn as nn
from mmengine._strategy import DeepSpeedStrategy as MMEngineDeepSpeedStrategy
from mmengine.optim import BaseOptimWrapper, _ParamScheduler


class DeepSpeedStrategy(MMEngineDeepSpeedStrategy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from transformers.integrations.deepspeed import HfDeepSpeedConfig

        # hf_deepspeed_config has to be saved as an attribute.
        self.hf_deepspeed_config = HfDeepSpeedConfig(self.config)

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
        """
        if self._prepared:
            return self._prepared_components()
        assert dispatch_kwargs is not None
        self.dispatch_kwargs.update(dispatch_kwargs)

        model = self.build_model(model)
        model = self._init_model_weights(model)

        if optim_wrapper is not None:
            self.optim_wrapper = self.build_optim_wrapper(optim_wrapper, model)
            # Remove param_groups whose all parameters do not requires
            # gradient, to ensure compatibility with DeepSpeed
            self.optim_wrapper.optimizer.param_groups = [
                group for group in self.optim_wrapper.optimizer.param_groups
                if any(param.requires_grad for param in group['params'])
            ]

            self.model = self._wrap_model(model)

            self.optim_wrapper.model = self.model  # type: ignore

        else:
            self.model = self._wrap_model(model)

        if param_scheduler is not None:
            self.param_schedulers = self.build_param_scheduler(
                param_scheduler, self.optim_wrapper)
        self._prepared = True
        return self._prepared_components()

    def _wrap_model(self, model):
        wrapper = super()._wrap_model(model)
        # hard code for deepspeed zero3
        # When utilizing Zero3, the model isn't allocated to CUDA within the
        # `deepspeed.initialize` process.
        assert hasattr(wrapper.model, 'data_preprocessor')
        wrapper.model.data_preprocessor.cuda()
        return wrapper
