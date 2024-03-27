# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmengine._strategy import DeepSpeedStrategy as MMEngineDeepSpeedStrategy

from xtuner import DS_CEPH_DIR
from xtuner.parallel.sequence import init_sequence_parallel
from xtuner.utils.fileio import patch_fileio


class DeepSpeedStrategy(MMEngineDeepSpeedStrategy):

    def __init__(self, *args, **kwargs):
        sequence_parallel_size = kwargs.pop('sequence_parallel_size', 1)
        self.sequence_parallel_size = sequence_parallel_size

        super().__init__(*args, **kwargs)

        from transformers.integrations.deepspeed import HfDeepSpeedConfig

        # hf_deepspeed_config has to be saved as an attribute.
        self.hf_deepspeed_config = HfDeepSpeedConfig(self.config)

    def _wrap_model(self, model):
        wrapper = super()._wrap_model(model)
        # hard code for deepspeed zero3
        # When utilizing Zero3, the model isn't allocated to CUDA within the
        # `deepspeed.initialize` process.
        assert hasattr(wrapper.model, 'data_preprocessor')
        wrapper.model.data_preprocessor.cuda()
        return wrapper

    def save_checkpoint(self, *args, **kwargs) -> None:
        if DS_CEPH_DIR:
            from os import path as osp
            work_dir_prefix = osp.split(self.work_dir)[0]

            filename = kwargs['filename'].replace(work_dir_prefix, DS_CEPH_DIR)
            kwargs['filename'] = filename
            with patch_fileio():
                super().save_checkpoint(*args, **kwargs)
        else:
            super().save_checkpoint(*args, **kwargs)

    def load_checkpoint(self, *args, **kwargs) -> None:
        if DS_CEPH_DIR:

            with patch_fileio():
                checkpoint = super().load_checkpoint(*args, **kwargs)
        else:
            checkpoint = super().load_checkpoint(*args, **kwargs)
        return checkpoint

    def resume(self, *args, **kwargs) -> None:
        if DS_CEPH_DIR:

            with patch_fileio():
                checkpoint = super().resume(*args, **kwargs)
        else:
            checkpoint = super().resume(*args, **kwargs)
        return checkpoint

    def _setup_distributed(  # type: ignore
        self,
        launcher: Optional[str] = None,
        backend: str = 'nccl',
        **kwargs,
    ):
        super()._setup_distributed(launcher, backend, **kwargs)
        init_sequence_parallel(self.sequence_parallel_size)
