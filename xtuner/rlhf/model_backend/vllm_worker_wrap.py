# Adapted from https://github.com/OpenLLMAI/OpenRLHF/blob/v0.2.5/openrlhf/trainer/ray/vllm_worker_wrap.py  # noqa: E501
import importlib

import torch
from vllm.model_executor.weight_utils import hf_model_weights_iterator
from vllm.worker.worker import Worker

from ..logger import init_logger
from .dist_utils import init_process_group

logger = init_logger(__name__)


def _hf_model_weights_iterator_wrap(model_name_or_path, *args, **kwargs):
    if isinstance(model_name_or_path, dict):
        yield from model_name_or_path.items()
    else:
        yield from hf_model_weights_iterator(model_name_or_path, *args,
                                             **kwargs)


class VllmWorkerWrap(Worker):

    def __init__(self, *args, **kwargs):
        # Monkey patch hf_model_weights_iterator to allow update single weight
        # NOTE: In 0.2.5, vLLM introduce lazy model loader
        # https://github.com/vllm-project/vllm/pull/2044
        from vllm.model_executor.models import _MODELS, ModelRegistry

        load_model_cls = ModelRegistry.load_model_cls

        def patched_load_model_cls(model_arch: str):
            module_name, _ = _MODELS[model_arch]
            module = importlib.import_module(
                f'vllm.model_executor.models.{module_name}')
            module.hf_model_weights_iterator = _hf_model_weights_iterator_wrap
            logger.info(
                f'Monkey patch hf_model_weights_iterator for module {module_name}'  # noqa: E501
            )

            return load_model_cls(model_arch)

        ModelRegistry.load_model_cls = patched_load_model_cls

        super().__init__(*args, **kwargs)

    def init_process_group(self, master_address, master_port, rank_offset,
                           world_size, group_name):
        """Init torch process group for model weights update."""
        assert torch.distributed.is_initialized(
        ), 'default torch process group must be initialized'
        assert group_name != '', 'group name must not be empty'

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_group = init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_address}:{master_port}',
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        logger.info(
            f'init_process_group: master_address={master_address}, master_port={master_port}, '  # noqa: E501
            f'rank={rank}, world_size={world_size}, group_name={group_name}')

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor
        model)"""
        if torch.distributed.get_rank() == 0:
            logger.debug(
                f'update weight: {name}, dtype: {dtype}, shape: {shape}')

        weight = torch.empty(shape, dtype=dtype, device='cuda')
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)
        self.model_runner.model.load_weights(model_name_or_path={name: weight})

        del weight
