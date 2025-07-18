# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from xtuner.v1.float8.float8_tensor import ScalingGranularity
from xtuner.v1.float8.fsdp_utils import (
    precompute_tensorwise_float8_scale_for_fsdp,
    precompute_tilewise_float8_scale_for_fsdp,
)
from xtuner.v1.utils import get_logger, is_evenly_distributed


logger = get_logger()


def _is_sm89_or_later():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


def default_linear_filter_fn(mod: nn.Module, fqn: str):
    return fqn != "lm_head" and fqn[-4:] != "gate"


def default_grouped_linear_filter_fn(mod: nn.Module, fqn: str):
    return True


# handler 要跟 Engine 一一对应？
class Float8Handler:
    scaling_granularity_gemm: ScalingGranularity
    scaling_granularity_grouped_gemm: ScalingGranularity
    fsdp_mesh: Optional[DeviceMesh] = None
    tilewise_reduce_mesh_devided_64: Optional[DeviceMesh] = None
    tilewise_reduce_mesh_mapping: Dict[Tuple[int, int], DeviceMesh] = {}

    def __init__(
        self,
        scaling_granularity_gemm: Optional[ScalingGranularity] = None,
        scaling_granularity_grouped_gemm: Optional[ScalingGranularity] = None,
    ) -> None:
        self.enabled = False

        if not _is_sm89_or_later():
            logger.warning(
                "Failed to enable float8 training because float8 is only supported on SM89 or later",
            )
            return

        assert scaling_granularity_gemm in (ScalingGranularity.TILEWISE, ScalingGranularity.TENSORWISE), (
            "scaling_granularity_gemm must be TILEWISE or TENSORWISE."
        )
        assert scaling_granularity_grouped_gemm in (ScalingGranularity.TILEWISE, ScalingGranularity.TENSORWISE), (
            "scaling_granularity_grouped_gemm must be TILEWISE or TENSORWISE."
        )

        self.scaling_granularity_gemm = scaling_granularity_gemm
        self.scaling_granularity_grouped_gemm = scaling_granularity_grouped_gemm
        self.is_tilewise_fp8 = (
            scaling_granularity_gemm == ScalingGranularity.TILEWISE
            or scaling_granularity_grouped_gemm == ScalingGranularity.TILEWISE
        )
        self.is_tensorwise_fp8 = scaling_granularity_gemm == ScalingGranularity.TENSORWISE
        self.enabled = True

    @staticmethod
    def get_num_features_after_pad(tensor_size, fsdp_shard_dim, num_chunks, fp8_block_size=128):
        total_size = tensor_size[fsdp_shard_dim]
        if total_size < fp8_block_size:
            # 对于小 tensor，需要 pad 到 fp8_block_size （实际场景几乎不会出现）
            total_size = fp8_block_size
        ideal_chunk_size = math.ceil(total_size / num_chunks)
        if ideal_chunk_size > fp8_block_size:
            # 如果大于base_size，则只允许是 n * 128 或 n * 128 + 64，64 是为了reduce的时候相对简单
            if ideal_chunk_size % fp8_block_size == 0:
                chunk_size = ideal_chunk_size
            elif ideal_chunk_size % fp8_block_size > 64:
                chunk_size = math.ceil(ideal_chunk_size / fp8_block_size) * fp8_block_size
            else:
                chunk_size = ideal_chunk_size // fp8_block_size * fp8_block_size + 64
        else:
            # 如果小于base_size，找到大于等于 ideal_chunk_size 的 128 的因数
            factors = [1, 2, 4, 8, 16, 32, 64, 128]
            chunk_size = next(size for size in factors if size >= ideal_chunk_size)
        return chunk_size * num_chunks

    def pad_for_fsdp(self, model: nn.Module, fsdp_mesh: DeviceMesh):
        from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
        from xtuner.v1.float8.float8_linear_tensor_wise import TensorWiseFloat8Linear
        from xtuner.v1.float8.float8_linear_tile_wise import TileWiseFloat8Linear

        if not self.enabled:
            logger.warning("Float8 training is not enabled.")
            return

        for module in model.modules():
            if isinstance(module, (TileWiseFloat8Linear, TileWiseFloat8GroupedLinear, TensorWiseFloat8Linear)):
                # make fsdp compatible with block-wise fp8
                # use size(-1) to support hsdp
                if isinstance(module.weight, DTensor):
                    assert is_evenly_distributed(module.weight), (
                        "Currently only support even distributed TP or EP weight for float8 training."
                    )
                    tensor_size = module.weight._local_tensor.size()
                    parallel_size = module.weight.device_mesh.size()
                else:
                    tensor_size = module.weight.size()
                    parallel_size = 1
                padded_out_features = self.get_num_features_after_pad(tensor_size, 0, fsdp_mesh.size(-1), 128)
                padded_out_features *= parallel_size
                module.pad_for_fsdp(padded_out_features=padded_out_features)

    # def convert_to_float8_training(
    #         self,
    #         model: nn.Module,
    #         fsdp_mesh: Optional[DeviceMesh] = None,
    #         linear_filter_fn: Optional[callable] = default_linear_filter_fn,
    #         grouped_linear_filter_fn: Optional[callable] = default_grouped_linear_filter_fn,
    # ):
    #     """
    #     Convert the model to use float8 training.
    #     Args:
    #         model (nn.Module): The model to convert.
    #         fsdp_mesh (DeviceMesh, optional): The FSDP mesh. If None, will use the default device mesh.
    #         linear_filter_fn (callable, optional): A filter function for linear modules.
    #         grouped_linear_filter_fn (callable, optional): A filter function for grouped linear modules.
    #     """
    #     from xtuner.v1.float8.float8_linear_tile_wise import TileWiseFloat8Linear
    #     from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear

    #     if not self.enabled:
    #         logger.warning("Float8 training is not enabled.")
    #         return

    #     def convert_to_tilewise_float8_linear_training(module: nn.Module, prefix: str):
    #         for name, child in module.named_children():
    #             if isinstance(child, nn.Linear) and linear_filter_fn(child, prefix + name):
    #                 ctx = torch.device('meta') if child.weight.device == torch.device('meta') else nullcontext()
    #                 with ctx:
    #                     # make fsdp compatible with block-wise fp8
    #                     # use size(-1) to support hsdp
    #                     padded_out_features = self.get_num_features_after_pad(
    #                         child.weight.size(), 0, fsdp_mesh.size(-1), 128)
    #                     fp8_linear = TileWiseFloat8Linear.from_float(
    #                         child,
    #                         padded_out_features=padded_out_features
    #                     )
    #                 module.add_module(name, fp8_linear)
    #             else:
    #                 convert_to_tilewise_float8_linear_training(child, prefix + name + ".")

    #     def convert_to_tilewise_float8_grouped_linear_training(module: nn.Module, prefix: str):
    #         for name, child in module.named_children():
    #             if isinstance(child, GroupedLinear) and grouped_linear_filter_fn(child, prefix + name):
    #                 ctx = torch.device('meta') if child.weight.device == torch.device('meta') else nullcontext()
    #                 with ctx:
    #                     # make fsdp compatible with block-wise fp8
    #                     # use size(-1) to support hsdp
    #                     padded_out_features = self.get_num_features_after_pad(
    #                         child.weight.size(), 0, fsdp_mesh.size(-1), 128)
    #                     fp8_grouped_linear = TileWiseFloat8GroupedLinear.from_float(
    #                         child,
    #                         padded_out_features=padded_out_features
    #                     )
    #                 module.add_module(name, fp8_grouped_linear)
    #             else:
    #                 convert_to_tilewise_float8_grouped_linear_training(child, prefix + name + ".")

    #     if self.scaling_granularity_gemm == ScalingGranularity.TILEWISE:
    #         assert fsdp_mesh is not None
    #         convert_to_tilewise_float8_linear_training(model, "")
    #         logger.info("Tile-wise FP8 Linear training enabled.")
    #     if self.scaling_granularity_grouped_gemm == ScalingGranularity.TILEWISE:
    #         assert fsdp_mesh is not None
    #         convert_to_tilewise_float8_grouped_linear_training(model, "")
    #         logger.info("Tile-wise FP8 Grouped Linear training enabled.")

    def build_reduce_mesh(self, model: nn.Module, fsdp_mesh: DeviceMesh):
        if not self.enabled:
            logger.warning("Float8 training is not enabled.")
            return

        self.fsdp_mesh = fsdp_mesh
        if self.is_tilewise_fp8:
            self._build_reduce_mesh_devided_64(fsdp_mesh)
            self._build_reduce_mesh_mapping(model, fsdp_mesh)

    def _build_reduce_mesh_devided_64(self, fsdp_mesh: DeviceMesh):
        # 为了支持 moe 参数被 fsdp 和 ep 切成 dout = n * 128 + 64 (n >= 1) 的情况
        # fsdp rank 0 的后 64 个 dim 要跟 fsdp rank 1 的前 64 个 dim 共同组成一个 block
        # 计算 absmax 的时候要 reduce max
        if not self.enabled:
            logger.warning("Float8 training is not enabled.")
            return
        if not self.is_tilewise_fp8:
            logger.warning("Scaling granularity is not TILEWISE, no need to build reduce group.")
            return

        world_size = dist.get_world_size()

        assert fsdp_mesh.ndim in (1, 2)
        # use size(-1) to support hsdp
        assert fsdp_mesh.size(-1) % 2 == 0, (
            f"Currently only support fsdp_shard_size % 2 == 0, got fsdp_mesh.shape {fsdp_mesh.shape}."
        )

        device_mesh = init_device_mesh(
            "cuda",
            (fsdp_mesh.size() // 2, 2, world_size // fsdp_mesh.size()),
            mesh_dim_names=("_", "reduce", "ep_or_tp"),
        )["reduce"]
        self.tilewise_reduce_mesh_devided_64 = device_mesh

    def _build_reduce_mesh_mapping(self, model: nn.Module, fsdp_mesh: DeviceMesh):
        if not self.enabled:
            logger.warning("Float8 training is not enabled.")
            return
        if not self.is_tilewise_fp8:
            logger.warning("Scaling granularity is not TILEWISE, no need to build reduce group.")
            return

        from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
        from xtuner.v1.float8.float8_linear_tile_wise import TileWiseFloat8Linear

        world_size = dist.get_world_size()
        tilewise_reduce_mesh_mapping: Dict[Tuple[int, int], DeviceMesh] = {}
        SHARD_DIM = 0
        for module in model.modules():
            if isinstance(module, (TileWiseFloat8Linear, TileWiseFloat8GroupedLinear)):
                assert isinstance(module.weight, DTensor), (
                    "`build_reduce_mesh_mapping` should be called after apply fully_shard to the model."
                )
                # 不同 rank 的 local shape 是相同的，因为在 convert_to_float8_training 中已经进行了 pad
                local_shape = module.weight._local_tensor.shape
                if local_shape[SHARD_DIM] >= 128:
                    assert local_shape[SHARD_DIM] % 128 in (0, 64), (
                        f"Currently only local_shape[SHARD_DIM] % 128 == 0 or "
                        f"local_shape[SHARD_DIM] % 128 == 64 is supported, got {local_shape}. Please contact us."
                    )
                    continue
                assert 128 % local_shape[SHARD_DIM] == 0, (
                    f"Currently only local_shape[SHARD_DIM] % 128 == 0 is supported, got {local_shape}. Please contact us."
                )
                reduce_world_size = 128 // local_shape[SHARD_DIM]
                if local_shape in tilewise_reduce_mesh_mapping:
                    assert tilewise_reduce_mesh_mapping[local_shape].size() == reduce_world_size, (
                        f"Local shape {local_shape} already exists in tilewise_reduce_mesh_mapping, "
                        f"but the world size is {dist.get_world_size(group=tilewise_reduce_mesh_mapping[local_shape].get_group())}, "
                        f"expected {reduce_world_size}."
                    )
                    continue
                assert fsdp_mesh.size(-1) >= reduce_world_size and fsdp_mesh.size(-1) % reduce_world_size == 0, (
                    f"Expect fsdp_mesh.size(-1) >= reduce_world_size and fsdp_mesh.size(-1) % reduce_world_size == 0, "
                    f"got fsdp_mesh.size(-1) = {fsdp_mesh.size(-1)}, reduce_world_size = {reduce_world_size}. Please contact us."
                )

                device_mesh = init_device_mesh(
                    "cuda",
                    (fsdp_mesh.size() // reduce_world_size, reduce_world_size, world_size // fsdp_mesh.size()),
                    mesh_dim_names=("_", "tilewise_reduce", "ep_or_tp"),
                )["tilewise_reduce"]
                tilewise_reduce_mesh_mapping[local_shape] = device_mesh
        self.tilewise_reduce_mesh_mapping = tilewise_reduce_mesh_mapping

    def precompute_float8_dynamic_scale_for_fsdp(self, model: Union[nn.Module, List[nn.Module]]):
        if not self.enabled:
            return

        models = [model] if isinstance(model, nn.Module) else model

        for m in models:
            if self.is_tilewise_fp8:
                precompute_tilewise_float8_scale_for_fsdp(
                    m, self.tilewise_reduce_mesh_mapping, self.tilewise_reduce_mesh_devided_64
                )
            if self.is_tensorwise_fp8:
                assert self.fsdp_mesh is not None, "FSDP mesh must be set for tensorwise float8 training."
                precompute_tensorwise_float8_scale_for_fsdp(m, self.fsdp_mesh)
