# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from xtuner.v1.float8.config import ScalingGranularity
from xtuner.v1.float8.fsdp_utils import (
    precompute_tensorwise_float8_scale_for_fsdp,
    precompute_tilewise_float8_scale_for_fsdp,
)
from xtuner.v1.utils import get_logger, is_evenly_distributed, log_rank0
from xtuner.v1.utils.interleaved_shard import RuntimeLayout

from .fsdp_utils import WeightWithDynamicTensorWiseFloat8CastTensor, WeightWithDynamicTilewiseFloat8CastTensor


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
    scaling_granularity_gemm: ScalingGranularity | None
    scaling_granularity_grouped_gemm: ScalingGranularity | None
    fsdp_mesh: Optional[DeviceMesh] = None
    tilewise_reduce_mesh_devided_64: Optional[DeviceMesh] = None
    tilewise_reduce_mesh_mapping: Dict[Tuple[int, int], DeviceMesh] = {}
    # Decoupled EP/FSDP layout only: routed experts are FSDP-sharded on `expert_fsdp_mesh`
    # (efsdp, strided by ep) instead of `fsdp_mesh`, so they need their own reduce meshes.
    expert_fsdp_mesh: Optional[DeviceMesh] = None
    expert_tilewise_reduce_mesh_devided_64: Optional[DeviceMesh] = None
    expert_tilewise_reduce_mesh_mapping: Optional[Dict[Tuple[int, int], DeviceMesh]] = None

    def __init__(
        self,
        scaling_granularity_gemm: Optional[ScalingGranularity] = None,
        scaling_granularity_grouped_gemm: Optional[ScalingGranularity] = None,
    ) -> None:
        torch.serialization.add_safe_globals(
            [
                WeightWithDynamicTilewiseFloat8CastTensor,
                WeightWithDynamicTensorWiseFloat8CastTensor,
            ]
        )

        if not _is_sm89_or_later():
            log_rank0.warning(
                "Failed to enable float8 training because float8 is only supported on SM89 or later",
            )
            return

        assert scaling_granularity_gemm in (ScalingGranularity.TILEWISE, ScalingGranularity.TENSORWISE, None), (
            "scaling_granularity_gemm must be TILEWISE, TENSORWISE or None."
        )
        assert scaling_granularity_grouped_gemm in (
            ScalingGranularity.TILEWISE,
            ScalingGranularity.TENSORWISE,
            None,
        ), "scaling_granularity_grouped_gemm must be TILEWISE or TENSORWISE."

        self.scaling_granularity_gemm = scaling_granularity_gemm
        self.scaling_granularity_grouped_gemm = scaling_granularity_grouped_gemm
        self.is_tilewise_fp8 = (
            scaling_granularity_gemm == ScalingGranularity.TILEWISE
            or scaling_granularity_grouped_gemm == ScalingGranularity.TILEWISE
        )
        self.is_tensorwise_fp8 = scaling_granularity_gemm == ScalingGranularity.TENSORWISE

    @staticmethod
    def get_num_features_after_pad(tensor_size, fsdp_shard_dim, num_chunks):
        fp8_block_size = 128
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
                if (chunk_size * num_chunks) % 128 != 0:
                    chunk_size += 64
        else:
            # 如果小于base_size，找到大于等于 ideal_chunk_size 的 128 的因数
            factors = [1, 2, 4, 8, 16, 32, 64, 128]
            for size in factors:
                # 找到大于等于 ideal_chunk_size 的 128 的因数，同时要求 chunk_size * num_chunks（即dout_pad）能被 128 整除
                if ideal_chunk_size <= size and (size * num_chunks) % 128 == 0:
                    chunk_size = size
                    break
        return chunk_size * num_chunks

    @staticmethod
    def get_shard_size_on_dim(tensor: torch.Tensor | DTensor, dim: int) -> int:
        if not isinstance(tensor, DTensor):
            return 1
        return RuntimeLayout.from_dtensor(tensor).shard_size(dim)

    @staticmethod
    def pad_for_fsdp(
        model: nn.Module,
        fsdp_mesh: DeviceMesh,
        callback_after_pad: Callable | None = None,
        expert_fsdp_mesh: DeviceMesh | None = None,
    ):
        from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
        from xtuner.v1.float8.float8_linear_tensor_wise import TensorWiseFloat8Linear
        from xtuner.v1.float8.float8_linear_tile_wise import TileWiseFloat8Linear

        for module in model.modules():
            if isinstance(module, (TileWiseFloat8Linear, TileWiseFloat8GroupedLinear, TensorWiseFloat8Linear)):
                # make fsdp compatible with block-wise fp8
                # use size(-1) to support hsdp
                if isinstance(module.weight, DTensor):
                    assert is_evenly_distributed(module.weight), (
                        "Currently only support even distributed TP or EP weight for float8 training."
                    )
                    tensor_size = module.weight._local_tensor.size()
                    parallel_size = Float8Handler.get_shard_size_on_dim(module.weight, dim=0)
                else:
                    tensor_size = module.weight.size()
                    parallel_size = 1
                num_fsdp_chunks = fsdp_mesh.size(-1)
                if expert_fsdp_mesh is not None and isinstance(module, TileWiseFloat8GroupedLinear):
                    # Decoupled EP/FSDP: routed experts are sharded `efsdp` ways, not `dp_shard` ways.
                    num_fsdp_chunks = expert_fsdp_mesh.size(-1)
                padded_out_features = Float8Handler.get_num_features_after_pad(tensor_size, 0, num_fsdp_chunks)
                padded_out_features *= parallel_size
                module.pad_for_fsdp(padded_out_features=padded_out_features)

        if callback_after_pad is not None:
            callback_after_pad()

    def build_reduce_mesh(self, model: nn.Module, fsdp_mesh: DeviceMesh, expert_fsdp_mesh: DeviceMesh | None = None):
        self.fsdp_mesh = fsdp_mesh
        self.expert_fsdp_mesh = expert_fsdp_mesh
        if self.is_tilewise_fp8:
            if expert_fsdp_mesh is not None:
                self._build_decoupled_reduce_meshes(model, fsdp_mesh, expert_fsdp_mesh)
                return
            if fsdp_mesh.size(-1) >= 2:
                self._build_reduce_mesh_devided_64(fsdp_mesh)
            self._build_reduce_mesh_mapping(model, fsdp_mesh)

    def _build_decoupled_reduce_meshes(
        self, model: nn.Module, fsdp_mesh: DeviceMesh, expert_fsdp_mesh: DeviceMesh
    ) -> None:
        # Decoupled EP/FSDP layout (root mesh `(replicate, efsdp, ep)`):
        #   * dense fp8 weights are sharded over `dp_shard` = flatten(efsdp, ep): consecutive FSDP
        #     shards live on consecutive ranks (stride 1);
        #   * routed-expert fp8 weights are sharded over `efsdp`, whose ranks are `ep` apart.
        # Each class gets its own "reduce max" meshes, built with the matching rank stride.
        from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
        from xtuner.v1.float8.float8_linear_tile_wise import TileWiseFloat8Linear

        world_size = dist.get_world_size()
        dense_shard_size = fsdp_mesh.size(-1)
        expert_shard_size = expert_fsdp_mesh.size(-1)
        expert_stride = world_size // expert_fsdp_mesh.size()  # == ep_size

        self.tilewise_reduce_mesh_devided_64 = (
            self._build_strided_reduce_mesh(2, 1) if dense_shard_size >= 2 and dense_shard_size % 2 == 0 else None
        )
        self.expert_tilewise_reduce_mesh_devided_64 = (
            self._build_strided_reduce_mesh(2, expert_stride)
            if expert_shard_size >= 2 and expert_shard_size % 2 == 0
            else None
        )
        self.tilewise_reduce_mesh_mapping = self._build_strided_reduce_mesh_mapping(
            model, (TileWiseFloat8Linear,), dense_shard_size, 1
        )
        self.expert_tilewise_reduce_mesh_mapping = self._build_strided_reduce_mesh_mapping(
            model, (TileWiseFloat8GroupedLinear,), expert_shard_size, expert_stride
        )

    @staticmethod
    def _build_strided_reduce_mesh(num_ranks: int, stride: int) -> DeviceMesh:
        # Groups of `num_ranks` ranks spaced `stride` apart, e.g. (r, r + stride, ...).
        world_size = dist.get_world_size()
        assert world_size % (num_ranks * stride) == 0, (world_size, num_ranks, stride)
        return init_device_mesh(
            "cuda",
            (world_size // (num_ranks * stride), num_ranks, stride),
            mesh_dim_names=("_", "tilewise_reduce", "ep_or_tp"),
        )["tilewise_reduce"]

    def _build_strided_reduce_mesh_mapping(
        self, model: nn.Module, module_types: Tuple[type, ...], shard_size: int, stride: int
    ) -> Dict[Tuple[int, int], DeviceMesh]:
        SHARD_DIM = 0
        mapping: Dict[Tuple[int, int], DeviceMesh] = {}
        for module in model.modules():
            if not isinstance(module, module_types):
                continue
            assert isinstance(module.weight, DTensor), (
                "`build_reduce_mesh` should be called after apply fully_shard to the model."
            )
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
            if local_shape in mapping:
                assert mapping[local_shape].size() == reduce_world_size
                continue
            assert shard_size >= reduce_world_size and shard_size % reduce_world_size == 0, (
                f"Expect FSDP shard size >= reduce_world_size and shard size % reduce_world_size == 0, "
                f"got shard size = {shard_size}, reduce_world_size = {reduce_world_size}. Please contact us."
            )
            mapping[local_shape] = self._build_strided_reduce_mesh(reduce_world_size, stride)
        return mapping

    def _build_reduce_mesh_devided_64(self, fsdp_mesh: DeviceMesh):
        # 为了支持 moe 参数被 fsdp 和 ep 切成 dout = n * 128 + 64 (n >= 1) 的情况
        # fsdp rank 0 的后 64 个 dim 要跟 fsdp rank 1 的前 64 个 dim 共同组成一个 block
        # 计算 absmax 的时候要 reduce max
        if not self.is_tilewise_fp8:
            log_rank0.warning("Scaling granularity is not TILEWISE, no need to build reduce group.")
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
        if not self.is_tilewise_fp8:
            log_rank0.warning("Scaling granularity is not TILEWISE, no need to build reduce group.")
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
        models = [model] if isinstance(model, nn.Module) else model

        for m in models:
            if self.is_tilewise_fp8:
                if self.expert_tilewise_reduce_mesh_mapping is None:
                    precompute_tilewise_float8_scale_for_fsdp(
                        m, self.tilewise_reduce_mesh_mapping, self.tilewise_reduce_mesh_devided_64
                    )
                else:
                    from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
                    from xtuner.v1.float8.float8_linear_tile_wise import TileWiseFloat8Linear

                    precompute_tilewise_float8_scale_for_fsdp(
                        m,
                        self.tilewise_reduce_mesh_mapping,
                        self.tilewise_reduce_mesh_devided_64,
                        module_types=(TileWiseFloat8Linear,),
                    )
                    precompute_tilewise_float8_scale_for_fsdp(
                        m,
                        self.expert_tilewise_reduce_mesh_mapping,
                        self.expert_tilewise_reduce_mesh_devided_64,
                        module_types=(TileWiseFloat8GroupedLinear,),
                    )
            if self.is_tensorwise_fp8:
                assert self.fsdp_mesh is not None, "FSDP mesh must be set for tensorwise float8 training."
                precompute_tensorwise_float8_scale_for_fsdp(m, self.fsdp_mesh)
