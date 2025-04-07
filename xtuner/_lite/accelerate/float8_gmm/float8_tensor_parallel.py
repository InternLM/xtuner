# Copyright (c) OpenMMLab. All rights reserved.
# Copied from https://github.com/pytorch/ao/blob/v0.8.0/torchao/float8/float8_tensor_parallel.py
import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
)

from xtuner._lite.accelerate.float8_gmm.config import ScalingType, e4m3_dtype
from xtuner._lite.accelerate.float8_gmm.float8_scaling_utils import (
    NoopFwToFloat8BwDynamic,
    hp_tensor_to_float8_dynamic,
)
from xtuner._lite.accelerate.float8_gmm.float8_tensor import GemmInputRole

# subclass the ColwiseParallel and RowwiseParallel classes
# to add the float8 support
# The parameter sharding stays the same as the core
# ColwiseParallel and RowwiseParallel, the only difference
# here is that in input/output handling we do casting after
# creating the DTensor.

# NOTE: This only works and tested with the dynamic scaling


def _float8_linear_supports_float8_allgather(m):
    # TODO(future): add support for delayed scaling for activations
    # and gradients
    return (
        m.scaling_type_input == ScalingType.DYNAMIC
        and m.scaling_type_grad_output == ScalingType.DYNAMIC
    )


class Float8ColwiseParallel(ColwiseParallel):
    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        input_tensor = hp_tensor_to_float8_dynamic(
            input_tensor,
            mod.config.cast_config_input.target_dtype,
            mod.linear_mm_config,
            gemm_input_role=GemmInputRole.INPUT,
        )  # DTensor(Float8Tensor)

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(
                placements=output_layouts, async_op=True
            )  # DTensor(torch.Tensor)

        # fwd noop bwd cast to DTensor(Float8Tensor)
        outputs = NoopFwToFloat8BwDynamic.apply(
            outputs,
            mod.linear_mm_config,
            mod.config.cast_config_grad_output.target_dtype,
        )

        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from xtuner._lite.accelerate.float8_gmm.float8_linear import Float8Linear

        if not isinstance(module, Float8Linear):
            raise ValueError(
                f"Expecting module to be Float8Linear but found {type(module)}"
            )
        elif isinstance(
            module, Float8Linear
        ) and not _float8_linear_supports_float8_allgather(module):
            raise AssertionError("unsupported")

        return super()._apply(module, device_mesh)


class Float8RowwiseParallel(RowwiseParallel):
    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        input_tensor = hp_tensor_to_float8_dynamic(
            input_tensor,
            mod.config.cast_config_input.target_dtype,
            mod.linear_mm_config,
            gemm_input_role=GemmInputRole.INPUT,
        )  # DTensor(Float8Tensor)

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)

        # fwd noop bwd cast to DTensor(Float8Tensor)
        outputs = NoopFwToFloat8BwDynamic.apply(
            outputs,
            mod.linear_mm_config,
            mod.config.cast_config_grad_output.target_dtype,
        )

        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from xtuner._lite.accelerate.float8_gmm.float8_linear import Float8Linear

        if not isinstance(module, Float8Linear):
            raise ValueError(
                f"Expecting module to be Float8Linear but found {type(module)}"
            )
        elif isinstance(
            module, Float8Linear
        ) and not _float8_linear_supports_float8_allgather(module):
            raise AssertionError("unsupported")

        return super()._apply(module, device_mesh)


class PrepareFloat8ModuleInput(PrepareModuleInput):
    # subclass the PrepareModuleInput classes to implement fp8 specific logic, the only difference is that
    # after we prepare the input DTensor, we cast the input to DTensor(Float8Tensor)
    # This is to ensure the float8 cast happens before the all-gather (i.e. Shard -> Replicate)
    # so that if there are multiple float8 users of the input activation, we perform fp8 allgather
    # only once.
    # FP8 Args:
    #   float8_dtype (torch.dtype, optional): control what float8 dtype to cast to when prepare the module input,
    #       we currently only support torch.float8_e4m3fn. default: torch.float8_e4m3fn
    #   fwd_config_submodule_fqn (str, optional): the fqn of the submodule that contains the forward config used
    #       for the float8 cast. If not specified, we will search for the Float8Linear in the submodules
    #       and use the forward config from that module, in this case all module's forward config must be
    #       the same.

    def __init__(
        self,
        *,
        input_layouts=None,
        desired_input_layouts=None,
        input_kwarg_layouts=None,
        desired_input_kwarg_layouts=None,
        use_local_output=False,
        float8_dtype=torch.float8_e4m3fn,
        fwd_config_submodule_fqn=None,
    ):
        super().__init__(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_output,
        )

        # fp8 specific fields
        self.float8_dtype = float8_dtype
        self.linear_mm_config = None
        self.fwd_config_submodule_fqn = fwd_config_submodule_fqn

        if self.float8_dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                "PrepareFloat8ModuleInput only support casting to float8_e4m3fn for now"
            )

    def _prepare_input_arg(self, input, mesh, input_layout, desired_layout):
        if input_layout is not None:
            if isinstance(input, DTensor):
                # TODO: re-enable the check once we fix the compile path
                # assert inp.placements[0] == input_layout
                dt_inp = input
            else:
                assert isinstance(
                    input, torch.Tensor
                ), "expecting input to be a torch.Tensor!"
                dt_inp = DTensor.from_local(
                    input, mesh, (input_layout,), run_check=False
                )

            dt_inp = hp_tensor_to_float8_dynamic(
                dt_inp,
                e4m3_dtype,
                self.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
            )  # DTensor(Float8Tensor)
            if desired_layout is not None and input_layout != desired_layout:
                dt_inp = dt_inp.redistribute(placements=(desired_layout,))

            return dt_inp.to_local() if self.use_local_output else dt_inp
        else:
            return input

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from xtuner._lite.accelerate.float8_gmm.float8_linear import Float8Linear

        if self.fwd_config_submodule_fqn is not None:
            fwd_linear = module.get_submodule(self.fwd_config_submodule_fqn)
            assert isinstance(fwd_linear, Float8Linear)
            self.linear_mm_config = fwd_linear.linear_mm_config
        else:
            # search for ScaledMM configs for all the submodules and make sure they are the same
            for mod in module.modules():
                if isinstance(mod, Float8Linear):
                    if self.linear_mm_config is None:
                        self.linear_mm_config = mod.linear_mm_config
                    else:
                        assert (
                            self.linear_mm_config == mod.linear_mm_config
                        ), "All the Float8Linear modules should have same linear_mm_config!"

        assert self.linear_mm_config is not None
        super()._apply(module, device_mesh)
        return module
