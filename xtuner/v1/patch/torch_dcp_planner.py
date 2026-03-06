import inspect
import warnings
from typing import Optional


import torch
import torch.distributed as dist
from torch.distributed.checkpoint import  state_dict_saver
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.logger import _dcp_method_logger
from torch.distributed.checkpoint.metadata import Metadata, STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter
from torch.distributed.checkpoint.utils import _DistWrapper
import torch.distributed.checkpoint.default_planner as torch_default_runner


def fake_validate_global_plan(*args, **kwargs):
    return True


def patch_default_save_plan():
    torch_default_runner._validate_global_plan = fake_validate_global_plan

def _xtuner_save_state_dict(
    state_dict: STATE_DICT_TYPE,
    storage_writer: StorageWriter,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[SavePlanner] = None,
) -> Metadata:
    torch._C._log_api_usage_once("torch.distributed.checkpoint.save_state_dict")

    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultSavePlanner()
    assert planner is not None

    global_metadata = None

    ckpt_kwargs = {}
    if (ckpt_id := getattr(storage_writer, "checkpoint_id", None)) is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id
        ckpt_kwargs["process_group"] = distW.group

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        assert planner is not None
        storage_meta = storage_writer.storage_meta()
        if "storage_meta" not in inspect.signature(planner.set_up_planner).parameters:
            warnings.warn(
                "The function definition for SavePlanner.set_up_planner has been updated"
                " to include the storage_meta argument. Please update your implementation"
                " to include this parameter."
            )
            planner.set_up_planner(state_dict, distW.is_coordinator)  # type: ignore[call-arg, arg-type]
        else:
            planner.set_up_planner(
                state_dict=state_dict,
                storage_meta=storage_meta,
                is_coordinator=distW.is_coordinator,
            )
        storage_writer.set_up_storage_writer(distW.is_coordinator)

        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        nonlocal global_metadata

        assert planner is not None
        all_local_plans, global_metadata = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan: SavePlan = distW.reduce_scatter("plan", local_step, global_step)

    @_dcp_method_logger(**ckpt_kwargs)
    def write_data():
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_writes = storage_writer.write_data(final_local_plan, planner)

        all_writes.wait()
        return all_writes.value()

    @_dcp_method_logger(**ckpt_kwargs)
    def finish_checkpoint(all_results):
        assert global_metadata is not None
        storage_writer.finish(metadata=global_metadata, results=all_results)
        # return global_metadata
        return Metadata(state_dict_metadata={}) # This is a patch to avoid broadcast overhead.

    return distW.all_reduce("write", write_data, finish_checkpoint)


def patch_dcp_save_state_dict():
    if hasattr(state_dict_saver, "_save_state_dict"):
        original = getattr(state_dict_saver, "_save_state_dict")
        if callable(original):
            state_dict_saver._save_state_dict = _xtuner_save_state_dict