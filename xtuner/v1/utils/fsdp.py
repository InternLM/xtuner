import torch

from xtuner.v1.utils.device import get_torch_device_module


DEVICE_MODULE = get_torch_device_module()


def release_deferred_fsdp_all_gathers(model: torch.nn.Module) -> tuple[int, int]:
    """Release FSDP2 all-gather buffers that were prefetched but not
    consumed."""

    try:
        from torch.distributed._composable_state import _get_module_state
    except Exception:
        return 0, 0

    released_comm_states = 0
    released_param_groups = 0
    seen_comm_ctx: set[int] = set()
    seen_param_groups: set[int] = set()

    def wait_all_gather_result(all_gather_result) -> None:
        event = getattr(all_gather_result, "all_gather_event", None)
        if event is not None:
            try:
                torch.accelerator.current_stream().wait_event(event)
            except Exception:
                DEVICE_MODULE.synchronize()
        work = getattr(all_gather_result, "all_gather_work", None)
        if work is not None and hasattr(work, "wait"):
            try:
                work.wait()
            except Exception:
                DEVICE_MODULE.synchronize()

    for module in model.modules():
        try:
            state = _get_module_state(module)
        except Exception:
            continue
        if state is None:
            continue

        comm_ctx = getattr(state, "_comm_ctx", None)
        if comm_ctx is not None and id(comm_ctx) not in seen_comm_ctx:
            seen_comm_ctx.add(id(comm_ctx))
            all_gather_state = getattr(comm_ctx, "all_gather_state", None)
            if all_gather_state is not None:
                event = getattr(all_gather_state, "event", None)
                if event is not None:
                    try:
                        event.synchronize()
                    except Exception:
                        DEVICE_MODULE.synchronize()
                comm_ctx.all_gather_state = None
                released_comm_states += 1

        param_group = getattr(state, "_fsdp_param_group", None)
        if param_group is None or id(param_group) in seen_param_groups:
            continue
        seen_param_groups.add(id(param_group))
        all_gather_result = getattr(param_group, "_all_gather_result", None)
        if all_gather_result is None:
            continue
        wait_all_gather_result(all_gather_result)
        param_group._all_gather_result = None
        released_param_groups += 1

    return released_comm_states, released_param_groups
