"""Reload selected train-side NCCL process groups without rebuilding FSDP2.

FSDP2 and DeviceMesh keep Python references to process groups. Destroying those groups directly leaves stale objects in
FSDP hooks and functional collective registries, so this module gives PyTorch a stable wrapper object and swaps only
the wrapped NCCL process group when train memory is released/reloaded.

The default/world process group is intentionally not managed here. It is used by Ray/torch distributed control paths
and is not replayed safely from this layer.
"""

import os
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.distributed as dist


class ReloadableProcessGroup(torch.distributed.ProcessGroup):
    """Stable ProcessGroup proxy whose inner NCCL group can be destroyed."""

    _GROUPS_BY_PID: dict[int, list["ReloadableProcessGroup"]] = {}
    _GROUPS_BY_NAME: dict[int, dict[str, "ReloadableProcessGroup"]] = {}

    def __init__(self, group: dist.ProcessGroup, ranks: list[int], backend: str | None = None):
        super().__init__(rank=group.rank(), size=group.size())  # type: ignore[call-arg]
        self.group: dist.ProcessGroup | None = group
        self._stable_group_name = group.group_name
        self._stable_group_desc = getattr(group, "group_desc", "")
        self.group_info = {
            "ranks": list(ranks),
            "backend": backend or _backend_for_reload(group),
        }
        # DeviceMesh reads these attributes directly instead of going through
        # methods, so expose stable Python properties on the wrapper itself.
        try:
            self._set_group_name(self._stable_group_name)
            self._set_group_desc(self._stable_group_desc)
        except RuntimeError:
            pass
        self._copy_public_pg_attrs(group)

        pid = os.getpid()
        self._GROUPS_BY_PID.setdefault(pid, []).append(self)
        self._GROUPS_BY_NAME.setdefault(pid, {})[self._stable_group_name] = self
        _register_wrapper_in_world(self, group)

    @property
    def group_name(self) -> str:
        return self._stable_group_name

    @property
    def group_desc(self) -> str:
        return self._stable_group_desc

    @property
    def _device_types(self):
        # all_gather_object uses this private field to select a CPU fallback.
        # Proxying it avoids "No backend type associated with device type cpu".
        return self._inner()._device_types

    def _copy_public_pg_attrs(self, group: dist.ProcessGroup) -> None:
        if hasattr(group, "bound_device_id"):
            self.bound_device_id = group.bound_device_id  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> Any:
        group = self._inner()
        return getattr(group, name)

    def _inner(self) -> dist.ProcessGroup:
        if self.group is None:
            raise RuntimeError("ReloadableProcessGroup inner process group is destroyed; reload it before use.")
        return self.group

    def _fwd(self, method: str, *args, **kwargs):
        return getattr(self._inner(), method)(*args, **kwargs)

    def rank(self) -> int:
        return self._inner().rank()

    def size(self) -> int:
        return self._inner().size()

    def name(self) -> str:
        return self._inner().name()

    def get_group_store(self):
        return self._inner().get_group_store()

    def _get_backend(self, *args, **kwargs):
        return self._inner()._get_backend(*args, **kwargs)

    def _get_backend_name(self, *args, **kwargs):
        return self._inner()._get_backend_name(*args, **kwargs)

    def _get_sequence_number_for_group(self, *args, **kwargs):
        return self._inner()._get_sequence_number_for_group(*args, **kwargs)

    def _set_sequence_number_for_group(self, *args, **kwargs):
        return self._inner()._set_sequence_number_for_group(*args, **kwargs)

    def shutdown(self) -> None:
        if self.group is not None:
            getattr(self.group, "shutdown")()

    def abort(self) -> None:
        if self.group is not None:
            getattr(self.group, "abort")()

    def split_group(self, *args, **kwargs):
        return self._inner().split_group(*args, **kwargs)

    def barrier(self, *args, **kwargs):
        return self._fwd("barrier", *args, **kwargs)

    def broadcast(self, *args, **kwargs):
        return self._fwd("broadcast", *args, **kwargs)

    def allreduce(self, *args, **kwargs):
        return self._fwd("allreduce", *args, **kwargs)

    def allreduce_coalesced(self, *args, **kwargs):
        return self._fwd("allreduce_coalesced", *args, **kwargs)

    def reduce(self, *args, **kwargs):
        return self._fwd("reduce", *args, **kwargs)

    def allgather(self, *args, **kwargs):
        return self._fwd("allgather", *args, **kwargs)

    def allgather_coalesced(self, *args, **kwargs):
        return self._fwd("allgather_coalesced", *args, **kwargs)

    def allgather_into_tensor_coalesced(self, *args, **kwargs):
        return self._fwd("allgather_into_tensor_coalesced", *args, **kwargs)

    def _allgather_base(self, *args, **kwargs):
        return self._fwd("_allgather_base", *args, **kwargs)

    def gather(self, *args, **kwargs):
        return self._fwd("gather", *args, **kwargs)

    def scatter(self, *args, **kwargs):
        return self._fwd("scatter", *args, **kwargs)

    def reduce_scatter(self, *args, **kwargs):
        return self._fwd("reduce_scatter", *args, **kwargs)

    def reduce_scatter_tensor_coalesced(self, *args, **kwargs):
        return self._fwd("reduce_scatter_tensor_coalesced", *args, **kwargs)

    def _reduce_scatter_base(self, *args, **kwargs):
        return self._fwd("_reduce_scatter_base", *args, **kwargs)

    def alltoall(self, *args, **kwargs):
        return self._fwd("alltoall", *args, **kwargs)

    def alltoall_base(self, *args, **kwargs):
        return self._fwd("alltoall_base", *args, **kwargs)

    def send(self, *args, **kwargs):
        return self._fwd("send", *args, **kwargs)

    def recv(self, *args, **kwargs):
        return self._fwd("recv", *args, **kwargs)

    def recv_anysource(self, *args, **kwargs):
        return self._fwd("recv_anysource", *args, **kwargs)

    def monitored_barrier(self, *args, **kwargs):
        return self._fwd("monitored_barrier", *args, **kwargs)


_ORIGINALS_BY_PID: dict[int, dict[str, Any]] = {}
_SPECS_BY_PID: dict[int, list[dict[str, Any]]] = {}


def monkey_patch_reloadable_process_groups() -> None:
    """Install per-process patches before DeviceMesh/FSDP2 creates
    subgroups."""

    pid = os.getpid()
    if pid in _ORIGINALS_BY_PID:
        return

    import torch.distributed.device_mesh as device_mesh
    import torch.distributed.distributed_c10d as c10d

    originals = {
        "c10d_new_group": c10d.new_group,
        "c10d_split_group": c10d.split_group,
        "c10d_resolve_process_group": c10d._resolve_process_group,
    }
    if hasattr(dist, "split_group"):
        originals["dist_split_group"] = dist.split_group
    _ORIGINALS_BY_PID[pid] = originals

    def new_group(*args, **kwargs):
        group = originals["c10d_new_group"](*args, **kwargs)
        wrapper_or_group = _wrap_group_if_needed(group, _extract_new_group_ranks(args, kwargs), kwargs.get("backend"))
        if _should_record_spec(group, args, kwargs):
            # Every rank must replay every subgroup creation call in the same
            # order, including nonmember calls that return NON_GROUP_MEMBER.
            _SPECS_BY_PID.setdefault(os.getpid(), []).append(
                {
                    "kind": "new_group",
                    "args": args,
                    "kwargs": dict(kwargs),
                    "wrapper": wrapper_or_group if isinstance(wrapper_or_group, ReloadableProcessGroup) else None,
                }
            )
        return wrapper_or_group

    def split_group(*args, **kwargs):
        group = originals["c10d_split_group"](*args, **kwargs)
        wrapper_or_group = _wrap_group_if_needed(group, None, None)
        _SPECS_BY_PID.setdefault(os.getpid(), []).append(
            {
                "kind": "split_group",
                "args": args,
                "kwargs": dict(kwargs),
                "wrapper": wrapper_or_group if isinstance(wrapper_or_group, ReloadableProcessGroup) else None,
            }
        )
        return wrapper_or_group

    def resolve_process_group(group_name: str):
        # Functional collectives store only the group name. Keep that name
        # resolving to the stable wrapper even after the inner PG is recreated.
        wrapper = ReloadableProcessGroup._GROUPS_BY_NAME.get(os.getpid(), {}).get(group_name)
        if wrapper is not None:
            return wrapper
        resolve_process_group_fn = cast(Callable[[str], Any], originals["c10d_resolve_process_group"])
        return resolve_process_group_fn(group_name)

    dist.new_group = new_group
    c10d.new_group = new_group
    device_mesh.new_group = new_group
    c10d.split_group = split_group
    device_mesh.split_group = split_group
    c10d._resolve_process_group = resolve_process_group
    device_mesh._resolve_process_group = resolve_process_group  # type: ignore[attr-defined]
    if "dist_split_group" in originals:
        dist.split_group = split_group  # type: ignore[attr-defined]


def destroy_reloadable_process_groups() -> list[dict[str, object]]:
    """Destroy wrapped NCCL groups while leaving the wrapper objects cached."""

    details: list[dict[str, object]] = []
    for wrapper in reversed(ReloadableProcessGroup._GROUPS_BY_PID.get(os.getpid(), [])):
        if wrapper.group is None:
            continue
        inner = wrapper.group
        ranks = list(wrapper.group_info["ranks"])
        backend = str(wrapper.group_info["backend"])
        try:
            dist.destroy_process_group(inner)
            wrapper.group = None
            details.append({"ranks": ranks, "backend": backend, "group_name": wrapper.group_name})
        except ValueError:
            wrapper.group = None
            details.append(
                {"ranks": ranks, "backend": backend, "group_name": wrapper.group_name, "already_gone": True}
            )
    return details


def reload_process_groups() -> list[dict[str, object]]:
    """Replay recorded subgroup creation and attach fresh inners to
    wrappers."""

    pid = os.getpid()
    originals = _ORIGINALS_BY_PID.get(pid)
    if originals is None:
        return []
    if not any(group.group is None for group in ReloadableProcessGroup._GROUPS_BY_PID.get(pid, [])):
        return []

    details: list[dict[str, object]] = []
    for spec in _SPECS_BY_PID.get(pid, []):
        wrapper = spec.get("wrapper")
        if wrapper is not None and wrapper.group is not None:
            continue
        if spec["kind"] == "new_group":
            group = originals["c10d_new_group"](
                *_resolve_args_for_reload(spec["args"]),
                **_resolve_kwargs_for_reload(spec["kwargs"]),
            )
        elif spec["kind"] == "split_group":
            group = originals["c10d_split_group"](
                *_resolve_args_for_reload(spec["args"]),
                **_resolve_kwargs_for_reload(spec["kwargs"]),
            )
        else:
            continue
        if wrapper is None:
            continue
        if group is None or group is dist.GroupMember.NON_GROUP_MEMBER:
            continue
        # New inner PGs get new private names/tags. FSDP2 caches the wrapper,
        # while functional collectives resolve by name, so both registries must
        # be moved back to the wrapper's original identity.
        _restore_inner_group_name(wrapper, group)
        wrapper.group = group
        wrapper._copy_public_pg_attrs(group)
        _register_wrapper_in_world(wrapper, group)
        _restore_inner_tag(wrapper, group)
        details.append(
            {
                "ranks": list(wrapper.group_info["ranks"]),
                "backend": wrapper.group_info["backend"],
                "group_name": wrapper.group_name,
            }
        )
    return details


def reloadable_process_group_status() -> dict[str, object]:
    groups = ReloadableProcessGroup._GROUPS_BY_PID.get(os.getpid(), [])
    return {
        "total": len(groups),
        "alive": sum(group.group is not None for group in groups),
        "destroyed": sum(group.group is None for group in groups),
        "specs": len(_SPECS_BY_PID.get(os.getpid(), [])),
    }


def _wrap_group_if_needed(
    group: dist.ProcessGroup | None,
    ranks: list[int] | None,
    backend_arg: object,
) -> dist.ProcessGroup | None:
    if group is None or group is dist.GroupMember.NON_GROUP_MEMBER:
        return group
    if group is dist.group.WORLD:
        return group
    backend = str(backend_arg or _backend_for_reload(group)).lower()
    if "gloo" in backend and "nccl" not in backend:
        return group
    if "nccl" not in backend and "cuda" not in backend:
        return group
    ranks = ranks or _ranks_from_world(group)
    if len(ranks) <= 1:
        return group
    return ReloadableProcessGroup(group, ranks, _backend_for_reload(group))


def _should_record_spec(group: dist.ProcessGroup | None, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    # Nonmember calls must still be recorded to keep future reload calls
    # collectively ordered across ranks.
    backend = str(kwargs.get("backend", "")).lower()
    if "gloo" in backend and "nccl" not in backend and "cuda" not in backend:
        return False
    if group is None:
        return False
    if group is dist.group.WORLD:
        return False
    if group is dist.GroupMember.NON_GROUP_MEMBER:
        return True
    try:
        backend = str(dist.get_backend(group)).lower()
    except Exception:
        return False
    return "nccl" in backend or "cuda" in backend


def _resolve_args_for_reload(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(_resolve_obj_for_reload(arg) for arg in args)


def _resolve_kwargs_for_reload(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: _resolve_obj_for_reload(value) for key, value in kwargs.items()}


def _resolve_obj_for_reload(value: Any) -> Any:
    # Replay creation with current inner groups, not stale wrappers whose
    # inners may have been destroyed.
    if isinstance(value, ReloadableProcessGroup):
        return value._inner()
    if isinstance(value, list):
        return [_resolve_obj_for_reload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_obj_for_reload(item) for item in value)
    if isinstance(value, dict):
        return {key: _resolve_obj_for_reload(item) for key, item in value.items()}
    return value


def _extract_new_group_ranks(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[int] | None:
    if args and args[0] is not None:
        return list(args[0])
    if kwargs.get("ranks") is not None:
        return list(kwargs["ranks"])
    if dist.is_initialized():
        return list(range(dist.get_world_size()))
    return None


def _ranks_from_world(group: dist.ProcessGroup) -> list[int]:
    import torch.distributed.distributed_c10d as c10d

    rank_map = c10d._world.pg_group_ranks.get(group)
    if rank_map is None:
        return list(range(group.size()))
    return [global_rank for global_rank, _ in sorted(rank_map.items(), key=lambda item: item[1])]


def _backend_for_reload(group: dist.ProcessGroup) -> str:
    backend = str(dist.get_backend(group)).lower()
    if "nccl" in backend or "cuda" in backend:
        return "nccl"
    return backend


def _register_wrapper_in_world(wrapper: ReloadableProcessGroup, inner: dist.ProcessGroup) -> None:
    import torch.distributed.distributed_c10d as c10d

    # c10d helpers query these Python maps directly. Registering the wrapper
    # makes dist.* calls and DeviceMesh code treat it like the original PG.
    world = c10d._world
    if inner in world.pg_map:
        world.pg_map[wrapper] = world.pg_map[inner]
    if inner in world.pg_names:
        world.pg_names[wrapper] = wrapper.group_name
    if inner in world.pg_group_ranks:
        world.pg_group_ranks[wrapper] = dict(world.pg_group_ranks[inner])
    if inner in world.pg_backend_config:
        world.pg_backend_config[wrapper] = world.pg_backend_config[inner]
    if inner in world.pg_to_tag:
        tag = world.pg_to_tag[inner]
        world.pg_to_tag[wrapper] = tag
        world.tags_to_pg.setdefault(tag, [])
        if wrapper not in world.tags_to_pg[tag]:
            world.tags_to_pg[tag].append(wrapper)


def _restore_inner_group_name(wrapper: ReloadableProcessGroup, inner: dist.ProcessGroup) -> None:
    import torch.distributed.distributed_c10d as c10d

    # The newly created inner receives a fresh c10d name. Restore the original
    # name so functional collectives using that name keep resolving correctly.
    old_name = process_group_name(inner)
    stable_name = wrapper.group_name
    if old_name == stable_name:
        return

    world = c10d._world
    if old_name != "unnamed":
        try:
            c10d._unregister_process_group(old_name)
        except Exception:
            pass
    try:
        inner._set_group_name(stable_name)
    except RuntimeError:
        pass
    if inner in world.pg_names:
        world.pg_names[inner] = stable_name
    try:
        c10d._register_process_group(stable_name, inner)
    except Exception:
        pass


def _restore_inner_tag(wrapper: ReloadableProcessGroup, inner: dist.ProcessGroup) -> None:
    import torch.distributed.distributed_c10d as c10d

    # all_to_all_single_autograd and similar paths resolve by tag/name instead
    # of by object identity, so the tag has to move to the fresh inner PG.
    world = c10d._world
    stable_tag = world.pg_to_tag.get(wrapper)
    if stable_tag is None:
        stable_tag = f"ptd:{wrapper.group_name}"
    old_tag = world.pg_to_tag.get(inner)
    if old_tag is not None and old_tag in world.tags_to_pg:
        try:
            world.tags_to_pg[old_tag].remove(inner)
        except ValueError:
            pass
    world.pg_to_tag[inner] = stable_tag
    world.tags_to_pg.setdefault(stable_tag, [])
    if inner not in world.tags_to_pg[stable_tag]:
        world.tags_to_pg[stable_tag].append(inner)


def process_group_name(group: dist.ProcessGroup) -> str:
    try:
        return str(group.group_name)
    except RuntimeError:
        return "unnamed"
