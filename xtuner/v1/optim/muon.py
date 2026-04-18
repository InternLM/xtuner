# ================================================================
# Copyright (c) 2026 InternLM. All rights reserved.
#
# This file is developed based on and inspired by the following projects:
#
# 1. Muon Optimizer
#    Author: Keller Jordan
#    Blog: https://kellerjordan.github.io/posts/muon/
#
# 2. Moonlight
#    https://github.com/MoonshotAI/Moonlight
#    Licensed under the MIT License
#    Copyright (c) 2025 Moonshot AI
#
# 3. DION
#    https://github.com/microsoft/dion
#    Licensed under the MIT License
#    Copyright (c) 2025 Microsoft
#
# Portions of this code are adapted from the above projects.
# All rights and licenses of the original works are respected.
# ================================================================


import math
from collections import defaultdict
from itertools import chain, product
from typing import Callable, Generator, Iterator, Literal, Sequence, cast, overload

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.optim.optimizer import Optimizer, ParamsT

from xtuner.v1.utils.dtensor import group_tensors_by_device_mesh_and_placements


def batch_to_local(tensor: list[Tensor]) -> list[Tensor]:
    return [t.to_local() if isinstance(t, DTensor) else t for t in tensor]


def create_param_batches(params: Sequence[Tensor], batch_size: int) -> Generator[list[Tensor], None, None]:
    """Batch parameters into groups of size `batch_size`.

    Tensors in each batch will have identical shape, sharding, and dtype.
    """
    # Group parameters by shape, sharding, and dtype
    groups = defaultdict(list)
    for p in params:
        sharding = p.placements if isinstance(p, DTensor) else None
        groups[(p.shape, sharding, p.dtype)].append(p)

    # Create batches from grouped parameters
    for group in groups.values():
        for i in range(0, len(group), batch_size):
            batch = group[i : i + batch_size]
            yield batch


def pad_batch(batch: list[Tensor], batch_size: int) -> list[Tensor]:
    """Insert dummy tensors so the batch has exactly `batch_size` elements."""
    assert len(batch) > 0
    assert len(batch) <= batch_size
    while len(batch) < batch_size:
        batch.append(torch.zeros_like(batch[0]))
    return batch


class AsyncTask:
    """AsyncTask wraps a Python generator to run until the next yield
    statement.

    This is used to allow other tasks to run while waiting for distributed operations.
    """

    def __init__(self, generator: Generator[None, None, None]):
        self._generator = generator
        self.run()  # Start running the generator

    def run(self) -> bool:
        # Run the next step of the async task.
        # Returns True if the task is still running and False if completed.
        try:
            next(self._generator)
            return True
        except StopIteration:
            pass
        return False


class AsyncRuntime:
    """Event loop for running multiple async tasks concurrently."""

    def __init__(self, task_gen: Iterator["AsyncTask"], max_concurrent_tasks: int):
        # Initialize runtime with a generator that produces AsyncTask objects
        if max_concurrent_tasks <= 0:
            raise ValueError(f"{max_concurrent_tasks=} cannot be <= 0")
        self._task_gen = task_gen
        self._max_concurrent_tasks = max_concurrent_tasks

    def _get_next_task(self) -> "AsyncTask | None":
        try:
            task = next(self._task_gen)
            return task
        except StopIteration:
            return None

    def run(self):
        # Run the event loop until all tasks are completed
        have_new_tasks = True
        previous_tasks: list["AsyncTask"] = []

        while have_new_tasks or previous_tasks:
            # See if we can add another task
            running_tasks = []
            if have_new_tasks and len(previous_tasks) < self._max_concurrent_tasks:
                new_task = self._get_next_task()
                if new_task is not None:
                    # Add new task to the queue
                    running_tasks.append(new_task)
                else:
                    # No more tasks left
                    have_new_tasks = False

            # Run all previous tasks for one step
            for task in previous_tasks:
                still_running = task.run()
                if still_running:
                    running_tasks.append(task)

            # Update task list for next iteration
            previous_tasks = running_tasks


def adamw_update(
    X: Tensor,  # Model weights (modified in place)
    G: Tensor,  # Gradient
    M: Tensor,  # Momentum buffer (modified in place)
    V: Tensor,  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
):
    """AdamW optimizer algorithm."""
    assert X.shape == G.shape
    assert X.shape == M.shape

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    M.lerp_(G.to(M.dtype), 1 - beta1)
    # V = beta2 * V + (1 - beta2) * G * G
    V.mul_(beta2).addcmul_(G, G, value=1 - beta2)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = V.sqrt().div_(bias_correction2_sqrt).add_(epsilon)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    # Apply weight decay
    X.mul_(1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    X.addcdiv_(M, denom, value=-adj_lr)


def adamw_update_foreach(  # type: ignore
    X: list[Tensor],  # Model weights (modified in place)
    G: list[Tensor],  # Gradient
    M: list[Tensor],  # Momentum buffer (modified in place)
    V: list[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
):
    """AdamW optimizer algorithm (foreach implementation)."""
    batch_size = len(X)
    assert batch_size == len(G)
    assert batch_size == len(M)
    assert batch_size == len(V)

    M_dtype = M[0].dtype
    V_dtype = V[0].dtype

    # Update momentum and variance
    # M = beta1 * M + (1 - beta1) * G
    G = [g.to(dtype=M_dtype) for g in G]
    torch._foreach_lerp_(M, G, [1 - beta1] * batch_size)

    # V = beta2 * V + (1 - beta2) * G * G
    G_square = torch._foreach_mul(G, G)
    G_square = [g.to(dtype=V_dtype) for g in G_square]
    torch._foreach_lerp_(V, G_square, [1 - beta2] * batch_size)

    # Bias correction
    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step
    bias_correction2_sqrt = bias_correction2.sqrt()

    # The goal is to compute the following in-place:
    # M = M / bias_correction1
    # V = V / bias_correction2
    # X = X - lr * M / (sqrt(V) + epsilon)

    # Compute the denominator for the weight update
    # sqrt(V / bias_correction2) = sqrt(V) / sqrt(bias_correction2)
    denom = torch._foreach_sqrt(V)
    torch._foreach_div_(denom, bias_correction2_sqrt)
    torch._foreach_add_(denom, [epsilon] * batch_size)

    # Adjust learning rate to include bias correction 1
    adj_lr = lr / bias_correction1

    # Apply weight decay
    torch._foreach_mul_(X, 1 - lr * weight_decay)

    # Weight update
    # X = X - adj_lr * M / denom
    M_div = torch._foreach_div(M, denom)
    torch._foreach_mul_(M_div, adj_lr)
    torch._foreach_sub_(X, M_div)


class Muon(Optimizer):
    """Distributed Muon optimizer for PyTorch FSDP2.

    All parameters must be DTensors. The optimizer extracts device mesh and process group
    information directly from DTensor metadata.

    Args:
        params (ParamsT): Parameters for the optimizer. Can be a list of parameters or a list
            of parameter groups. Each parameter group can specify 'num_experts' to enable
            per-expert orthogonalization for MoE models.
        lr (float): Base learning rate. For Muon, this will be scaled based on the matrix dimensions.
            For element-wise update rules, this is the actual learning rate and no additional scaling is done.
        mu (float): Momentum factor for Muon algorithm.
        betas (tuple[float, float]): tuple of (beta1, beta2) for AdamW algorithms.
        weight_decay (float): Weight decay factor.
        epsilon (float): Small value to avoid division by zero.
        nesterov (bool): Whether to use Nesterov momentum.
        adjust_lr (str): How to adjust the learning rate for Muon updates ("spectral_norm", "rms_norm", or "none").
            "spectral_norm": Adjust based on spectral norm, for learning rate transfer across model scale.
            "rms_norm": Adjust based on RMS norm, for learning rate compatibility with Adam/AdamW.
            "none": Do not adjust the learning rate.
        flatten (bool): Whether to flatten 3D+ tensors to 2D for Muon updates.
            True: Tensors with 3+ dimensions are flattened to 2D. Use this for convolutional layers.
            False: Tensors are not flattened. 3D+ tensors are treated as batches of 2D matrices.
        use_triton (bool): Whether to use Triton kernel for Newton-Schulz. Ignored if custom function is provided.
        newton_schulz_func (Callable | None): Use a custom Newton-Schulz function for orthogonalization.
            Signature is `func(input: Tensor, epsilon: float, num_experts: int) -> Tensor`.

    Muon optimizer algorithm by Keller Jordan: https://kellerjordan.github.io/posts/muon/
    FSDP2 Muon uses all-to-all communications: https://www.essential.ai/blog/infra
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.01,
        epsilon: float = 1e-8,
        nesterov: bool = False,
        adjust_lr: Literal["rms_norm", "spectral_norm", "none"] = "rms_norm",
        flatten: bool = False,
        use_triton: bool = False,
        newton_schulz_func: Callable | None = None,
    ):
        # Check hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if mu < 0.0:
            raise ValueError(f"Invalid momentum factor (mu): {mu}")
        if len(betas) != 2 or betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError(f"Invalid betas: {betas}")
        if adjust_lr not in ("spectral_norm", "rms_norm", "none"):
            raise ValueError(f"Invalid adjust_lr value: {adjust_lr}. Must be 'spectral_norm', 'rms_norm', or 'none'.")

        # Default arguments for each param group
        defaults = dict(
            lr=lr,
            mu=mu,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            algorithm="muon",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            adjust_lr=adjust_lr,
            num_experts=1,  # Default: no MoE expert handling
        )
        super().__init__(params, defaults)

        # Pre-compute lr adjustment ratios for each Muon parameter based on global shape.
        # This must happen at init time because DTensor.shape here is guaranteed to be
        # the global (unsharded) shape, whereas inside the async task the shape may
        # reflect a local shard after distributed communication.
        for group in self.param_groups:
            if group["algorithm"] != "muon":
                continue
            adj = group["adjust_lr"]
            ne = group.get("num_experts", 1)
            for p in group["params"]:
                state = self.state[p]
                if adj == "none":
                    state["lr_ratio"] = 1.0
                elif adj == "spectral_norm":
                    fan_out = p.shape[-2] // ne
                    fan_in = p.shape[-1]
                    state["lr_ratio"] = math.sqrt(fan_out / fan_in)
                elif adj == "rms_norm":
                    A = p.shape[-2] // ne
                    B = p.shape[-1]
                    state["lr_ratio"] = 0.2 * math.sqrt(max(A, B))

        # Newton-Schulz configuration
        if newton_schulz_func is not None:
            if not callable(newton_schulz_func):
                raise TypeError(f"newton_schulz_func must be a callable function, got {type(newton_schulz_func)}")
            self._newton_schulz_func = newton_schulz_func
        elif use_triton:
            from .newton_schulz_triton import newton_schulz_triton

            self._newton_schulz_func = newton_schulz_triton
        else:
            self._newton_schulz_func = zeropower_via_newtonschulz5

        # Pre-create sub-group process groups for MoE sub-group all-gather optimization.
        # This must happen in __init__ so that all ranks call dist.new_group collectively.
        self._subgroup_cache: dict[tuple[int, int, int], tuple[ProcessGroup, int, int]] = {}
        self._init_moe_subgroups()

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_groups = []
        adamw_groups = []

        for group in self.param_groups:
            # Increment step
            group["step"] += 1

            # Split parameter groups by algorithm
            algo = group["algorithm"]
            if algo == "muon":
                muon_groups.append(group)
            elif algo == "adamw":
                adamw_groups.append(group)
            else:
                raise ValueError(f"Unknown algorithm: {algo}")

        # Create async tasks for each algorithm
        muon_tasks = self._create_muon_tasks(muon_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(muon_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        """Get optimizer state for the given parameter tensor, or lazy-
        initialize it if it doesn't exist."""
        state = self.state[param]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(param)
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    @staticmethod
    def _find_fsdp_mesh_dim(device_mesh: DeviceMesh) -> int | None:
        for dim, name in enumerate(device_mesh.mesh_dim_names or ()):
            if "fsdp" in name:
                return dim
        return None

    def _init_moe_subgroups(self) -> None:
        """Pre-create sub-group process groups for MoE parameters needing sub-group all-gather.

        When ``fsdp_size % ns_num_experts == 0``, each expert is split across
        ``fsdp_size // ns_num_experts`` FSDP ranks.  Instead of a padded all-to-all
        over all FSDP ranks, we can do a small all-gather within the sub-group that
        shares each expert, dramatically reducing communication volume.

        This method is called once in ``__init__`` so that all ranks participate in
        the collective ``dist.new_group`` calls together.
        """
        for group in self.param_groups:
            if group["algorithm"] != "muon":
                continue
            num_experts = group.get("num_experts", 1)
            if num_experts <= 1:
                continue

            mesh_groups = group_tensors_by_device_mesh_and_placements(group["params"])
            for (device_mesh, placements), _ in mesh_groups.items():
                shard_placements = [
                    (i, p) for i, p in enumerate(placements) if p.is_shard() and device_mesh.size(i) > 1
                ]
                if not shard_placements:
                    continue

                ns_num_experts = num_experts
                sharded_mesh_dim: int

                if len(shard_placements) == 1:
                    sharded_mesh_dim = shard_placements[0][0]
                elif len(shard_placements) == 2:
                    fsdp_mesh_dim = self._find_fsdp_mesh_dim(device_mesh)
                    if fsdp_mesh_dim is None:
                        continue
                    sharded_mesh_dim = fsdp_mesh_dim
                    sharded_tensor_dim = cast(Shard, placements[fsdp_mesh_dim]).dim
                    for i, p in shard_placements:
                        if i != fsdp_mesh_dim and cast(Shard, p).dim == sharded_tensor_dim:
                            assert ns_num_experts % device_mesh.size(i) == 0
                            ns_num_experts = ns_num_experts // device_mesh.size(i)
                else:
                    # >2 shard dims will be caught as error in _create_muon_tasks
                    continue

                fsdp_size = device_mesh.size(sharded_mesh_dim)

                # Case A: each rank holds complete experts → no sub-groups needed
                if ns_num_experts % fsdp_size == 0:
                    continue

                # Case B: sub-group all-gather
                if fsdp_size % ns_num_experts == 0:
                    subgroup_size = fsdp_size // ns_num_experts
                    self._create_subgroup_pgs(device_mesh, sharded_mesh_dim, subgroup_size)
                else:
                    raise ValueError(
                        f"Number of local experts ({ns_num_experts}) and FSDP size ({fsdp_size}) "
                        f"must satisfy either n_experts % fsdp_size == 0 or fsdp_size % n_experts == 0. "
                        f"Please adjust expert count or world size accordingly."
                    )

    def _create_subgroup_pgs(
        self,
        device_mesh: DeviceMesh,
        sharded_mesh_dim: int,
        subgroup_size: int,
    ) -> None:
        """Create and cache sub-group process groups for a given mesh dimension.

        Every rank in the world must participate in ``dist.new_group`` calls, so
        this iterates over *all* sub-groups across all non-FSDP dimension
        combinations and creates them collectively.
        """
        cache_key = (id(device_mesh), sharded_mesh_dim, subgroup_size)
        if cache_key in self._subgroup_cache:
            return

        fsdp_size = device_mesh.size(sharded_mesh_dim)
        fsdp_local_rank = device_mesh.get_local_rank(sharded_mesh_dim)
        mesh_tensor = device_mesh.mesh
        ndim = mesh_tensor.ndim

        other_dims = [d for d in range(ndim) if d != sharded_mesh_dim]
        other_dim_ranges = [range(mesh_tensor.size(d)) for d in other_dims]

        current_rank = dist.get_rank()
        my_pg: ProcessGroup | None = None

        for combo in product(*other_dim_ranges):
            idx: list[int | slice] = [slice(None)] * ndim
            for d, c in zip(other_dims, combo):
                idx[d] = c
            fsdp_ranks = mesh_tensor[tuple(idx)].tolist()

            for base in range(0, fsdp_size, subgroup_size):
                sg_ranks = fsdp_ranks[base : base + subgroup_size]
                pg = dist.new_group(sg_ranks)
                if current_rank in sg_ranks:
                    my_pg = pg

        assert my_pg is not None, f"Rank {current_rank} not found in any sub-group"
        subgroup_rank = fsdp_local_rank % subgroup_size
        self._subgroup_cache[cache_key] = (my_pg, subgroup_rank, subgroup_size)

    def _create_muon_tasks(
        self,
        param_groups: list[dict],
        algo_name: str = "muon",
    ) -> Generator["AsyncTask", None, None]:
        """Helper function to create batches of Muon matrices and generate
        AsyncTask objects so we can process multiple batches concurrently."""
        for group in param_groups:
            assert group["algorithm"] == algo_name
            assert all(p.ndim >= 2 for p in group["params"]), "Muon optimizer only supports matrix parameters."

            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            lr = torch.tensor(group["lr"])
            mu = torch.tensor(group["mu"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = torch.tensor(group["epsilon"])
            nesterov = group["nesterov"]
            flatten = group["flatten"]
            num_experts = group.get("num_experts", 1)

            # Process DTensor parameters grouped by (device_mesh, placements)
            mesh_groups = group_tensors_by_device_mesh_and_placements(group_params)
            for (device_mesh, placements), mesh_params in mesh_groups.items():
                # Extract communication primitives from the group's device_mesh
                # Find the sharded placement and get its mesh and tensor dimensions
                # Skip any Shard() placements on size-1 mesh dimension = Replicate()
                shard_placements = [
                    (i, p) for i, p in enumerate(placements) if p.is_shard() and device_mesh.size(i) > 1
                ]

                sharded_mesh_dim = None
                sharded_tensor_dim = None
                ns_num_experts = num_experts

                if len(shard_placements) == 1:
                    # Standard case: single shard dim (FSDP only, or FSDP+EP with Replicate on EP)
                    sharded_mesh_dim = shard_placements[0][0]
                    sharded_tensor_dim = cast(Shard, shard_placements[0][1]).dim
                elif len(shard_placements) > 1:
                    if len(shard_placements) > 2:
                        raise NotImplementedError(
                            f"Muon optimizer supports at most 2 active shard dimensions (EP + FSDP), "
                            f"got {len(shard_placements)}. Tensor parallelism is not yet supported."
                        )
                    # Multi-shard case: FSDP + EP for MoE params
                    # Identify FSDP mesh dim by name and use it for all-to-all
                    fsdp_mesh_dim = self._find_fsdp_mesh_dim(device_mesh)
                    if fsdp_mesh_dim is None:
                        raise RuntimeError(
                            "Could not identify FSDP mesh dimension for multi-shard DTensor. "
                            "Ensure the DTensor device mesh has a dimension named 'fsdp'."
                        )
                    fsdp_placement = placements[fsdp_mesh_dim]
                    sharded_mesh_dim = fsdp_mesh_dim
                    sharded_tensor_dim = cast(Shard, fsdp_placement).dim

                    # Newton-Schulz needs the LOCAL number of experts after EP sharding,
                    # so each block corresponds to one complete expert.
                    for i, p in shard_placements:
                        if i != fsdp_mesh_dim and cast(Shard, p).dim == sharded_tensor_dim:
                            assert ns_num_experts % device_mesh.size(i) == 0
                            ns_num_experts = ns_num_experts // device_mesh.size(i)

                # Optimization: if each FSDP rank already holds complete experts,
                # we can skip all-to-all and orthogonalize locally.
                skip_communication = False
                use_subgroup_allgather = False
                subgroup_process_group: ProcessGroup | None = None
                subgroup_size = 1
                subgroup_rank = 0

                if sharded_mesh_dim is not None and ns_num_experts > 1:
                    fsdp_size = device_mesh.size(sharded_mesh_dim)
                    if ns_num_experts % fsdp_size == 0:
                        # Case A: each rank holds complete experts → zero communication
                        ns_num_experts = ns_num_experts // fsdp_size
                        sharded_mesh_dim = None
                        sharded_tensor_dim = None
                        skip_communication = True
                    elif fsdp_size % ns_num_experts == 0:
                        if len(mesh_params) < fsdp_size:
                            # Case B: each expert spans a sub-group of ranks → sub-group all-gather
                            # Guard: only when params can't fill a batch, otherwise batched all-to-all
                            # is more efficient (fewer kernel launches, better bandwidth utilization).
                            sg_size = fsdp_size // ns_num_experts
                            ns_num_experts = 1  # After all-gather each rank has 1 complete expert
                            use_subgroup_allgather = True
                            cache_key = (id(device_mesh), sharded_mesh_dim, sg_size)
                            subgroup_process_group, subgroup_rank, subgroup_size = self._subgroup_cache[cache_key]
                        # else: fall through to all-to-all (enough params for full batch)
                    else:
                        raise ValueError(
                            f"Number of local experts ({ns_num_experts}) and FSDP size ({fsdp_size}) "
                            f"must satisfy either n_experts % fsdp_size == 0 or fsdp_size % n_experts == 0. "
                            f"Please adjust expert count or world size accordingly."
                        )

                # Helper to extract batch metadata for task creation
                def _batch_info(params: list[Tensor]) -> tuple[list[Tensor], list[Tensor], float]:
                    gradients: list[Tensor] = [g for p in params if (g := p.grad) is not None]
                    states = [self._get_or_initialize_state(p, algo_name) for p in params]
                    momentums = [s["momentum"] for s in states]
                    lr_ratios = [s["lr_ratio"] for s in states]
                    assert len(set(lr_ratios)) == 1, f"Found different lr_ratios: {set(lr_ratios)}"
                    return gradients, momentums, lr_ratios[0]

                if skip_communication or use_subgroup_allgather:
                    # Each rank processes its local experts independently, no all-to-all needed.
                    for params in create_param_batches(mesh_params, batch_size=1):
                        gradients, momentums, batch_lr_ratio = _batch_info(params)
                        yield AsyncTask(
                            muon_update_batch_async(
                                X=pad_batch(params, 1),
                                G=pad_batch(gradients, 1),
                                M=pad_batch(momentums, 1),
                                lr=lr,
                                lr_ratio=batch_lr_ratio,
                                momentum=mu,
                                weight_decay=weight_decay,
                                epsilon=epsilon,
                                nesterov=nesterov,
                                flatten=flatten,
                                device_rank=0,
                                world_size=1,
                                shard_dim=sharded_tensor_dim,
                                process_group=None,
                                newton_schulz_func=self._newton_schulz_func,
                                num_experts=ns_num_experts,
                                subgroup_process_group=subgroup_process_group,
                                subgroup_size=subgroup_size,
                                subgroup_rank=subgroup_rank,
                            )
                        )

                elif sharded_mesh_dim is not None and ns_num_experts <= 1:
                    # Non-expert sharded params: use all-to-all by default (padded when
                    # necessary). All-gather is only used for shape groups where the
                    # remainder is very small relative to fsdp_size — in that case the
                    # all-to-all padding overhead (zeros_like + wasted communication)
                    # exceeds the cost of redundant Newton-Schulz iterations.
                    #
                    # Decision rationale:
                    #   - All-to-all: each rank does 1 NS per batch (efficient compute),
                    #     but pads with (fsdp_size - remainder) dummy zeros.
                    #   - All-gather: each rank does `remainder` NS iterations (redundant
                    #     compute), but zero padding overhead.
                    #   All-gather only wins when `remainder` is tiny.
                    fsdp_pg = device_mesh.get_group(sharded_mesh_dim)
                    fsdp_size = device_mesh.size(sharded_mesh_dim)
                    fsdp_rank = device_mesh.get_local_rank(sharded_mesh_dim)

                    # Threshold: use all-gather only when the remainder is at most
                    # this many tensors.  Beyond this, the redundant NS compute from
                    # all-gather outweighs the padding savings.
                    allgather_threshold = max(1, fsdp_size // 16)

                    # Split params by shape group into routing categories
                    shape_groups: dict[tuple, list[Tensor]] = defaultdict(list)
                    for p in mesh_params:
                        sharding = p.placements if isinstance(p, DTensor) else None
                        shape_groups[(p.shape, sharding, p.dtype)].append(p)

                    all2all_exact_params: list[Tensor] = []  # Exact multiple → no padding
                    all2all_padded_params: list[Tensor] = []  # Large remainder → padded all2all
                    allgather_params: list[Tensor] = []  # Tiny remainder → all-gather
                    for group_list in shape_groups.values():
                        remainder = len(group_list) % fsdp_size
                        if remainder == 0:
                            all2all_exact_params.extend(group_list)
                        elif remainder <= allgather_threshold:
                            # Split: full batches to all2all (no padding), remainder to all-gather
                            full_batch_count = (len(group_list) // fsdp_size) * fsdp_size
                            all2all_exact_params.extend(group_list[:full_batch_count])
                            allgather_params.extend(group_list[full_batch_count:])
                        else:
                            all2all_padded_params.extend(group_list)

                    # All-to-all tasks — exact multiple batches (no padding)
                    for params in create_param_batches(all2all_exact_params, batch_size=fsdp_size):
                        gradients, momentums, batch_lr_ratio = _batch_info(params)
                        yield AsyncTask(
                            muon_update_batch_async(
                                X=params,
                                G=gradients,
                                M=momentums,
                                lr=lr,
                                lr_ratio=batch_lr_ratio,
                                momentum=mu,
                                weight_decay=weight_decay,
                                epsilon=epsilon,
                                nesterov=nesterov,
                                flatten=flatten,
                                device_rank=fsdp_rank,
                                world_size=fsdp_size,
                                shard_dim=sharded_tensor_dim,
                                process_group=fsdp_pg,
                                newton_schulz_func=self._newton_schulz_func,
                                num_experts=ns_num_experts,
                            )
                        )

                    # All-to-all tasks — padded batches (remainder too large for all-gather)
                    for params in create_param_batches(all2all_padded_params, batch_size=fsdp_size):
                        gradients, momentums, batch_lr_ratio = _batch_info(params)
                        yield AsyncTask(
                            muon_update_batch_async(
                                X=pad_batch(params, fsdp_size),
                                G=pad_batch(gradients, fsdp_size),
                                M=pad_batch(momentums, fsdp_size),
                                lr=lr,
                                lr_ratio=batch_lr_ratio,
                                momentum=mu,
                                weight_decay=weight_decay,
                                epsilon=epsilon,
                                nesterov=nesterov,
                                flatten=flatten,
                                device_rank=fsdp_rank,
                                world_size=fsdp_size,
                                shard_dim=sharded_tensor_dim,
                                process_group=fsdp_pg,
                                newton_schulz_func=self._newton_schulz_func,
                                num_experts=ns_num_experts,
                            )
                        )

                    # All-gather tasks (tiny remainder — redundant NS cost acceptable)
                    for params in create_param_batches(allgather_params, batch_size=fsdp_size):
                        gradients, momentums, batch_lr_ratio = _batch_info(params)
                        yield AsyncTask(
                            muon_update_allgather_async(
                                X=params,
                                G=gradients,
                                M=momentums,
                                lr=lr,
                                lr_ratio=batch_lr_ratio,
                                momentum=mu,
                                weight_decay=weight_decay,
                                epsilon=epsilon,
                                nesterov=nesterov,
                                flatten=flatten,
                                shard_dim=sharded_tensor_dim,
                                process_group=fsdp_pg,
                                world_size=fsdp_size,
                                local_rank=fsdp_rank,
                                newton_schulz_func=self._newton_schulz_func,
                                num_experts=ns_num_experts,
                            )
                        )

                elif sharded_mesh_dim is not None:
                    # Expert params going through all-to-all (Case B fallthrough:
                    # fsdp_size % ns_num_experts == 0 but enough params for full batch)
                    group_process_group = device_mesh.get_group(sharded_mesh_dim)
                    group_world_size = device_mesh.size(sharded_mesh_dim)
                    group_device_rank = device_mesh.get_local_rank(sharded_mesh_dim)

                    for params in create_param_batches(mesh_params, batch_size=group_world_size):
                        gradients, momentums, batch_lr_ratio = _batch_info(params)
                        yield AsyncTask(
                            muon_update_batch_async(
                                X=pad_batch(params, group_world_size),
                                G=pad_batch(gradients, group_world_size),
                                M=pad_batch(momentums, group_world_size),
                                lr=lr,
                                lr_ratio=batch_lr_ratio,
                                momentum=mu,
                                weight_decay=weight_decay,
                                epsilon=epsilon,
                                nesterov=nesterov,
                                flatten=flatten,
                                device_rank=group_device_rank,
                                world_size=group_world_size,
                                shard_dim=sharded_tensor_dim,
                                process_group=group_process_group,
                                newton_schulz_func=self._newton_schulz_func,
                                num_experts=ns_num_experts,
                            )
                        )

                else:
                    # Not sharded on any active mesh dim; fall back to FSDP mesh dim
                    fsdp_dim = self._find_fsdp_mesh_dim(device_mesh)
                    if fsdp_dim is not None:
                        group_process_group = device_mesh.get_group(fsdp_dim)
                        group_world_size = device_mesh.size(fsdp_dim)
                        group_device_rank = device_mesh.get_local_rank(fsdp_dim)
                    else:
                        group_process_group = None
                        group_world_size = 1
                        group_device_rank = 0

                    for params in create_param_batches(mesh_params, batch_size=group_world_size):
                        gradients, momentums, batch_lr_ratio = _batch_info(params)
                        yield AsyncTask(
                            muon_update_batch_async(
                                X=pad_batch(params, group_world_size),
                                G=pad_batch(gradients, group_world_size),
                                M=pad_batch(momentums, group_world_size),
                                lr=lr,
                                lr_ratio=batch_lr_ratio,
                                momentum=mu,
                                weight_decay=weight_decay,
                                epsilon=epsilon,
                                nesterov=nesterov,
                                flatten=flatten,
                                device_rank=group_device_rank,
                                world_size=group_world_size,
                                shard_dim=sharded_tensor_dim,
                                process_group=group_process_group,
                                newton_schulz_func=self._newton_schulz_func,
                                num_experts=ns_num_experts,
                            )
                        )

    def _create_adamw_tasks(
        self,
        param_groups: list[dict],
        algo_name: str = "adamw",
    ) -> Generator["AsyncTask", None, None]:
        """Helper function to generate AsyncTask objects for AdamW updates."""
        for group in param_groups:
            assert group["algorithm"] == algo_name

            # Get parameters and optimizer states
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, algo_name) for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            lr = torch.tensor(group["lr"])
            beta1 = torch.tensor(group["beta1"])
            beta2 = torch.tensor(group["beta2"])
            weight_decay = torch.tensor(group["weight_decay"])
            epsilon = group["epsilon"]  # Keep as float
            step = group["step"]  # Keep as int

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=batch_to_local(params),
                    G=batch_to_local(gradients),
                    M=batch_to_local(momentums),
                    V=batch_to_local(variances),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    weight_decay=weight_decay,
                    step=step,
                    epsilon=epsilon,
                )
            )


def muon_update_batch_async(
    X: list[Tensor],  # Model weights (modified in place)
    G: list[Tensor],  # Gradient
    M: list[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    lr_ratio: float,  # Pre-computed lr adjustment ratio
    momentum: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    nesterov: bool,  # Whether to use Nesterov momentum
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    device_rank: int,  # Rank of the current device
    world_size: int,  # Total number of devices to parallelize over
    newton_schulz_func: Callable,  # Newton-Schulz function for orthogonalization
    shard_dim: int | None = None,  # Shard dimension for DTensor (if applicable)
    process_group: ProcessGroup | None = None,
    num_experts: int = 1,  # Number of experts for MoE models
    subgroup_process_group: ProcessGroup | None = None,  # Sub-group PG for MoE all-gather
    subgroup_size: int = 1,  # Number of ranks in the sub-group
    subgroup_rank: int = 0,  # This rank's position within the sub-group
) -> Generator[None, None, None]:
    """Batched version of Muon update.

    Batch size should be equal to number of GPUs. All tensors in a batch should have identical shape, sharding, and
    dtype. Identical hyperparameters are used for all tensors in the batch.
    """

    assert len(X) == len(G)
    assert len(X) == len(M)
    assert len(X) == world_size

    # Update momentum and compute the inputs for orthogonalization
    U = muon_update_pre_orthogonalize(
        G=batch_to_local(G),
        M=batch_to_local(M),
        momentum=momentum,
        nesterov=nesterov,
    )

    # Get one whole matrix for each device to orthogonalize
    if shard_dim is not None:
        assert isinstance(X[0], DTensor), "X should contain DTensors"
        assert not isinstance(U[0], DTensor), "U should contain local shards"

        if subgroup_process_group is not None:
            # Sub-group all-gather path: reconstruct a complete expert from a small
            # group of FSDP ranks that together hold all shards of one expert, then
            # orthogonalize locally and slice back.  No global all-to-all needed.
            assert world_size == 1, "Sub-group all-gather expects world_size=1 (no batch padding)"
            local_shard = U[0]

            # All-gather within sub-group to reconstruct full expert
            gathered = torch.empty(
                (subgroup_size,) + local_shard.shape,
                dtype=local_shard.dtype,
                device=local_shard.device,
            )
            work = dist.all_gather_into_tensor(
                gathered, local_shard.contiguous(), group=subgroup_process_group, async_op=True
            )
            yield
            work.wait()  # type: ignore[union-attr]

            # Reconstruct full expert along shard_dim
            if shard_dim == 0:
                full_expert = gathered.flatten(0, 1)
            else:
                full_expert = torch.cat(gathered.unbind(0), dim=shard_dim)

            # Orthogonalize (num_experts=1: we reconstructed exactly one expert)
            full_expert = muon_update_newton_schulz(
                full_expert,
                newton_schulz_func=newton_schulz_func,
                flatten=flatten,
                epsilon=epsilon,
                num_experts=num_experts,
            )

            # Slice back to local shard — no reverse communication needed
            shard_size = full_expert.size(shard_dim) // subgroup_size
            U[0] = full_expert.narrow(shard_dim, subgroup_rank * shard_size, shard_size)

        else:
            # All-to-all paths: use all-to-all to transform from a batch of shards
            # to a single whole matrix per rank.
            # https://www.essential.ai/blog/infra
            assert process_group is not None, "process_group must be provided for sharded DTensors"

            global_shard_dim_size = X[0].size(shard_dim)

            if global_shard_dim_size >= world_size and global_shard_dim_size % world_size == 0:
                # Standard path: use all-to-all for evenly sharded tensors
                # Pack the list of shards into a single contiguous tensor
                # U is currently [Shard_for_Rank0, Shard_for_Rank1, ...]
                # Stack creates shape: (World_Size, *Shard_Shape)
                U_packed = torch.stack(U)

                # Allocate buffer to receive parts of the "Single Matrix"
                # Shape: (World_Size, *Shard_Shape)
                single_matrix_parts = torch.empty_like(U_packed)

                # Perform optimized All-to-All
                # This sends one large contiguous buffer instead of many small ones
                work = dist.all_to_all_single(single_matrix_parts, U_packed, group=process_group, async_op=True)
                yield
                work.wait()  # type: ignore[union-attr]

                # Reconstruct the full matrix
                # single_matrix_parts has shape (World_Size, D0, D1...)
                if shard_dim == 0:
                    # Optimization: If sharded on dim 0, we can simply flatten the batch dim
                    # to reconstruct the full matrix. This is a Zero-Copy View.
                    single_matrix = single_matrix_parts.flatten(0, 1)
                else:
                    # General case (e.g., Col-wise sharding): We must concatenate along shard_dim.
                    # This requires a memory copy.
                    single_matrix = torch.cat(single_matrix_parts.unbind(0), dim=shard_dim)

                # 5. Perform Newton-Schulz Orthogonalization
                single_matrix = muon_update_newton_schulz(
                    single_matrix,
                    newton_schulz_func=newton_schulz_func,
                    flatten=flatten,
                    epsilon=epsilon,
                    num_experts=num_experts,
                )

                # Prepare to scatter results back
                if shard_dim == 0:
                    # Optimization: View back to (World_Size, Shard_Size, ...)
                    # This is a Zero-Copy View.
                    single_matrix_shards_packed = single_matrix.view(world_size, -1, *single_matrix.shape[1:])
                else:
                    # General case: Split back into chunks and stack them.
                    # We use stack to ensure the output is contiguous (World_Size, ...) for NCCL
                    single_matrix_shards_packed = torch.stack(single_matrix.chunk(world_size, dim=shard_dim))

                # Ensure contiguity is preserved (crucial for NCCL)
                if not single_matrix_shards_packed.is_contiguous():
                    single_matrix_shards_packed = single_matrix_shards_packed.contiguous()

                # Allocate buffer for receiving updated gradients
                U_packed_back = torch.empty_like(single_matrix_shards_packed)

                # Perform optimized All-to-All (Scatter back)
                work = dist.all_to_all_single(
                    U_packed_back, single_matrix_shards_packed, group=process_group, async_op=True
                )
                yield
                work.wait()  # type: ignore[union-attr]

                # Unpack back to list form for the post-processing function
                # unbind(0) is a view operation (slicing)
                U = list(U_packed_back.unbind(0))

            else:
                # Uneven sharding path: use all-to-all with padding to handle uneven shards.
                # Each rank still only orthogonalizes one matrix (no redundant computation).

                # Calculate padded shard size (ceil division) so all ranks have same-sized tensors
                padded_shard_size = (global_shard_dim_size + world_size - 1) // world_size

                # Compute true local sizes for each rank using DTensor's sharding logic
                local_sizes = []
                for r in range(world_size):
                    start = r * padded_shard_size
                    end = min((r + 1) * padded_shard_size, global_shard_dim_size)
                    local_sizes.append(max(0, end - start))

                # Pad all local shards to the same size for uniform all-to-all
                U_padded = []
                for u in U:
                    current_size = u.size(shard_dim)
                    if current_size < padded_shard_size:
                        pad_size = padded_shard_size - current_size
                        pad_shape = list(u.shape)
                        pad_shape[shard_dim] = pad_size
                        padding = torch.zeros(pad_shape, dtype=u.dtype, device=u.device)
                        u_padded = torch.cat([u, padding], dim=shard_dim)
                    else:
                        u_padded = u
                    U_padded.append(u_padded)

                # Stack into single tensor: (world_size, padded_shard_size, ...)
                U_packed = torch.stack(U_padded)
                single_matrix_parts = torch.empty_like(U_packed)

                # All-to-all: each rank sends its shard of each matrix, receives all shards of one matrix
                work = dist.all_to_all_single(single_matrix_parts, U_packed, group=process_group, async_op=True)
                yield
                work.wait()  # type: ignore[union-attr]

                # Reconstruct the full matrix by unpadding and concatenating
                if shard_dim == 0:
                    # Flatten then unpad: total padded size = world_size * padded_shard_size
                    single_matrix = single_matrix_parts.flatten(0, 1).narrow(0, 0, global_shard_dim_size)
                else:
                    # General case: unpad each shard then concatenate
                    shards = []
                    for r in range(world_size):
                        true_size = local_sizes[r]
                        if true_size > 0:
                            shards.append(single_matrix_parts[r].narrow(shard_dim, 0, true_size))
                    single_matrix = torch.cat(shards, dim=shard_dim)

                # Orthogonalize
                single_matrix = muon_update_newton_schulz(
                    single_matrix,
                    newton_schulz_func=newton_schulz_func,
                    flatten=flatten,
                    epsilon=epsilon,
                    num_experts=num_experts,
                )

                # Split back into padded shards for all-to-all scatter
                if shard_dim == 0:
                    # Pad back to world_size * padded_shard_size, then view
                    pad_total = world_size * padded_shard_size - global_shard_dim_size
                    if pad_total > 0:
                        pad_shape = list(single_matrix.shape)
                        pad_shape[0] = pad_total
                        padding = torch.zeros(pad_shape, dtype=single_matrix.dtype, device=single_matrix.device)
                        single_matrix_padded = torch.cat([single_matrix, padding], dim=0)
                    else:
                        single_matrix_padded = single_matrix
                    single_matrix_shards_packed = single_matrix_padded.view(
                        world_size, padded_shard_size, *single_matrix.shape[1:]
                    )
                else:
                    # General case: split and pad each shard
                    shards_padded = []
                    offset = 0
                    for r in range(world_size):
                        true_size = local_sizes[r]
                        if true_size > 0:
                            shard = single_matrix.narrow(shard_dim, offset, true_size)
                            offset += true_size
                            # Pad to padded_shard_size if needed
                            if true_size < padded_shard_size:
                                pad_shape = list(shard.shape)
                                pad_shape[shard_dim] = padded_shard_size - true_size
                                padding = torch.zeros(pad_shape, dtype=shard.dtype, device=shard.device)
                                shard = torch.cat([shard, padding], dim=shard_dim)
                        else:
                            # Create zero-padded shard
                            shape = list(single_matrix.shape)
                            shape[shard_dim] = padded_shard_size
                            shard = torch.zeros(shape, dtype=single_matrix.dtype, device=single_matrix.device)
                        shards_padded.append(shard)
                    single_matrix_shards_packed = torch.stack(shards_padded)

                if not single_matrix_shards_packed.is_contiguous():
                    single_matrix_shards_packed = single_matrix_shards_packed.contiguous()

                U_packed_back = torch.empty_like(single_matrix_shards_packed)

                # All-to-all scatter back
                work = dist.all_to_all_single(
                    U_packed_back, single_matrix_shards_packed, group=process_group, async_op=True
                )
                yield
                work.wait()  # type: ignore[union-attr]

                # Unpad and unpack to list form
                my_size = local_sizes[device_rank]
                U = []
                for i in range(world_size):
                    shard = U_packed_back[i]
                    if my_size > 0 and my_size < padded_shard_size:
                        shard = shard.narrow(shard_dim, 0, my_size)
                    elif my_size == 0:
                        shape = list(shard.shape)
                        shape[shard_dim] = 0
                        shard = torch.empty(shape, dtype=shard.dtype, device=shard.device)
                    U.append(shard)

    else:
        # Matrices are not sharded, so we can directly orthogonalize
        # Get a single matrix corresponding to this device
        single_matrix = U[device_rank]
        assert not isinstance(single_matrix, DTensor)

        single_matrix = muon_update_newton_schulz(
            single_matrix,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
            num_experts=num_experts,
        )

        if process_group is not None and process_group.size() > 1:
            # Ensure input is contiguous
            input_tensor = single_matrix.contiguous()

            # Create output buffer
            gathered_U = torch.empty(
                (world_size,) + input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device
            )

            # Use efficient all_gather
            work = dist.all_gather_into_tensor(gathered_U, input_tensor, group=process_group, async_op=True)
            yield
            work.wait()  # type: ignore[union-attr]

            # Unbind to list for compatibility
            U = list(gathered_U.unbind(0))

        else:
            # Single GPU case, no need to gather
            assert world_size == 1
            U = [single_matrix]

    # Compute adjusted learning rate
    adjusted_lr = lr * lr_ratio

    # Update model parameters with orthogonalized output
    muon_update_post_orthogonalize(
        X=batch_to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


def muon_update_allgather_async(
    X: list[Tensor],  # Model weights (DTensors, modified in place)
    G: list[Tensor],  # Gradient
    M: list[Tensor],  # Momentum buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    lr_ratio: float,  # Pre-computed lr adjustment ratio
    momentum: Tensor,  # Momentum factor (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    epsilon: Tensor,  # Epsilon (scalar tensor)
    nesterov: bool,  # Whether to use Nesterov momentum
    flatten: bool,  # Whether to flatten 3D+ tensors to 2D
    shard_dim: int,  # Shard dimension for DTensor
    process_group: ProcessGroup,  # FSDP process group
    world_size: int,  # Number of FSDP ranks
    local_rank: int,  # This rank's position in FSDP group
    newton_schulz_func: Callable,  # Newton-Schulz function for orthogonalization
    num_experts: int = 1,  # Number of experts for MoE models
) -> Generator[None, None, None]:
    """Muon update using all-gather communication for sharded parameters.

    Unlike the all-to-all path, this gathers each parameter independently from all
    FSDP ranks, performs Newton-Schulz orthogonalization on the full matrix, then
    slices back to the local shard. This avoids the padding overhead of all-to-all
    when the number of same-shape parameters is not a multiple of the FSDP world size.
    """
    assert len(X) == len(G)
    assert len(X) == len(M)

    # Update momentum and compute inputs for orthogonalization
    U = muon_update_pre_orthogonalize(
        G=batch_to_local(G),
        M=batch_to_local(M),
        momentum=momentum,
        nesterov=nesterov,
    )

    # Phase 1: All-gather all tensors, reconstructing full matrices
    full_matrices: list[Tensor] = []
    global_dim_sizes: list[int] = []
    max_shard_sizes: list[int] = []

    for i in range(len(U)):
        local_shard = U[i]

        # Get the global shard dimension size from the DTensor to handle uneven
        # FSDP sharding (e.g., gate.weight with dim 0 size < world_size).
        # all_gather_into_tensor requires uniform input sizes across ranks, so
        # we pad local shards to ceil(global_size / world_size) when needed.
        assert isinstance(X[i], DTensor)
        global_dim_size = X[i].shape[shard_dim]
        max_shard_size = (global_dim_size + world_size - 1) // world_size
        local_size = local_shard.size(shard_dim)

        global_dim_sizes.append(global_dim_size)
        max_shard_sizes.append(max_shard_size)

        # Pad local shard to uniform size for NCCL all-gather
        if local_size < max_shard_size:
            pad_shape = list(local_shard.shape)
            pad_shape[shard_dim] = max_shard_size - local_size
            local_shard = torch.cat(
                [local_shard, torch.zeros(pad_shape, dtype=local_shard.dtype, device=local_shard.device)],
                dim=shard_dim,
            )

        # All-gather from all FSDP ranks to reconstruct the full parameter
        gathered = torch.empty(
            (world_size,) + local_shard.shape,
            dtype=local_shard.dtype,
            device=local_shard.device,
        )
        work = dist.all_gather_into_tensor(gathered, local_shard.contiguous(), group=process_group, async_op=True)
        yield
        work.wait()  # type: ignore[union-attr]

        # Reconstruct full matrix along shard_dim, trimming any padding
        if shard_dim == 0:
            full_matrix = gathered.flatten(0, 1).narrow(0, 0, global_dim_size)
        else:
            shards = []
            remaining = global_dim_size
            for r in range(world_size):
                true_size = min(max_shard_size, remaining)
                if true_size > 0:
                    shards.append(gathered[r].narrow(shard_dim, 0, true_size))
                    remaining -= true_size
            full_matrix = torch.cat(shards, dim=shard_dim)

        full_matrices.append(full_matrix)

    # Phase 2: Batched Newton-Schulz — stack all R matrices and run NS once
    # with num_experts=R, leveraging batch matmul instead of R separate calls.
    # All tensors in a batch have identical shape (grouped by create_param_batches),
    # so stacking is safe.
    R = len(full_matrices)
    if R > 0:
        # Stack along dim 0: (R, *shape) then flatten to (R*M, N) for NS batch processing
        stacked = torch.stack(full_matrices, dim=0)  # (R, M, N) or (R, dim0, dim1)
        stacked_shape = stacked.shape
        # Reshape to 2D-like for muon_update_newton_schulz: (R*M, N)
        stacked_2d = stacked.reshape(-1, stacked_shape[-1])

        stacked_2d = muon_update_newton_schulz(
            stacked_2d,
            newton_schulz_func=newton_schulz_func,
            flatten=flatten,
            epsilon=epsilon,
            num_experts=R * num_experts,
        )

        # Reshape back to (R, M, N) and split
        stacked = stacked_2d.reshape(stacked_shape)
        full_matrices = list(stacked.unbind(0))

    # Phase 3: Slice back to local shards — no reverse communication needed
    for i in range(len(U)):
        full_matrix = full_matrices[i]
        local_offset = local_rank * max_shard_sizes[i]
        local_true_size = min(max_shard_sizes[i], max(0, global_dim_sizes[i] - local_offset))
        if local_true_size > 0:
            U[i] = full_matrix.narrow(shard_dim, local_offset, local_true_size)
        else:
            out_shape = list(full_matrix.shape)
            out_shape[shard_dim] = 0
            U[i] = torch.empty(out_shape, dtype=full_matrix.dtype, device=full_matrix.device)

    # Compute adjusted learning rate
    adjusted_lr = lr * lr_ratio

    # Update model parameters with orthogonalized output
    muon_update_post_orthogonalize(
        X=batch_to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


def adamw_update_foreach_async(
    X: list[Tensor],  # Model weights (modified in place)
    G: list[Tensor],  # Gradient
    M: list[Tensor],  # Momentum buffer (modified in place)
    V: list[Tensor],  # Variance buffer (modified in place)
    lr: Tensor,  # Learning rate (scalar tensor)
    beta1: Tensor,  # Beta 1 (scalar tensor)
    beta2: Tensor,  # Beta 2 (scalar tensor)
    weight_decay: Tensor,  # Weight decay (scalar tensor)
    step: int,
    epsilon: float,
) -> Generator[None, None, None]:
    """Async wrapper around foreach AdamW update."""
    adamw_update_foreach(X, G, M, V, lr, beta1, beta2, weight_decay, step, epsilon)
    yield


def muon_update_pre_orthogonalize(
    G: list[Tensor],
    M: list[Tensor],
    momentum: Tensor,
    nesterov: bool,
) -> list[Tensor]:
    """Update momentum with gradient and compute the input to
    orthogonalization."""
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    # Update momentum with new gradient
    torch._foreach_mul_(M, momentum)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, momentum)
        torch._foreach_add_(U, G)
    else:
        U = M

    # Convert to bfloat16 before communication
    U = [u.to(dtype=torch.bfloat16) for u in U]

    return U


def muon_update_post_orthogonalize(
    X: list[Tensor],
    U: list[Tensor],
    base_lr: Tensor,
    adjusted_lr: Tensor,
    weight_decay: Tensor,
):
    """Apply weight decay and weight update after orthogonalization."""
    # Apply weight decay
    torch._foreach_mul_(X, 1 - base_lr * weight_decay)

    # Weight update
    torch._foreach_mul_(U, adjusted_lr)
    torch._foreach_sub_(X, U)


def muon_update_newton_schulz(
    X: Tensor,
    newton_schulz_func: Callable,
    flatten: bool,
    epsilon: Tensor,
    num_experts: int = 1,
) -> Tensor:
    """Flatten the input tensor if needed and call the Newton-Schulz
    function."""
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        # Flatten 3D+ tensors to 2D matrix
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        # Given 4D+ batch, flatten to 3D batch
        X = X.flatten(end_dim=-3)

    return newton_schulz_func(X, epsilon=epsilon, num_experts=num_experts).reshape(original_shape)


def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7, num_experts: int = 1):
    """Newton-Schulz iteration to approximate the orthogonalization of X.

    This function handles both regular matrices and MoE expert weight matrices.
    For MoE models, each expert's weight matrix is orthogonalized independently,
    rather than orthogonalizing the concatenated large matrix.

    Unified algorithm for both cases:
    1. Reshape input to (num_experts, M, N) - for regular case this is (1, M, N)
    2. Apply Newton-Schulz iteration to each expert matrix independently using
       batch matrix multiplication
    3. Reshape back to original shape

    Mathematical equivalence:
    - num_experts=1:  X.view(1, M, N) -> process -> X.view(M, N)
      This is mathematically equivalent to processing X directly, but allows
      unified code path with the MoE case.
    - num_experts>1:  X.view(num_experts, M, N) -> process each expert -> X.view(num_experts*M, N)
      Each expert matrix is orthogonalized independently with its own spectral norm.

    Args:
        G: Input tensor to orthogonalize. Shape: (num_experts * M, N) for MoE,
           or (M, N) for regular matrices.
        epsilon: Small value to avoid division by zero.
        num_experts: Number of experts for MoE models. Default 1 for regular matrices.
            When > 1, the input is treated as concatenated expert matrices.
    """
    # Newton-Schulz constants - fixed coefficients for 5th order iteration
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    original_shape = X.shape

    # Unified handling: reshape to (num_experts, M, N) for both cases
    # For regular case (num_experts=1), this adds a batch dimension of size 1
    N = X.size(-1)
    X = X.view(num_experts, -1, N)

    # Transpose if needed (when rows > cols) for numerical stability in NS iteration
    # This ensures X @ X.mT produces a smaller square matrix
    need_transpose = X.size(-2) > X.size(-1)
    if need_transpose:
        X = X.mT  # (num_experts, N, M) if rows > cols, else (num_experts, M, N)

    # Ensure spectral norm is at most 1 for each expert matrix independently
    # norm shape: (num_experts, 1, 1) - each expert has its own normalization factor
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    # Newton-Schulz iteration: orthogonalize each expert matrix
    # Using batch matrix multiplication (@) to process all experts in parallel
    for a, b, c in ns_consts:
        # A = X @ X^T: compute Gram matrix for each expert
        A = X @ X.mT  # shape: (num_experts, M, M) or (num_experts, N, N)
        # B = b * A + c * A @ A: polynomial combination for convergence
        B = b * A + c * (A @ A)
        # X = a * X + B @ X: update step
        X = a * X + B @ X

    # Undo transpose if applied
    if need_transpose:
        X = X.mT

    # Reshape back to original shape: (num_experts, M, N) -> (num_experts * M, N)
    X = X.view(original_shape)

    return X
