from datetime import timedelta
from typing import Any, Optional, Union

from torch.distributed.distributed_c10d import (Backend, PrefixStore, Store,
                                                _new_process_group_helper,
                                                _world, default_pg_timeout,
                                                rendezvous)


# Adapted from https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py  # noqa: E501
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = '',
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (
        init_method is None), 'Cannot specify both init_method and store.'

    if store is not None:
        assert world_size > 0, 'world_size must be positive if using store'
        assert rank >= 0, 'rank must be non-negative if using store'
    elif init_method is None:
        init_method = 'env://'

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend('undefined')

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(
            init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    pg = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        pg_options=pg_options,
        timeout=timeout,
    )

    pg = pg[0] if isinstance(pg, tuple) else pg
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg
