import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field

from xtuner.v1.data_proto.rl_data import RolloutState, Status, update_group_status


@dataclass
class StorageIndices:
    # 为不同存储后段提供统一的索引接口
    task_name: str | None = None
    group_status: Status | None = None
    tags: dict = field(default_factory=dict)  # 非等于的条件则使用 scores_gt > 0.8


class Storage(ABC):
    @abstractmethod
    async def put(self, items: list[RolloutState], storage_indices: StorageIndices): ...
    @abstractmethod
    async def get(self, count: int, storage_indices: StorageIndices) -> list[list[RolloutState]]: ...
    @abstractmethod
    def count(self, storage_indices: StorageIndices) -> int: ...
    @abstractmethod
    def __len__(self): ...


class NaiveStorage(Storage):
    def __init__(self):
        self._storage = defaultdict(list)

    def _hash_storage_indices(self, indices: StorageIndices) -> tuple:
        base = (indices.task_name, indices.group_status)

        if indices.tags:
            sorted_tags = tuple(sorted(indices.tags.items()))
            return base + sorted_tags
        return base

    async def put(self, items: list[RolloutState], storage_indices: StorageIndices):
        indices = self._hash_storage_indices(storage_indices)
        self._storage[indices].append(items)

    async def get(self, count: int, storage_indices: StorageIndices) -> list[list[RolloutState]]:
        indices = self._hash_storage_indices(storage_indices)
        target_list = self._storage[indices]
        target_count = min(count, len(target_list))
        result = target_list[:target_count]
        self._storage[indices] = target_list[target_count:]
        return result

    def __len__(self):
        return sum(len(v) for v in self._storage.values())

    def count(self, storage_indices: StorageIndices) -> int:
        indices = self._hash_storage_indices(storage_indices)
        return len(self._storage[indices])


class FIFOBackend(NaiveStorage):
    # 普通的先进先出，用完就丢，不持久保存，目前同步应该就够用了
    def __init__(self, limit: int = 0):
        self.limit = limit
        self._storage = defaultdict(lambda: deque(maxlen=limit) if limit > 0 else deque())

    async def put(self, items: list[RolloutState], storage_indices: StorageIndices):
        await super().put(items, storage_indices)

    async def get(self, count: int, storage_indices: StorageIndices) -> list[list[RolloutState]]:
        indices = self._hash_storage_indices(storage_indices)
        target_count = min(count, len(self._storage[indices]))
        return [self._storage[indices].popleft() for _ in range(target_count)]


class StalenessBackend(NaiveStorage):
    # xtuner v1的异步的replay buffer的实现，同样不持久保存
    # TODO(@duanyanhui): 还没实现completed/aborted/expired状态的切换，这个考虑下在哪里完成
    def __init__(self, limit: int = 0, max_staleness: int = 0, min_staleness: int = 0):
        self.limit = limit
        self.max_staleness = max_staleness
        self.min_staleness = min_staleness
        self._storage = defaultdict(lambda: {i: deque() for i in range(min_staleness, max_staleness + 1)})
        self._bucket_counts: defaultdict[tuple, int] = defaultdict(int)

    async def put(self, items: list[RolloutState], storage_indices: StorageIndices):
        indices = self._hash_storage_indices(storage_indices)
        group_seq_staleness = max([item.seq_staleness for item in items])
        self._storage[indices][group_seq_staleness].append(items)
        self._bucket_counts[indices] += len(items)

    async def get(self, count: int, storage_indices: StorageIndices) -> list[list[RolloutState]]:
        indices = self._hash_storage_indices(storage_indices)
        if self._bucket_counts[indices] == 0:
            return []

        target_items = []
        needed = count

        for s in range(self.max_staleness, self.min_staleness - 1, -1):
            if needed <= 0:
                break
            cur_bucket = self._storage[indices][s]
            take = min(len(cur_bucket), needed)
            for _ in range(take):
                target_items.append(cur_bucket.popleft())
            self._bucket_counts[indices] -= take
            needed -= take
        return target_items

    def count(self, storage_indices: StorageIndices) -> int:
        indices = self._hash_storage_indices(storage_indices)
        total_len = 0
        for s in range(self.min_staleness, self.max_staleness + 1):
            total_len += len(self._storage[indices][s])
        return total_len

    def __len__(self):
        return sum(count for count in self._bucket_counts.values())


class ReplayBuffer:
    def __init__(self, storage_backend: Storage | None = None):
        self._storage = FIFOBackend() if storage_backend is None else storage_backend
        self._lock = asyncio.Lock()

    async def put(self, items: list[RolloutState], task_name: str, **kwargs) -> None:
        group_status = update_group_status(items)
        indices = StorageIndices(task_name=task_name, group_status=group_status, tags=kwargs)
        async with self._lock:
            await self._storage.put(items, indices)

    async def get(self, batch_size: int, task_name: str, group_status: Status, **kwargs) -> list[list[RolloutState]]:
        indices = StorageIndices(task_name=task_name, group_status=group_status, tags=kwargs)
        async with self._lock:
            return await self._storage.get(batch_size, indices)

    async def count(self, task_name: str, group_status: Status, **kwargs) -> int:
        indices = StorageIndices(task_name=task_name, group_status=group_status, tags=kwargs)
        async with self._lock:
            return self._storage.count(indices)
