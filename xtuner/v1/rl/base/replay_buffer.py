import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

from xtuner.v1.data_proto.rl_data import RolloutState, Status


@dataclass(frozen=True)
class StorageIndices:
    # 为不同存储后段提供统一的接口
    task_name: str | None = None
    group_status: Status | None = None

    def get_key(self):
        # 给用户留出重新定义索引的接口
        return (self.task_name, self.group_status)


class StorageBackend(ABC):
    @abstractmethod
    def put(self, items: list[RolloutState], storage_indices: StorageIndices): ...
    @abstractmethod
    def get(self, count: int, storage_indices: StorageIndices) -> list[RolloutState]: ...


class FIFOStorageBackend(StorageBackend):
    # 普通的先进先出，用完就丢，不持久保存，目前同步应该就够用了
    def __init__(self, limit: int = 0):
        self.limit = limit
        self._storage = defaultdict(list)

    def put(self, items: list[RolloutState], storage_indices: StorageIndices):
        indices = storage_indices.get_key()
        target_list = self._storage[indices]
        target_list.extend(items)
        if self.limit > 0 and len(target_list) > self.limit:
            self._storage[indices] = target_list[-self.limit :]

    def get(self, count: int, storage_indices: StorageIndices) -> list[RolloutState]:
        indices = storage_indices.get_key()
        target_count = min(count, len(self._storage[indices]))
        target_items = self._storage[indices][:target_count]
        self._storage[indices] = self._storage[indices][target_count:]
        return target_items


class StalenessStorageBackend(StorageBackend):
    # xtuner v1的异步的replay buffer的实现，同样不持久保存
    # TODO(@duanyanhui): 还没实现completed/aborted/expired状态的切换，这个考虑下在哪里完成
    def __init__(self, limit: int = 0, max_staleness: int = 0, min_staleness: int = 0):
        self.limit = limit
        self.max_staleness = max_staleness
        self.min_staleness = min_staleness
        self._storage = defaultdict(lambda: {i: [] for i in range(min_staleness, max_staleness + 1)})
        self._bucket_counts = defaultdict(int)

    def put(self, items: list[RolloutState], storage_indices: StorageIndices):
        indices = storage_indices.get_key()
        group_seq_staleness = max([item.seq_staleness for item in items])
        self._storage[indices][group_seq_staleness].extend(items)
        self._bucket_counts[indices] += len(items)

    def get(self, count: int, storage_indices: StorageIndices) -> list[RolloutState]:
        indices = storage_indices.get_key()
        if self._bucket_counts[indices] == 0:
            return []

        target_items = []
        for s in range(self.max_staleness, self.min_staleness - 1, -1):
            cur_bucket = self._storage[indices][s]
            needed = count - len(target_items)
            take = min(len(cur_bucket), needed)
            target_items.extend(cur_bucket[:take])
            self._storage[indices][s] = self._storage[indices][s][take:]
            self._bucket_counts[indices] -= take

            if len(target_items) >= count:
                break
        return target_items


class ReplayBuffer:
    def __init__(self, storage_backend: StorageBackend = None):
        self._storage = FIFOStorageBackend() if storage_backend is None else storage_backend
        self._lock = asyncio.Lock()

    async def put(self, items: list[RolloutState], task_name: str, group_status: Status):
        async with self._lock:
            self._storage.put(items, StorageIndices(task_name=task_name, group_status=group_status))

    async def get(self, batch_size: int, task_name: str, group_status: Status) -> list[RolloutState]:
        async with self._lock:
            return self._storage.get(batch_size, StorageIndices(task_name=task_name, group_status=group_status))
