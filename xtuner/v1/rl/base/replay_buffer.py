import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from itertools import count
from typing import List

from pydantic import BaseModel

from xtuner.v1.data_proto.rl_data import RolloutState, Status, update_group_status
from xtuner.v1.rl.utils import DSLRule, DSLRuleType


@dataclass
class StorageItem:
    # 存储类型
    item: List[RolloutState]
    uid: int
    timestamp_id: int
    task_name: str
    status: Status
    staleness: int


@dataclass
class QueryItem:
    # 查询类型
    task_name: DSLRuleType | str | None = None  # e.g. {"$eq": "math"}
    status: DSLRuleType | Status | None = None  # e.g. {"$eq": "completed"}
    staleness: DSLRuleType | int | None = None  # e.g. {"$between": [1, 5]}

    def match_storage(self, record: StorageItem) -> bool:
        if self.task_name is not None and not DSLRule.match(record.task_name, self.task_name):
            return False
        if self.status is not None and not DSLRule.match(record.status, self.status):
            return False
        if self.staleness is not None and not DSLRule.match(record.staleness, self.staleness):
            return False
        return True


class StorageBackend(ABC):
    @abstractmethod
    async def put(self, item: StorageItem) -> int: ...

    @abstractmethod
    async def get(self, query: QueryItem) -> List[StorageItem]: ...

    @abstractmethod
    async def delete(self, uids: list[int]) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...


class ReplayPolicy(ABC):
    @abstractmethod
    async def put(self, item: StorageItem, storage_backend: StorageBackend) -> None: ...

    @abstractmethod
    async def get(self, count: int, query: QueryItem, storage_backend: StorageBackend) -> list[list[RolloutState]]: ...

    async def count(self, query: QueryItem, storage_backend: StorageBackend) -> int:
        return len(await storage_backend.get(query))


class NaiveStorage(StorageBackend):
    def __init__(self):
        self._uid_gen = count(1)
        self._timestamp_id_gen = count(1)
        self._items: list[StorageItem] = []

    async def put(self, item: StorageItem) -> int:
        uid = next(self._uid_gen)
        timestamp_id = next(self._timestamp_id_gen)
        stored = replace(item, uid=uid, timestamp_id=timestamp_id)
        self._items.append(stored)
        return uid

    async def get(self, query: QueryItem) -> list[StorageItem]:
        return [record for record in self._items if query.match_storage(record)]

    async def delete(self, uids: list[int]) -> None:
        if not uids:
            return
        id_set = set(uids)
        self._items = [record for record in self._items if record.uid not in id_set]

    def __len__(self) -> int:
        return len(self._items)


class FIFOBackend(ReplayPolicy):
    async def put(self, item: StorageItem, storage_backend: StorageBackend) -> None:
        if not item.item:
            return
        await storage_backend.put(item)

    async def get(self, count: int, query: QueryItem, storage_backend: StorageBackend) -> list[list[RolloutState]]:
        records = await storage_backend.get(query)
        records.sort(key=lambda r: r.timestamp_id)
        selected = records[:count]
        await storage_backend.delete([record.uid for record in selected])
        return [record.item for record in selected]


class StalenessBackend(ReplayPolicy):
    def __init__(self, max_staleness: int = 0, min_staleness: int = 0):
        self.max_staleness = max_staleness
        self.min_staleness = min_staleness

    async def put(self, item: StorageItem, storage_backend: StorageBackend) -> None:
        if not item.item:
            return
        await storage_backend.put(item)

    async def get(self, count: int, query: QueryItem, storage_backend: StorageBackend) -> list[list[RolloutState]]:
        # TODO: 目前get性能较差，测试异步功能时再优化
        records = await storage_backend.get(query)
        records = [r for r in records if self.min_staleness <= r.staleness <= self.max_staleness]
        records.sort(key=lambda r: (-r.staleness, r.timestamp_id))
        selected = records[:count]
        await storage_backend.delete([record.uid for record in selected])
        return [record.item for record in selected]

    async def count(self, query: QueryItem, storage_backend: StorageBackend) -> int:
        records = await storage_backend.get(query)
        return sum(1 for record in records if self.min_staleness <= record.staleness <= self.max_staleness)


class ReplayBuffer:
    def __init__(
        self,
        policy: ReplayPolicy,
        storage_backend: StorageBackend,
    ):
        self._policy = policy
        self._storage = storage_backend
        self._lock = asyncio.Lock()

    async def put(self, items: list[RolloutState], task_name: str) -> None:
        if not items:
            return
        storage_item = StorageItem(
            item=items,
            uid=0,  # 占位
            timestamp_id=0,  # 占位
            task_name=task_name,
            status=update_group_status(items),
            staleness=max(item.seq_staleness for item in items),
        )
        async with self._lock:
            await self._policy.put(storage_item, self._storage)

    async def get(self, batch_size: int, task_name: str, group_status: Status) -> list[list[RolloutState]]:
        query = QueryItem(task_name=task_name, status=group_status)
        async with self._lock:
            return await self._policy.get(batch_size, query, self._storage)

    async def count(self, task_name: str, group_status: Status) -> int:
        query = QueryItem(task_name=task_name, status=group_status)
        async with self._lock:
            return await self._policy.count(query, self._storage)

    def __len__(self) -> int:
        return len(self._storage)


class SyncReplayBufferConfig(BaseModel):
    def build(self):
        policy = FIFOBackend()
        storage = NaiveStorage()
        replay_buffer = ReplayBuffer(policy=policy, storage_backend=storage)
        return replay_buffer


class AsyncReplayBufferConfig(BaseModel):
    min_staleness: int = 0
    max_staleness: int = 0

    def build(self):
        policy = StalenessBackend(max_staleness=self.max_staleness, min_staleness=self.min_staleness)
        storage = NaiveStorage()
        replay_buffer = ReplayBuffer(policy=policy, storage_backend=storage)
        return replay_buffer
