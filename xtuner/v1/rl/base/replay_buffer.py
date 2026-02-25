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
    async def get(self, count: int, storage_indices: StorageIndices) -> list[RolloutState]: ...
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


class PandasStorage(Storage):
    def __init__(self, limit: int = 0):
        raise NotImplementedError("PandasStorageBackend is under development and not yet implemented.")
        import pandas as pd

        self._df = pd.DataFrame(columns=["task_name", "group_status", "data"])

    def __len__(self): ...
    async def put(self, items: list[RolloutState], indices: StorageIndices):
        import pandas as pd

        new_rows = []
        base_info = {"task_name": indices.task_name, "group_status": indices.group_status, **indices.tags}

        for item in items:
            row = base_info.copy()
            row["data"] = item
            new_rows.append(row)

        new_df = pd.DataFrame(new_rows)
        self._df = pd.concat([self._df, new_df], ignore_index=True, sort=False)

    async def get(self, count: int, indices: StorageIndices) -> list[RolloutState]:
        if self._df.empty:
            return []
        mask = (self._df["task_name"] == indices.task_name) & (self._df["group_status"] == indices.group_status)
        for key, value in indices.tags.items():
            if key in self._df.columns:
                mask &= self._df[key] == value
            else:
                return []
        target_df = self._df[mask].head(count)
        if target_df.empty:
            return []
        result = target_df["data"].tolist()
        self._df.drop(target_df.index, inplace=True)
        return result


class SQLStorage(Storage):
    def __init__(self, db_path: str = ":memory:"):
        raise NotImplementedError("SQLStorageBackend is under development and not yet implemented.")
        self.db_path = db_path
        self._init_db()

    def _init_db(self): ...
    def _serialize_item(self, item: RolloutState) -> bytes: ...
    def _deserialize_item(self, blob: bytes) -> RolloutState: ...
    def __len__(self): ...

    async def put(self, items: list[RolloutState], indices: StorageIndices):
        import json
        import sqlite3

        rows = []
        tags_json = json.dumps(indices.tags)

        for item in items:
            data_blob = self._serialize_item(item)
            rows.append((indices.task_name, indices.group_status, tags_json, data_blob))

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT INTO replay_buffer (task_name, group_status, tags, data) VALUES (?, ?, ?, ?)", rows
            )

    async def get(self, count: int, indices: StorageIndices) -> list[RolloutState]:
        import sqlite3

        # 构建动态查询
        query = "SELECT id, data FROM replay_buffer WHERE task_name = ? AND group_status = ?"
        params = [indices.task_name, indices.group_status]

        # SQLite 的 JSON 查询语法 (需要 SQLite 3.38+，如果是旧版本需要用 LIKE 模拟或不做 DB 级过滤)
        # 这里演示简单的方法：如果在 Python 端过滤 tags 效率低，但在 SQL 端过滤 JSON 语法较复杂。
        # 为了通用性，这里我只用 task 和 status 查出候选集，然后用 Python 过滤 Tags (如果 tags 很复杂建议把 tags 独立成列)
        # 或者使用 JSON_EXTRACT (推荐)
        for key, value in indices.tags.items():
            # 注意：JSON 中数值和字符串的区别。这里假设 value 都是简单类型。
            # $.key 取出对应的值
            query += f" AND json_extract(tags, '$.{key}') = ?"
            params.append(value)

        query += f" LIMIT {count}"

        results = []
        ids_to_delete = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            for row_id, data_blob in rows:
                results.append(self._deserialize_item(data_blob))
                ids_to_delete.append(row_id)

            if ids_to_delete:
                placeholders = ",".join("?" for _ in ids_to_delete)
                conn.execute(f"DELETE FROM replay_buffer WHERE id IN ({placeholders})", ids_to_delete)

        return results


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
        self._bucket_counts = defaultdict(int)

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
    def __init__(self, storage_backend: Storage = None):
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
