import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, replace
from itertools import count
from pathlib import Path
from typing import Any, List, TypeAlias, Union

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto.rl_data import RolloutState, Status, update_group_status
from xtuner.v1.rl.utils import (
    BetweenNode,
    ConditionNode,
    LogicNode,
    LogicOperator,
    Operators,
    QueryNode,
    ScalarNode,
    SetNode,
    calculate_seq_staleness,
    parse_query,
)
from xtuner.v1.utils import get_logger


logger = get_logger(__name__)


@dataclass
class StorageItem:
    # 存储类型
    item: List[RolloutState]
    uid: int
    timestamp_id: int
    task_name: str
    status: Status
    staleness: int


QUERY_KEYS = [f.name for f in fields(StorageItem)]
QueryKey = Union[str, LogicOperator]  # str 是 StorageItem 的字段名，LogicOperator 是 "$and", "$or" 等逻辑操作符

# 查询类型：
QueryDict: TypeAlias = dict[
    QueryKey,
    Union[
        Any,  # 直接匹配值，例如: {"task_name": "math"}
        dict[Operators, Any],  # 操作符匹配，例如: {"uid": {"$gt": 10}}
        List["QueryDict"],  # 逻辑组合，例如: {"$and": [{"a": 1}, {"b": 2}]}
    ],
]
QueryType = Union[QueryDict, QueryNode]


class StorageBackend(ABC):
    @abstractmethod
    async def put(self, item: StorageItem) -> int: ...

    @abstractmethod
    async def get(self, query: QueryType) -> List[StorageItem]: ...

    @abstractmethod
    async def count(self, query: QueryType) -> int: ...

    @abstractmethod
    async def delete(self, uids: list[int]) -> None: ...

    @abstractmethod
    async def update(self, items: list[StorageItem]) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class ReplayPolicy(ABC):
    @abstractmethod
    async def put(self, item: StorageItem, storage_backend: StorageBackend) -> None: ...

    @abstractmethod
    async def get(self, count: int, query: QueryType, storage_backend: StorageBackend) -> list[list[RolloutState]]: ...

    async def count(self, query: QueryType, storage_backend: StorageBackend) -> int:
        return await storage_backend.count(query)


class NaiveStorage(StorageBackend):
    def __init__(self):
        self._uid_gen = count(1)
        self._timestamp_id_gen = count(1)
        self._items: dict[int, StorageItem] = {}

    async def put(self, item: StorageItem) -> int:
        uid = next(self._uid_gen)
        stored = replace(item, uid=uid, timestamp_id=next(self._timestamp_id_gen))
        self._items[uid] = stored
        return uid

    def _evaluate(self, item: StorageItem, query_node: QueryNode) -> bool:
        """NaiveStorage 实现的原生 Python 对象过滤树遍历."""
        if isinstance(query_node, LogicNode):
            if not query_node.conditions:
                return query_node.relation == "$and"

            if query_node.relation == "$and":
                return all(self._evaluate(item, child) for child in query_node.conditions)
            else:
                return any(self._evaluate(item, child) for child in query_node.conditions)

        elif isinstance(query_node, ConditionNode):
            if query_node.field not in QUERY_KEYS:
                raise ValueError(f"查询字段错误: 找不到属性 '{query_node.field}'。可用属性为: {QUERY_KEYS}")
            val = getattr(item, query_node.field)

            if isinstance(query_node, ScalarNode):
                if query_node.op == "$eq":
                    return val == query_node.value
                if query_node.op == "$ne":
                    return val != query_node.value
                if query_node.op == "$gt":
                    return val > query_node.value
                if query_node.op == "$gte":
                    return val >= query_node.value
                if query_node.op == "$lt":
                    return val < query_node.value
                if query_node.op == "$lte":
                    return val <= query_node.value

            elif isinstance(query_node, SetNode):
                if query_node.op == "$in":
                    return val in query_node.value
                if query_node.op == "$not_in":
                    return val not in query_node.value

            elif isinstance(query_node, BetweenNode):
                return query_node.lower <= val <= query_node.upper

        return False

    async def get(self, query: QueryType) -> list[StorageItem]:
        ast = parse_query(query)
        return [item for item in self._items.values() if self._evaluate(item, ast)]

    async def count(self, query: QueryType) -> int:
        ast = parse_query(query)
        return sum(1 for item in self._items.values() if self._evaluate(item, ast))

    async def delete(self, uids: list[int]) -> None:
        if not uids:
            return
        for uid in uids:
            self._items.pop(uid, None)

    async def update(self, items: list[StorageItem]) -> None:
        for item in items:
            old_item = self._items.get(item.uid)
            if old_item is None:
                continue
            # 原地更新保留 uid/timestamp，避免刷新 staleness 改变 replay 顺序。
            self._items[item.uid] = replace(item, uid=old_item.uid, timestamp_id=old_item.timestamp_id)

    def __len__(self) -> int:
        return len(self._items)

    def state_dict(self) -> dict[str, Any]:
        max_uid = max(self._items, default=0)
        max_timestamp_id = max((item.timestamp_id for item in self._items.values()), default=0)
        return {
            "items": list(self._items.values()),
            "next_uid": max_uid + 1,
            "next_timestamp_id": max_timestamp_id + 1,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        items: list[StorageItem] = state["items"]
        self._items = {item.uid: item for item in items}
        self._uid_gen = count(state["next_uid"])
        self._timestamp_id_gen = count(state["next_timestamp_id"])


class PandasStorage(StorageBackend):
    def __init__(self):
        self._uid_gen = count(1)
        self._timestamp_id_gen = count(1)
        self._df = pd.DataFrame(columns=["uid", "timestamp_id", "task_name", "status", "staleness", "item"])
        self._buffer: list[dict] = []

    def _flush_buffer(self):
        if self._buffer:
            new_df = pd.DataFrame(self._buffer)
            self._df = new_df if self._df.empty else pd.concat([self._df, new_df], ignore_index=True)
            self._buffer.clear()

    async def put(self, item: StorageItem) -> int:
        uid = next(self._uid_gen)
        row = {
            "uid": uid,
            "timestamp_id": next(self._timestamp_id_gen),
            "task_name": item.task_name,
            "status": item.status,
            "staleness": item.staleness,
            "item": item.item,
        }
        self._buffer.append(row)
        return uid

    def _evaluate(self, query_node: QueryNode, df: pd.DataFrame) -> pd.Series:
        """PandasStorage 实现的向量化 DataFrame 过滤树遍历."""
        if isinstance(query_node, LogicNode):
            if not query_node.conditions:
                return (
                    pd.Series(True, index=df.index)
                    if query_node.relation == "$and"
                    else pd.Series(False, index=df.index)
                )

            mask = self._evaluate(query_node.conditions[0], df)
            for child in query_node.conditions[1:]:
                child_mask = self._evaluate(child, df)
                if query_node.relation == "$and":
                    mask = mask & child_mask
                else:
                    mask = mask | child_mask
            return mask

        elif isinstance(query_node, ConditionNode):
            field = query_node.field
            if field not in QUERY_KEYS:
                raise ValueError(f"查询字段错误: 找不到属性 '{query_node.field}'。可用属性为: {QUERY_KEYS}")
            series = df[query_node.field]

            if isinstance(query_node, ScalarNode):
                if query_node.op == "$eq":
                    return series == query_node.value
                if query_node.op == "$ne":
                    return series != query_node.value
                if query_node.op == "$gt":
                    return series > query_node.value
                if query_node.op == "$gte":
                    return series >= query_node.value
                if query_node.op == "$lt":
                    return series < query_node.value
                if query_node.op == "$lte":
                    return series <= query_node.value

            elif isinstance(query_node, SetNode):
                if query_node.op == "$in":
                    return series.isin(query_node.value)
                if query_node.op == "$not_in":
                    return ~series.isin(query_node.value)

            elif isinstance(query_node, BetweenNode):
                return series.between(query_node.lower, query_node.upper)
        else:
            raise ValueError(f"不支持的查询节点类型: {type(query_node)}")

    async def get(self, query: QueryType) -> list[StorageItem]:
        self._flush_buffer()
        if self._df.empty:
            return []

        ast = parse_query(query)
        filtered_df = self._df[self._evaluate(ast, self._df)]
        return [
            StorageItem(
                item=row["item"],
                uid=row["uid"],
                timestamp_id=row["timestamp_id"],
                task_name=row["task_name"],
                status=row["status"],
                staleness=row["staleness"],
            )
            for _, row in filtered_df.iterrows()
        ]

    async def count(self, query: QueryType) -> int:
        self._flush_buffer()
        if self._df.empty:
            return 0
        ast = parse_query(query)
        return int(self._evaluate(ast, self._df).sum())

    async def delete(self, uids: list[int]) -> None:
        self._flush_buffer()
        if not uids or self._df.empty:
            return
        self._df = self._df[~self._df["uid"].isin(uids)]

    async def update(self, items: list[StorageItem]) -> None:
        self._flush_buffer()
        if not items or self._df.empty:
            return
        for item in items:
            mask = self._df["uid"] == item.uid
            if not mask.any():
                continue
            for row_idx in self._df.index[mask]:
                self._df.at[row_idx, "status"] = item.status
                self._df.at[row_idx, "staleness"] = item.staleness
                self._df.at[row_idx, "item"] = item.item

    def __len__(self) -> int:
        return len(self._df) + len(self._buffer)

    def state_dict(self) -> dict[str, Any]:
        self._flush_buffer()
        max_uid = int(self._df["uid"].max()) if not self._df.empty else 0
        max_timestamp_id = int(self._df["timestamp_id"].max()) if not self._df.empty else 0
        return {
            "df": self._df.copy(deep=True),
            "next_uid": max_uid + 1,
            "next_timestamp_id": max_timestamp_id + 1,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._df = state["df"].copy(deep=True)
        self._buffer = []
        self._uid_gen = count(state["next_uid"])
        self._timestamp_id_gen = count(state["next_timestamp_id"])


class FIFOReplayPolicy(ReplayPolicy):
    async def put(self, item: StorageItem, storage_backend: StorageBackend) -> None:
        if not item.item:
            return
        await storage_backend.put(item)

    async def get(self, count: int, query: QueryType, storage_backend: StorageBackend) -> list[list[RolloutState]]:
        if count <= 0:
            return []
        records = await storage_backend.get(query)
        records.sort(key=lambda r: r.timestamp_id)
        selected = records[:count]
        if selected:
            await storage_backend.delete([record.uid for record in selected])
        return [record.item for record in selected]


class StalenessReplayPolicy(ReplayPolicy):
    async def put(self, item: StorageItem, storage_backend: StorageBackend) -> None:
        if not item.item:
            return
        await storage_backend.put(item)

    async def get(self, count: int, query: QueryType, storage_backend: StorageBackend) -> list[list[RolloutState]]:
        if count <= 0:
            return []

        records = await storage_backend.get(query)
        records.sort(key=lambda r: (-r.staleness, r.timestamp_id))
        selected = records[:count]
        if selected:
            await storage_backend.delete([record.uid for record in selected])
        return [record.item for record in selected]

    async def count(self, query: QueryType, storage_backend: StorageBackend) -> int:
        return await storage_backend.count(query)


class ReplayBuffer:
    _SAVE_PATH = "replay_buffer.pth"

    def __init__(self, policy: ReplayPolicy, storage_backend: StorageBackend):
        self._policy = policy
        self._storage = storage_backend
        self._lock = asyncio.Lock()

    async def put(self, items: list[RolloutState], task_name: str) -> None:
        if not items:
            return
        storage_item = StorageItem(
            item=items,
            uid=0,
            timestamp_id=0,
            task_name=task_name,
            status=update_group_status(items),
            staleness=max(item.seq_staleness for item in items),
        )
        async with self._lock:
            await self._policy.put(storage_item, self._storage)

    async def get(self, batch_size: int, task_name: str, group_status: Status) -> list[list[RolloutState]]:
        # 使用 DSL 字典进行查询
        query_dsl: QueryDict = {"$and": [{"task_name": task_name}, {"status": group_status}]}
        async with self._lock:
            return await self._policy.get(batch_size, query_dsl, self._storage)

    async def count(self, task_name: str, group_status: Status) -> int:
        # 使用 DSL 字典进行查询
        query_dsl: QueryDict = {"$and": [{"task_name": task_name}, {"status": group_status}]}
        async with self._lock:
            return await self._policy.count(query_dsl, self._storage)

    @staticmethod
    def _refresh_seq_staleness(items: list[RolloutState], current_train_step: int) -> None:
        for item in items:
            response_model_steps = getattr(item, "response_model_steps", None) or []
            if response_model_steps:
                item.seq_staleness = calculate_seq_staleness(min(response_model_steps), current_train_step)
            elif hasattr(item, "seq_staleness"):
                item.seq_staleness = 0

    async def refresh_completed_staleness(
        self,
        task_name: str,
        current_train_step: int,
        stale_threshold: int,
        statuses: list[Status] | None = None,
    ) -> int:
        # 保留历史方法名；可复用的 completed / aborted buffer 样本都需要随 train_step 刷新过期状态。
        if stale_threshold <= 0:
            raise ValueError(f"stale_threshold must be positive, got {stale_threshold}.")
        if statuses is None:
            statuses = [Status.COMPLETED, Status.ABORTED]
        query_dsl: QueryDict = {
            "$and": [
                {"task_name": task_name},
                {"status": {"$in": statuses}},
            ]
        }
        async with self._lock:
            records = await self._storage.get(query_dsl)
            updated_records: list[StorageItem] = []
            expired_count = 0
            for record in records:
                self._refresh_seq_staleness(record.item, current_train_step)
                staleness = max((getattr(item, "seq_staleness", 0) for item in record.item), default=0)
                should_expire = any(getattr(item, "seq_staleness", 0) >= stale_threshold for item in record.item)
                if should_expire:
                    # completed / aborted 样本超过 step 级阈值时整组翻转，后续 sampler 可按 EXPIRED 重新取样。
                    for item in record.item:
                        item.status = Status.EXPIRED
                    status = Status.EXPIRED
                    expired_count += 1
                else:
                    status = update_group_status(record.item)
                updated_records.append(replace(record, status=status, staleness=staleness))
            await self._storage.update(updated_records)
            return expired_count

    def __len__(self) -> int:
        return len(self._storage)

    async def save(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        replay_buffer_path = file_path / self._SAVE_PATH
        async with self._lock:
            state = {
                "policy": type(self._policy).__name__,
                "storage": type(self._storage).__name__,
                "storage_state": self._storage.state_dict(),
            }
        await asyncio.to_thread(torch.save, state, replay_buffer_path)
        logger.info(f"Replay buffer saved to {replay_buffer_path}")

    async def resume(self, path: str | Path) -> None:
        if len(self._storage) > 0:
            raise RuntimeError("Cannot resume into a non-empty buffer")

        file_path = Path(path)
        replay_buffer_path = file_path / self._SAVE_PATH
        state = await asyncio.to_thread(torch.load, replay_buffer_path, map_location="cpu", weights_only=False)
        if state["policy"] != type(self._policy).__name__:
            raise ValueError(f"Replay policy mismatch: expected {type(self._policy).__name__}, got {state['policy']}")

        if state["storage"] != type(self._storage).__name__:
            raise ValueError(
                f"Storage backend mismatch: expected {type(self._storage).__name__}, got {state['storage']}"
            )

        async with self._lock:
            self._storage.load_state_dict(state["storage_state"])
        logger.info(f"Replay buffer resumed from {replay_buffer_path}")


class SyncReplayBufferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def build(self):
        return ReplayBuffer(policy=FIFOReplayPolicy(), storage_backend=NaiveStorage())


class AsyncReplayBufferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def build(self):
        policy = StalenessReplayPolicy()
        return ReplayBuffer(policy=policy, storage_backend=NaiveStorage())
