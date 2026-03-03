import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, replace
from itertools import count
from typing import Any, List, TypeAlias, Union

import pandas as pd
from pydantic import BaseModel

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
    parse_query,
)


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
    def __len__(self) -> int: ...


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

    def __len__(self) -> int:
        return len(self._items)


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

    def __len__(self) -> int:
        return len(self._df) + len(self._buffer)


class FIFOBackend(ReplayPolicy):
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


class StalenessBackend(ReplayPolicy):
    def __init__(self, max_staleness: int = 0, min_staleness: int = 0):
        self.max_staleness = max_staleness
        self.min_staleness = min_staleness

    async def put(self, item: StorageItem, storage_backend: StorageBackend) -> None:
        if not item.item:
            return
        await storage_backend.put(item)

    def _hybrid_query(self, base_query: QueryType) -> QueryType:
        staleness_cond: QueryDict = {"staleness": {"$between": [self.min_staleness, self.max_staleness]}}

        # 1. 修复：如果穿进来的是已经解析的 AST 节点
        if isinstance(base_query, QueryNode):
            return LogicNode(
                "$and",
                [
                    base_query,
                    parse_query(staleness_cond),  # 将新的字典也解析为节点后组合
                ],
            )

        # 2. 如果传进来的是字典形式的 DSL
        base_dict = base_query if isinstance(base_query, dict) else {}
        return {
            "$and": [
                base_dict,  # type: ignore
                staleness_cond,
            ]
        }

    async def get(self, count: int, query: QueryType, storage_backend: StorageBackend) -> list[list[RolloutState]]:
        if count <= 0:
            return []

        hybrid_query = self._hybrid_query(query)
        records = await storage_backend.get(hybrid_query)
        records.sort(key=lambda r: (-r.staleness, r.timestamp_id))
        selected = records[:count]
        if selected:
            await storage_backend.delete([record.uid for record in selected])
        return [record.item for record in selected]

    async def count(self, query: QueryType, storage_backend: StorageBackend) -> int:
        hybrid_query = self._hybrid_query(query)
        return await storage_backend.count(hybrid_query)


class ReplayBuffer:
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

    def __len__(self) -> int:
        return len(self._storage)


class SyncReplayBufferConfig(BaseModel):
    def build(self):
        return ReplayBuffer(policy=FIFOBackend(), storage_backend=NaiveStorage())


class AsyncReplayBufferConfig(BaseModel):
    min_staleness: int = 0
    max_staleness: int = 0

    def build(self):
        assert self.max_staleness >= self.min_staleness, "max_staleness must be greater than or equal to min_staleness"
        policy = StalenessBackend(max_staleness=self.max_staleness, min_staleness=self.min_staleness)
        return ReplayBuffer(policy=policy, storage_backend=NaiveStorage())
