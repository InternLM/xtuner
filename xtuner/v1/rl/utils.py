import atexit
import signal
import subprocess
import typing
from abc import ABC
from typing import Any, List, Literal, Union

import torch.nn.functional as F

from xtuner.v1.utils.logger import get_logger


ScalarOperator = Literal["$eq", "$ne", "$gt", "$gte", "$lt", "$lte"]
SetOperator = Literal["$in", "$not_in"]
BetweenOperator = Literal["$between"]
Operators = Union[ScalarOperator, SetOperator, BetweenOperator]
LogicOperator = Literal["$and", "$or"]


class QueryNode(ABC):
    """查询语法树的基类，仅作数据结构标记."""

    pass


class ConditionNode(QueryNode):
    """代表一个具体的查询条件."""

    field: str


class ScalarNode(ConditionNode):
    def __init__(self, field: str, op: ScalarOperator, value: Any):
        self.field = field
        self.op = op
        self.value = value


class SetNode(ConditionNode):
    def __init__(self, field: str, op: SetOperator, value: List[Any]):
        self.field = field
        self.op = op
        self.value = value


class BetweenNode(ConditionNode):
    def __init__(self, field: str, lower: Any, upper: Any):
        if lower > upper:
            raise ValueError("lower bound must be less than or equal to upper bound")
        self.field = field
        self.op = "$between"
        self.lower = lower
        self.upper = upper


class LogicNode(QueryNode):
    """复合逻辑组."""

    def __init__(self, relation: LogicOperator, conditions: List[QueryNode]):
        self.relation = relation
        self.conditions = conditions


def parse_query(expr: Union[dict, QueryNode]) -> QueryNode:
    """将基于字典的 DSL 解析为纯粹的 AST 节点树 (ConditionNode, LogicNode)"""
    if isinstance(expr, QueryNode):
        return expr

    if isinstance(expr, dict):
        conditions: list[QueryNode] = []
        for key, value in expr.items():
            if key in ("$and", "$or"):
                if isinstance(value, list):
                    sub_asts = [parse_query(sub_expr) for sub_expr in value]
                    conditions.append(LogicNode(key, sub_asts))  # type: ignore
                else:
                    raise ValueError(f"逻辑操作符 {key} 的值必须是一个列表")
            else:
                if isinstance(value, dict):
                    # 例如: {"staleness": {"$lt": 5, "$gt": 0}}
                    for op, op_val in value.items():
                        if op in typing.get_args(ScalarOperator):
                            conditions.append(ScalarNode(field=key, op=op, value=op_val))
                        elif op in typing.get_args(SetOperator):
                            if not isinstance(op_val, (list, tuple)):
                                raise ValueError(f"操作符 '{op}' 需要传入一个列表或元组")
                            conditions.append(SetNode(field=key, op=op, value=op_val))
                        elif op == "$between":
                            if not isinstance(op_val, (list, tuple)) or len(op_val) != 2:
                                raise ValueError("操作符 '$between' 需要传入包含2个元素的列表或元组")
                            conditions.append(BetweenNode(field=key, lower=op_val[0], upper=op_val[1]))
                        else:
                            raise ValueError(f"不支持的操作符: {op}")
                else:
                    # 隐式等值，例如: {"task_name": "math"} -> "$eq"
                    conditions.append(ScalarNode(field=key, op="$eq", value=value))

        if len(conditions) > 1:
            # 默认多个条件之间是 AND 关系，例如: {"uid": "123", "status": {"$in": ["pending", "running]}}}
            return LogicNode("$and", conditions)  # type: ignore
        return conditions[0] if conditions else LogicNode("$and", [])

    raise ValueError(f"不支持的查询表达式格式: {expr}")


def gather_logprobs(logits, shifted_labels):
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs = logprobs.gather(dim=-1, index=shifted_labels.clip(min=0).unsqueeze(-1)).squeeze(-1)
    return logprobs


logger = get_logger()


def close_ray():
    """Clean up the ray resource."""
    import ray

    # 1. Shutdown ray if initialized
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown successfully")
    except Exception as e:
        logger.warning(f"Error during ray.shutdown(): {e}")

    # 2. Stop ray launched by CLI
    try:
        result = subprocess.run(["ray", "stop", "--force"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.warning(f"Ray stop failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"Error stopping ray cluster: {e}")


def register_cleanup():
    """Register cleanup handlers for Ray on exit and signals."""
    _cleaned = False

    def cleanup_once():
        nonlocal _cleaned
        if not _cleaned:
            _cleaned = True
            close_ray()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        cleanup_once()
        import sys

        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    atexit.register(cleanup_once)
