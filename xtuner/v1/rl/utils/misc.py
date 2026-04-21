import importlib
import json
import random
import socket
import typing
from abc import ABC
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, List, Literal, Union

import torch.nn.functional as F

from xtuner.v1.data_proto import RolloutState, Status
from xtuner.v1.utils.logger import get_logger


logger = get_logger()
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
    def __init__(self, field: str, op: SetOperator, value: list[Any] | tuple[Any]):
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


def load_function(path):
    """Load a function from a module.

    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def find_free_ports(
    *,
    nums: int = 1,
    host: str = "127.0.0.1",
    start_port: int | None = None,
    end_port: int | None = None,
    contiguous: bool = False,
) -> list[int]:
    """Return available TCP ports on the given host.

    The candidate sockets are kept open until all requested ports are found so
    one call cannot return duplicate ports. Set ``contiguous=True`` to require
    the returned ports to be a continuous range.
    """
    if nums < 1:
        raise ValueError("nums must be greater than 0.")
    if start_port is not None:
        if end_port is None:
            raise ValueError("end_port must be set when start_port is set.")
        if end_port - start_port < nums:
            raise ValueError("The port range must contain at least nums ports.")

    def try_bind_ports(candidate_ports: list[int]) -> list[int] | None:
        ports: list[int] = []
        sockets: list[socket.socket] = []
        try:
            for candidate_port in candidate_ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind((host, candidate_port))
                    sock.listen(1)
                except OSError:
                    sock.close()
                    return None

                sockets.append(sock)
                ports.append(int(sock.getsockname()[1]))
            return ports
        finally:
            for sock in sockets:
                sock.close()

    if contiguous:
        if start_port is None:
            for _ in range(100):
                candidate = random.randint(20000, 60000 - nums)
                bound_ports = try_bind_ports(list(range(candidate, candidate + nums)))
                if bound_ports is not None:
                    return bound_ports
        else:
            assert end_port is not None
            for candidate in range(start_port, end_port - nums + 1):
                bound_ports = try_bind_ports(list(range(candidate, candidate + nums)))
                if bound_ports is not None:
                    return bound_ports
    else:
        available_ports: list[int] = []
        sockets: list[socket.socket] = []
        try:
            if start_port is None:
                candidates: range | list[int] = [0] * nums
            else:
                assert end_port is not None
                candidates = range(start_port, end_port)

            for candidate_port in candidates:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind((host, candidate_port))
                    sock.listen(1)
                except OSError:
                    sock.close()
                    continue

                sockets.append(sock)
                available_ports.append(int(sock.getsockname()[1]))
                if len(available_ports) >= nums:
                    return available_ports
        finally:
            for sock in sockets:
                sock.close()

    if start_port is None:
        raise RuntimeError(f"Could not find {nums} available ports.")
    raise RuntimeError(f"Could not find {nums} available ports from {start_port} to {end_port}.")


def get_eos_token(model_path: str) -> int | List[int]:
    generation_config_path = Path(model_path) / "generation_config.json"
    if not generation_config_path.exists():
        logger.warning(
            f"Config {generation_config_path} does not exist and thus cannot get eos_token. You must provide eos_token manually."
        )
        return []
    with open(generation_config_path) as f:
        generation_config = json.load(f)
    eos_token_id = generation_config.get("eos_token_id")
    if eos_token_id is None:
        raise ValueError(
            f"eos_token_id is not found in {generation_config_path}. You must provide eos_token manually."
        )
    return eos_token_id


def chat_trace_records_to_rollout_states(
    rollout_state: RolloutState,
    records: list[Any],
    *,
    tokenizer: Any | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> list[RolloutState]:
    """Convert Gateway chat trace records into trainable rollout states.

    The records may be ``ChatTraceRecord`` dataclass instances or serialized
    dictionaries returned by ``/trace_store``.
    """
    normalized_records = []
    for record in records:
        if isinstance(record, dict):
            normalized_records.append(record)
        elif not isinstance(record, type) and is_dataclass(record):
            normalized_records.append(asdict(record))
        elif hasattr(record, "__dict__"):
            normalized_records.append(dict(record.__dict__))
        else:
            raise TypeError(f"Unsupported chat trace record type: {type(record)}")

    trace_count = len(normalized_records)
    trace_summary = [
        {
            "request_id": record.get("request_id"),
            "finish_reason": record.get("finish_reason"),
            "status": record.get("status"),
            "prompt_ids": record.get("prompt_ids", []),
            "response_ids": record.get("response_ids", []),
        }
        for record in normalized_records
    ]

    states: list[RolloutState] = []
    for index, record in enumerate(normalized_records):
        prompt_ids = record.get("prompt_ids")
        response_ids = record.get("response_ids")
        if not prompt_ids or not response_ids:
            raise RuntimeError(f"Gateway trace record {index} is missing prompt_ids or response_ids.")

        logprobs = record.get("logprobs")
        if not isinstance(logprobs, list) or len(logprobs) != len(response_ids):
            logprobs = None

        status_value = record.get("status")
        if isinstance(status_value, Status):
            status = status_value
        elif isinstance(status_value, str):
            try:
                status = Status(status_value)
            except ValueError:
                status = Status.FAILED
        else:
            status = Status.FAILED

        request_id = record.get("request_id")
        try:
            uid = int(request_id) if request_id is not None else None
        except (TypeError, ValueError):
            uid = None

        response = record.get("output_text")
        if response is None and tokenizer is not None:
            try:
                response = tokenizer.decode(response_ids)
            except Exception:
                response = None

        normalized = rollout_state.model_copy(deep=True)
        normalized.uid = uid
        normalized.prompt_ids = list(prompt_ids)
        normalized.tokens = list(prompt_ids)
        normalized.response_ids = list(response_ids)
        normalized.response_mask = [1] * len(response_ids)
        normalized.logprobs = logprobs
        normalized.response = response
        normalized.finish_reason = record.get("finish_reason")
        normalized.status = status
        normalized.error_msg = None if status == Status.COMPLETED else f"Gateway trace status={status.value}"
        normalized.reward = None
        normalized.extra_fields = {
            **deepcopy(rollout_state.extra_fields),
            "gateway_trace_index": index,
            "gateway_trace_count": trace_count,
            "gateway_trace_records": deepcopy(trace_summary),
            "gateway_request_id": record.get("request_id"),
            "gateway_request_snapshot": record.get("request_snapshot"),
            "gateway_response_snapshot": record.get("response_snapshot"),
            **deepcopy(extra_fields or {}),
        }
        states.append(normalized)
    return states
