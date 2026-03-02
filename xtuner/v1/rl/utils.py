import atexit
import signal
import subprocess
from typing import Any, Literal, TypeAlias, cast

import torch.nn.functional as F

from xtuner.v1.utils.logger import get_logger


DSLOp: TypeAlias = Literal["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$not_in", "$between"]
DSLRuleType: TypeAlias = dict[DSLOp, Any]
ALLOWED_DSL_OPS: set[str] = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nnot_inin", "$between"}


class DSLRule:
    @staticmethod
    def normalize(rule_or_literal: Any) -> DSLRuleType:
        if isinstance(rule_or_literal, dict) and rule_or_literal:
            if len(rule_or_literal) != 1:
                raise ValueError(
                    f"Rule must use exactly one operator key. Supported operators: {sorted(ALLOWED_DSL_OPS)}"
                )
            op = next(iter(rule_or_literal))
            if op not in ALLOWED_DSL_OPS:
                raise ValueError(f"Unsupported DSL operator: {op}. Use one of: {sorted(ALLOWED_DSL_OPS)}.")
            return cast(DSLRuleType, rule_or_literal)
        return {"$eq": rule_or_literal}

    @classmethod
    def match(cls, actual: Any, rule_or_literal: Any) -> bool:
        rule = cls.normalize(rule_or_literal)
        op, expected = next(iter(rule.items()))

        if op == "$eq":
            return actual == expected
        if op == "$ne":
            return actual != expected
        if op == "$gt":
            return actual > expected
        if op == "$gte":
            return actual >= expected
        if op == "$lt":
            return actual < expected
        if op == "$lte":
            return actual <= expected
        if op == "$in":
            return actual in expected
        if op == "$not_in":
            return actual not in expected
        if op == "$between":
            if not isinstance(expected, (list, tuple)) or len(expected) != 2:
                raise ValueError("$between expects [lower, upper].")
            lower, upper = expected
            return lower <= actual <= upper

        raise ValueError(f"Unsupported DSL operator: {op}")


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
