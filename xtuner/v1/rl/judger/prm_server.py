"""Serve a value-head PRM checkpoint over HTTP for step-level scoring.

lmdeploy only serves lm_head generation models, so a value-head PRM is scored
through the xtuner engine forward on a single GPU. The server is deliberately
thin: step location and aggregation helpers are pure functions (unit-tested
without a GPU), and ``PRMScorer`` only wires tokenization, one forward pass,
and value lookup at step-end positions.

Protocol (consumed by ``PRMJudger`` in ``prm.py``):

- ``POST /score`` with ``{"text": "<response>", "step_separator": "\\n\\n"}``
  (``step_separator`` optional) returns ``{"step_scores": [...],
  "process_score": <mean in [0, 1]>, "n_steps": <int>}``. The ``[0, 1]`` ->
  ``[-1, 1]`` mapping and reward fusion stay on the judger/merge side.
- ``GET /health`` returns ``{"status": "ok"}``.

Launch::

    python -m xtuner.v1.rl.judger.prm_server --config prm_server_cfg.py

The config file is a plain Python module loaded with ``Config.fromfile`` and
must define ``model_cfg`` (a model config whose text tower uses a value head),
``ckpt`` (HF checkpoint of the trained PRM) and ``tokenizer_path``; ``host``,
``port`` and ``max_length`` (input guard, in tokens) are optional.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, cast

import torch
import torch.distributed as dist
from aiohttp import web
from cyclopts import App, Parameter
from cyclopts.group import Group

from xtuner.v1.utils.logger import get_logger


logger = get_logger()

app = App()


def split_steps(text: str, step_separator: str) -> list[str]:
    """Split a response into non-empty reasoning steps."""
    return [step for step in text.split(step_separator) if step.strip()]


def step_end_char_offsets(text: str, steps: list[str]) -> list[int]:
    """Return the exclusive end char offset of each step in the original text.

    Steps are located by scanning forward from the previous step's end, so repeated identical steps map to their own
    occurrences.
    """
    ends: list[int] = []
    cursor = 0
    for step in steps:
        start = text.index(step, cursor)
        cursor = start + len(step)
        ends.append(cursor)
    return ends


def locate_step_positions(offset_mapping: list[list[int]], ends: list[int]) -> list[int]:
    """Map step-end char offsets to token positions via the tokenizer offset
    mapping.

    The step score is read from the token covering the step's last character. Steps whose last character is covered by
    no token are skipped.
    """
    positions: list[int] = []
    for end in ends:
        target = end - 1
        position = next((i for i, (start, stop) in enumerate(offset_mapping) if start <= target < stop), None)
        if position is not None:
            positions.append(position)
    return positions


class PRMScorer:
    """Single-GPU value-head PRM scorer backed by the xtuner training
    engine."""

    def __init__(
        self,
        model_cfg: Any,
        ckpt: str | Path,
        tokenizer_path: str | Path,
        max_length: int | None = None,
    ) -> None:
        from transformers import AutoTokenizer
        from xtuner.v1.config import AdamWConfig, FSDPConfig
        from xtuner.v1.engine.train_engine import TrainEngine

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29778")
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        torch.cuda.set_device(0)

        self.engine = TrainEngine(
            model_cfg=model_cfg,
            optim_cfg=AdamWConfig(lr=1e-6, foreach=False, weight_decay=0.0),
            fsdp_cfg=FSDPConfig(
                cpu_offload=False,
                torch_compile=False,
                fp32_value_head=True,
                recompute_ratio=1.0,
            ),
        )
        self.engine.from_hf(ckpt, strict=True)
        self.engine.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.max_length = max_length

    @torch.no_grad()
    def score(self, text: str, step_separator: str = "\n\n") -> dict[str, Any]:
        from xtuner.v1.data_proto import SequenceContext
        from xtuner.v1.loss import BaseLossContext

        if not isinstance(text, str) or not text:
            raise ValueError("PRM scoring requires a non-empty text.")
        if not step_separator:
            raise ValueError("PRM scoring requires a non-empty step_separator.")
        steps = split_steps(text, step_separator)
        if not steps:
            return {"step_scores": [], "process_score": 0.0, "n_steps": 0}
        ends = step_end_char_offsets(text, steps)
        encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        input_ids = encoding["input_ids"]
        if self.max_length is not None and len(input_ids) > self.max_length:
            raise ValueError(f"PRM input has {len(input_ids)} tokens, exceeding max_length {self.max_length}.")
        positions = locate_step_positions(encoding["offset_mapping"], ends)

        ids = cast(torch.LongTensor, torch.tensor([input_ids], dtype=torch.long))
        seq_ctx = SequenceContext.from_input_ids((ids,), device="cuda")
        shifted_labels = torch.full_like(ids, -100)
        returns = torch.zeros(ids.shape, dtype=torch.float32)
        loss_ctx = cast(
            dict[str, BaseLossContext],
            self.engine.model.build_loss_ctx_batch(
                [{"seq_ctx": seq_ctx, "shifted_labels": shifted_labels, "returns": returns}], None
            )[0],
        )
        output = self.engine.model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
        if output.logits is None:
            raise RuntimeError("PRM forward returned no value logits.")
        values = output.logits.squeeze(-1).float().cpu()[0]
        step_scores = [values[position].item() for position in positions]
        process_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
        return {
            "step_scores": step_scores,
            "process_score": process_score,
            "n_steps": len(step_scores),
        }


@app.default()
def main(
    *,
    config: Annotated[Path, Parameter(group=Group("config-path", sort_key=0))],
    host: str | None = None,
    port: int | None = None,
):
    from xtuner.v1.utils import Config

    cfg = Config.fromfile(config)
    scorer = PRMScorer(
        cfg.model_cfg,
        cfg.ckpt,
        cfg.tokenizer_path,
        max_length=cfg.get("max_length"),
    )
    serve_host = host or cfg.get("host", "0.0.0.0")
    serve_port = port or cfg.get("port", 18001)

    async def handle_score(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            text = payload["text"]
            step_separator = payload.get("step_separator", "\n\n")
        except Exception as error:
            return web.json_response({"error": f"bad request: {error}"}, status=400)
        try:
            return web.json_response(scorer.score(text, step_separator))
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)

    async def handle_health(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    application = web.Application()
    application.router.add_post("/score", handle_score)
    application.router.add_get("/health", handle_health)
    logger.info(f"[prm-server] serving {cfg.ckpt} on {serve_host}:{serve_port}")
    web.run_app(application, host=serve_host, port=serve_port, print=None)


if __name__ == "__main__":
    app(exit_on_error=False)
