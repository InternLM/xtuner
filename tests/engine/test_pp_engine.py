"""Distributed tests for the pipeline-parallel engine (PPEngine).

Covers pipeline parallel on its own and combined with expert parallel, plus the HuggingFace
checkpoint round-trip for a pipeline-split model. Run on 4 GPUs.
"""

import json
import tempfile
from itertools import chain
from pathlib import Path

import parametrize
import torch
import torch.distributed as dist

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, PipelineParallelConfig
from xtuner.v1.engine.pipeline_engine import PPEngine
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.utils.device import get_device


DEVICE = get_device()


class TestPPEngine(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 4

    def _build_engine(self, pp_size: int, ep_size: int) -> PPEngine:
        moe_cfg = Qwen3MoE30BA3Config(num_hidden_layers=4, ep_size=ep_size, compile_cfg=False)
        engine = PPEngine(
            model_cfg=moe_cfg,
            optim_cfg=AdamWConfig(lr=1e-3),
            pp_cfg=PipelineParallelConfig(pp_size=pp_size),
            ep_size=ep_size,
        )
        engine.init_model_weights()
        return engine

    def _make_batches(self, vocab_size: int, n_microbatches: int, engine: PPEngine) -> list[ModelItem]:
        batches: list[ModelItem] = []
        for _ in range(n_microbatches):
            ids = torch.randint(0, vocab_size, (1, 129), dtype=torch.int64, device=DEVICE)
            seq_ctx = SequenceContext.from_input_ids(input_ids=(ids[:, :-1],))
            colate = [{"seq_ctx": seq_ctx, "shifted_labels": ids[:, 1:]}]
            loss_ctx = engine.model.build_loss_ctx_batch(colate)[0]
            batches.append({"seq_ctx": seq_ctx, "loss_ctx": loss_ctx})
        return batches

    @parametrize.parametrize(
        "device,pp_size,ep_size",
        [
            ("cuda", 2, 2),
            ("cuda", 4, 1),
        ],
    )
    def test_pp_engine_train(self, device, pp_size, ep_size):
        self.create_pg(device)
        engine = self._build_engine(pp_size, ep_size)

        n_microbatches = pp_size  # the schedule requires n_microbatches >= num_stages
        for step in range(3):
            torch.manual_seed(100 + step)
            batches = self._make_batches(engine.model_cfg.vocab_size, n_microbatches, engine)
            info = engine.train_step(batches)
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)

            assert torch.isfinite(torch.tensor(info["total_loss"])), f"non-finite loss at step {step}"
            assert torch.isfinite(grad_norm), f"non-finite grad_norm at step {step}"

    @parametrize.parametrize(
        "device,pp_size,ep_size",
        [
            ("cuda", 2, 2),
            ("cuda", 4, 1),
        ],
    )
    def test_pp_engine_save_hf_roundtrip(self, device, pp_size, ep_size):
        self.create_pg(device)
        engine = self._build_engine(pp_size, ep_size)

        tmp = [None]
        if dist.get_rank() == 0:
            tmp[0] = tempfile.mkdtemp(prefix="pp_ckpt_")
        dist.broadcast_object_list(tmp, src=0)
        save_dir = Path(tmp[0]) / "hf"

        engine.save_hf(save_dir)
        dist.barrier()

        if dist.get_rank() == 0:
            weight_map = json.loads((save_dir / "model.safetensors.index.json").read_text())["weight_map"]

            full = Qwen3MoE30BA3Config(num_hidden_layers=4, ep_size=1, compile_cfg=False).build()
            expected = set(chain(*map(full.to_hf_key_list, full.state_dict())))
            assert not (expected - set(weight_map)), "merged checkpoint index is missing keys"

            full2 = Qwen3MoE30BA3Config(num_hidden_layers=4, ep_size=1, compile_cfg=False).build()
            _, unloaded, missing = full2.from_hf(save_dir, strict=True)
            assert not unloaded and not missing, f"reload incomplete: unloaded={unloaded} missing={missing}"

        dist.barrier()
