from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import ray
import torch

from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.rl.trainer.controller import TrainingController
from xtuner.v1.rl.trainer.ppo_config import PPOConfig
from xtuner.v1.train.rl_trainer import BaseRLTrainer


def test_truncated_siblings_do_not_free_retained_pixel_ref() -> None:
    """K rollouts share prompt pixels, so pruning one must not free all."""
    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(num_cpus=1, include_dashboard=False, logging_level="ERROR")

    try:
        shared_pixels = np.arange(12, dtype=np.float32).reshape(3, 4)
        shared_pixel_ref = ray.put(shared_pixels)
        group = []
        for rollout_id in range(8):
            group.append(
                RolloutState(
                    rollout_id=rollout_id,
                    group_id=0,
                    message=[{"role": "user", "content": "multimodal prompt"}],
                    prompt_ids=[10, 11],
                    response="response",
                    response_ids=[20, 21],
                    response_mask=[1, 1],
                    logprobs=[-0.1, -0.2],
                    reward={"score": 0.0},
                    status=Status.COMPLETED,
                    finish_reason="length",
                    mm_info={
                        "pixel_values": shared_pixel_ref,
                        "image_grid_thw": np.array([[1, 1, 1]], dtype=np.int64),
                    },
                    position_ids=np.array([[[0, 1]], [[0, 1]], [[0, 1]]], dtype=np.int64),
                    routed_experts=ray.put(np.array([rollout_id], dtype=np.int64)),
                )
            )

        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer._train_worker_cfg = SimpleNamespace(ppo_cfg=PPOConfig(selection_seed=42))
        trainer._cur_step = 0
        trainer.logger = MagicMock()
        trainer.tokenizer = MagicMock()

        data_batches, info = trainer._prepare_ppo_train_data([group], pack_max_length=16)

        assert len(data_batches) == 1
        assert info["ppo/retained_truncated"] == 1.0
        assert info["ppo/dropped_truncated"] == 7.0
        assert sum(state.routed_experts is not None for state in group) == 1
        retained_pixel_ref = data_batches[0]["seq_ctx"].pixel_values
        np.testing.assert_array_equal(ray.get(retained_pixel_ref), shared_pixels)
    finally:
        if started_ray:
            ray.shutdown()


def test_failed_fit_does_not_manual_free_pixels_while_sibling_workers_may_run() -> None:
    seq_ctx = SequenceContext.from_input_ids((torch.tensor([[1, 2]]),), device="cpu")
    seq_ctx.pixel_values = [MagicMock(name="pixel_ref")]
    packed = {
        "seq_ctx": seq_ctx,
        "shifted_labels": torch.tensor([[-100, 2]]),
        "advantages": torch.zeros(1, 2),
        "rollout_logprobs": None,
    }
    worker = SimpleNamespace(
        get_data_replicate_size=SimpleNamespace(remote=MagicMock(return_value="replicate_handle")),
        fit=SimpleNamespace(remote=MagicMock(return_value="fit_handle")),
    )
    controller = TrainingController([worker])
    controller._packing = MagicMock(return_value=[packed])

    with (
        patch("xtuner.v1.rl.trainer.controller.ray.get", side_effect=[1, RuntimeError("fit failed")]),
        patch("xtuner.v1.rl.trainer.controller.free_object_refs") as free_refs,
        pytest.raises(RuntimeError, match="fit failed"),
    ):
        controller.fit([packed], pack_max_length=2, rollout_idx=1)

    free_refs.assert_not_called()
