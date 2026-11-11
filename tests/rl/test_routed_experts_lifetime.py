"""Shared routed experts survive borrowers and retire after the last holder."""

import asyncio
import base64
import gc
import importlib.util
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
import ray

from xtuner.v1.data_proto.rl_data import RolloutState, Status, discard_rollout_state
from xtuner.v1.rl.distillation import DistillationTrainerAdapter
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore
from xtuner.v1.rl.rollout.utils import PartialRolloutHandler
from xtuner.v1.rl.trainer.pack import RankPlan
from xtuner.v1.rl.trainer.train_batch import TrainBatch, resolve_pack_refs
from xtuner.v1.rl.trainer.worker import TrainingWorker
from xtuner.v1.train.rl_trainer import BaseRLTrainer


@pytest.fixture(scope="module", autouse=True)
def local_ray() -> Iterator[None]:
    ray.init(address="local", num_cpus=2, include_dashboard=False, object_store_memory=128 * 1024**2)
    yield
    ray.shutdown()


@ray.remote(num_cpus=0)
class BatchConsumer:
    def load(self, refs: list[ray.ObjectRef]) -> None:
        state = RolloutState(message=[], routed_experts=refs[0])
        state.set_training_fields(input_ids=[1, 2, 3], labels=[-100, 2, 3], logprobs=None)
        self.batch: TrainBatch | None = TrainBatch(
            [state],
            RankPlan(global_indices=(0,), steps=(((0,),),)),
            {
                "rollout_idx": 0,
                "use_3d_position_ids": False,
                "pack_loss_keys": [],
                "has_routed_experts": True,
            },
            DistillationTrainerAdapter(None),
            pack_max_length=2,
            model_cfg=SimpleNamespace(n_routed_experts=256, num_hidden_layers=128, num_experts_per_tok=128),
            teacher_loss_keys=[],
            distillation_config=None,
        )

    def read(self) -> int:
        assert self.batch is not None
        pack = self.batch.materialize((0,))
        seq_ctx = pack.full_data["seq_ctx"]
        resolve_pack_refs(seq_ctx, model_cfg=self.batch.model_cfg, pack_max_length=self.batch.pack_max_length)
        return int(seq_ctx.rollout_routed_experts.long().sum())

    def release(self) -> None:
        TrainingWorker.release_train_batch(self)
        assert self.batch is None


class TestRoutedExpertsLifetime:
    @pytest.mark.parametrize("cleanup", [RolloutState.reset_response, discard_rollout_state])
    @pytest.mark.parametrize("as_list", [False, True])
    def test_reset_or_discard_preserves_sibling_routes(
        self, cleanup: Callable[[RolloutState], object], as_list: bool
    ) -> None:
        data = np.ones((4096, 2, 2), dtype=np.int64)
        ref = ray.put(data)
        first = RolloutState(message=[], routed_experts=[ref] if as_list else ref)
        sibling = RolloutState(message=[], routed_experts=[ref] if as_list else ref)
        first.set_training_fields(input_ids=[1, 2], labels=[-100, 2], logprobs=[0.0, -0.1])
        cleanup(first)
        assert first.routed_experts is None
        assert first.input_ids is None and first.labels is None and first.train_meta is None
        borrowed = sibling.routed_experts[0] if as_list else sibling.routed_experts
        np.testing.assert_array_equal(ray.get(borrowed, timeout=5), data)

    def test_partial_rollout_concat_preserves_borrowed_inputs(self) -> None:
        history = np.array([[[1]], [[2]]])
        current = np.array([[[1]], [[2]], [[3]]])
        history_ref, current_ref = ray.put(history), ray.put(current)
        state = RolloutState(
            message=[],
            response="old",
            response_ids=[1, 2],
            logprobs=[0.1, 0.2],
            routed_experts=history_ref,
            status=Status.ABORTED,
        )
        asyncio.run(
            PartialRolloutHandler().postprocess(
                state,
                response="new",
                response_ids=[3],
                logprobs=[0.3],
                routed_experts=current_ref,
                finish_reason="stop",
                status=Status.COMPLETED,
                prompt_tokens=3,
                completion_tokens=1,
            )
        )
        np.testing.assert_array_equal(ray.get(state.routed_experts, timeout=5), current)
        np.testing.assert_array_equal(ray.get(history_ref, timeout=5), history)
        np.testing.assert_array_equal(ray.get(current_ref, timeout=5), current)

    def test_trace_overwrite_and_batch_release_preserve_other_consumers(self) -> None:
        store = RolloutTraceStore.remote()
        first, second = BatchConsumer.remote(), BatchConsumer.remote()
        try:
            ref = ray.put(np.ones((2, 128, 128), dtype=np.int64))
            ray.get(store.insert.remote("s", "prompt", {"routed_experts": ref}))
            ray.get([first.load.remote([ref]), second.load.remote([ref])])
            del ref
            ray.get(store.insert.remote("s", "prompt", {"routed_experts": None}))
            ray.get(store.release.remote("s"))
            ray.get(first.release.remote())
            # Each pass rematerializes routes from TrainBatch's pristine samples.
            for _ in range(3):
                assert ray.get(second.read.remote(), timeout=10) == 32768
            ray.get(second.release.remote())
            ray.get(second.release.remote())
        finally:
            for actor in (store, first, second):
                ray.kill(actor)

    def test_repeated_batches_drop_all_route_references(self) -> None:
        store = RolloutTraceStore.remote()
        consumer = BatchConsumer.remote()
        object_ids = []
        try:
            for index in range(8):
                ref = ray.put(np.ones((2, 128, 128), dtype=np.int64))
                object_ids.append(ref.hex())
                ray.get(store.insert.remote(str(index), "prompt", {"routed_experts": ref}))
                ray.get(consumer.load.remote([ref]))
                del ref
                ray.get(store.release.remote(str(index)))
                assert ray.get(consumer.read.remote()) == 32768
                ray.get(consumer.release.remote())
            # Recreating ObjectRefs to inspect them would itself keep them alive.
            deadline = time.monotonic() + 10
            while True:
                gc.collect()
                counts = ray._private.worker.global_worker.core_worker.get_all_reference_counts()
                remaining = set(object_ids).intersection(counts)
                if not remaining or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            assert not remaining
            assert ray.get(store.list_sessions.remote()) == []
        finally:
            ray.kill(store)
            ray.kill(consumer)

    @pytest.mark.parametrize("traced", [False, True])
    def test_trained_batch_retires_refs_but_preserves_external_borrower(self, tmp_path: Path, traced: bool) -> None:
        store = RolloutTraceStore.remote()
        ref = ray.put(np.ones((2, 128, 128), dtype=np.int64))
        state = RolloutState(message=[], session_id=1 if traced else None, routed_experts=[ref])
        if traced:
            ray.get(store.insert.remote("1", "prompt", {"routed_experts": ref}))
        trainer = BaseRLTrainer.__new__(BaseRLTrainer)
        trainer._meta = SimpleNamespace(latest_exp=SimpleNamespace(exp_dir=tmp_path))
        trainer.logger = Mock()
        trainer._train_worker_cfg = SimpleNamespace(pack_max_length=4)
        trainer._prepare_train_batch = Mock(return_value=(0, 0))
        trainer._save_trajectories = Mock()
        trainer.train_controller = SimpleNamespace(fit=Mock(return_value=([], {})))
        try:
            with patch("xtuner.v1.rl.rollout.trace_store.get_existing_store", return_value=store):
                trainer._train_one_batch([[state]], train_step=1, step_timer_dict={})
            assert state.routed_experts is None
            assert ray.get(store.list_sessions.remote()) == []
            assert ray.get(ref, timeout=5).sum() == 32768
        finally:
            ray.kill(store)


@pytest.fixture
def vllm_worker_class() -> type:
    # Only the optional server-startup imports are stubbed. Execute the actual
    # response handler without requiring a vLLM installation or a GPU server.
    from xtuner.v1.rl.rollout import worker

    module_names = (
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.api_server",
        "vllm.entrypoints.openai.cli_args",
        "vllm.entrypoints.utils",
        "vllm.utils",
    )
    spec = importlib.util.spec_from_file_location(
        "xtuner.v1.rl.rollout._vllm_response_test", Path(worker.__file__).with_name("vllm.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: Mock() for name in module_names}):
        spec.loader.exec_module(module)
    return module.vLLMWorker


class TestVLLMRoutedExpertsResponse:
    @pytest.mark.parametrize("encoded", [False, True])
    def test_array_response_creates_reference(self, vllm_worker_class: type, encoded: bool) -> None:
        data = np.ones((2, 1, 1), dtype=np.int64)
        payload = base64.b64encode(ray.cloudpickle.dumps(data)).decode() if encoded else data.tolist()
        worker = SimpleNamespace(enable_return_routed_experts=True)
        state = RolloutState(message=[])
        response = self._response(payload)
        result = asyncio.run(vllm_worker_class._handle_non_stream_response(worker, state, response))
        assert result.status == Status.COMPLETED
        assert isinstance(result.routed_experts, ray.ObjectRef)
        np.testing.assert_array_equal(ray.get(result.routed_experts, timeout=5), data)

    def test_serialized_object_ref_is_rejected(self, vllm_worker_class: type) -> None:
        ref = ray.put(np.ones((2, 1, 1), dtype=np.int64))
        payload = base64.b64encode(ray.cloudpickle.dumps(ref)).decode()
        worker = SimpleNamespace(enable_return_routed_experts=True)
        try:
            with pytest.raises(ValueError, match="not serialized Ray ObjectRefs"):
                asyncio.run(
                    vllm_worker_class._handle_non_stream_response(
                        worker, RolloutState(message=[]), self._response(payload)
                    )
                )
        finally:
            # cloudpickle's out-of-band serialization pins the object. This is
            # precisely the format the handler must reject, not a normal borrower.
            ray._private.internal_api.free([ref], local_only=False)

    @staticmethod
    def _response(payload: Any) -> SimpleNamespace:
        return SimpleNamespace(
            json=lambda: {
                "choices": [
                    {
                        "message": {"content": "answer"},
                        "finish_reason": "stop",
                        "token_ids": [2],
                        "logprobs": {"content": [{"logprob": -0.1}]},
                        "routed_experts": payload,
                    }
                ]
            }
        )
