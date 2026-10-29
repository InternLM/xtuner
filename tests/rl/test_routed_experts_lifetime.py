"""Shared routed experts survive individual consumers and retire after the last one."""

import asyncio
import gc
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import ray

from xtuner.v1.data_proto.rl_data import RolloutState, Status, discard_rollout_state, reset_rollout_response
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore
from xtuner.v1.rl.rollout.utils import PartialRolloutHandler
from xtuner.v1.rl.trainer.worker import TrainingWorker
from xtuner.v1.train.rl_trainer import RLColocateTrainer


@pytest.fixture(scope="module", autouse=True)
def local_ray():
    ray.init(address="local", num_cpus=2, include_dashboard=False, object_store_memory=128 * 1024**2)
    yield
    ray.shutdown()


@pytest.mark.parametrize("cleanup", [reset_rollout_response, discard_rollout_state])
def test_reset_or_discard_preserves_sibling_routes(cleanup):
    data = np.ones((4096, 2, 2), dtype=np.int64)
    ref = ray.put(data)
    first = RolloutState(message=[], routed_experts=[ref])
    sibling = RolloutState(message=[], routed_experts=[ref])
    cleanup(first)
    assert first.routed_experts is None
    np.testing.assert_array_equal(ray.get(sibling.routed_experts[0], timeout=5), data)


def test_partial_rollout_concat_preserves_borrowed_inputs():
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


@ray.remote(num_cpus=0)
class BatchConsumer:
    def load(self, refs):
        self._cached_samples = refs
        self._cached_pack_plan = []
        self._cached_batch_attr = {}

    def read(self):
        return int(ray.get(self._cached_samples[0]).sum())

    def release(self):
        TrainingWorker.release_train_batch(self)


def test_trace_overwrite_and_batch_release_preserve_other_consumers():
    store = RolloutTraceStore.remote()
    first, second = BatchConsumer.remote(), BatchConsumer.remote()
    try:
        ref = ray.put(np.ones(32768, dtype=np.int64))
        ray.get(store.insert.remote("s", "prompt", {"routed_experts": ref}))
        ray.get([first.load.remote([ref]), second.load.remote([ref])])
        del ref
        ray.get(store.insert.remote("s", "prompt", {"routed_experts": None}))
        ray.get(store.release.remote("s"))
        ray.get(first.release.remote())
        assert ray.get(second.read.remote(), timeout=10) == 32768
        ray.get(second.release.remote())
    finally:
        for actor in (store, first, second):
            ray.kill(actor)


def test_repeated_batches_drop_all_route_references():
    store = RolloutTraceStore.remote()
    consumer = BatchConsumer.remote()
    object_ids = []
    try:
        for index in range(8):
            ref = ray.put(np.ones(32768, dtype=np.int64))
            object_ids.append(ref.hex())
            ray.get(store.insert.remote(str(index), "prompt", {"routed_experts": ref}))
            ray.get(consumer.load.remote([ref]))
            del ref
            ray.get(store.release.remote(str(index)))
            assert ray.get(consumer.read.remote()) == 32768
            ray.get(consumer.release.remote())
        # Inspect the owner's reference table without recreating ObjectRefs,
        # which would itself keep these objects alive.
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


def test_trained_batch_retires_refs_but_preserves_external_borrower(tmp_path):
    store = RolloutTraceStore.remote()
    ref = ray.put(np.ones(32768, dtype=np.int64))
    state = RolloutState(message=[], session_id=1, routed_experts=ref)
    ray.get(store.insert.remote("1", "prompt", {"routed_experts": ref}))
    trainer = RLColocateTrainer.__new__(RLColocateTrainer)
    trainer._meta = SimpleNamespace(latest_exp=SimpleNamespace(exp_dir=tmp_path))
    trainer.logger = Mock()
    trainer._train_worker_cfg = SimpleNamespace(pack_max_length=4)
    trainer._prepare_train_batch = Mock(return_value=(0, 0))
    trainer._save_trajectories = Mock()
    trainer.train_controller = SimpleNamespace(fit=Mock(return_value=([], {})))
    trainer._release_trace_sessions_after_train_batch = lambda batch: ray.get(store.release_sessions.remote(["1"]))
    try:
        trainer._train_one_batch([[state]], train_step=1, step_timer_dict={})
        assert state.routed_experts is None
        assert ray.get(store.list_sessions.remote()) == []
        assert ray.get(ref, timeout=5).sum() == 32768
    finally:
        ray.kill(store)
