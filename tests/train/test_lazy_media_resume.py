"""CPU-only checkpoint/resume coverage for one-batch async lookahead."""

import io
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from xtuner.v1.datasets.dataloader import Dataloader
from xtuner.v1.datasets.preset_sampler import PresetSampler
from xtuner.v1.datasets.sampler import LengthGroupedSampler, ParallelSampler
from xtuner.v1.train.trainer import Trainer


class Records(Dataset):
    longest = [8, 3, 6, 1]

    def __init__(self):
        self.reads = []
        self.condition = threading.Condition()

    def __len__(self):
        return 4

    def __getitem__(self, index):
        with self.condition:
            self.reads.append(index)
            self.condition.notify_all()
        return index


def make_trainer(tmp_path, kind, batch_size, steps=7, dp_mesh=None):
    dataset = Records()
    global_batch_size = batch_size * (dp_mesh.size() if dp_mesh is not None else 1)
    if kind == "preset":
        order = tmp_path / "order.npy"
        np.save(order, np.array([2, 0, 3, 1]))
        sampler = PresetSampler(dataset, str(order), global_batch_size=global_batch_size, dp_mesh=dp_mesh)
    else:
        cls = ParallelSampler if kind == "parallel" else LengthGroupedSampler
        sampler = cls(dataset, global_batch_size=global_batch_size, seed=17, dp_mesh=dp_mesh)
    loader = Dataloader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=list, num_workers=0)
    trainer = object.__new__(Trainer)
    trainer._dataloader = loader
    trainer._cur_epoch = 1
    trainer._cur_step = 0
    trainer._total_step = steps
    trainer.sp_mesh = SimpleNamespace(get_local_rank=lambda: 0)
    return trainer


def consume(trainer, iterator):
    batch = next(iterator)
    result = (trainer._cur_epoch, tuple(batch))
    if trainer._dataloader._defer_consumption:
        trainer._dataloader.commit_batch(len(batch))
    trainer._cur_step += 1
    return result


def checkpoint(trainer):
    # Serialize actual sampler/data state, not a live-object copy.
    buffer = io.BytesIO()
    torch.save((trainer._cur_epoch, trainer._cur_step, trainer._dataloader.get_state_dict()), buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


@pytest.mark.parametrize("kind", ["parallel", "length", "preset"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("cut", [1, 2, 4, 5])
def test_resume_matches_uninterrupted_across_epoch_boundaries(monkeypatch, tmp_path, kind, batch_size, cut):
    monkeypatch.setenv("XTUNER_LAZY_MEDIA_PREFETCH", "1")
    reference = make_trainer(tmp_path, kind, batch_size)
    iterator = reference._prefetched_data_iter()
    expected = [consume(reference, iterator) for _ in range(reference.total_step)]
    iterator.close()

    interrupted = make_trainer(tmp_path, kind, batch_size)
    iterator = interrupted._prefetched_data_iter()
    prefix = [consume(interrupted, iterator) for _ in range(cut)]
    state = checkpoint(interrupted)
    iterator.close()
    assert state[2]["total_consumed_samples"] == cut * batch_size

    resumed = make_trainer(tmp_path, kind, batch_size)
    resumed._cur_epoch, resumed._cur_step, loader_state = state
    resumed._dataloader.load_state_dict(loader_state)
    iterator = resumed._prefetched_data_iter()
    suffix = [consume(resumed, iterator) for _ in range(resumed.total_step - resumed._cur_step)]
    iterator.close()
    assert prefix + suffix == expected
    assert not any(thread.name.startswith("lazy-media-lookahead") for thread in threading.enumerate())


def test_prefetched_and_failed_current_batch_are_not_checkpointed(monkeypatch, tmp_path):
    monkeypatch.setenv("XTUNER_LAZY_MEDIA_PREFETCH", "1")
    trainer = make_trainer(tmp_path, "parallel", 1)
    iterator = trainer._prefetched_data_iter()
    first = consume(trainer, iterator)
    dataset = trainer._dataloader.dataset
    with dataset.condition:
        assert dataset.condition.wait_for(lambda: len(dataset.reads) >= 2, timeout=5)
    assert checkpoint(trainer)[2]["total_consumed_samples"] == 1
    failed_batch = next(iterator)  # Simulate failure before optimizer/commit.
    state = checkpoint(trainer)
    iterator.close()
    assert state[2]["total_consumed_samples"] == 1
    resumed = make_trainer(tmp_path, "parallel", 1)
    resumed._cur_epoch, resumed._cur_step, loader_state = state
    resumed._dataloader.load_state_dict(loader_state)
    iterator = resumed._prefetched_data_iter()
    assert consume(resumed, iterator) == (first[0], tuple(failed_batch))
    iterator.close()


def test_lookahead_does_not_advance_epoch_before_last_batch_commit(monkeypatch, tmp_path):
    monkeypatch.setenv("XTUNER_LAZY_MEDIA_PREFETCH", "1")
    trainer = make_trainer(tmp_path, "parallel", 1)
    iterator = trainer._prefetched_data_iter()
    for _ in range(3):
        consume(trainer, iterator)
    last = next(iterator)
    # Producer can reach StopIteration, but only the consumer switches epoch.
    assert trainer._cur_epoch == 1
    assert trainer._dataloader.sampler.epoch == 0
    assert checkpoint(trainer)[2]["total_consumed_samples"] == 3
    trainer._dataloader.commit_batch(len(last))
    trainer._cur_step += 1
    state = checkpoint(trainer)
    assert state[2]["sampler"]["step"] == 4
    assert state[2]["sampler"]["epoch"] == 0
    assert consume(trainer, iterator)[0] == 2
    iterator.close()


def test_disabled_keeps_delivered_cursor_and_checkpoint_schema(monkeypatch, tmp_path):
    monkeypatch.delenv("XTUNER_LAZY_MEDIA_PREFETCH", raising=False)
    trainer = make_trainer(tmp_path, "parallel", 1)
    iterator = trainer._prefetched_data_iter()
    next(iterator)
    assert not trainer._dataloader._defer_consumption
    state = checkpoint(trainer)[2]
    assert state["total_consumed_samples"] == 1
    assert set(state) == {"sampler", "dataset", "total_consumed_samples"}
    iterator.close()


@pytest.mark.parametrize("kind", ["parallel", "length", "preset"])
@pytest.mark.parametrize("rank", [0, 1])
def test_dp_global_offset_with_cpu_mocked_reduction(monkeypatch, tmp_path, kind, rank):
    from xtuner.v1.datasets import dataloader as module

    monkeypatch.setenv("XTUNER_LAZY_MEDIA_PREFETCH", "1")
    monkeypatch.setattr(module, "reduce_sum_across_dp_group", lambda mesh, value: value * 2)
    mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: rank)
    trainer = make_trainer(tmp_path, kind, 1, dp_mesh=mesh)
    iterator = trainer._prefetched_data_iter()
    consume(trainer, iterator)
    state = checkpoint(trainer)
    expected = consume(trainer, iterator)
    iterator.close()
    assert state[2]["total_consumed_samples"] == 2
    assert state[2]["sampler"]["step"] == 2
    resumed = make_trainer(tmp_path, kind, 1, dp_mesh=mesh)
    resumed._cur_epoch, resumed._cur_step, loader_state = state
    resumed._dataloader.load_state_dict(loader_state)
    iterator = resumed._prefetched_data_iter()
    assert consume(resumed, iterator) == expected
    iterator.close()


def test_no_commit_replays_first_batch(monkeypatch, tmp_path):
    monkeypatch.setenv("XTUNER_LAZY_MEDIA_PREFETCH", "1")
    trainer = make_trainer(tmp_path, "parallel", 1)
    iterator = trainer._prefetched_data_iter()
    failed = next(iterator)
    state = checkpoint(trainer)
    iterator.close()
    assert state[2]["total_consumed_samples"] == 0
    resumed = make_trainer(tmp_path, "parallel", 1)
    resumed._cur_epoch, resumed._cur_step, loader_state = state
    resumed._dataloader.load_state_dict(loader_state)
    iterator = resumed._prefetched_data_iter()
    assert consume(resumed, iterator)[1] == tuple(failed)
    iterator.close()


@pytest.mark.parametrize("optimizer_fails", [False, True])
def test_training_loop_commits_only_after_optimizer(monkeypatch, tmp_path, optimizer_fails):
    from contextlib import nullcontext
    from unittest.mock import Mock

    from xtuner.v1.train import trainer as module

    class StopTraining(Exception):
        pass

    monkeypatch.setenv("XTUNER_LAZY_MEDIA_PREFETCH", "1")
    monkeypatch.setattr(module, "ProberList", Mock())
    monkeypatch.setattr(module, "DEVICE_MODULE", Mock())
    trainer = make_trainer(tmp_path, "parallel", 1)
    trainer._async_hf_export = trainer._async_checkpoint = False
    trainer._offload_optimizer_first_step = False
    trainer._local_total_consumed_tokens = 0
    trainer._do_clip = False
    trainer._grad_norm_dtype = torch.float32
    trainer.hooks_config = Mock()
    trainer.hooks_config.get_hooks.return_value = []
    trainer._engine = Mock()
    trainer._engine.train_step.return_value = {"step_consumed_tokens": 1, "step_seqlen_tokens": 1}
    trainer._engine.clip_grad_norm.return_value = torch.tensor(0.0)
    trainer._maybe_profiling = nullcontext
    trainer._prepare_model_input = lambda batch: batch
    trainer._maybe_pop_model_internal_metrics = lambda inputs: None
    trainer._compute_performance_metrics = Mock(return_value={})
    trainer._log_step = Mock()
    trainer._lr_scheduler = Mock()
    trainer._maybe_check_health = Mock()
    trainer._maybe_save_hf = Mock()
    trainer._maybe_save = Mock(side_effect=StopTraining("checkpoint reached"))
    if optimizer_fails:
        trainer._engine.step_optimizer.side_effect = StopTraining("optimizer failed")
    iterator = trainer._prefetched_data_iter()
    trainer._prefetched_data_iter = lambda: iterator
    try:
        with pytest.raises(StopTraining, match="optimizer failed" if optimizer_fails else "checkpoint reached"):
            trainer.fit()
        state = checkpoint(trainer)
        assert state[1] == (0 if optimizer_fails else 1)
        assert state[2]["total_consumed_samples"] == (0 if optimizer_fails else 1)
        assert trainer._maybe_save.call_count == (0 if optimizer_fails else 1)
    finally:
        iterator.close()
