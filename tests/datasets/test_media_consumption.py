from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset, Sampler

from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.datasets.dataloader import Dataloader
from xtuner.v1.datasets.media import MaterializedMedia, MediaEntry, MediaPlan, MediaSource, TensorSpec
from xtuner.v1.datasets.sampler import ParallelSampler
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.train.trainer import Trainer


class CursorDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return index

    def get_state_dict(self):
        return {"manifest": "fixture"}

    def load_state_dict(self, state):
        assert state["manifest"] == "fixture"


class CursorSampler(Sampler):
    def __init__(self):
        self.offset = 0

    def __iter__(self):
        return iter(range(self.offset, 4))

    def __len__(self):
        return 4 - self.offset

    def get_state_dict(self, consumed):
        return {"offset": consumed}

    def load_state_dict(self, state):
        self.offset = state["offset"]


def make_loader(batch_size=1):
    return Dataloader(CursorDataset(), sampler=CursorSampler(), batch_size=batch_size, collate_fn=list, num_workers=0)


def test_uncommitted_lookahead_is_replayed_on_resume():
    loader = make_loader()
    loader.enable_explicit_consumption()
    iterator = iter(loader)
    assert next(iterator) == [0]
    assert next(iterator) == [1]
    assert loader.get_state_dict()["total_consumed_samples"] == 0
    loader.commit_batch(1)
    state = loader.get_state_dict()
    assert state["total_consumed_samples"] == 1
    assert state["media_consumption"]["uncommitted_envelopes"] == 1
    resumed = make_loader()
    resumed.load_state_dict(state)
    resumed.enable_explicit_consumption()
    assert next(iter(resumed)) == [1]


def test_commit_uses_envelope_sample_units_and_rejects_wrong_count():
    loader = make_loader(batch_size=2)
    loader.enable_explicit_consumption()
    assert next(iter(loader)) == [0, 1]
    with pytest.raises(ValueError, match="sample count"):
        loader.commit_batch(1)
    loader.commit_batch(2)
    assert loader.get_state_dict()["total_consumed_samples"] == 2


def test_native_delivered_cursor_is_unchanged_when_disabled():
    loader = make_loader()
    assert next(iter(loader)) == [0]
    assert loader.get_state_dict()["total_consumed_samples"] == 1
    assert "media_consumption" not in loader.get_state_dict()


class EpochLoader:
    def __init__(self, values):
        self.values = values
        self.reads = []
        self.epochs = []

    def __iter__(self):
        for value in self.values:
            self.reads.append(value)
            yield value

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def make_trainer(loader, steps):
    trainer = object.__new__(Trainer)
    trainer._dataloader = loader
    trainer._cur_step = 0
    trainer._cur_epoch = 1
    trainer._total_step = steps
    return trainer


def test_prefetch_does_not_read_past_remaining_optimizer_steps():
    loader = EpochLoader([0, 1, 2, 3])
    trainer = make_trainer(loader, 3)
    output = []
    for batch in trainer._media_data_iter():
        output.append(batch)
        trainer._cur_step += 1
    assert output == [0, 1, 2]
    assert loader.reads == [0, 1, 2]
    assert trainer._media_lookahead is None


def test_prefetch_epoch_transition_waits_until_current_is_consumed():
    loader = EpochLoader([0, 1])
    trainer = make_trainer(loader, 3)
    iterator = trainer._media_data_iter()
    assert next(iterator) == 0
    assert loader.epochs == []
    trainer._cur_step += 1
    assert next(iterator) == 1
    assert loader.epochs == []
    trainer._cur_step += 1
    assert next(iterator) == 0
    assert loader.epochs == [2]
    trainer._cur_step += 1
    with pytest.raises(StopIteration):
        next(iterator)
    assert loader.reads == [0, 1, 0]


def test_empty_epoch_fails_instead_of_spinning():
    trainer = make_trainer(EpochLoader([]), 1)
    with pytest.raises(RuntimeError, match="empty"):
        next(trainer._media_data_iter())


def make_epoch_loader():
    dataset = CursorDataset()
    sampler = ParallelSampler(dataset, global_batch_size=1, shuffle=True, seed=17)
    loader = Dataloader(dataset, sampler=sampler, batch_size=1, collate_fn=list, num_workers=0)
    loader.enable_explicit_consumption()
    return loader


def checkpoint_cursor(trainer, tmp_path, monkeypatch):
    trainer._media_pipeline = SimpleNamespace(enabled=True)
    trainer._engine = SimpleNamespace(_media_runtime=None)
    monkeypatch.setattr("xtuner.v1.train.trainer.get_rank", lambda: 0)
    captured = {}
    monkeypatch.setattr(torch, "save", lambda state, path: captured.update(state))
    trainer._save_dataloader(tmp_path / "cursor.pt")
    return captured


@pytest.mark.parametrize("committed", [2, 4])
def test_checkpoint_restores_next_sample_and_epoch_at_middle_and_boundary(tmp_path, monkeypatch, committed):
    loader = make_epoch_loader()
    trainer = make_trainer(loader, 6)
    iterator = trainer._media_data_iter()
    for _ in range(committed):
        next(iterator)
        loader.commit_batch(1)
        trainer._cur_step += 1
    state = checkpoint_cursor(trainer, tmp_path, monkeypatch)
    expected = next(iterator)
    expected_epoch = trainer._cur_epoch
    resumed_loader = make_epoch_loader()
    resumed = make_trainer(resumed_loader, 6)
    resumed._cur_step = committed
    resumed._media_pipeline = SimpleNamespace(enabled=True)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: state)
    checkpoint = tmp_path / "cursor.pt"
    checkpoint.touch()
    resumed._resume_dataloader(checkpoint)
    resumed_iterator = resumed._media_data_iter()
    assert next(resumed_iterator) == expected
    assert resumed._cur_epoch == expected_epoch
    assert state["media_consumption"]["at_epoch_boundary"] == (committed == 4)
    iterator.close()
    resumed_iterator.close()


def test_committed_epoch_boundary_is_known_without_reading_beyond_final_step(tmp_path, monkeypatch):
    loader = make_epoch_loader()
    trainer = make_trainer(loader, 4)
    iterator = trainer._media_data_iter()
    for _ in range(4):
        next(iterator)
        loader.commit_batch(1)
        trainer._cur_step += 1
    state = checkpoint_cursor(trainer, tmp_path, monkeypatch)
    assert state["media_consumption"]["at_epoch_boundary"]
    assert state["media_consumption"]["next_epoch"] == 2
    assert state["sampler"]["epoch"] == 2
    assert state["sampler"]["step"] == 0
    assert trainer._cur_epoch == 1
    iterator.close()


def test_epoch_cannot_advance_over_uncommitted_envelope():
    loader = make_epoch_loader()
    iterator = iter(loader)
    next(iterator)
    with pytest.raises(RuntimeError, match="uncommitted"):
        loader.set_epoch(2)
    iterator.close()


def test_engine_lease_keeps_plan_only_original_and_releases_after_backward():
    spec = TensorSpec("pixel_values", "float32", (2, 3))
    entry = MediaEntry("one:0", (MediaSource("fixture://one", "frozen", 16, 16),), (spec,), 1024)
    plan = MediaPlan("fixture", "1", "{}", "processor", "manifest", (entry,), (spec,))
    context = SequenceContext.from_input_ids((torch.tensor([[1, 2]]),), device="cpu")
    context.media_plan = plan
    context.media_pack_id = "pack-0"
    weight = torch.nn.Parameter(torch.tensor(2.0))
    events = []

    class FakeRuntime:
        @contextmanager
        def consume(self, requested_plan, **kwargs):
            assert requested_plan is plan
            events.append("acquire")
            try:
                yield MaterializedMedia({"pixel_values": torch.ones(2, 3)})
            finally:
                assert weight.grad is not None
                events.append("release")

    engine = object.__new__(TrainEngine)
    engine._media_runtime = FakeRuntime()
    engine._media_epoch = 1
    engine._media_step = 0
    with engine._consume_media([context], 0) as resolved:
        assert resolved[0] is not context
        assert resolved[0].media_plan is None
        assert context.media_plan is plan and context.pixel_values is None
        (resolved[0].pixel_values.sum() * weight).backward()
        events.append("backward")
    assert events == ["acquire", "backward", "release"]


def test_engine_forward_only_uses_same_consumption_entry():
    context = SequenceContext.from_input_ids((torch.tensor([[1, 2]]),), device="cpu")
    engine = object.__new__(TrainEngine)
    engine._media_runtime = None
    engine.model = lambda **kwargs: SimpleNamespace(context=kwargs["seq_ctx"])
    assert engine.forward_only(context, None).context is context
