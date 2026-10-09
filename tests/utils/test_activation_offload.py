"""Tests for saved-tensor offload lifecycle management."""

from unittest import mock

import pytest

from xtuner.v1.utils.activation_offload import OffloadItem, OffloadManager, SingletonMeta


@pytest.fixture
def manager():
    # OffloadManager is a process-level singleton; rebuild it per test so no
    # runtime state leaks across cases.
    SingletonMeta._instances.pop(OffloadManager, None)
    yield OffloadManager()
    SingletonMeta._instances.pop(OffloadManager, None)


class TestOffloadManager:
    def test_clear_step_releases_runtime_state_and_preserves_pin_cache(self, manager):
        manager.items["text_0_0"] = OffloadItem()
        manager.may_npu_tensors["text_0_1"] = OffloadItem()
        manager.items["other_0_0"] = OffloadItem()
        manager.pin_memory_cache["text_0_0"] = mock.sentinel.pinned_buffer

        manager.clear_step(group="text")

        assert "text_0_0" not in manager.items
        assert "text_0_1" not in manager.may_npu_tensors
        assert "other_0_0" in manager.items
        assert manager.pin_memory_cache["text_0_0"] is mock.sentinel.pinned_buffer

    def test_clear_step_is_noop_without_offload(self, manager):
        manager.clear_step()  # no entries; must not raise
        assert not manager.items
