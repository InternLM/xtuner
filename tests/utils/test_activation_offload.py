"""Tests for saved-tensor offload lifecycle management."""

from unittest import mock

import pytest

from xtuner.v1.utils.activation_offload import OffloadItem, OffloadManager, SingletonMeta


@pytest.fixture
def manager():
    # OffloadManager is a process-level singleton; rebuild it per test so no
    # runtime state (including registered streams) leaks across cases.
    SingletonMeta._instances.pop(OffloadManager, None)
    yield OffloadManager()
    SingletonMeta._instances.pop(OffloadManager, None)


class TestOffloadManager:
    def test_clear_step_synchronizes_and_preserves_pin_cache(self, manager):
        stream = mock.Mock()
        manager.register_stream("text", stream)
        manager.items["text_0_0"] = OffloadItem()
        manager.may_npu_tensors["text_0_1"] = OffloadItem()
        manager.items["other_0_0"] = OffloadItem()
        manager.pin_memory_cache["text_0_0"] = mock.sentinel.pinned_buffer

        manager.clear_step(group="text")

        stream.synchronize.assert_called_once_with()
        assert "text_0_0" not in manager.items
        assert "text_0_1" not in manager.may_npu_tensors
        assert "other_0_0" in manager.items
        assert manager.pin_memory_cache["text_0_0"] is mock.sentinel.pinned_buffer

    def test_clear_releases_offload_streams(self, manager):
        stream = mock.Mock()
        manager.register_stream("text", stream)
        manager.items["text_0_0"] = OffloadItem()

        manager.clear(group="text")

        assert "text" not in manager.offload_streams

    def test_clear_step_is_noop_without_offload(self, manager):
        manager.clear_step()  # no stream, no entries; must not raise
        assert not manager.items
        assert not manager.offload_streams
