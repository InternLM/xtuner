"""Tests for saved-tensor offload lifecycle management."""

from unittest import mock

from xtuner.v1.utils.activation_offload import OffloadItem, OffloadManager


class TestOffloadManager:
    def test_clear_step_synchronizes_and_preserves_pin_cache(self):
        manager = OffloadManager()
        manager.clear(clear_pin_memory_cache=True)
        stream = mock.Mock()
        try:
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
        finally:
            manager.clear(clear_pin_memory_cache=True)
