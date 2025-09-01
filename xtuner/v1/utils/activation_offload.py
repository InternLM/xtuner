"""
This file is adapted from: https://gitee.com/ascend/MindSpeed-MM/blob/master/mindspeed_mm/utils/async_offload.py
Original Author: liyx616
Original License: MIT
Modifications: To enable compatibility on both GPU and NPU, replace all torch.npu with torch.cuda, and then use the transfer_to_npu interface
"""

import torch
from torch.autograd.graph import saved_tensors_hooks


if torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "npu":
    from torch_npu.contrib import transfer_to_npu  # noqa


def base_check_fn(tensor):
    if isinstance(tensor._base, torch.nn.parameter.Parameter) or isinstance(tensor, torch.nn.parameter.Parameter):
        return False
    if tensor.storage().size() <= 0:
        return False
    return True


class GetCnt:
    def __init__(self):
        self._block_idx = -1
        self._block_tensor_nums = {}  # offload tensors per block

    def get_cnt(self, block_idx):
        after_block = False
        if block_idx > self._block_idx:
            self._block_tensor_nums[block_idx] = 1
            if block_idx != 0:
                after_block = True
            self._block_idx = block_idx
        elif block_idx == self._block_idx:
            self._block_tensor_nums[block_idx] += 1
        else:
            # one step end
            self._block_idx = block_idx
            self._block_tensor_nums = {block_idx: 1}

        offload_tensor_key = f"{self._block_idx}_{self._block_tensor_nums[self._block_idx] - 1}"
        return offload_tensor_key, after_block

    def get_prefetch_keys(self, block_idx, tensor_idx):
        prefetch_block_idx = max((idx for idx in self._block_tensor_nums.keys() if idx < block_idx), default=None)

        if prefetch_block_idx is None:
            return []

        prefetch_block_tensor_nums = self._block_tensor_nums[prefetch_block_idx]
        block_tensor_nums = self._block_tensor_nums[block_idx]
        start = tensor_idx * prefetch_block_tensor_nums // block_tensor_nums
        end = (tensor_idx + 1) * prefetch_block_tensor_nums // block_tensor_nums
        prefetch_idxs = list(range(start, end))
        return [f"{block_idx - 1}_{prefetch_idx}" for prefetch_idx in prefetch_idxs]


class SwapTensor:
    def __init__(self, tensor, key):
        self.tensor = tensor
        self.size = tensor.size()
        self.storage_size = tensor.storage().size()
        self.tensor_cpu = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True, device="cpu")

        self.is_slice_tensor = tensor.storage().size() != tensor.numel()
        self.stat = "device"
        self.key = key

        self.h2d_event = torch.cuda.Event()

    # device to host
    def launch_d2h(self, stream):
        if self.stat != "device":
            return

        forward_event = torch.cuda.Event()
        forward_event.record()
        with torch.no_grad():
            with torch.cuda.stream(stream):
                stream.wait_event(forward_event)
                if self.is_slice_tensor:
                    self.tensor_cpu.copy_(self.tensor, non_blocking=True)
                else:
                    self.tensor_cpu.storage().copy_(self.tensor.storage(), non_blocking=True)
                self.stat = "host"

    # synchronize d2h and resize 0
    def wait_d2h_finished(self, stream, flag):
        if self.stat != "host":
            return
        if flag:
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.default_stream().wait_stream(stream)
        self.tensor.storage().resize_(0)
        self.stat = "host"

    # resize storage_size and host to device
    def launch_h2d(self, h2d_stream, flag, working_stream):
        if self.stat != "host":
            return
        backward_event = torch.cuda.Event()
        backward_event.record()
        if flag:
            self.tensor.storage().resize_(self.storage_size)
        with torch.no_grad():
            with torch.cuda.stream(h2d_stream):
                h2d_stream.wait_event(backward_event)
                if self.is_slice_tensor:
                    self.tensor.copy_(self.tensor_cpu, non_blocking=True)
                else:
                    self.tensor.storage().copy_(self.tensor_cpu.storage(), non_blocking=True)
                self.h2d_event.record()
                self.stat = "device"

                working_stream.wait_stream(h2d_stream)

    # resize storage_size and host to device
    def prefetch_launch_h2d(self, h2dstream, flag):
        if self.stat != "host":
            return
        backward_event = torch.cuda.Event()
        backward_event.record()
        if flag:
            self.tensor.storage().resize_(self.storage_size)
        with torch.no_grad():
            with torch.cuda.stream(h2dstream):
                h2dstream.wait_event(backward_event)
                if self.is_slice_tensor:
                    self.tensor.copy_(self.tensor_cpu, non_blocking=True)
                else:
                    self.tensor.storage().copy_(self.tensor_cpu.storage(), non_blocking=True)
                self.h2d_event.record()
                self.stat = "device"
                self.tensor.record_stream(h2dstream)

    # synchronize h2d
    def wait_h2d_finished(self):
        if self.stat != "device":
            return
        if self.h2d_event:
            torch.cuda.current_stream().wait_event(self.h2d_event)
            torch.cuda.default_stream().wait_event(self.h2d_event)
        self.stat = "device"


class SingletonMeta(type):
    """Single meta class."""

    _instances = {}  # type: ignore

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]


class OffloadItem:
    """Class for offload item."""

    def __init__(self, act=None, ref_cnt=0, event=None):
        self.act = act
        self.ref_cnt = ref_cnt
        self.event = event

    def get_event(self):
        return self.event

    def has_event(self):
        return self.event is not None


class OffloadManager(metaclass=SingletonMeta):
    """Class for offload manager."""

    def __init__(self, check=False):
        self.items = {}
        self.check = check
        self.device_item = []
        self.getcnt = GetCnt()

    def get_cnt(self, block_idx):
        return self.getcnt.get_cnt(block_idx)

    def assert_exist(self, key):
        if key not in self.items:
            raise RuntimeError(f"Key {key} does not exist in items")

    def exist(self, key):
        return key in self.items

    def assert_not_exist(self, key):
        if key not in self.items:
            raise RuntimeError(f"Key {key} already exist in items")

    def put(self, key, act, event=None):
        if key in self.items:
            self.items[key].act = act
            self.items[key].ref_cnt += 1
            self.items[key].event = event
        else:
            self.items[key] = OffloadItem(act, 1, event)

    def put_npu_tensor(self, act):
        self.device_item.append(act)

    def del_npu_tensor(self, prefile_key, d2h_stream):
        for key in self.items.keys():
            if key.startswith(prefile_key):
                self.items[key].act.wait_d2h_finished(d2h_stream, True)

    def get(self, key):
        self.assert_exist(key)
        item = self.items[key]

        act = item.act
        if item.has_event():
            item.get_event().wait()

        item.ref_cnt -= 1
        if item.ref_cnt == 0:
            self.clear(key)
        return act

    def prefetch_get(self, block_idx, tensor_idx, h2d_stream, d2h_stream):
        prefetch_keys = self.getcnt.get_prefetch_keys(block_idx, tensor_idx)
        for prefetch_key in prefetch_keys:
            if self.exist(prefetch_key):
                prefetch_swap_tensor = self.get(prefetch_key)
                d2h_stream.wait_stream(h2d_stream)
                prefetch_swap_tensor.prefetch_launch_h2d(h2d_stream, True)
                prefetch_swap_tensor.tensor.record_stream(h2d_stream)

    def empty(self):
        return len(self.items) == 0

    def clear(self, key=None):
        if key is None:
            self.items.clear()
        else:
            self.assert_exist(key)
            self.items.pop(key)

    # event interface #

    def get_event(self, key):
        self.assert_exist(key)
        item = self.items[key]
        event = item.get_event()
        return event

    def has_event(self, key):
        if not self.exist(key):
            return False
        item = self.items[key]
        return item.has_event()


class async_save_on_cpu(saved_tensors_hooks):
    def __init__(self, h2d_stream, d2h_stream, block_idx, depth, custom_check_fn=None, prefetch=True) -> None:
        def _pack_to_cpu(tensor):
            if not base_check_fn(tensor):
                return tensor

            if (custom_check_fn is not None) and (not custom_check_fn(tensor)):
                return tensor

            key, after_block = OffloadManager().get_cnt(block_idx)

            if after_block:
                OffloadManager().del_npu_tensor(f"{block_idx - 1}_", d2h_stream)

            swap_tensor = SwapTensor(tensor, key)

            if block_idx < depth - 1:
                working_stream = torch.cuda.current_stream()
                d2h_stream.wait_stream(working_stream)
                swap_tensor.launch_d2h(d2h_stream)

            OffloadManager().put(key, swap_tensor)
            return swap_tensor

        def _unpack_from_cpu(swap_tensor) -> torch.Tensor:
            if isinstance(swap_tensor, torch.Tensor):
                return swap_tensor

            working_stream = torch.cuda.current_stream()
            working_stream.wait_stream(h2d_stream)  # make sure all d2h copy is done before into backward

            h2d_stream.wait_stream(working_stream)

            swap_tensor.launch_h2d(h2d_stream, True, working_stream)

            if prefetch:
                block_idx, tensor_idx = swap_tensor.key.split("_")
                OffloadManager().prefetch_get(int(block_idx), int(tensor_idx), h2d_stream, d2h_stream)
            return swap_tensor.tensor

        super().__init__(_pack_to_cpu, _unpack_from_cpu)
