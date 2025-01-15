from loguru import logger

GB_SHIFT = 30
MB_SHIFT = 20


class CudaMemoryStats(dict):
    # see: https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html  # noqa: E501
    # def add_memory_stats(self, key, device):
    #     import torch
    #     status = torch.cuda.memory_stats(device=device)
    #     self.__setattr__(key, status)

    @property
    def num_gpus(self):
        return len(self.keys())

    @property
    def total_current_bytes(self):
        CURRENT_BYTE_KEY = 'allocated_bytes.all.current'
        total = 0
        for _, v in self.items():
            total += v.get(CURRENT_BYTE_KEY, 0)
        return total

    @property
    def total_current_gb(self):
        return self.total_current_bytes >> GB_SHIFT

    @property
    def total_current_mb(self):
        return self.total_current_bytes >> MB_SHIFT

    @property
    def avg_current_bytes(self):
        return self.total_current_bytes / self.num_gpus if self.num_gpus != 0 else 0  # noqa: E501

    def __repr__(self):
        return f'CudaMemoryStats: {self.num_gpus} GPU takes {self.total_current_mb} MiB'  # noqa: E501


def merge_cuda_memory_stats_list(
        dict_list: list[CudaMemoryStats]) -> CudaMemoryStats:
    if isinstance(dict_list, CudaMemoryStats):
        logger.warning('dict_list is a CudaMemoryStatus instead of a list')
        return dict_list
    memory_stats_dict: CudaMemoryStats = dict_list[0]
    assert isinstance(memory_stats_dict, CudaMemoryStats)
    if len(dict_list) > 1:
        for m in dict_list[1:]:
            memory_stats_dict.update(m)
    return memory_stats_dict
