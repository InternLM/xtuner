import torch


class CacheDataset(torch.utils.data.Dataset):

    @property
    def cached_dir(self):
        pass

    @property
    def cached(self):
        pass

    def cache(self, cache_dir):
        pass

    def load_cache(self):
        pass

    @classmethod
    def from_cache(self, cache_dir):
        pass
