from mmengine.utils import ManagerMixin


class AttentionContext(ManagerMixin):

    def __init__(self) -> None:
        self._cumulative_len = None
        self._max_seqlen = None

    def update(self, seqlen_list):
        cumulative_len = [0]
        max_seqlen = 0
        for seqlen in seqlen_list:
            cumulative_len.append(cumulative_len[-1] + seqlen)
            max_seqlen = max(max_seqlen, seqlen)
        self._cumulative_len = cumulative_len
        self._max_seqlen = max_seqlen

    def clear(self):
        self._cumulative_len = None
        self._max_seqlen = None

    @property
    def cumulative_len(self):
        return self._cumulative_len

    @property
    def max_seqlen(self):
        return self._max_seqlen
