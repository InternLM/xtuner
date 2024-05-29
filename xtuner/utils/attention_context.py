from mmengine.utils import ManagerMixin


class MessageHub(ManagerMixin):

    def __init__(self, name: str = '', **kwargs):
        super().__init__(name, **kwargs)
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


class AttentionContext:

    message_hub = MessageHub.get_instance('attention_context')

    @classmethod
    def update(cls, seqlen_list):
        cls.message_hub.update(seqlen_list)

    @classmethod
    def clear(cls):
        cls.message_hub.clear()

    @classmethod
    def get_max_seqlen(cls):
        return cls.message_hub.max_seqlen

    @classmethod
    def get_cumulative_len(cls):
        return cls.message_hub.cumulative_len
