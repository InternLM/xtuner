from . import StrEnum


class ForwardState(StrEnum):
    PREFILLING = "prefilling"
    DECODING = "decoding"
    TRAINING = "training"
