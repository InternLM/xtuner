from . import StrEnum


class State(StrEnum):
    PREFILLING = "prefilling"
    DECODING = "decoding"
    TRAINING = "training"
