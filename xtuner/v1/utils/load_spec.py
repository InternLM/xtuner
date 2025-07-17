import torch.distributed as dist
from pydantic import BaseModel, ConfigDict

from .enum_helper import StrEnum


class LoadEnum(StrEnum):
    FUSED = "fused"
    SAME = "same"
    SHARD = "shard"


class LoadSpec(BaseModel):
    # TODO: (yehaochen) Add more description
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    hf_keys: list[str]
    shape: tuple[int, ...]
    dim: int | None = None
    load_enum: LoadEnum
    shard_start: int | None = None
    shard_end: int | None = None
    group: dist.ProcessGroup | None = None

    def model_post_init(self, _) -> None:
        if self.load_enum == LoadEnum.SAME:
            assert len(self.hf_keys) == 1, "hf_keys should have exactly one key when load_enum is SAME"
        elif self.load_enum == LoadEnum.FUSED:
            if self.dim is None:
                self.dim = 0
            assert self.dim == 0, "dim should be 0 when load_enum is FUSED"
        elif self.load_enum == LoadEnum.SHARD:
            assert self.dim is not None, "dim should not be None when load_enum is SHARD"
            assert len(self.hf_keys) > 1, "hf_keys should have more than one key when load_enum is SHARD"
            assert self.shard_start is not None, "shard_start should not be None when load_enum is SHARD"
            assert self.shard_end is not None, "shard_end should not be None when load_enum is SHARD"
