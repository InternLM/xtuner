from pathlib import Path
import json

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def get_dtensor_meta(dtensor: torch.Tensor):
    if not isinstance(dtensor, DTensor):
        return {}
    
    dtensor: DTensor
    return {
        "local_shape": dtensor._local_tensor.shape,
        "device_mesh": str(dtensor.device_mesh),
        "placements": str(dtensor.placements),
    }
    
        


class AccProber:
    dump_dir: Path | None = None
    profile_step: list[int] | None = None
    cur_step: int = -1

    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int]):
        cls.dump_dir = dump_home / "acc_prober"
        cls.dump_dir.mkdir(parents=True, exist_ok=True)
        cls.profile_step = profile_step
    
    @classmethod
    def set_step(cls, step: int):
        cls.cur_step = step

    @classmethod
    def before_clip_grad_norm(cls, model):
        assert cls.dump_dir is not None, "dump_dir is not set"
        assert cls.profile_step is not None, "profile_step is not set"
        assert cls.cur_step != -1, "cur_step is not set"

        if cls.cur_step not in cls.profile_step:
            return

        if dist.get_rank() != 0:
            return

        res = []
        trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        for name, param in trainable_params:
            assert param.grad is not None, "Internal Error: param.grad must not be None"
            print(f"name: {name}, grad: {param.grad}")
            grad = param.grad.detach().clone().view(-1)
            grad_sum = grad.float().sum()
            cur_json = {
                "name": name,
                "grad_sum": grad_sum.item(),
                "weight_sum": param.detach().clone().float().sum().item(),
                "shape": param.shape,
                "dtype": str(param.dtype),
                "param_info": str(param),
                # "grad_dtensor_meta": get_dtensor_meta(param.grad),
            }
            res.append(cur_json)
        
        dump_file = cls.dump_dir.joinpath(f"STEP_{cls.cur_step}_RANK_{dist.get_rank()}_before_clip_grad_norm.json")
        with open(dump_file, "w", encoding="utf-8") as f:
            for line in res:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print("Dump before_clip_grad_norm to ", dump_file)


    @classmethod
    def after_clip_grad_norm(cls, model):
        pass