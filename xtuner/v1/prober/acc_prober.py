from pathlib import Path
import json

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
import torch.nn as nn


def get_dtensor_meta(dtensor: torch.Tensor):
    if not isinstance(dtensor, DTensor):
        return {}
    
    dtensor: DTensor
    return {
        "local_shape": dtensor._local_tensor.shape,
        "device_mesh": str(dtensor.device_mesh),
        "placements": str(dtensor.placements),
    }
    

class Prober:
    dump_dir: Path | None = None
    profile_step: list[int] | None = None
    model: nn.Module | None = None
    initialized: bool = False
    cur_step: int = -1
    cur_micro_batch_iter: int = 0

    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int], model):
        cls.dump_dir = dump_home
        cls.dump_dir.mkdir(parents=True, exist_ok=True)
        cls.profile_step = profile_step
        cls.model = model
        cls.initialized = True
    
    @classmethod
    def skip(cls):
        if cls.cur_step not in cls.profile_step:
            return True
        if dist.get_rank() != 0:
            return True
        return False
    
    @classmethod
    def set_step(cls, step: int):
        cls.cur_step = step
    
    @classmethod
    def set_micro_batch_iter(cls, iter: int):
        cls.cur_micro_batch_iter = iter
    
    @classmethod
    def record_tensor(cls, tensor: torch.Tensor, name: str):
        if cls.skip():
            return
        assert cls.initialized, "AccProber is not initialized, please call setup() first"
        tensor = tensor.detach().clone()
        cur_json = {
            "name": name,
            "tensor_sum": tensor.float().sum().item(),
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
            "step": cls.cur_step,
            "micro_batch_iter": cls.cur_micro_batch_iter,
            "tensor_info": str(tensor),
        }
        cls.forward_records.append(json.dumps(cur_json, ensure_ascii=False))


class AccProber(Prober):
    forward_records: list = []

    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int], model):
        super().setup(dump_home, profile_step, model)
        cls.dump_dir = cls.dump_dir / "acc_prober"
        cls.dump_dir.mkdir(parents=True, exist_ok=True)
        cls.forward_records = []

    # Below is for checking forward pass activations
    @classmethod
    def before_embed_tokens(cls, input_ids: torch.Tensor):
        cls.record_tensor(input_ids, "[embed_tokens][before]input_ids")

    @classmethod
    def after_embed_tokens(cls, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, "[embed_tokens][after]hidden_states")

    @classmethod
    def before_lm_head(cls, hidden_states: torch.Tensor, shifted_labels: torch.Tensor):
        cls.record_tensor(hidden_states, "[lm_head][before]hidden_states")
        cls.record_tensor(shifted_labels, "[lm_head][before]shifted_labels")

    @classmethod
    def after_lm_head(cls, loss: torch.Tensor, logits: torch.Tensor):
        cls.record_tensor(loss, "[lm_head][after]loss")
        cls.record_tensor(logits, "[lm_head][after]logits")

    @classmethod
    def dump_forward_records(cls):
        if cls.skip():
            return
        assert cls.initialized, "AccProber is not initialized, please call setup() first"
        dump_file = cls.dump_dir.joinpath(f"Step_{cls.cur_step}_MicroIter_{cls.cur_micro_batch_iter}_RANK_{dist.get_rank()}_forward_records.jsonl")
        with open(dump_file, "w", encoding="utf-8") as f:
            for record in cls.forward_records:
                f.write(record + "\n")
        print(f"Dump forward records to {dump_file}")

        cls.forward_records = []

    # Below is for checking gradient
    @classmethod
    def grad_dump(cls, suffix: str):
        assert cls.initialized, "AccProber is not initialized, please call setup() first"
        model = cls.model

        if cls.skip():
            return

        res = []
        trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        for name, param in trainable_params:
            assert param.grad is not None, f"Error: {name} param.grad must not be None"
            # print(f"name: {name}, grad: {param.grad}")
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
        
        dump_file = cls.dump_dir.joinpath(f"STEP_{cls.cur_step}_RANK_{dist.get_rank()}_{suffix}.jsonl")
        with open(dump_file, "w", encoding="utf-8") as f:
            for line in res:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"Dump {suffix} to {dump_file}")

    @classmethod
    def before_clip_grad_norm(cls):
        cls.grad_dump("before_clip_grad_norm")

    @classmethod
    def after_clip_grad_norm(cls):
        cls.grad_dump("after_clip_grad_norm")