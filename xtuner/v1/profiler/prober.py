from pathlib import Path
import json
import time
from abc import ABC, abstractmethod
from typing import ClassVar

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


class BaseProber(ABC):
    """
    抽象基类 - 定义Prober接口规范
    每个子类都是单例（通过类方法实现）
    """
    dump_dir: ClassVar[Path | None] = None
    profile_step: ClassVar[list[int] | None] = None
    model: ClassVar[nn.Module | None] = None
    initialized: ClassVar[bool] = False
    
    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int], model: nn.Module):
        """子类必须实现setup方法，用于初始化自己的dump_dir"""
        cls.dump_dir = dump_home
        cls.dump_dir.mkdir(parents=True, exist_ok=True)
        cls.profile_step = profile_step
        cls.model = model
        cls.initialized = True
    
    @classmethod
    def skip(cls) -> bool:
        if cls.profile_step is None or cls.cur_step not in cls.profile_step:
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
    
    # 钩子方法 - 子类选择性重写
    @classmethod
    def record_tensor(cls, tensor: torch.Tensor, name: str):
        pass

    @classmethod
    def before_embed_tokens(cls, input_ids: torch.Tensor):
        pass
    
    @classmethod
    def after_embed_tokens(cls, hidden_states: torch.Tensor):
        pass
    
    @classmethod
    def before_lm_head(cls, hidden_states: torch.Tensor, shifted_labels: torch.Tensor):
        pass
    
    @classmethod
    def after_lm_head(cls, loss: torch.Tensor, logits: torch.Tensor):
        pass
    
    @classmethod
    def dump_micro_iter_forward(cls):
        pass
    
    @classmethod
    def before_clip_grad_norm(cls):
        pass
    
    @classmethod
    def after_clip_grad_norm(cls):
        pass


class ProberList:
    prober_list: ClassVar[list[type[BaseProber]]] = []
    
    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int], model: nn.Module, 
              prober_classes: list[type[BaseProber]]):
        cls.prober_list = prober_classes
        # 初始化每个Prober
        for prober_cls in cls.prober_list:
            prober_cls.setup(dump_home, profile_step, model)
        
        print(f"ProberList initialized with {len(cls.prober_list)} probers: "
              f"{[p.__name__ for p in cls.prober_list]}")
    
    @classmethod
    def set_step(cls, step: int):
        for prober_cls in cls.prober_list:
            prober_cls.set_step(step)
    
    @classmethod
    def set_micro_batch_iter(cls, iter: int):
        for prober_cls in cls.prober_list:
            prober_cls.set_micro_batch_iter(iter)
    
    @classmethod
    def record_tensor(cls, tensor: torch.Tensor, name: str):
        for prober_cls in cls.prober_list:
            prober_cls.record_tensor(tensor, name)

    @classmethod
    def before_embed_tokens(cls, input_ids: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_embed_tokens(input_ids)
    
    @classmethod
    def after_embed_tokens(cls, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_embed_tokens(hidden_states)
    
    @classmethod
    def before_lm_head(cls, hidden_states: torch.Tensor, shifted_labels: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_lm_head(hidden_states, shifted_labels)
    
    @classmethod
    def after_lm_head(cls, loss: torch.Tensor, logits: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_lm_head(loss, logits)
    
    @classmethod
    def dump_micro_iter_forward(cls):
        for prober_cls in cls.prober_list:
            prober_cls.dump_micro_iter_forward()
    
    @classmethod
    def before_clip_grad_norm(cls):
        for prober_cls in cls.prober_list:
            prober_cls.before_clip_grad_norm()
    
    @classmethod
    def after_clip_grad_norm(cls):
        for prober_cls in cls.prober_list:
            prober_cls.after_clip_grad_norm()


class AccProber(BaseProber):
    forward_records: ClassVar[list] = []
    
    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int], model: nn.Module):
        super().setup(dump_home, profile_step, model)
        cls.dump_dir = dump_home / "acc_prober"
        cls.dump_dir.mkdir(parents=True, exist_ok=True)
        cls.forward_records = []
        print(f"AccProber initialized at {cls.dump_dir}")
    
    @classmethod
    def record_tensor(cls, tensor: torch.Tensor, name: str):
        """记录张量信息"""
        if cls.skip():
            return
        assert cls.initialized, "AccProber is not initialized, please call setup() first"
        tensor = tensor.detach().clone()
        cur_json = {
            "name": name,
            "tensor_sum": tensor.float().sum().item(),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "step": cls.cur_step,
            "micro_batch_iter": cls.cur_micro_batch_iter,
            "tensor_info": str(tensor),
        }
        cls.forward_records.append(json.dumps(cur_json, ensure_ascii=False))
    
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
    def dump_micro_iter_forward(cls):
        if cls.skip():
            return
        assert cls.initialized, "AccProber is not initialized, please call setup() first"
        dump_file = cls.dump_dir.joinpath(
            f"Step_{cls.cur_step}_MicroIter_{cls.cur_micro_batch_iter}_"
            f"RANK_{dist.get_rank()}_forward_records.jsonl"
        )
        with open(dump_file, "w", encoding="utf-8") as f:
            for record in cls.forward_records:
                f.write(record + "\n")
        print(f"[AccProber] Dump forward records to {dump_file}")
        cls.forward_records = []
    
    @classmethod
    def _grad_dump(cls, suffix: str):
        if cls.skip():
            return
        assert cls.initialized, "AccProber is not initialized, please call setup() first"
        
        res = []
        trainable_params = [
            (name, param) for name, param in cls.model.named_parameters() 
            if param.requires_grad
        ]
        for name, param in trainable_params:
            assert param.grad is not None, f"Error: {name} param.grad must not be None"
            grad = param.grad.detach().clone().view(-1)
            grad_sum = grad.float().sum()
            cur_json = {
                "name": name,
                "grad_sum": grad_sum.item(),
                "weight_sum": param.detach().clone().float().sum().item(),
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "param_info": str(param),
            }
            res.append(cur_json)
        
        dump_file = cls.dump_dir.joinpath(
            f"STEP_{cls.cur_step}_RANK_{dist.get_rank()}_{suffix}.jsonl"
        )
        with open(dump_file, "w", encoding="utf-8") as f:
            for line in res:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"[AccProber] Dump {suffix} to {dump_file}")
    
    @classmethod
    def before_clip_grad_norm(cls):
        cls._grad_dump("before_clip_grad_norm")
    
    @classmethod
    def after_clip_grad_norm(cls):
        cls._grad_dump("after_clip_grad_norm")


class TimeProber(BaseProber):
    """
    时间探测器 - 记录各阶段耗时
    """
    timings: ClassVar[dict[str, list[float]]] = {}
    start_times: ClassVar[dict[str, float]] = {}
    
    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int], model: nn.Module):
        super().setup(dump_home, profile_step, model)
        cls.dump_dir = dump_home / "time_prober"
        cls.dump_dir.mkdir(parents=True, exist_ok=True)
        cls.timings = {}
        cls.start_times = {}
        print(f"TimeProber initialized at {cls.dump_dir}")
    
    @classmethod
    def _start_timer(cls, name: str):
        if cls.skip():
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cls.start_times[name] = time.perf_counter()
    
    @classmethod
    def _end_timer(cls, name: str):
        if cls.skip():
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if name not in cls.start_times:
            print(f"[TimeProber] Warning: {name} timer not started")
            return
        
        elapsed = time.perf_counter() - cls.start_times[name]
        if name not in cls.timings:
            cls.timings[name] = []
        cls.timings[name].append(elapsed)
    
    @classmethod
    def before_embed_tokens(cls, input_ids: torch.Tensor):
        cls._start_timer("embed_tokens")
    
    @classmethod
    def after_embed_tokens(cls, hidden_states: torch.Tensor):
        cls._end_timer("embed_tokens")
    
    @classmethod
    def before_lm_head(cls, hidden_states: torch.Tensor, shifted_labels: torch.Tensor):
        cls._start_timer("lm_head")
    
    @classmethod
    def after_lm_head(cls, loss: torch.Tensor, logits: torch.Tensor):
        cls._end_timer("lm_head")
    
    @classmethod
    def before_clip_grad_norm(cls):
        cls._start_timer("clip_grad_norm")
    
    @classmethod
    def after_clip_grad_norm(cls):
        cls._end_timer("clip_grad_norm")
    
    @classmethod
    def dump_micro_iter_forward(cls):
        """转储计时信息"""
        if cls.skip():
            return
        assert cls.initialized, "TimeProber is not initialized, please call setup() first"
        
        # 计算统计信息
        stats = {}
        for name, times in cls.timings.items():
            if times:
                stats[name] = {
                    "count": len(times),
                    "total_ms": sum(times) * 1000,
                    "avg_ms": sum(times) / len(times) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                }
        
        dump_file = cls.dump_dir.joinpath(
            f"Step_{cls.cur_step}_MicroIter_{cls.cur_micro_batch_iter}_"
            f"RANK_{dist.get_rank()}_timings.json"
        )
        with open(dump_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[TimeProber] Dump timings to {dump_file}")
        
        # 清空本次记录
        cls.timings = {}
        cls.start_times = {}


# ==================== 使用示例 ====================

def example_usage():
    """
    使用示例 - 类似Java IO的组合方式
    
    Java IO:
        InputStream in = new BufferedInputStream(
                           new DataInputStream(
                             new FileInputStream("file.txt")));
    
    这里:
        ProberManager.setup(dump_home, steps, model, [AccProber, TimeProber])
    """
    dump_home = Path("./prober_dumps")
    profile_step = [0, 10, 20, 50, 100]
    model = None  # 你的模型
    
    # 初始化ProberManager，传入需要的Prober类列表
    ProberList.setup(
        dump_home=dump_home,
        profile_step=profile_step,
        model=model,
        prober_classes=[
            AccProber,      # 准确性探测
            TimeProber,     # 时间探测
            # MemoryProber, # 内存探测（待实现）
            # DistProber,   # 分布式探测（待实现）
        ]
    )
    
    # 在训练循环中使用
    for step in range(100):
        ProberList.set_step(step)
        
        for micro_iter in range(4):
            ProberList.set_micro_batch_iter(micro_iter)
            
            # 模拟训练过程
            input_ids = torch.randn(2, 128)
            ProberList.before_embed_tokens(input_ids)
            
            hidden_states = torch.randn(2, 128, 768)
            ProberList.after_embed_tokens(hidden_states)
            
            shifted_labels = torch.randint(0, 50000, (2, 128))
            ProberList.before_lm_head(hidden_states, shifted_labels)
            
            loss = torch.tensor(3.14)
            logits = torch.randn(2, 128, 50000)
            ProberList.after_lm_head(loss, logits)
            
            ProberList.dump_micro_iter_forward()
        
        # 梯度相关
        # ProberManager.before_clip_grad_norm()
        # ... 梯度裁剪 ...
        # ProberManager.after_clip_grad_norm()