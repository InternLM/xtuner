from pathlib import Path
import re
import sys
import json
import time
from abc import ABC, abstractmethod
from typing import ClassVar

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
import torch.nn as nn

from xtuner.v1.utils import get_logger

logger = get_logger()


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
        # if dist.get_rank() != 0:
        #     return True
        return False
    
    @classmethod
    def set_step(cls, step: int):
        cls.cur_step = step
    
    @classmethod
    def set_micro_batch_iter(cls, iter: int):
        cls.cur_micro_batch_iter = iter
    
    @classmethod
    def record_tensor(cls, tensor: torch.Tensor, name: str):
        pass

    ############################## forward hooks #################################
    @classmethod
    def before_embed_tokens(cls, input_ids: torch.Tensor):
        pass
    
    @classmethod
    def after_embed_tokens(cls, hidden_states: torch.Tensor):
        pass

    @classmethod
    def before_layer(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        pass

    @classmethod
    def after_layer(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        pass
    
    # ******************************* Attention Block *******************************
    @classmethod
    def before_input_layernorm(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        pass
    
    @classmethod
    def after_input_layernorm(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        pass
    
    @classmethod
    def before_self_attn(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        pass
    
    @classmethod
    def after_self_attn(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        pass

    # ******************************* MoE Block *******************************
    @classmethod
    def before_post_attention_layernorm(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        pass
    
    @classmethod
    def after_post_attention_layernorm(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        pass
    
    @classmethod
    def before_router_gate(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        pass
    
    @classmethod
    def after_router_gate(cls, layer_idx: str|int, logits: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        pass

    @classmethod
    def before_dispatch(cls, layer_idx: str|int, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        pass
    
    @classmethod
    def after_dispatch(cls, layer_idx: str|int, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor, row_ids_map: torch.Tensor, topk_weights: torch.Tensor):
        pass

    @classmethod
    def before_experts(cls, layer_idx: str|int, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        pass
    
    @classmethod
    def after_experts(cls, layer_idx: str|int, experts_out: torch.Tensor):
        pass
    
    @classmethod
    def before_combine(cls, layer_idx: str|int, experts_out: torch.Tensor, row_ids_map: torch.Tensor, topk_weights: torch.Tensor):
        pass
    
    @classmethod
    def after_combine(cls, layer_idx: str|int, combined_hidden_states: torch.Tensor):
        pass
    
    # ******************************* LM Head Block *******************************
    @classmethod
    def before_lm_head(cls, hidden_states: torch.Tensor, shifted_labels: torch.Tensor):
        pass
    
    @classmethod
    def after_lm_head(cls, loss: torch.Tensor, logits: torch.Tensor):
        pass

    @classmethod
    def before_balancing_loss(cls, router_logits: torch.Tensor):
        pass
    
    @classmethod
    def after_balancing_loss(cls, loss: torch.Tensor, routing_weights_mean_global: torch.Tensor, tokens_per_expert_global: torch.Tensor, scale_global: torch.Tensor):
        pass

    @classmethod
    def before_z_loss(cls, router_logits: torch.Tensor):
        pass
    
    @classmethod
    def after_z_loss(cls, z_loss: torch.Tensor):
        pass
    
    ############################## hooks for gradient #################################
    @classmethod
    def before_clip_grad_norm(cls):
        pass
    
    @classmethod
    def after_clip_grad_norm(cls):
        pass

    ############################## hooks for step and iter #################################
    @classmethod
    def after_micro_iter_forward(cls):  # usually used for dumping forward activation records
        pass

    @classmethod
    def after_step(cls):  # usually used for dumping timing records
        pass


class ProberList:
    prober_list: ClassVar[list[type[BaseProber]]] = []
    
    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int], model: nn.Module, 
              prober_class_names: list[str]):
        prober_classes = []
        for prober_class_name in prober_class_names:
            prober_classes.append(getattr(sys.modules[__name__], prober_class_name))
        cls.prober_list = prober_classes
        # 初始化每个Prober
        for prober_cls in cls.prober_list:
            prober_cls.setup(dump_home, profile_step, model)
        
        logger.info(f"ProberList initialized with {len(cls.prober_list)} probers: "
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

    ############################## forward hooks #################################
    @classmethod
    def before_embed_tokens(cls, input_ids: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_embed_tokens(input_ids)
    
    @classmethod
    def after_embed_tokens(cls, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_embed_tokens(hidden_states)
    
    @classmethod
    def before_layer(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_layer(layer_idx, hidden_states)

    @classmethod
    def after_layer(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_layer(layer_idx, hidden_states)

    # ******************************* Attention Block *******************************
    @classmethod
    def before_input_layernorm(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_input_layernorm(layer_idx, hidden_states)
    
    @classmethod
    def after_input_layernorm(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_input_layernorm(layer_idx, hidden_states)
    
    @classmethod
    def before_self_attn(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_self_attn(layer_idx, hidden_states)
    
    @classmethod
    def after_self_attn(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_self_attn(layer_idx, hidden_states)
    
    # ******************************* MoE Block *******************************
    @classmethod
    def before_post_attention_layernorm(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_post_attention_layernorm(layer_idx, hidden_states)
    
    @classmethod
    def after_post_attention_layernorm(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_post_attention_layernorm(layer_idx, hidden_states)
    
    @classmethod
    def before_router_gate(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_router_gate(layer_idx, hidden_states)
    
    @classmethod
    def after_router_gate(cls, layer_idx: str|int, logits: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_router_gate(layer_idx, logits, topk_weights, topk_ids)
    
    @classmethod
    def before_dispatch(cls, layer_idx: str|int, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_dispatch(layer_idx, hidden_states, topk_ids, topk_weights)
    
    @classmethod
    def after_dispatch(cls, layer_idx: str|int, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor, row_ids_map: torch.Tensor, topk_weights: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_dispatch(layer_idx, hidden_states, tokens_per_expert, row_ids_map, topk_weights)
    
    @classmethod
    def before_experts(cls, layer_idx: str|int, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_experts(layer_idx, hidden_states, tokens_per_expert)
    
    @classmethod
    def after_experts(cls, layer_idx: str|int, experts_out: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_experts(layer_idx, experts_out)
    
    @classmethod
    def before_combine(cls, layer_idx: str|int, experts_out: torch.Tensor, row_ids_map: torch.Tensor, topk_weights: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_combine(layer_idx, experts_out, row_ids_map, topk_weights)
    
    @classmethod
    def after_combine(cls, layer_idx: str|int, combined_hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_combine(layer_idx, combined_hidden_states)

    # ******************************* LM Head Block *******************************
    @classmethod
    def before_lm_head(cls, hidden_states: torch.Tensor, shifted_labels: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_lm_head(hidden_states, shifted_labels)
    
    @classmethod
    def after_lm_head(cls, loss: torch.Tensor, logits: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_lm_head(loss, logits)
    
    @classmethod
    def before_balancing_loss(cls, router_logits: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_balancing_loss(router_logits)
    
    @classmethod
    def after_balancing_loss(cls, loss: torch.Tensor, routing_weights_mean_global: torch.Tensor, tokens_per_expert_global: torch.Tensor, scale_global: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_balancing_loss(loss, routing_weights_mean_global, tokens_per_expert_global, scale_global)
    
    @classmethod
    def before_z_loss(cls, router_logits: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_z_loss(router_logits)
    
    @classmethod
    def after_z_loss(cls, z_loss: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_z_loss(z_loss)
    
    ############################## hooks for gradient #################################
    @classmethod
    def before_clip_grad_norm(cls):
        for prober_cls in cls.prober_list:
            prober_cls.before_clip_grad_norm()
    
    @classmethod
    def after_clip_grad_norm(cls):
        for prober_cls in cls.prober_list:
            prober_cls.after_clip_grad_norm()
    
    ############################## hooks for step and iter #################################
    @classmethod
    def after_micro_iter_forward(cls):
        for prober_cls in cls.prober_list:
            prober_cls.after_micro_iter_forward()
    
    @classmethod
    def after_step(cls):
        for prober_cls in cls.prober_list:
            prober_cls.after_step()


class AccProber(BaseProber):
    forward_records: ClassVar[list] = []
    
    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int], model: nn.Module):
        super().setup(dump_home, profile_step, model)
        cls.dump_dir = dump_home / "acc_prober"
        cls.dump_dir.mkdir(parents=True, exist_ok=True)
        cls.forward_records = []
        logger.info(f"AccProber initialized at {cls.dump_dir}")
    
    @classmethod
    def record_tensor(cls, tensor: torch.Tensor | None, name: str):
        """记录张量信息"""
        if cls.skip():
            return
        assert cls.initialized, "AccProber is not initialized, please call setup() first"
        if tensor is None:
            logger.warning(f"[AccProber] Warning: {name} is None, skip recording")
            return
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
    
    ############################## forward hooks #################################
    @classmethod
    def before_embed_tokens(cls, input_ids: torch.Tensor):
        cls.record_tensor(input_ids, "[embed_tokens][before]input_ids")
    
    @classmethod
    def after_embed_tokens(cls, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, "[embed_tokens][after]hidden_states")
    
    @classmethod
    def before_layer(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}][before]hidden_states")
    
    @classmethod
    def after_layer(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}][after]hidden_states")
    
    # ******************************* Attention Block *******************************
    @classmethod
    def before_input_layernorm(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}.input_layernorm][before]hidden_states")
    
    @classmethod
    def after_input_layernorm(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}.input_layernorm][after]hidden_states")
    
    @classmethod
    def before_self_attn(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}.self_attn][before]hidden_states")
    
    @classmethod
    def after_self_attn(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}.self_attn][after]hidden_states")
    
    # ******************************* MoE Block *******************************
    @classmethod
    def before_post_attention_layernorm(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}.post_attention_layernorm][before]hidden_states")
    
    @classmethod
    def after_post_attention_layernorm(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}.post_attention_layernorm][after]hidden_states")
    
    @classmethod
    def before_router_gate(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}.router_gate][before]hidden_states")
    
    @classmethod
    def after_router_gate(cls, layer_idx: str|int, logits: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        cls.record_tensor(logits, f"[layers.{layer_idx}.router_gate][after]logits")
        cls.record_tensor(topk_weights, f"[layers.{layer_idx}.router_gate][after]topk_weights")
        cls.record_tensor(topk_ids, f"[layers.{layer_idx}.router_gate][after]topk_ids")
    
    @classmethod
    def before_dispatch(cls, layer_idx: str|int, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}.dispatch][before]hidden_states")
        cls.record_tensor(topk_ids, f"[layers.{layer_idx}.dispatch][before]topk_ids")
        cls.record_tensor(topk_weights, f"[layers.{layer_idx}.dispatch][before]topk_weights")
    
    @classmethod
    def after_dispatch(cls, layer_idx: str|int, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor, row_ids_map: torch.Tensor, topk_weights: torch.Tensor):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}.dispatch][after]hidden_states")
        cls.record_tensor(tokens_per_expert, f"[layers.{layer_idx}.dispatch][after]tokens_per_expert")
        cls.record_tensor(row_ids_map, f"[layers.{layer_idx}.dispatch][after]row_ids_map")
        cls.record_tensor(topk_weights, f"[layers.{layer_idx}.dispatch][after]topk_weights")
    
    @classmethod
    def before_experts(cls, layer_idx: str|int, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}.experts][before]hidden_states")
        cls.record_tensor(tokens_per_expert, f"[layers.{layer_idx}.experts][before]tokens_per_expert")
    
    @classmethod
    def after_experts(cls, layer_idx: str|int, experts_out: torch.Tensor):
        cls.record_tensor(experts_out, f"[layers.{layer_idx}.experts][after]experts_out")
    
    @classmethod
    def before_combine(cls, layer_idx: str|int, experts_out: torch.Tensor, row_ids_map: torch.Tensor, topk_weights: torch.Tensor):
        cls.record_tensor(experts_out, f"[layers.{layer_idx}.combine][before]experts_out")
        cls.record_tensor(row_ids_map, f"[layers.{layer_idx}.combine][before]row_ids_map")
        cls.record_tensor(topk_weights, f"[layers.{layer_idx}.combine][before]topk_weights")
    
    @classmethod
    def after_combine(cls, layer_idx: str|int, combined_hidden_states: torch.Tensor):
        cls.record_tensor(combined_hidden_states, f"[layers.{layer_idx}.combine][after]combined_hidden_states")
    
    # ******************************* LM Head Block *******************************
    @classmethod
    def before_lm_head(cls, hidden_states: torch.Tensor, shifted_labels: torch.Tensor):
        cls.record_tensor(hidden_states, "[lm_head][before]hidden_states")
        cls.record_tensor(shifted_labels, "[lm_head][before]shifted_labels")
    
    @classmethod
    def after_lm_head(cls, loss: torch.Tensor, logits: torch.Tensor):
        cls.record_tensor(loss, "[lm_head][after]loss")
        cls.record_tensor(logits, "[lm_head][after]logits")
    
    @classmethod
    def before_balancing_loss(cls, router_logits: torch.Tensor):
        cls.record_tensor(router_logits, "[balancing_loss][before]router_logits")
    
    @classmethod
    def after_balancing_loss(cls, loss: torch.Tensor, routing_weights_mean_global: torch.Tensor, tokens_per_expert_global: torch.Tensor, scale_global: torch.Tensor):
        cls.record_tensor(loss, "[balancing_loss][after]loss")
        cls.record_tensor(routing_weights_mean_global, "[balancing_loss][after]routing_weights_mean_global")
        cls.record_tensor(tokens_per_expert_global, "[balancing_loss][after]tokens_per_expert_global")
        cls.record_tensor(scale_global, "[balancing_loss][after]scale_global")
    
    @classmethod
    def before_z_loss(cls, router_logits: torch.Tensor):
        cls.record_tensor(router_logits, "[z_loss][before]router_logits")
    
    @classmethod
    def after_z_loss(cls, z_loss: torch.Tensor):
        cls.record_tensor(z_loss, "[z_loss][after]z_loss")
    
    ############################## hooks for step and iter #################################
    @classmethod
    def after_micro_iter_forward(cls):
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
        # logger.info(f"[AccProber] Dump forward records to {dump_file}")
        cls.forward_records = []
    
    ############################## hooks for gradient #################################
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
        # logger.info(f"[AccProber] Dump {suffix} to {dump_file}")
    
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
    max_step: ClassVar[int] = 0
    
    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int], model: nn.Module):
        super().setup(dump_home, profile_step, model)
        cls.dump_dir = dump_home / "time_prober"
        cls.dump_dir.mkdir(parents=True, exist_ok=True)
        cls.timings = {}
        cls.start_times = {}
        cls.max_step = max(profile_step)
        logger.info(f"TimeProber initialized at {cls.dump_dir}")
    
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
            logger.warning(f"[TimeProber] Warning: {name} timer not started")
            return
        
        elapsed = time.perf_counter() - cls.start_times[name]
        if name not in cls.timings:
            cls.timings[name] = []
        cls.timings[name].append(elapsed)
    
    ############################## forward hooks #################################
    @classmethod
    def before_embed_tokens(cls, input_ids: torch.Tensor):
        cls._start_timer("embed_tokens")
    
    @classmethod
    def after_embed_tokens(cls, hidden_states: torch.Tensor):
        cls._end_timer("embed_tokens")
    
    @classmethod
    def before_layer(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        cls._start_timer(f"layer.{layer_idx}")
    
    @classmethod
    def after_layer(cls, layer_idx: str|int, hidden_states: torch.Tensor):
        cls._end_timer(f"layer.{layer_idx}")
    
    @classmethod
    def before_lm_head(cls, hidden_states: torch.Tensor, shifted_labels: torch.Tensor):
        cls._start_timer("lm_head")
    
    @classmethod
    def after_lm_head(cls, loss: torch.Tensor, logits: torch.Tensor):
        cls._end_timer("lm_head")
    
    ############################## hooks for gradient #################################
    @classmethod
    def before_clip_grad_norm(cls):
        cls._start_timer("clip_grad_norm")
    
    @classmethod
    def after_clip_grad_norm(cls):
        cls._end_timer("clip_grad_norm")
    
    @classmethod
    def after_step(cls):
        """转储计时信息"""
        if cls.skip():
            return
        assert cls.initialized, "TimeProber is not initialized, please call setup() first"

        # if cls.cur_step < cls.max_step:
            # return
        
        # 计算统计信息
        stats = {}
        for name, times in cls.timings.items():
            if not times:
                continue
            stats[name] = {
                "count": len(times),
                "total_ms": sum(times) * 1000,
                "avg_ms": sum(times) / len(times) * 1000,
                "min_ms": min(times) * 1000,
                "max_ms": max(times) * 1000,
            }
            if 'layer' not in name:
                continue
            # 聚合所有layer信息，将 "layer.{idx}.xxx" 去掉idx，转换成 "layer.xxx"
            # 注意 {idx} 是多位数字
            layer_name = re.sub(r'layer\.(\d+)', 'layer', name)
            if layer_name not in stats:
                stats[layer_name] = {
                    "count": 0,
                    "total_ms": 0,
                    "avg_ms": 0,
                    "min_ms": float('inf'),
                    "max_ms": float('-inf'),
                }
            last_count = stats[layer_name]["count"]
            last_total_ms = stats[layer_name]["total_ms"]
            last_min_ms = stats[layer_name]["min_ms"]
            last_max_ms = stats[layer_name]["max_ms"]
            stats[layer_name]["count"] = last_count + len(times)
            stats[layer_name]["total_ms"] = last_total_ms + sum(times) * 1000
            stats[layer_name]["avg_ms"] = stats[layer_name]["total_ms"] / stats[layer_name]["count"]
            stats[layer_name]["min_ms"] = min(last_min_ms, min(times) * 1000)
            stats[layer_name]["max_ms"] = max(last_max_ms, max(times) * 1000)
        
        dump_file = cls.dump_dir.joinpath(
            f"Step_{cls.cur_step}_"
            f"RANK_{dist.get_rank()}_timings.json"
        )
        with open(dump_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        # logger.info(f"[TimeProber] Dump timings to {dump_file}")
        
        # 清空本次记录
        cls.timings = {}
        cls.start_times = {}


# ==================== 使用示例 ====================

def example_usage():
    """
    使用示例
    """
    dump_home = Path("./prober_dumps")
    profile_step = [0, 10, 20, 50, 100]
    model = None  # 你的模型
    
    # 初始化ProberManager，传入需要的Prober类列表
    ProberList.setup(
        dump_home=dump_home,
        profile_step=profile_step,
        model=model,
        prober_class_names=[
            "AccProber",      # 准确性探测
            "TimeProber",     # 时间探测
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
            
            ProberList.after_micro_iter_forward()

        
        # 梯度相关
        # ProberManager.before_clip_grad_norm()
        # ... 梯度裁剪 ...
        # ProberManager.after_clip_grad_norm()

        ProberList.after_step()