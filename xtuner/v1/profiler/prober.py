import functools
import json
import pydoc
import re
import time
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from xtuner.v1.utils import get_logger


if TYPE_CHECKING:
    # Don't import model and module here to avoid circular import.
    # Because the model and module may import ProberList to debug.
    from xtuner.v1.module.decoder_layer.moe_decoder_layer import RouterResults
else:
    RouterResults = dict


logger = get_logger()


class BaseProber(ABC):
    """
    抽象基类 - 定义Prober接口规范
    每个子类都是单例（通过类方法实现）
    """

    dump_dir: ClassVar[Path | None] = None
    profile_step: ClassVar[list[int] | None] = None
    initialized: ClassVar[bool] = False
    cur_step: ClassVar[int] = 0
    cur_micro_batch_iter: ClassVar[int] = 0

    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int]):
        """子类必须实现setup方法，用于初始化自己的dump_dir."""
        cls.dump_dir = dump_home
        cls.dump_dir.mkdir(parents=True, exist_ok=True)
        cls.profile_step = profile_step
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
    def before_embed_tokens(cls, name: str, input_ids: torch.Tensor):
        pass

    @classmethod
    def after_embed_tokens(cls, name: str, hidden_states: torch.Tensor):
        pass

    @classmethod
    def before_rotary_emb(cls, name: str, x: torch.Tensor, position_ids: torch.Tensor):
        pass

    @classmethod
    def after_rotary_emb(cls, name: str, cos: torch.Tensor, sin: torch.Tensor):
        pass

    @classmethod
    def before_layer(cls, name: str, hidden_states: torch.Tensor):
        pass

    @classmethod
    def after_layer(cls, name: str, hidden_states: torch.Tensor):
        pass

    @classmethod
    def before_rms_norm(cls, name: str, hidden_states: torch.Tensor):
        pass

    @classmethod
    def after_rms_norm(cls, name: str, hidden_states: torch.Tensor):
        pass

    # ******************************* Attention Block *******************************
    @classmethod
    def before_attention(cls, name: str, hidden_states: torch.Tensor):
        pass

    @classmethod
    def after_attention(cls, name: str, outputs: torch.Tensor):
        pass

    # ******************************* MoE Block *******************************
    @classmethod
    def before_moe_gate(cls, name: str, hidden_states: torch.Tensor):
        pass

    @classmethod
    def after_moe_gate(cls, name: str, router_results: RouterResults):
        pass

    @classmethod
    def before_dispatch(
        cls, layer_idx: str | int, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ):
        pass

    @classmethod
    def after_dispatch(
        cls,
        layer_idx: str | int,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        row_ids_map: torch.Tensor | None,
        topk_weights: torch.Tensor,
    ):
        pass

    @classmethod
    def before_experts(cls, name: str, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        pass

    @classmethod
    def after_experts(cls, name: str, experts_out: torch.Tensor):
        pass

    @classmethod
    def before_combine(
        cls,
        layer_idx: str | int,
        experts_out: torch.Tensor,
        row_ids_map: torch.Tensor | None,
        topk_weights: torch.Tensor,
    ):
        pass

    @classmethod
    def after_combine(cls, layer_idx: str | int, combined_hidden_states: torch.Tensor):
        pass

    # ******************************* LM Head Block *******************************
    @classmethod
    def before_lm_head(cls, name: str, hidden_states: torch.Tensor, shifted_labels: torch.Tensor | None):
        pass

    @classmethod
    def after_lm_head(cls, name: str, loss: torch.Tensor, logits: torch.Tensor | None):
        pass

    @classmethod
    def before_balancing_loss(cls, name: str, router_weights: torch.Tensor):
        pass

    @classmethod
    def after_balancing_loss(
        cls,
        name: str,
        loss: torch.Tensor,
    ):
        pass

    @classmethod
    def before_z_loss(cls, name: str, router_logits: torch.Tensor):
        pass

    @classmethod
    def after_z_loss(cls, name: str, z_loss: torch.Tensor):
        pass

    ############################## hooks for gradient #################################
    @classmethod
    def before_clip_grad_norm(cls, model: nn.Module):
        pass

    @classmethod
    def after_clip_grad_norm(cls, model: nn.Module, grad_norm: torch.Tensor):
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
    def setup(cls, dump_home: Path, profile_step: list[int] | None, prober_class_names: list[str]):
        prober_classes = []
        for prober_class_name in prober_class_names:
            prober_class = pydoc.locate(f"{__name__}.{prober_class_name}")
            if prober_class is None:
                raise ValueError(f"Prober class {prober_class_name} not found")
            prober_classes.append(cast(type[BaseProber], prober_class))
        cls.prober_list = prober_classes

        # 初始化每个Prober
        profile_step = profile_step if profile_step is not None else []
        for prober_cls in cls.prober_list:
            prober_cls.setup(dump_home, profile_step)

        logger.info(
            f"ProberList initialized with {len(cls.prober_list)} probers: {[p.__name__ for p in cls.prober_list]}"
        )

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

    ############################# wrappers for forward hooks #################################
    @classmethod
    def wrap_embedding_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            ProberList.before_embed_tokens(name, args[0])
            hidden_states = forward(*args, **kwargs)
            ProberList.after_embed_tokens(name, hidden_states)
            return hidden_states

        return wrapped_forward
    
    @classmethod
    def wrap_rotary_emb_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            x, position_ids = args[0], args[1]
            ProberList.before_rotary_emb(name, x, position_ids)
            outputs = forward(*args, **kwargs)
            cos, sin = outputs
            ProberList.after_rotary_emb(name, cos, sin)
            return outputs
        return wrapped_forward

    @classmethod
    def wrap_decoder_layer_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            if len(args) >= 1:
                hidden_states = args[0]
            else:
                hidden_states = kwargs["hidden_states"]
            ProberList.before_layer(name, hidden_states)
            outputs = forward(*args, **kwargs)
            if isinstance(outputs, tuple):  # for MoEDecoderLayer
                hidden_states = outputs[0]
            else:  # for DenseDecoderLayer
                hidden_states = outputs
            ProberList.after_layer(name, hidden_states)
            return outputs

        return wrapped_forward

    @classmethod
    def wrap_rms_norm_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            if len(args) >= 1:
                hidden_states = args[0]
            else:
                hidden_states = kwargs["hidden_states"]
            ProberList.before_rms_norm(name, hidden_states)
            hidden_states = forward(*args, **kwargs)
            ProberList.after_rms_norm(name, hidden_states)
            return hidden_states

        return wrapped_forward

    @classmethod
    def wrap_attention_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            if len(args) >= 1:
                hidden_states = args[0]
            else:
                hidden_states = kwargs["hidden_states"]
            ProberList.before_attention(name, hidden_states)
            outputs = forward(*args, **kwargs)
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs
            ProberList.after_attention(name, hidden_states)
            return outputs

        return wrapped_forward

    @classmethod
    def wrap_moe_gate_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            if len(args) >= 1:
                hidden_states = args[0]
            else:
                hidden_states = kwargs["hidden_states"]
            ProberList.before_moe_gate(name, hidden_states)
            router_results = forward(*args, **kwargs)
            ProberList.after_moe_gate(name, router_results)
            return router_results

        return wrapped_forward

    @classmethod
    def wrap_moe_block_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            if len(args) >= 1:
                hidden_states = args[0]
            else:
                hidden_states = kwargs["x"]
            if len(args) >= 2:
                tokens_per_expert = args[1]
            else:
                tokens_per_expert = kwargs["tokens_per_expert"]
            ProberList.before_experts(name, hidden_states, tokens_per_expert)
            outputs = forward(*args, **kwargs)
            ProberList.after_experts(name, outputs)
            return outputs

        return wrapped_forward

    @classmethod
    def wrap_lm_head_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            if len(args) >= 1:
                hidden_states = args[0]
            else:
                hidden_states = kwargs["hidden_states"]
            if len(args) >= 2:
                loss_ctx = args[1]
            else:
                loss_ctx = kwargs.get("loss_ctx", None)
            ProberList.before_lm_head(
                name, hidden_states, loss_ctx.loss_kwargs.shifted_labels if loss_ctx is not None else None
            )
            outputs = forward(*args, **kwargs)
            loss, (logits, extra_info) = outputs
            ProberList.after_lm_head(name, loss, logits)
            return outputs

        return wrapped_forward

    @classmethod
    def wrap_balancing_loss_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            if len(args) >= 1:
                router_weights = args[0]
            else:
                router_weights = kwargs["router_weights"]
            ProberList.before_balancing_loss(name, router_weights)
            loss = forward(*args, **kwargs)
            ProberList.after_balancing_loss(name, loss)
            return loss

        return wrapped_forward

    @classmethod
    def wrap_z_loss_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            if len(args) >= 1:
                router_logits = args[0]
            else:
                router_logits = kwargs["router_logits"]
            ProberList.before_z_loss(name, router_logits)
            z_loss = forward(*args, **kwargs)
            ProberList.after_z_loss(name, z_loss)
            return z_loss

        return wrapped_forward

    ############################## forward hooks #################################
    @classmethod
    def before_embed_tokens(cls, name: str, input_ids: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_embed_tokens(name, input_ids)

    @classmethod
    def after_embed_tokens(cls, name: str, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_embed_tokens(name, hidden_states)

    @classmethod
    def before_rotary_emb(cls, name: str, x: torch.Tensor, position_ids: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_rotary_emb(name, x, position_ids)

    @classmethod
    def after_rotary_emb(cls, name: str, cos: torch.Tensor, sin: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_rotary_emb(name, cos, sin)

    @classmethod
    def before_layer(cls, name: str, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_layer(name, hidden_states)

    @classmethod
    def after_layer(cls, name: str, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_layer(name, hidden_states)

    @classmethod
    def before_rms_norm(cls, name: str, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_rms_norm(name, hidden_states)

    @classmethod
    def after_rms_norm(cls, name: str, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_rms_norm(name, hidden_states)

    # ******************************* Attention Block *******************************
    @classmethod
    def before_attention(cls, name: str, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_attention(name, hidden_states)

    @classmethod
    def after_attention(cls, name: str, outputs: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_attention(name, outputs)

    # ******************************* MoE Block *******************************
    @classmethod
    def before_moe_gate(cls, name: str, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_moe_gate(name, hidden_states)

    @classmethod
    def after_moe_gate(cls, name: str, router_results: RouterResults):
        for prober_cls in cls.prober_list:
            prober_cls.after_moe_gate(name, router_results)

    @classmethod
    def before_dispatch(
        cls, layer_idx: str | int, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ):
        for prober_cls in cls.prober_list:
            prober_cls.before_dispatch(layer_idx, hidden_states, topk_ids, topk_weights)

    @classmethod
    def after_dispatch(
        cls,
        layer_idx: str | int,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        row_ids_map: torch.Tensor | None,
        topk_weights: torch.Tensor,
    ):
        for prober_cls in cls.prober_list:
            prober_cls.after_dispatch(layer_idx, hidden_states, tokens_per_expert, row_ids_map, topk_weights)

    @classmethod
    def before_experts(cls, name: str, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_experts(name, hidden_states, tokens_per_expert)

    @classmethod
    def after_experts(cls, name: str, experts_out: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_experts(name, experts_out)

    @classmethod
    def before_combine(
        cls,
        layer_idx: str | int,
        experts_out: torch.Tensor,
        row_ids_map: torch.Tensor | None,
        topk_weights: torch.Tensor,
    ):
        for prober_cls in cls.prober_list:
            prober_cls.before_combine(layer_idx, experts_out, row_ids_map, topk_weights)

    @classmethod
    def after_combine(cls, layer_idx: str | int, combined_hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_combine(layer_idx, combined_hidden_states)

    # ******************************* LM Head Block *******************************
    @classmethod
    def before_lm_head(cls, name: str, hidden_states: torch.Tensor, shifted_labels: torch.Tensor | None):
        for prober_cls in cls.prober_list:
            prober_cls.before_lm_head(name, hidden_states, shifted_labels)

    @classmethod
    def after_lm_head(cls, name: str, loss: torch.Tensor, logits: torch.Tensor | None):
        for prober_cls in cls.prober_list:
            prober_cls.after_lm_head(name, loss, logits)

    @classmethod
    def before_balancing_loss(cls, name: str, router_weights: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_balancing_loss(name, router_weights)

    @classmethod
    def after_balancing_loss(
        cls,
        name: str,
        loss: torch.Tensor,
    ):
        for prober_cls in cls.prober_list:
            prober_cls.after_balancing_loss(name, loss)

    @classmethod
    def before_z_loss(cls, name: str, router_logits: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_z_loss(name, router_logits)

    @classmethod
    def after_z_loss(cls, name: str, z_loss: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_z_loss(name, z_loss)

    ############################## hooks for gradient #################################
    @classmethod
    def before_clip_grad_norm(cls, model: nn.Module):
        for prober_cls in cls.prober_list:
            prober_cls.before_clip_grad_norm(model)

    @classmethod
    def after_clip_grad_norm(cls, model: nn.Module, grad_norm: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_clip_grad_norm(model, grad_norm)

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
    dump_dir: ClassVar[Path | None] = None
    profile_step: ClassVar[list[int] | None] = None
    initialized: ClassVar[bool] = False
    cur_step: ClassVar[int] = 0
    cur_micro_batch_iter: ClassVar[int] = 0

    forward_records: ClassVar[list] = []

    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int]):
        super().setup(dump_home / "acc_prober", profile_step)
        cls.forward_records = []
        logger.info(f"AccProber initialized at {cls.dump_dir}")

    @classmethod
    def record_tensor(cls, tensor: torch.Tensor | None, name: str):
        """记录张量信息."""
        if cls.skip():
            return
        assert cls.initialized, "AccProber is not initialized, please call setup() first"
        if tensor is None:
            # logger.warning(f"[AccProber] Warning: {name} is None, skip recording")
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
    def before_embed_tokens(cls, name: str, input_ids: torch.Tensor):
        cls.record_tensor(input_ids, f"[{name}][before]input_ids")

    @classmethod
    def after_embed_tokens(cls, name: str, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[{name}][after]hidden_states")
    
    @classmethod
    def before_rotary_emb(cls, name: str, x: torch.Tensor, position_ids: torch.Tensor):
        cls.record_tensor(x, f"[{name}][before]x")
        cls.record_tensor(position_ids, f"[{name}][before]position_ids")

    @classmethod
    def after_rotary_emb(cls, name: str, cos: torch.Tensor, sin: torch.Tensor):
        cls.record_tensor(cos, f"[{name}][after]cos")
        cls.record_tensor(sin, f"[{name}][after]sin")

    @classmethod
    def before_layer(cls, name: str | int, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[{name}][before]hidden_states")

    @classmethod
    def after_layer(cls, name: str | int, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[{name}][after]hidden_states")

    @classmethod
    def before_rms_norm(cls, name: str, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[{name}][before]hidden_states")

    @classmethod
    def after_rms_norm(cls, name: str, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[{name}][after]hidden_states")

    # ******************************* Attention Block *******************************
    @classmethod
    def before_attention(cls, name: str, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[{name}][before]hidden_states")

    @classmethod
    def after_attention(cls, name: str, outputs: torch.Tensor):
        cls.record_tensor(outputs, f"[{name}][after]outputs")

    # ******************************* MoE Block *******************************
    @classmethod
    def before_moe_gate(cls, name: str, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[{name}][before]hidden_states")

    @classmethod
    def after_moe_gate(cls, name: str, router_results: RouterResults):
        cls.record_tensor(router_results["logits"], f"[{name}][after]logits")
        cls.record_tensor(router_results["topk_weights"], f"[{name}][after]topk_weights")
        cls.record_tensor(router_results["topk_ids"], f"[{name}][after]topk_ids")

    @classmethod
    def before_dispatch(
        cls, layer_idx: str | int, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}.dispatch][before]hidden_states")
        cls.record_tensor(topk_ids, f"[layers.{layer_idx}.dispatch][before]topk_ids")
        cls.record_tensor(topk_weights, f"[layers.{layer_idx}.dispatch][before]topk_weights")

    @classmethod
    def after_dispatch(
        cls,
        layer_idx: str | int,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        row_ids_map: torch.Tensor | None,
        topk_weights: torch.Tensor,
    ):
        cls.record_tensor(hidden_states, f"[layers.{layer_idx}.dispatch][after]hidden_states")
        cls.record_tensor(tokens_per_expert, f"[layers.{layer_idx}.dispatch][after]tokens_per_expert")
        cls.record_tensor(row_ids_map, f"[layers.{layer_idx}.dispatch][after]row_ids_map")
        cls.record_tensor(topk_weights, f"[layers.{layer_idx}.dispatch][after]topk_weights")

    @classmethod
    def before_experts(cls, name: str, hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        cls.record_tensor(hidden_states, f"[{name}][before]hidden_states")
        cls.record_tensor(tokens_per_expert, f"[{name}][before]tokens_per_expert")

    @classmethod
    def after_experts(cls, name: str, experts_out: torch.Tensor):
        cls.record_tensor(experts_out, f"[{name}][after]experts_out")

    @classmethod
    def before_combine(
        cls,
        layer_idx: str | int,
        experts_out: torch.Tensor,
        row_ids_map: torch.Tensor | None,
        topk_weights: torch.Tensor,
    ):
        cls.record_tensor(experts_out, f"[layers.{layer_idx}.combine][before]experts_out")
        cls.record_tensor(row_ids_map, f"[layers.{layer_idx}.combine][before]row_ids_map")
        cls.record_tensor(topk_weights, f"[layers.{layer_idx}.combine][before]topk_weights")

    @classmethod
    def after_combine(cls, layer_idx: str | int, combined_hidden_states: torch.Tensor):
        cls.record_tensor(combined_hidden_states, f"[layers.{layer_idx}.combine][after]combined_hidden_states")

    # ******************************* LM Head Block *******************************
    @classmethod
    def before_lm_head(cls, name: str, hidden_states: torch.Tensor, shifted_labels: torch.Tensor | None):
        cls.record_tensor(hidden_states, f"[{name}][before]hidden_states")
        cls.record_tensor(shifted_labels, f"[{name}][before]shifted_labels")

    @classmethod
    def after_lm_head(cls, name: str, loss: torch.Tensor, logits: torch.Tensor | None):
        cls.record_tensor(loss, f"[{name}][after]loss")
        cls.record_tensor(logits, f"[{name}][after]logits")

    @classmethod
    def before_balancing_loss(cls, name: str, router_weights: torch.Tensor):
        cls.record_tensor(router_weights, f"[{name}][before]router_weights")

    @classmethod
    def after_balancing_loss(
        cls,
        name: str,
        loss: torch.Tensor,
        # routing_weights_mean_global: torch.Tensor,
        # tokens_per_expert_global: torch.Tensor,
        # scale_global: torch.Tensor,
    ):
        cls.record_tensor(loss, f"[{name}][after]loss")
        # cls.record_tensor(routing_weights_mean_global, "[balancing_loss][after]routing_weights_mean_global")
        # cls.record_tensor(tokens_per_expert_global, "[balancing_loss][after]tokens_per_expert_global")
        # cls.record_tensor(scale_global, "[balancing_loss][after]scale_global")

    @classmethod
    def before_z_loss(cls, name: str, router_logits: torch.Tensor):
        cls.record_tensor(router_logits, f"[{name}][before]router_logits")

    @classmethod
    def after_z_loss(cls, name: str, z_loss: torch.Tensor):
        cls.record_tensor(z_loss, f"[{name}][after]z_loss")

    ############################## hooks for step and iter #################################
    @classmethod
    def after_micro_iter_forward(cls):
        if cls.skip():
            return
        assert cls.initialized, "AccProber is not initialized, please call setup() first"
        dump_file = cls.dump_dir.joinpath(
            f"Step_{cls.cur_step}_MicroIter_{cls.cur_micro_batch_iter}_RANK_{dist.get_rank()}_forward_records.jsonl"
        )
        with open(dump_file, "w", encoding="utf-8") as f:
            for record in cls.forward_records:
                f.write(record + "\n")
        # logger.info(f"[AccProber] Dump forward records to {dump_file}")
        cls.forward_records = []

    ############################## hooks for gradient #################################
    @classmethod
    def _grad_dump(cls, model: nn.Module, suffix: str, grad_norm: torch.Tensor | None = None):
        if cls.skip():
            return
        assert cls.initialized, "AccProber is not initialized, please call setup() first"

        assert cls.dump_dir is not None
        res = []
        trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        for name, param in trainable_params:
            assert param.grad is not None, f"Error: {name} param.grad must not be None"
            grad = param.grad.detach().clone().view(-1)
            grad = grad.float()
            cur_json = {
                "name": name,
                "grad_sum": grad.sum().item(),
                "grad_mean": grad.mean().item(),
                "grad_std": grad.to_local().std().item() if isinstance(grad, DTensor) else grad.std().item(),
                "weight_sum": param.detach().clone().float().sum().item(),
                "shape": list(param.shape),
                "dtype": str(param.dtype),
                "param_info": str(param),
            }
            res.append(cur_json)
        if grad_norm is not None:
            res.append(
                {
                    "name": "grad_norm",
                    "grad_sum": grad_norm.detach().float().sum().item(),
                    "weight_sum": grad_norm.detach().float().sum().item(),
                    "shape": list(grad_norm.shape),
                    "dtype": str(grad_norm.dtype),
                    "param_info": str(grad_norm),
                }
            )

        dump_file = cls.dump_dir.joinpath(f"STEP_{cls.cur_step}_RANK_{dist.get_rank()}_{suffix}.jsonl")
        with open(dump_file, "w", encoding="utf-8") as f:
            for line in res:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        # logger.info(f"[AccProber] Dump {suffix} to {dump_file}")

    @classmethod
    def before_clip_grad_norm(cls, model: nn.Module):
        cls._grad_dump(model, "before_clip_grad_norm")

    @classmethod
    def after_clip_grad_norm(cls, model: nn.Module, grad_norm: torch.Tensor):
        cls._grad_dump(model, "after_clip_grad_norm", grad_norm)


class TimeProber(BaseProber):
    """
    时间探测器 - 记录各阶段耗时
    """

    dump_dir: ClassVar[Path | None] = None
    profile_step: ClassVar[list[int] | None] = None
    initialized: ClassVar[bool] = False
    cur_step: ClassVar[int] = 0
    cur_micro_batch_iter: ClassVar[int] = 0

    timings: ClassVar[dict[str, list[float]]] = {}
    start_times: ClassVar[dict[str, float]] = {}
    max_step: ClassVar[int] = 0

    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int]):
        super().setup(dump_home / "time_prober", profile_step)
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
    def before_embed_tokens(cls, name: str, input_ids: torch.Tensor):
        cls._start_timer(name)

    @classmethod
    def after_embed_tokens(cls, name: str, hidden_states: torch.Tensor):
        cls._end_timer(name)

    @classmethod
    def before_layer(cls, name: str, hidden_states: torch.Tensor):
        cls._start_timer(name)

    @classmethod
    def after_layer(cls, name: str, hidden_states: torch.Tensor):
        cls._end_timer(name)

    @classmethod
    def before_lm_head(cls, name: str, hidden_states: torch.Tensor, shifted_labels: torch.Tensor | None):
        cls._start_timer(name)

    @classmethod
    def after_lm_head(cls, name: str, loss: torch.Tensor, logits: torch.Tensor | None):
        cls._end_timer(name)

    ############################## hooks for gradient #################################
    @classmethod
    def before_clip_grad_norm(cls, model: nn.Module):
        cls._start_timer("clip_grad_norm")

    @classmethod
    def after_clip_grad_norm(cls, model: nn.Module, grad_norm: torch.Tensor):
        cls._end_timer("clip_grad_norm")

    @classmethod
    def after_step(cls):
        """转储计时信息."""
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
            if "layer" not in name:
                continue
            # 聚合所有layer信息，将 "layer.{idx}.xxx" 去掉idx，转换成 "layer.xxx"
            # 注意 {idx} 是多位数字
            layer_name = re.sub(r"layer\.(\d+)", "layer", name)
            if layer_name not in stats:
                stats[layer_name] = {
                    "count": 0,
                    "total_ms": 0,
                    "avg_ms": 0,
                    "min_ms": float("inf"),
                    "max_ms": float("-inf"),
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

        dump_file = cls.dump_dir.joinpath(f"Step_{cls.cur_step}_RANK_{dist.get_rank()}_timings.json")
        with open(dump_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        # logger.info(f"[TimeProber] Dump timings to {dump_file}")

        # 清空本次记录
        cls.timings = {}
        cls.start_times = {}


class PdbProber(BaseProber):
    dump_dir: ClassVar[Path | None] = None
    profile_step: ClassVar[list[int] | None] = None
    initialized: ClassVar[bool] = False
    cur_step: ClassVar[int] = 0
    cur_micro_batch_iter: ClassVar[int] = 0

    @classmethod
    def before_layer(cls, name: str, hidden_states: torch.Tensor):
        if cls.cur_step == 10 and cls.cur_micro_batch_iter == 0 and "layers.10" in name:
            dist.breakpoint()

    @classmethod
    def before_moe_gate(cls, name: str, hidden_states: torch.Tensor):
        dist.breakpoint()
