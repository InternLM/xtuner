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

from xtuner.v1.utils import get_logger, log_rank0


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
    # Pre-computed skip flag: True means "do not record".  Updated in set_step()
    # so that torch.compile guards on a bool that changes only at profile-step
    # transitions (not on cur_step, which changes every step).
    _skip_flag: ClassVar[bool] = True

    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int]):
        """子类必须实现setup方法，用于初始化自己的dump_dir."""
        cls.dump_dir = dump_home
        cls.dump_dir.mkdir(parents=True, exist_ok=True)
        cls.profile_step = profile_step
        cls.initialized = True
        cls._skip_flag = True  # reset; will be updated by first set_step() call

    @classmethod
    def skip(cls) -> bool:
        return cls._skip_flag

    @classmethod
    def set_step(cls, step: int):
        cls.cur_step = step
        if cls.profile_step is None:
            cls._skip_flag = True
        else:
            cls._skip_flag = step not in cls.profile_step

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

    # ******************************* Linear *******************************
    @classmethod
    def before_linear(cls, name: str, input: torch.Tensor):
        pass

    @classmethod
    def after_linear(cls, name: str, output: torch.Tensor):
        pass

    # ******************************* MoEMLP (shared experts) *******************************
    @classmethod
    def before_moe_mlp(cls, name: str, x: torch.Tensor):
        pass

    @classmethod
    def after_moe_mlp(cls, name: str, out: torch.Tensor):
        pass

    # ******************************* Attention Block *******************************
    @classmethod
    def before_attention(cls, name: str, hidden_states: torch.Tensor):
        pass

    @classmethod
    def after_attention(cls, name: str, outputs: torch.Tensor):
        pass

    # ******************************* GatedDeltaNet ops *******************************
    @classmethod
    def before_causal_conv1d(cls, name: str, x: torch.Tensor):
        pass

    @classmethod
    def after_causal_conv1d(cls, name: str, out: torch.Tensor):
        pass

    @classmethod
    def before_chunk_gated_delta_rule(
        cls,
        name: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None,
        beta: torch.Tensor | None,
    ):
        pass

    @classmethod
    def after_chunk_gated_delta_rule(cls, name: str, core_attn_out: torch.Tensor):
        pass

    @classmethod
    def before_fused_rms_norm_gated(cls, name: str, x: torch.Tensor, g: torch.Tensor):
        pass

    @classmethod
    def after_fused_rms_norm_gated(cls, name: str, out: torch.Tensor):
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

        log_rank0.info(
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
    def wrap_linear_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            input = args[0] if args else kwargs["input"]
            ProberList.before_linear(name, input)
            output = forward(*args, **kwargs)
            ProberList.after_linear(name, output)
            return output

        return wrapped_forward

    @classmethod
    def wrap_moe_mlp_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            x = args[0] if args else kwargs["x"]
            ProberList.before_moe_mlp(name, x)
            out = forward(*args, **kwargs)
            ProberList.after_moe_mlp(name, out)
            return out

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
    def wrap_causal_conv1d_fn(cls, fn: Callable, name: str) -> Callable:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            x = kwargs.get("x", args[0] if args else None)
            ProberList.before_causal_conv1d(name, x)
            out = fn(*args, **kwargs)
            ProberList.after_causal_conv1d(name, out)
            return out

        return wrapped

    @classmethod
    def wrap_chunk_gated_delta_rule(cls, fn: Callable, name: str) -> Callable:
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            q = args[0] if len(args) > 0 else kwargs.get("q")
            k = args[1] if len(args) > 1 else kwargs.get("k")
            v = args[2] if len(args) > 2 else kwargs.get("v")
            g = kwargs.get("g")
            beta = kwargs.get("beta")
            ProberList.before_chunk_gated_delta_rule(name, q, k, v, g, beta)
            out = fn(*args, **kwargs)
            ProberList.after_chunk_gated_delta_rule(name, out[0])
            return out

        return wrapped

    @classmethod
    def wrap_fused_rms_norm_gated_forward(cls, forward: Callable, name: str):
        @functools.wraps(forward)
        def wrapped_forward(self, *args, **kwargs):
            x = args[0] if args else kwargs["x"]
            g = args[1] if len(args) > 1 else kwargs.get("g")
            ProberList.before_fused_rms_norm_gated(name, x, g)
            out = forward(*args, **kwargs)
            ProberList.after_fused_rms_norm_gated(name, out)
            return out

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

    # ******************************* Linear *******************************
    @classmethod
    def before_linear(cls, name: str, input: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_linear(name, input)

    @classmethod
    def after_linear(cls, name: str, output: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_linear(name, output)

    # ******************************* MoEMLP (shared experts) *******************************
    @classmethod
    def before_moe_mlp(cls, name: str, x: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_moe_mlp(name, x)

    @classmethod
    def after_moe_mlp(cls, name: str, out: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_moe_mlp(name, out)

    # ******************************* Attention Block *******************************
    @classmethod
    def before_attention(cls, name: str, hidden_states: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_attention(name, hidden_states)

    @classmethod
    def after_attention(cls, name: str, outputs: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_attention(name, outputs)

    # ******************************* GatedDeltaNet ops *******************************
    @classmethod
    def before_causal_conv1d(cls, name: str, x: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_causal_conv1d(name, x)

    @classmethod
    def after_causal_conv1d(cls, name: str, out: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_causal_conv1d(name, out)

    @classmethod
    def before_chunk_gated_delta_rule(
        cls,
        name: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None,
        beta: torch.Tensor | None,
    ):
        for prober_cls in cls.prober_list:
            prober_cls.before_chunk_gated_delta_rule(name, q, k, v, g, beta)

    @classmethod
    def after_chunk_gated_delta_rule(cls, name: str, core_attn_out: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_chunk_gated_delta_rule(name, core_attn_out)

    @classmethod
    def before_fused_rms_norm_gated(cls, name: str, x: torch.Tensor, g: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.before_fused_rms_norm_gated(name, x, g)

    @classmethod
    def after_fused_rms_norm_gated(cls, name: str, out: torch.Tensor):
        for prober_cls in cls.prober_list:
            prober_cls.after_fused_rms_norm_gated(name, out)

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
    _skip_flag: ClassVar[bool] = True

    forward_records: ClassVar[list] = []
    # Tensors buffered during the forward pass; serialized in after_micro_iter_forward
    # so that no Python string / JSON ops occur inside torch.compile regions.
    _pending_tensors: ClassVar[list] = []

    @classmethod
    def setup(cls, dump_home: Path, profile_step: list[int]):
        super().setup(dump_home / "acc_prober", profile_step)
        cls.forward_records = []
        cls._pending_tensors = []
        log_rank0.info(f"AccProber initialized at {cls.dump_dir}")

    @classmethod
    def record_tensor(cls, tensor: torch.Tensor | None, name: str):
        """Buffer compact tensor stats for deferred serialization.

        Only light tensor ops happen here so this method is compatible with
        torch.compile.  Expensive Python ops (.item(), .tolist(), str) are
        deferred to after_micro_iter_forward(), which always runs in eager mode.

        What is stored per tensor:
          - tensor_sum  : scalar tensor (float32)
          - shape       : torch.Size  (compile-time constant in static-shape mode)
          - dtype_str   : str of tensor.dtype (compile-time constant)
          - first_10    : 1-D float32 tensor with up to 10 flattened elements

        torch.compile behavior:
          _skip_flag=True  → dead-code elimination; no graph breaks.
          _skip_flag=False → tensor ops are compiled; list.append is inlined
                             as a Python side effect (no graph break).
        """
        if cls.skip():
            return
        if tensor is None:
            return
        t = tensor.detach().float()
        tensor_sum = t.sum()
        first_10 = t.flatten()[:10].clone()
        cls._pending_tensors.append((name, tensor_sum, tensor.shape, str(tensor.dtype), first_10))

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

    # ******************************* Linear *******************************
    @classmethod
    def before_linear(cls, name: str, input: torch.Tensor):
        cls.record_tensor(input, f"[{name}][before]input")

    @classmethod
    def after_linear(cls, name: str, output: torch.Tensor):
        cls.record_tensor(output, f"[{name}][after]output")

    # ******************************* MoEMLP (shared experts) *******************************
    @classmethod
    def before_moe_mlp(cls, name: str, x: torch.Tensor):
        cls.record_tensor(x, f"[{name}][before]x")

    @classmethod
    def after_moe_mlp(cls, name: str, out: torch.Tensor):
        cls.record_tensor(out, f"[{name}][after]out")

    # ******************************* Attention Block *******************************
    @classmethod
    def before_attention(cls, name: str, hidden_states: torch.Tensor):
        cls.record_tensor(hidden_states, f"[{name}][before]hidden_states")

    @classmethod
    def after_attention(cls, name: str, outputs: torch.Tensor):
        if isinstance(outputs, dict):
            outputs = outputs["projected_output"]
        else:
            assert isinstance(outputs, torch.Tensor), f"Unsupported outputs type: {type(outputs)}"

        cls.record_tensor(outputs, f"[{name}][after]outputs")

    # ******************************* GatedDeltaNet ops *******************************
    @classmethod
    def before_causal_conv1d(cls, name: str, x: torch.Tensor):
        cls.record_tensor(x, f"[{name}][before]conv1d_x")

    @classmethod
    def after_causal_conv1d(cls, name: str, out: torch.Tensor):
        cls.record_tensor(out, f"[{name}][after]conv1d_out")

    @classmethod
    def before_chunk_gated_delta_rule(
        cls,
        name: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None,
        beta: torch.Tensor | None,
    ):
        cls.record_tensor(q, f"[{name}][before]q")
        cls.record_tensor(k, f"[{name}][before]k")
        cls.record_tensor(v, f"[{name}][before]v")
        cls.record_tensor(g, f"[{name}][before]g")
        cls.record_tensor(beta, f"[{name}][before]beta")

    @classmethod
    def after_chunk_gated_delta_rule(cls, name: str, core_attn_out: torch.Tensor):
        cls.record_tensor(core_attn_out, f"[{name}][after]core_attn_out")

    @classmethod
    def before_fused_rms_norm_gated(cls, name: str, x: torch.Tensor, g: torch.Tensor):
        cls.record_tensor(x, f"[{name}][before]norm_x")
        cls.record_tensor(g, f"[{name}][before]norm_g")

    @classmethod
    def after_fused_rms_norm_gated(cls, name: str, out: torch.Tensor):
        cls.record_tensor(out, f"[{name}][after]norm_out")

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

    ############################## hooks for step and iter #################################
    @classmethod
    def after_micro_iter_forward(cls):
        if cls.skip():
            return
        assert cls.initialized, "AccProber is not initialized, please call setup() first"
        # Serialize buffered tensors here — outside any compiled region, so full
        # Python / string ops are safe.  tensor_info (str(tensor)) is included
        # because we are in eager mode at this point.
        for name, tensor_sum, shape, dtype_str, first_10 in cls._pending_tensors:
            cur_json = {
                "name": name,
                "tensor_sum": tensor_sum.item(),
                "shape": list(shape),
                "dtype": dtype_str,
                "step": cls.cur_step,
                "micro_batch_iter": cls.cur_micro_batch_iter,
                "first_10": first_10.tolist(),
            }
            cls.forward_records.append(json.dumps(cur_json, ensure_ascii=False))
        cls._pending_tensors = []

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
        log_rank0.info(f"TimeProber initialized at {cls.dump_dir}")

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
