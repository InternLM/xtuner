from collections import defaultdict
from typing import Any
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from xtuner.v1.module import (
    RMSNorm,
    MultiHeadAttention,
    MultiLatentAttention,
    LMHead
)
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEGate, MoEBlock, MoEDecoderLayer
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.model import MoE
from xtuner.v1.model.base import ModelItem
from xtuner.v1.engine.train_engine import TrainEngine

RMS_NORM_MONITOR_MODULES = (
    nn.Embedding,
    DenseDecoderLayer,
    MoEDecoderLayer,
    LMHead,
    # MultiHeadAttention,
    # MultiLatentAttention,
    # MoEGate,
    # MoEBlock,
    # RMSNorm,
)

MOE_MODEL_CLS = (MoE,)
ATTENTION_CLS = (MultiHeadAttention, MultiLatentAttention)

class InternalMetricsRecorder:
    def __init__(self, engine: TrainEngine):
        self.model = engine.model
        self.intra_layer_micro_batch = engine.intra_layer_micro_batch
        self.hooks: list[Any] = []
        self.metrics: dict[str, dict[str, Any]] = dict[str, dict[str, Any]](
            weight_rms=dict[str, Any](),
            maxvio=dict[str, Any](),
            drop_ratio=dict[str, Any](),
            router_logits_max=dict[str, Any](),
            router_logits_mean=dict[str, Any](),
            attn_max_lse=dict[str, Any](),
            attn_max_logits=dict[str, Any](),
        )

    def register_weight_rms_hook(self, module, layer_name=None):
        """
        Register weight RMS hook as a pre-forward hook, as at this point, the parameters are should be 
        all-gathered into current rank.
        """
        if layer_name is None:
            layer_name = f"layer_{len(self.weight_rms_dict)}"

        def hook(module, args, kwargs=None):
            if layer_name in self.metrics['weight_rms']: # only calculate before the first batch
                return
            l2_norm = 0.0
            total_params = 0
            for param in module.parameters():
                if param.requires_grad:
                    l2_norm += torch.norm(param.detach().flatten().float(), p=2) ** 2
                    total_params += param.numel()
            if total_params > 0:
                rms = torch.sqrt(l2_norm / total_params)
                self.metrics['weight_rms'][layer_name] = rms

        hook_handle = module.register_forward_pre_hook(hook)
        self.hooks.append(hook_handle)

    def register_attn_extra_info_hook(self, module, layer_name=None):
        def hook(module, input, output):
            extra_info = output[1]
            if extra_info.get("softmax_lse", None) is not None:
                if layer_name not in self.metrics["attn_max_lse"]:
                    # original shape: [n_head, seq]
                    self.metrics["attn_max_lse"][layer_name] = extra_info["softmax_lse"].max()
                else:
                    prev_lse_max = self.metrics["attn_max_lse"][layer_name]
                    self.metrics["attn_max_lse"][layer_name] = max(prev_lse_max, extra_info["softmax_lse"].max())
            if extra_info.get("attn_logits", None) is not None:
                if layer_name not in self.metrics["attn_max_logits"]:
                    # original shape: [b, n_head, seq, seq]
                    self.metrics["attn_max_logits"][layer_name] = extra_info["attn_logits"].max()
                else:
                    prev_logits_max = self.metrics["attn_max_logits"][layer_name]
                    self.metrics["attn_max_logits"][layer_name] = max(prev_logits_max, extra_info["attn_logits"].max())

        hook_handle = module.register_forward_hook(hook)
        self.hooks.append(hook_handle)

    @torch.no_grad()
    def get_metrics(self, data_batches: list[ModelItem]):
        additional_kwargs = {}
        if isinstance(self.model, MoE):
            # for MoE model, add additional kwargs to return necessary stats
            additional_kwargs["return_tokens_per_expert_global"] = True
            additional_kwargs["return_router_logits"] = True
        
        # metrics before aggregation
        tokens_per_expert_global = None
        router_logits_max = defaultdict(list)
        router_logits_mean = defaultdict(list)

        # do dummy forward to get metrics
        for i in range(0, len(data_batches), self.intra_layer_micro_batch):
            data_batch = data_batches[i : i + self.intra_layer_micro_batch]
            seq_ctx_list = []
            loss_ctx_list = []
            for data in data_batch:
                seq_ctx = data["seq_ctx"]
                loss_ctx = data["loss_ctx"]
                seq_ctx_list.append(seq_ctx)
                loss_ctx_list.append(loss_ctx)
            if self.intra_layer_micro_batch == 1:
                output = self.model(seq_ctx=seq_ctx_list[0], loss_ctx=loss_ctx_list[0], **additional_kwargs)
            else:
                # although we dont need loss at this point, we still need loss_ctx for micro-batch forward
                output = self.model(
                    seq_ctx=seq_ctx_list,
                    loss_ctx=loss_ctx_list,
                    **additional_kwargs,
                )

            if output.get("tokens_per_expert_global", None) is not None:
                # At this point, tokens_per_expert_global is already all-reduced into current rank.
                # [num_layers, num_experts]
                if tokens_per_expert_global is None:
                    tokens_per_expert_global = output["tokens_per_expert_global"].float()
                else:
                    tokens_per_expert_global += output["tokens_per_expert_global"].float()


            if output.get("router_logits", None) is not None:
                for layer_name, router_logits in output["router_logits"].items():
                    # [bsz, packed_len, num_experts]
                    router_logits_max[layer_name].append(router_logits.max())
                    router_logits_mean[layer_name].append(router_logits.mean())

        if tokens_per_expert_global is not None:
            avg_count_load = tokens_per_expert_global.mean(1)
            max_load_i = torch.amax(tokens_per_expert_global, dim=1)
            maxvio_all_layers = (max_load_i - avg_count_load) / avg_count_load
            drop_ratio_all_layers = (tokens_per_expert_global - avg_count_load[:,None]).abs().mean(dim=1) / avg_count_load
            drop_ratio = drop_ratio_all_layers.mean()
            self.metrics["drop_ratio"].update(
                {f"layer{idx}": drop_ratio_all_layers[idx].item() for idx in range(drop_ratio_all_layers.shape[0])}
            )
            self.metrics["maxvio"].update(
                {f"layer{idx}": maxvio_all_layers[idx].item() for idx in range(max_load_i.shape[0])}
            )
            maxvio = maxvio_all_layers.mean()
            self.metrics["maxvio"]["total"] = maxvio
            self.metrics["drop_ratio"]["total"] = drop_ratio

        if len(router_logits_max) > 0:
            for layer_name, router_logits_list in router_logits_max.items():
                # [bsz/intra_layer_micro_batch, ]
                local_router_logits_max = torch.max(torch.stack(router_logits_list))
                dist.all_reduce(local_router_logits_max, op=dist.ReduceOp.MAX)
                self.metrics["router_logits_max"][layer_name] = local_router_logits_max.item()

        if len(router_logits_mean) > 0:
            for layer_name, router_logits_list in router_logits_mean.items():
                # [bsz/intra_layer_micro_batch, ]
                local_router_logits_mean = torch.mean(torch.stack(router_logits_list))
                dist.all_reduce(local_router_logits_mean.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
                self.metrics["router_logits_mean"][layer_name] = local_router_logits_mean.item()

        if len(self.metrics["attn_max_lse"]) > 0:
            for layer_name, local_attn_max_lse in self.metrics["attn_max_lse"].items():
                dist.all_reduce(local_attn_max_lse, op=dist.ReduceOp.MAX)
                self.metrics["attn_max_lse"][layer_name] = local_attn_max_lse.item()

        if len(self.metrics["attn_max_logits"]) > 0:
            for layer_name, local_attn_max_logits in self.metrics["attn_max_logits"].items():
                dist.all_reduce(local_attn_max_logits, op=dist.ReduceOp.MAX)
                self.metrics["attn_max_logits"][layer_name] = local_attn_max_logits.item()

        return self.metrics

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ATTENTION_CLS):
                self.register_attn_extra_info_hook(module, self._clean_module_name(name))
            if isinstance(module, RMS_NORM_MONITOR_MODULES):
                self.register_weight_rms_hook(module, self._clean_module_name(name))
            else:
                pass
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self.hooks:
            hook.remove()

    def _clean_module_name(self, name: str) -> str:
        if "._checkpoint_wrapped_module" in name:
            name = name.replace("._checkpoint_wrapped_module", "")
        if "._orig_mod" in name:
            name = name.replace("._orig_mod", "")
        return name
