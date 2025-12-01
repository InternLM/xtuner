from collections import defaultdict

import torch
import torch.distributed as dist
from mmengine.dist import get_world_size
from pydantic import BaseModel, ConfigDict, model_validator
from torch import nn
from torch.utils.hooks import RemovableHandle
from typing_extensions import TypedDict

from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.model import MoE
from xtuner.v1.model.base import ModelItem
from xtuner.v1.module import LMHead, MHAConfig, MLAConfig, MultiHeadAttention, MultiLatentAttention
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEDecoderLayer
from xtuner.v1.utils.device import get_device
from xtuner.v1.utils.grad_norm import cal_total_norm, group_tensors_by_device_mesh_and_placements


DEVICE = get_device()


RMS_NORM_MONITOR_MODULES = (
    nn.Embedding,
    DenseDecoderLayer,
    MoEDecoderLayer,
    LMHead,
)

SMALL_VAL = -1e9

MOE_MODEL_CLS = (MoE,)
ATTENTION_CLS = (MultiHeadAttention, MultiLatentAttention)

# In our prev tests, registering these stats dicts could lead to unexpected recompile behavior.
# We want to avoid accessing class members in hooks here, thus the global vars.
# TODO: This could be optimized in torch2.8 @nil0x9
ATTN_MAX_LSE: dict[str, torch.Tensor] = {}
ATTN_MAX_LOGITS: dict[str, torch.Tensor] = {}


class InternalMetrics(TypedDict, total=False):
    weight_rms: dict[str, float]
    maxvio: dict[str, float]
    drop_ratio: dict[str, float]
    router_logits_max: dict[str, float]
    router_logits_mean: dict[str, float]
    attn_max_lse: dict[str, float]
    attn_max_logits: dict[str, float]


class InternalMetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    internal_metrics_interval: int | None = None
    monitor_weights_rms_norm: bool = True
    monitor_attn_logits_stats: bool = True
    monitor_moe_router_logits_stats: bool | None = None  # only applies to MoE models
    monitor_moe_load_balance_stats: bool | None = None

    @model_validator(mode="after")
    def post_init(self):
        monitoring_fields = [
            self.monitor_weights_rms_norm,
            self.monitor_attn_logits_stats,
            self.monitor_moe_router_logits_stats,
            self.monitor_moe_load_balance_stats,
        ]

        if all(field is False or field is None for field in monitoring_fields):
            self.internal_metrics_interval = None

        return self


class InternalMetricsRecorder:
    def __init__(self, internal_metrics_cfg: InternalMetricsConfig, engine: TrainEngine):
        self.internal_metrics_cfg = internal_metrics_cfg
        self.model = engine.model
        self.intra_layer_micro_batch = engine.intra_layer_micro_batch
        self.hooks: list[RemovableHandle] = []
        self._attn_monitor_type: str | None = None
        self.metrics = self._init_metrics_dict()
        self._closed = False

    def _init_metrics_dict(self) -> InternalMetrics:
        metrics: InternalMetrics = {}
        if self.internal_metrics_cfg.monitor_weights_rms_norm:
            metrics["weight_rms"] = {}
        if self.internal_metrics_cfg.monitor_attn_logits_stats:
            attn_cfg: MHAConfig | MLAConfig = self.model.config.attention
            if isinstance(attn_cfg, MLAConfig):
                attn_impl = "flash_attention"
            else:
                attn_impl = attn_cfg.attn_impl
            if attn_impl == "eager_attention":
                # We typically won't use eager attn, but implement it here anyway
                self._attn_monitor_type = "attn_logits"
                metrics["attn_max_logits"] = {}
            elif not (DEVICE == "npu" and attn_impl == "flash_attention"):
                self._attn_monitor_type = "softmax_lse"
                metrics["attn_max_lse"] = {}
            for name, module in self.model.named_modules():
                if isinstance(module, ATTENTION_CLS):
                    if self._attn_monitor_type == "attn_logits":
                        ATTN_MAX_LOGITS[module.name] = torch.tensor(SMALL_VAL).to(DEVICE)
                    elif self._attn_monitor_type == "softmax_lse":
                        ATTN_MAX_LSE[module.name] = torch.tensor(SMALL_VAL).to(DEVICE)

        if isinstance(self.model, MoE):
            if self.internal_metrics_cfg.monitor_moe_router_logits_stats:
                metrics["router_logits_max"] = {}
                metrics["router_logits_mean"] = {}

            if self.internal_metrics_cfg.monitor_moe_load_balance_stats:
                metrics["maxvio"] = {}
                metrics["drop_ratio"] = {}

        return metrics

    @torch.no_grad()
    def calculate_module_weight_rms(self, module: nn.Module, layer_name: str, dtype: torch.dtype = torch.float32):
        """Calculate the RMS of the module's parameters."""
        self._check_closed()
        all_params = [param.data for param in module.parameters() if param.requires_grad]
        if not all_params:
            return
        grouped_params = group_tensors_by_device_mesh_and_placements(all_params)  # type: ignore[arg-type]
        total_norms = []
        total_numel = 0
        for params in grouped_params.values():
            total_norm = cal_total_norm(params, norm_type=2.0, foreach=True, dtype=dtype)
            total_norms.append(total_norm)
            total_numel += sum(p.numel() for p in params)
        param_l2_norm = torch.linalg.vector_norm(torch.stack(total_norms), ord=2.0, dtype=dtype)
        param_rms = param_l2_norm / total_numel**0.5
        self.metrics["weight_rms"][layer_name] = param_rms.item()

    def register_attn_output_hook(self, module: nn.Module):
        """Register attention output hook as a forward hook."""
        self._check_closed()

        def hook(module, input, output):
            if output.get("softmax_lse") is not None:
                ATTN_MAX_LSE[module.name] = torch.max(ATTN_MAX_LSE[module.name], output["softmax_lse"].max())
            if output.get("attn_logits") is not None:
                ATTN_MAX_LOGITS[module.name] = max(ATTN_MAX_LOGITS[module.name], output["attn_logits"].max())

        hook_handle: RemovableHandle = module.register_forward_hook(hook)
        self.hooks.append(hook_handle)

    @torch.no_grad()
    def pop_metrics(self, data_batches: list[ModelItem]):
        """Run a dummy forward to get metrics."""
        self._check_closed()
        for name, module in self.model.named_modules():
            if self.internal_metrics_cfg.monitor_attn_logits_stats and isinstance(module, ATTENTION_CLS):
                self.register_attn_output_hook(module)
            if self.internal_metrics_cfg.monitor_weights_rms_norm and isinstance(module, RMS_NORM_MONITOR_MODULES):
                self.calculate_module_weight_rms(module, self._clean_module_name(name), dtype=torch.float32)

        additional_kwargs = {}
        if self.internal_metrics_cfg.monitor_moe_router_logits_stats and isinstance(self.model, MoE):
            # for MoE model, add additional kwargs to return necessary stats
            additional_kwargs["return_router_logits"] = True

        # metrics before aggregation
        tokens_per_expert_global = None
        router_logits_max = defaultdict(list)
        router_logits_mean = defaultdict(list)

        # do dummy forward to get metrics
        if self.need_dummy_forward:
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

                if (
                    self.internal_metrics_cfg.monitor_moe_load_balance_stats
                    and output.get("tokens_per_expert_global") is not None
                ):
                    # At this point, tokens_per_expert_global is already all-reduced into current rank.
                    # [num_layers, num_experts]
                    if tokens_per_expert_global is None:
                        tokens_per_expert_global = output["tokens_per_expert_global"].float()
                    else:
                        tokens_per_expert_global += output["tokens_per_expert_global"].float()

                if (
                    self.internal_metrics_cfg.monitor_moe_router_logits_stats
                    and output.get("router_logits") is not None
                ):
                    for layer_name, router_logits in output["router_logits"].items():
                        # [bsz, packed_len, num_experts]
                        router_logits_max[layer_name].append(router_logits.max())
                        router_logits_mean[layer_name].append(router_logits.mean())

        if tokens_per_expert_global is not None:
            avg_count_load = tokens_per_expert_global.mean(1)
            max_load_i = torch.amax(tokens_per_expert_global, dim=1)
            maxvio_all_layers = (max_load_i - avg_count_load) / avg_count_load
            drop_ratio_all_layers = (tokens_per_expert_global - avg_count_load[:, None]).abs().mean(
                dim=1
            ) / avg_count_load
            drop_ratio = drop_ratio_all_layers.mean()
            self.metrics["drop_ratio"].update(
                {f"layer{idx}": drop_ratio_all_layers[idx].item() for idx in range(drop_ratio_all_layers.shape[0])}
            )
            self.metrics["maxvio"].update(
                {f"layer{idx}": maxvio_all_layers[idx].item() for idx in range(max_load_i.shape[0])}
            )
            maxvio = maxvio_all_layers.mean()
            self.metrics["maxvio"]["total"] = maxvio.item()
            self.metrics["drop_ratio"]["total"] = drop_ratio.item()

        if router_logits_max:
            for layer_name, router_logits_list in router_logits_max.items():
                # [bsz/intra_layer_micro_batch, ]
                local_router_logits_max = torch.max(torch.stack(router_logits_list))
                dist.all_reduce(local_router_logits_max, op=dist.ReduceOp.MAX)
                self.metrics["router_logits_max"][layer_name] = local_router_logits_max.item()

        if router_logits_mean:
            for layer_name, router_logits_list in router_logits_mean.items():
                # [bsz/intra_layer_micro_batch, ]
                local_router_logits_mean = torch.mean(torch.stack(router_logits_list))
                dist.all_reduce(local_router_logits_mean.div_(get_world_size()), op=dist.ReduceOp.SUM)
                self.metrics["router_logits_mean"][layer_name] = local_router_logits_mean.item()

        if self._attn_monitor_type == "softmax_lse":
            for layer_name, local_attn_max_lse in ATTN_MAX_LSE.items():
                dist.all_reduce(local_attn_max_lse, op=dist.ReduceOp.MAX)
                self.metrics["attn_max_lse"][layer_name] = local_attn_max_lse.item()

        if self._attn_monitor_type == "attn_logits":
            for layer_name, local_attn_max_logits in ATTN_MAX_LOGITS.items():
                dist.all_reduce(local_attn_max_logits, op=dist.ReduceOp.MAX)
                self.metrics["attn_max_logits"][layer_name] = local_attn_max_logits.item()

        self._maybe_reset_attn_max_lse_or_logits(ATTN_MAX_LSE)
        self._maybe_reset_attn_max_lse_or_logits(ATTN_MAX_LOGITS)

        for hook in self.hooks:
            hook.remove()

        return self.metrics

    def close(self):
        if not self._closed:
            del self.metrics
            global ATTN_MAX_LSE
            global ATTN_MAX_LOGITS
            ATTN_MAX_LSE = None
            ATTN_MAX_LOGITS = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _check_closed(self):
        if self._closed:
            raise RuntimeError("`InternalMetricsRecorder` is closed and cannot be used")

    def __del__(self):
        self.close()

    def _maybe_reset_attn_max_lse_or_logits(self, target: dict[str, torch.Tensor]):
        if not target:
            return
        for v in target.values():
            if isinstance(v, torch.Tensor):
                v.fill_(SMALL_VAL)
            else:
                raise TypeError("Only Tensor type is allowed!")

    def _clean_module_name(self, name: str) -> str:
        if "._checkpoint_wrapped_module" in name:
            name = name.replace("._checkpoint_wrapped_module", "")
        if "._orig_mod" in name:
            name = name.replace("._orig_mod", "")
        return name

    @property
    def need_dummy_forward(self) -> bool:
        internal_metrics_cfg = self.internal_metrics_cfg
        if (
            internal_metrics_cfg.monitor_attn_logits_stats
            or internal_metrics_cfg.monitor_moe_router_logits_stats
            or internal_metrics_cfg.monitor_moe_load_balance_stats
        ):
            return True
        else:
            return False


def flatten_internal_metrics_for_logs(metrics: InternalMetrics, sep: str = "/") -> dict:
    items = []
    for name, sub_metrics in metrics.items():
        if isinstance(sub_metrics, dict):
            for k, v in sub_metrics.items():
                if isinstance(v, (float, int)):
                    items.append((f"{name}{sep}{k}", v))
                else:
                    raise ValueError(f"Unsupported metric value type: expected float or int, but got {type(v)}")
        else:
            raise ValueError(
                f"Unsupported metric type for internal metrics: expected dict, but got {type(sub_metrics)}"
            )
    return dict(items)
