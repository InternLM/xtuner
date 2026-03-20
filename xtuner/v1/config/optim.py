from abc import abstractmethod
from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from typing_extensions import Annotated

from xtuner.v1.optim import Muon
from xtuner.v1.utils import get_logger


logger = get_logger()


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr: Annotated[float, Parameter(help="Learning rate for optimization")] = 1e-5
    max_grad_norm: Annotated[float, Parameter(help="Maximum gradient norm for gradient clipping")] = 1.0
    skip_grad_norm_threshold: Annotated[
        float | None, Parameter(help="Gradient norm threshold for skipping optimizer step.")
    ] = None

    @abstractmethod
    def build(self, params):
        pass


class AdamWConfig(OptimConfig):
    weight_decay: Annotated[float, Parameter(help="Weight decay coefficient for L2 regularization")] = 0.01
    betas: Annotated[Tuple[float, float], Parameter(help="Beta coefficients for Adam optimizer")] = (0.9, 0.95)
    eps: Annotated[float, Parameter(help="Epsilon value for numerical stability in Adam optimizer")] = 1e-8
    foreach: Annotated[Optional[bool], Parameter(help="Use foreach implementation for AdamW")] = None

    def build(self, model):
        params = [p for p in model.parameters() if p.requires_grad]

        trainable_parameters_names = model.trainable_parameters()
        trainable_names = [name for name, _ in trainable_parameters_names]
        untrainable_names = []
        num_total_requires_grad = 0
        num_total = 0
        for name, params_ in model.named_parameters():
            num_total += params_.numel()
            num_total_requires_grad += params_.numel() if name in trainable_names else 0
            if name not in trainable_names:
                untrainable_names.append(name)

        if dist.get_rank() == 0:
            logger.info(
                f"Total trainable parameters: {num_total_requires_grad // 1e6}M, total parameters: {num_total // 1e6}M"
            )
            logger.info(f"Untrainable parameters names: {untrainable_names}")
        return torch.optim.AdamW(
            params, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay, foreach=self.foreach
        )


class MuonConfig(OptimConfig):
    weight_decay: Annotated[float, Parameter(help="Weight decay coefficient for L2 regularization")] = 0.1
    momentum: Annotated[float, Parameter(help="Momentum coefficients for Muon optimizer")] = 0.95
    betas: Annotated[Tuple[float, float], Parameter(help="Beta coefficients for AdamW optimizer")] = (0.9, 0.95)
    eps: Annotated[float, Parameter(help="Epsilon value for numerical stability in Muon optimizer")] = 1e-8
    adjust_lr: Annotated[
        Literal["rms_norm", "spectral_norm", "none"], Parameter(help="Method for adjusting lr in Muon")
    ] = "rms_norm"

    def build(self, model):
        trainable_parameters_names = model.trainable_parameters()
        trainable_names = {name for name, _ in trainable_parameters_names}

        untrainable_names = []
        num_total = 0
        num_total_requires_grad = 0
        num_muon = 0
        num_adamw = 0

        muon_params = []
        adamw_params = []

        for name, p in model.named_parameters():
            n = p.numel()
            num_total += n
            if name in trainable_names:
                num_total_requires_grad += n
                # we want to avoid using Muon for 1D-tensors, as well as embed_tokens and lm_head.
                # effectively-1D tensors where one dimension accounts for all elements (e.g. shape [1, D]) should
                # also be excluded.
                is_muon_tensor = (
                    p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name and p.numel() not in p.shape
                )
                if is_muon_tensor:
                    muon_params.append(p)
                    num_muon += n
                else:
                    adamw_params.append(p)
                    num_adamw += n
            else:
                untrainable_names.append(name)

        param_groups = [
            dict(params=muon_params),
            dict(params=adamw_params, algorithm="adamw"),
        ]

        if dist.get_rank() == 0:
            logger.info(
                f"Total trainable parameters: {num_total_requires_grad / 1e6:.2f}M, "
                f"total parameters: {num_total / 1e6:.2f}M"
            )
            logger.info(f"Muon params: {num_muon / 1e6:.2f}M, AdamW params: {num_adamw / 1e6:.2f}M (counts by numel)")
            logger.info(f"Untrainable parameters names: {untrainable_names}")
            logger.info(
                f"using Muon optimizer distributed_mesh_size: {model.fsdp_mesh.size()}, "
                f"distributed_mesh: {model.fsdp_mesh}"
            )

        optimizer = Muon(
            param_groups,
            distributed_mesh=model.fsdp_mesh,  # TODO: 暂不支持 EP>1
            lr=self.lr,
            mu=self.momentum,
            betas=self.betas,
            weight_decay=self.weight_decay,
            nesterov=True,
            adjust_lr=self.adjust_lr,
            flatten=True,  # TODO:@nil0x9 would be nice if we have fine-grained control here.
            use_triton=False,
            epsilon=self.eps,
        )
        return optimizer


class LRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lr_type: Annotated[Literal["cosine", "linear", "constant"], Parameter(help="Type of learning rate schedule")] = (
        "constant"
    )
    warmup_ratio: Annotated[float, Parameter(help="Ratio of warmup steps to total training steps")] = 0.03
    lr_min: Annotated[float, Parameter(help="Minimum learning rate for optimization")] = 1e-6
