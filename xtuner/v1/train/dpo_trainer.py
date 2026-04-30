# Copyright (c) OpenMMLab. All rights reserved.
"""
DPO (Direct Preference Optimization) Trainer for XTuner v1.

This trainer supports offline preference learning with multiple loss types:
- sigmoid: Standard DPO loss
- bco_pair: Binary Classifier Optimization
- sft: Supervised Fine-Tuning loss
- hinge, ipo, robust: Other DPO variants

For MPO (Mixed Preference Optimization), use loss_types=["sigmoid", "bco_pair", "sft"]
with appropriate loss_weights.
"""
import json
import os
import gc
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import torch
import torch.distributed as dist
from mmengine import load
from mmengine.dist import get_rank, init_dist
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict, model_validator
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from typing_extensions import Self

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from xtuner.v1.config import FSDPConfig, OptimConfig, LRConfig
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.data_proto.utils import split_for_sequence_parallel, pad_to_multiple_of
from xtuner.v1.datasets import qwen3_vl_dpo_collator, DPOColateItem
from xtuner.v1.datasets.config import DataloaderConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.engine.vision_compose_train_engine import VisionComposeTrainEngine
from xtuner.v1.model.base import TransformerConfig
from xtuner.v1.model.compose.base import BaseComposeConfig
from xtuner.v1.loss.rl_loss import LogProbConfig
from xtuner.v1.rl.dpo import DPOLossConfig, DPOLossContext, DPOLossContextInputItem
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger, is_hf_model_path, record_git_info
from xtuner.v1.utils.device import get_device, get_torch_device_module
from torch.distributed.device_mesh import init_device_mesh

from .trainer import ExpHistory, ExpInfo, GitInfo, XTunerMeta


DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()
logger = get_logger()


class DPOTrainerConfig(BaseModel):
    """Configuration for DPO Trainer.

    Args:
        model_cfg: Model architecture configuration.
        optim_cfg: Optimizer configuration.
        loss_cfg: DPO loss configuration.
        lr_cfg: Learning rate scheduler configuration.
        fsdp_cfg: FSDP configuration for distributed training.
        load_from: Path to load model from.
        ref_load_from: Path to load reference model from (optional).
        tokenizer_path: Path to tokenizer.
        work_dir: Working directory for outputs.
        total_epochs: Total number of training epochs.
        global_batch_size: Global batch size across all devices.
        per_device_batch_size: Batch size per device.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        max_length: Maximum sequence length.
        save_interval: Interval for saving checkpoints.
        eval_interval: Interval for evaluation.
        log_interval: Interval for logging.
        seed: Random seed.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model_cfg: TransformerConfig | BaseComposeConfig
    optim_cfg: OptimConfig
    loss_cfg: DPOLossConfig
    lr_cfg: LRConfig
    fsdp_cfg: FSDPConfig
    load_from: str | Path
    ref_load_from: str | Path | None = None
    tokenizer_path: str | Path
    work_dir: Path | str = Path("./work_dir")
    log_dir: Path | str | None = None
    total_epochs: int = 3
    global_batch_size: int = 128
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_length: int = 4096
    save_interval: int | None = None
    eval_interval: int | None = None
    log_interval: int = 10
    seed: int = 42
    auto_resume: bool = False
    freeze_ref_model: bool = True
    num_workers: int = 4
    use_vlm_collator: bool = False  # Use qwen3_vl_dpo_collator for VLM
    sp_size: int = 1  # Sequence parallel size
    
    # Dataloader configuration (xtuner v1 style, optional)
    dataloader_cfg: DataloaderConfig | None = None

    @model_validator(mode="after")
    def _convert_paths(self):
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
        return self


class DPOTrainer:
    """DPO Trainer for offline preference learning.

    This trainer implements Direct Preference Optimization and its variants
    for offline preference alignment. It supports:
    - Standard DPO (sigmoid loss)
    - MPO (Mixed Preference Optimization with multiple loss types)
    - IPO, BCO, and other variants

    Args:
        config: DPOTrainerConfig instance.
        train_dataset: Training dataset with preference pairs.
        eval_dataset: Evaluation dataset (optional).
        collate_fn: Custom collate function (optional).

    Example:
        >>> from functools import partial
        >>> from xtuner.v1.datasets import (
        ...     PreferenceJsonlDataset,
        ...     PreferenceTokenizeFunction,
        ... )
        >>> 
        >>> # Prepare dataset
        >>> tokenize_fn = PreferenceTokenizeFunction(tokenizer, max_length=4096)
        >>> dataset = PreferenceJsonlDataset("data.jsonl", tokenize_fn=tokenize_fn)
        >>> collator = partial(dpo_llm_collator, pack_max_length=4096, padding_token_idx=0)
        >>> 
        >>> # Create trainer
        >>> config = DPOTrainerConfig(
        ...     model_cfg=model_cfg,
        ...     optim_cfg=AdamWConfig(lr=5e-7),
        ...     loss_cfg=DPOLossConfig(
        ...         loss_types=["sigmoid", "bco_pair", "sft"],
        ...         loss_weights=[0.8, 0.2, 1.0],
        ...     ),
        ...     lr_cfg=LRConfig(lr_type="cosine"),
        ...     fsdp_cfg=FSDPConfig(),
        ...     load_from="Qwen/Qwen3-VL-8B-Instruct",
        ...     tokenizer_path="Qwen/Qwen3-VL-8B-Instruct",
        ... )
        >>> trainer = DPOTrainer(config, dataset, collate_fn=collator)
        >>> trainer.fit()
    """

    META_PATH = ".xtuner_dpo"

    def __init__(
        self,
        config: DPOTrainerConfig,
        dataloader_cfg: DataloaderConfig,
    ):
        """Initialize DPO Trainer.
        
        This follows the same pattern as the SFT Trainer - accepting dataloader_cfg
        and building the dataloader internally using dataloader_cfg.build().
        
        Args:
            config: DPO trainer configuration.
            dataloader_cfg: Dataloader configuration containing dataset configs.
        """
        self.config = config
        self._dataloader_config = dataloader_cfg

        self._cur_step = 0
        self._cur_epoch = 0

        # Initialize distributed if needed
        if not dist.is_initialized():
            init_dist("pytorch")

        # Initialize
        self._set_deterministic()
        self._set_random_seed(config.seed)

        # Setup work directory
        if isinstance(config.work_dir, str):
            config.work_dir = Path(config.work_dir)
        self._work_dir = config.work_dir

        if get_rank() == 0:
            self._work_dir.mkdir(parents=True, exist_ok=True)

        # Synchronize before continuing
        if dist.is_initialized():
            dist.barrier()

        # Initialize meta for experiment tracking
        self._meta = self._init_xtuner_meta(self._work_dir, config.auto_resume)

        # Setup logging
        log_dir = config.log_dir or self.exp_dir
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        self.logger = self._init_logger(log_dir)

        # Load tokenizer
        self.logger.info(f"Loading tokenizer from {config.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path, trust_remote_code=True
        )

        # Initialize training engine (model + optimizer)
        # Use VisionComposeTrainEngine for VLM (compose) models, TrainEngine for text-only models
        self.logger.info(f"Initializing training engine from {config.load_from}")
        if isinstance(config.model_cfg, BaseComposeConfig):
            self.train_engine = VisionComposeTrainEngine(
                model_cfg=config.model_cfg,
                optim_cfg=config.optim_cfg,
                fsdp_cfg=config.fsdp_cfg,
            )
        else:
            self.train_engine = TrainEngine(
                model_cfg=config.model_cfg,
                optim_cfg=config.optim_cfg,
                fsdp_cfg=config.fsdp_cfg,
            )

        # Load model weights using from_hf (same as TrainEngine API)
        self.logger.info(f"Loading model weights from {config.load_from}")
        self.train_engine.from_hf(str(config.load_from))

        # Resolve dataloader config with tokenizer (set pad_token_id etc.)
        self._sp_size = config.sp_size
        self._resolve_dataloader_config(dataloader_cfg)

        # Initialize data mesh (same as SFT Trainer)
        tp_size = config.fsdp_cfg.tp_size if config.fsdp_cfg else 1
        sp_size = config.sp_size
        self.data_mesh = self._init_data_mesh(tp_size, sp_size, config.fsdp_cfg)
        self.sp_mesh = self.data_mesh["sp"]

        # Build dataloader using xtuner v1 pattern (same as SFT Trainer)
        self.logger.info("Building dataloader from dataloader_cfg...")
        micro_batch_size = config.per_device_batch_size
        
        # Get dp_mesh from data_mesh
        dp_mesh = self.data_mesh["dp"]
        
        self._dataloader = dataloader_cfg.build(
            tokenizer=self.tokenizer,
            dp_mesh=dp_mesh,
            global_batch_size=config.global_batch_size,
            micro_batch_size=micro_batch_size,
            seed=config.seed,
        )
        
        # Evaluation dataloader not supported yet in xtuner v1 pattern
        self.eval_dataloader = None

        # Build learning rate scheduler
        self._build_lr_scheduler()

        # Initialize reference model if needed
        self.ref_engine = None
        if not config.loss_cfg.reference_free:
            self._init_reference_model()

        # LogProbConfig for safe forward (avoids DTensor / regular Tensor mismatch
        # when loss_ctx=None causes direct F.linear on FSDP-sharded weights)
        self.logprob_cfg = LogProbConfig(
            mode=config.loss_cfg.mode,
            chunk_size=config.loss_cfg.chunk_size,
        )

        # Calculate training steps
        self._total_steps = len(self._dataloader) * config.total_epochs

        # Save config
        if get_rank() == 0:
            config_path = log_dir / "dpo_trainer_config.json"
            with config_path.open("w") as f:
                f.write(config.model_dump_json(indent=2))

        self.logger.info(f"DPO Trainer initialized")
        self.logger.info(f"  Total epochs: {config.total_epochs}")
        self.logger.info(f"  Total steps: {self._total_steps}")
        self.logger.info(f"  Loss types: {config.loss_cfg.loss_types}")
        self.logger.info(f"  Loss weights: {config.loss_cfg.loss_weights}")
        self.logger.info(f"  Reference free: {config.loss_cfg.reference_free}")
        self.logger.info(f"  Sequence parallel size: {config.sp_size}")
    
    def _resolve_dataloader_config(self, dataloader_cfg: DataloaderConfig):
        """Resolve dataloader config conflicts, similar to SFT Trainer."""
        if hasattr(self.tokenizer, "pad_token_id"):
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = 0
        
        if dataloader_cfg.pad_token_id is None:
            dataloader_cfg.pad_token_id = pad_token_id
        elif dataloader_cfg.pad_token_id != pad_token_id:
            self.logger.warning(
                f"Dataloader pad_token_id {dataloader_cfg.pad_token_id} is different from tokenizer "
                f"pad_token_id {pad_token_id}. Using tokenizer pad_token_id {pad_token_id}."
            )
            dataloader_cfg.pad_token_id = pad_token_id
        
        # Check sequence parallel constraints
        if self._sp_size > 1:
            if dataloader_cfg.pack_to_max_length is False:
                self.logger.warning(
                    "pack_to_max_length must be True when using sequence parallel. "
                    "Setting pack_to_max_length to True."
                )
                dataloader_cfg.pack_to_max_length = True

    def _init_data_mesh(self, tp_size: int, sp_size: int, fsdp_cfg: FSDPConfig | None):
        """Initialize data mesh for distributed training, same as SFT Trainer."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        if world_size % tp_size != 0:
            raise ValueError(
                f"Found tp_size {tp_size}, world_size {world_size}. "
                "Tensor parallel size must be a divisor of world size."
            )
        
        if world_size % sp_size != 0:
            raise ValueError(
                f"Found sp_size {sp_size}, world_size {world_size}. "
                "Sequence parallel size must be a divisor of world size."
            )
        
        if world_size % (tp_size * sp_size) != 0:
            raise ValueError(
                f"Found tp_size {tp_size}, sp_size {sp_size}, world_size {world_size}. "
                "`tp_size * sp_size` must be a divisor of world size."
            )
        
        dp_size = world_size // (tp_size * sp_size)
        
        # Use CPU for device mesh if not cpu_offload, else use DEVICE
        cpu_offload = fsdp_cfg.cpu_offload if fsdp_cfg else False
        device = str(DEVICE) if not cpu_offload else "cpu"
        
        data_mesh = init_device_mesh(
            device,
            (dp_size, sp_size, tp_size),
            mesh_dim_names=("dp", "sp", "tp"),
        )
        return data_mesh

    @classmethod
    def from_config(cls, config: DPOTrainerConfig) -> "DPOTrainer":
        """Create a DPOTrainer instance from a DPOTrainerConfig.
        
        This method follows the same pattern as SFT Trainer.from_config,
        passing the dataloader_cfg to __init__ which handles building internally.

        Args:
            config: DPOTrainerConfig instance containing all configuration parameters.

        Returns:
            DPOTrainer instance initialized with the provided config.
        
        Example:
            >>> from xtuner.v1.utils import Config
            >>> cfg = Config.fromfile("dpo_config.py")
            >>> trainer = DPOTrainer.from_config(cfg["trainer"])
            >>> trainer.fit()
        """
        # Validate config has dataloader_cfg
        if config.dataloader_cfg is None:
            raise ValueError(
                "DPOTrainerConfig.dataloader_cfg is required when using from_config(). "
                "Please configure dataloader_cfg with dataset_config_list."
            )
        
        # Create trainer - __init__ handles all building internally
        return cls(
            config=config,
            dataloader_cfg=config.dataloader_cfg,
        )

    def _build_lr_scheduler(self):
        """Build learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR

        # Calculate total OPTIMIZER steps (not batch steps!)
        # lr_scheduler.step() is called once per gradient_accumulation_steps batches
        steps_per_epoch = len(self._dataloader)
        total_batches = steps_per_epoch * self.config.total_epochs
        total_steps = total_batches // self.config.gradient_accumulation_steps

        self.logger.info(f"LR Scheduler: total_batches={total_batches}, "
                        f"gradient_accumulation_steps={self.config.gradient_accumulation_steps}, "
                        f"total_optimizer_steps={total_steps}")

        lr_cfg = self.config.lr_cfg
        optimizer = self.train_engine.optimizer

        # Calculate warmup steps
        if lr_cfg.warmup_ratio < 1:
            warmup_steps = int(lr_cfg.warmup_ratio * total_steps)
        else:
            warmup_steps = int(lr_cfg.warmup_ratio)

        self.logger.info(f"LR Scheduler: warmup_ratio={lr_cfg.warmup_ratio}, warmup_steps={warmup_steps}")

        # Warmup function - linear warmup from 0 to base_lr (same as SFT Trainer)
        def warmup_fn(x):
            return x / warmup_steps if x < warmup_steps else 1

        warmup_scheduler = LambdaLR(optimizer, warmup_fn)

        # Main scheduler
        main_steps = total_steps - warmup_steps
        if lr_cfg.lr_type == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=lr_cfg.lr_min / lr_cfg.lr if hasattr(lr_cfg, 'lr') else 0.01,
                total_iters=main_steps,
            )
        elif lr_cfg.lr_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=main_steps, eta_min=lr_cfg.lr_min)
        elif lr_cfg.lr_type == "constant":
            scheduler = LambdaLR(optimizer, lambda x: 1.0)
        else:
            raise ValueError(f"Unsupported lr type: {lr_cfg.lr_type}")

        self.lr_scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_steps],
        )

    def _init_reference_model(self):
        """Initialize reference model for KL divergence computation."""
        ref_load_from = self.config.ref_load_from or self.config.load_from

        self.logger.info(f"Initializing reference model from {ref_load_from}")

        # Create a separate engine for reference model
        # Use VisionComposeTrainEngine for VLM models, TrainEngine for text-only models
        ref_fsdp_cfg = deepcopy(self.config.fsdp_cfg)

        if isinstance(self.config.model_cfg, BaseComposeConfig):
            self.ref_engine = VisionComposeTrainEngine(
                model_cfg=self.config.model_cfg,
                optim_cfg=self.config.optim_cfg,
                fsdp_cfg=ref_fsdp_cfg,
            )
        else:
            self.ref_engine = TrainEngine(
                model_cfg=self.config.model_cfg,
                optim_cfg=self.config.optim_cfg,
                fsdp_cfg=ref_fsdp_cfg,
            )
        self.ref_engine.from_hf(str(ref_load_from))

        # Freeze reference model
        if self.config.freeze_ref_model:
            for param in self.ref_engine.model.parameters():
                param.requires_grad = False
            self.ref_engine.model.eval()

    # _setup_dataloader removed - dataloader is now built in __init__ using dataloader_cfg.build()

    def _compute_ref_logprobs(
        self,
        chosen_seq_ctx: SequenceContext,
        rejected_seq_ctx: SequenceContext,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reference model log probabilities.

        Uses LogProbConfig to build loss_ctx so the forward goes through the
        LossHead module (triggering FSDP hooks) instead of raw F.linear on
        DTensor weights.
        """
        if self.ref_engine is None:
            return None, None

        def _get_field(out: Any, key: str):
            if isinstance(out, dict):
                return out[key]
            return getattr(out, key)

        with torch.no_grad():
            chosen_logprob_ctx = self.logprob_cfg.build(data={"shifted_labels": chosen_labels})
            rejected_logprob_ctx = self.logprob_cfg.build(data={"shifted_labels": rejected_labels})

            ref_chosen_output = self.ref_engine.forward_only(
                chosen_seq_ctx, loss_ctx={"lm": chosen_logprob_ctx}
            )
            ref_rejected_output = self.ref_engine.forward_only(
                rejected_seq_ctx, loss_ctx={"lm": rejected_logprob_ctx}
            )

            # output["loss"] contains per-token logprobs [B, L]
            ref_chosen_token_logprobs = _get_field(ref_chosen_output, "loss")
            ref_rejected_token_logprobs = _get_field(ref_rejected_output, "loss")

            chosen_mask = (chosen_labels != -100)
            rejected_mask = (rejected_labels != -100)

            local_chosen_sum = (ref_chosen_token_logprobs * chosen_mask.float()).sum(dim=-1)
            local_rejected_sum = (ref_rejected_token_logprobs * rejected_mask.float()).sum(dim=-1)
            local_chosen_count = chosen_mask.sum(dim=-1).float()
            local_rejected_count = rejected_mask.sum(dim=-1).float()

            if self.sp_mesh.size() > 1:
                dist.all_reduce(local_chosen_sum, op=dist.ReduceOp.SUM, group=self.sp_mesh.get_group())
                dist.all_reduce(local_rejected_sum, op=dist.ReduceOp.SUM, group=self.sp_mesh.get_group())
                dist.all_reduce(local_chosen_count, op=dist.ReduceOp.SUM, group=self.sp_mesh.get_group())
                dist.all_reduce(local_rejected_count, op=dist.ReduceOp.SUM, group=self.sp_mesh.get_group())

            if self.config.loss_cfg.use_average_log_prob:
                ref_chosen_logprobs = local_chosen_sum / local_chosen_count.clamp(min=1)
                ref_rejected_logprobs = local_rejected_sum / local_rejected_count.clamp(min=1)
            else:
                ref_chosen_logprobs = local_chosen_sum
                ref_rejected_logprobs = local_rejected_sum

        return ref_chosen_logprobs, ref_rejected_logprobs

    def _gather_logprobs(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Gather log probabilities for the given labels."""
        import torch.nn.functional as F

        # Shift logits and labels for causal LM
        # logits: [batch, seq_len, vocab_size]
        # labels: [batch, seq_len] (already shifted in collator)
        logprobs = F.log_softmax(logits, dim=-1)

        # Gather log probs for target tokens
        # Handle -100 (ignore index) by clipping
        gathered = logprobs.gather(
            dim=-1, index=labels.clip(min=0).unsqueeze(-1)
        ).squeeze(-1)

        return gathered

    def _train_step(self, batch: List[DPOColateItem]) -> Dict[str, float]:
        """Perform a single training step."""
        total_loss = torch.tensor(0.0, device=DEVICE)
        all_metrics = {}

        def _get_field(out: Any, key: str):
            if isinstance(out, dict):
                return out[key]
            return getattr(out, key)

        for item in batch:
            # Extract data from DPOColateItem
            chosen_seq_ctx = item["chosen_seq_ctx"]
            rejected_seq_ctx = item["rejected_seq_ctx"]
            chosen_labels = item["chosen_shifted_labels"]
            rejected_labels = item["rejected_shifted_labels"]

            # Move to device
            chosen_seq_ctx = chosen_seq_ctx.to(DEVICE)
            rejected_seq_ctx = rejected_seq_ctx.to(DEVICE)
            chosen_labels = chosen_labels.to(DEVICE)
            rejected_labels = rejected_labels.to(DEVICE)

            # Apply sequence parallel split if enabled
            if self.sp_mesh.size() > 1:
                sp_size = self.sp_mesh.size()

                chosen_seq_ctx = chosen_seq_ctx.split(sequence_parallel_mesh=self.sp_mesh)
                rejected_seq_ctx = rejected_seq_ctx.split(sequence_parallel_mesh=self.sp_mesh)
                chosen_labels = pad_to_multiple_of(chosen_labels, -100, sp_size, dim=1)
                chosen_labels = split_for_sequence_parallel(chosen_labels, dim=1, sp_mesh=self.sp_mesh)
                rejected_labels = pad_to_multiple_of(rejected_labels, -100, sp_size, dim=1)
                rejected_labels = split_for_sequence_parallel(rejected_labels, dim=1, sp_mesh=self.sp_mesh)

            # Compute reference log probs if needed
            ref_chosen_logprobs = item.get("ref_chosen_logps")
            ref_rejected_logprobs = item.get("ref_rejected_logps")

            if ref_chosen_logprobs is None and self.ref_engine is not None:
                ref_chosen_logprobs, ref_rejected_logprobs = self._compute_ref_logprobs(
                    chosen_seq_ctx, rejected_seq_ctx, chosen_labels, rejected_labels
                )

            # Create DPO input item
            # NOTE: chosen_labels and rejected_labels are ALREADY SP-split above,
            # so we should NOT call dpo_input.sp_split() again (double split bug)
            dpo_input = DPOLossContextInputItem(
                chosen_shifted_labels=chosen_labels,
                rejected_shifted_labels=rejected_labels,
                ref_chosen_logprobs=ref_chosen_logprobs,
                ref_rejected_logprobs=ref_rejected_logprobs,
            )

            # Build loss kwargs
            loss_kwargs_list = DPOLossContext.build_batches_loss_kwargs(
                [dpo_input], self.config.loss_cfg,
                sp_mesh=self.sp_mesh if self.sp_mesh.size() > 1 else None,
            )

            # Build LogProbContexts for policy forward (goes through LossHead module,
            # which triggers FSDP hooks to properly unshard DTensor weights)
            chosen_logprob_ctx = self.logprob_cfg.build(data={"shifted_labels": chosen_labels})
            rejected_logprob_ctx = self.logprob_cfg.build(data={"shifted_labels": rejected_labels})

            chosen_output = self.train_engine.model(
                seq_ctx=chosen_seq_ctx, loss_ctx={"lm": chosen_logprob_ctx}
            )
            rejected_output = self.train_engine.model(
                seq_ctx=rejected_seq_ctx, loss_ctx={"lm": rejected_logprob_ctx}
            )

            # output["loss"] contains per-token logprobs [B, L]
            chosen_token_logprobs = _get_field(chosen_output, "loss")
            rejected_token_logprobs = _get_field(rejected_output, "loss")

            # Concatenate to match DPOLossKwargs shifted_labels layout [chosen; rejected]
            all_logprobs = torch.cat([chosen_token_logprobs, rejected_token_logprobs], dim=1)

            loss_kwargs = loss_kwargs_list[0]
            shifted_labels = loss_kwargs.shifted_labels
            chosen_mask = loss_kwargs.chosen_mask
            rejected_mask = loss_kwargs.rejected_mask
            loss_weight = loss_kwargs.loss_weight

            # Compute local log probs (sum over local tokens)
            chosen_logprobs = all_logprobs * chosen_mask.float()
            rejected_logprobs = all_logprobs * rejected_mask.float()
            
            # Sum local log probs
            local_chosen_sum = chosen_logprobs.sum(dim=-1)
            local_rejected_sum = rejected_logprobs.sum(dim=-1)
            local_chosen_count = chosen_mask.sum(dim=-1).float()
            local_rejected_count = rejected_mask.sum(dim=-1).float()

            # Aggregate across SP ranks if using sequence parallelism
            if self.sp_mesh.size() > 1:
                # All-reduce to get global sum of log probs across SP group
                dist.all_reduce(local_chosen_sum, op=dist.ReduceOp.SUM, group=self.sp_mesh.get_group())
                dist.all_reduce(local_rejected_sum, op=dist.ReduceOp.SUM, group=self.sp_mesh.get_group())
                dist.all_reduce(local_chosen_count, op=dist.ReduceOp.SUM, group=self.sp_mesh.get_group())
                dist.all_reduce(local_rejected_count, op=dist.ReduceOp.SUM, group=self.sp_mesh.get_group())

            # Compute final log probs (average or sum based on config)
            if self.config.loss_cfg.use_average_log_prob:
                policy_chosen_logps = local_chosen_sum / local_chosen_count.clamp(min=1)
                policy_rejected_logps = local_rejected_sum / local_rejected_count.clamp(min=1)
            else:
                policy_chosen_logps = local_chosen_sum
                policy_rejected_logps = local_rejected_sum

            ref_chosen_logps = loss_kwargs.ref_chosen_logprobs
            ref_rejected_logps = loss_kwargs.ref_rejected_logprobs
            if ref_chosen_logps is None:
                ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            if ref_rejected_logps is None:
                ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

            loss_ctx = DPOLossContext(self.config.loss_cfg, loss_kwargs)
            loss = torch.tensor(0.0, device=all_logprobs.device, dtype=all_logprobs.dtype)
            extra_info: dict[str, Any] = {}

            for loss_type, weight in zip(self.config.loss_cfg.loss_types, self.config.loss_cfg.loss_weights):
                if loss_type == "sigmoid":
                    _l = loss_ctx._dpo_loss_sigmoid(  # type: ignore[attr-defined]
                        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
                    ).mean() * weight
                    extra_info["dpo_sigmoid_loss"] = _l.detach()
                    loss = loss + _l
                elif loss_type == "robust":
                    _l = loss_ctx._dpo_loss_robust(  # type: ignore[attr-defined]
                        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
                    ).mean() * weight
                    extra_info["dpo_robust_loss"] = _l.detach()
                    loss = loss + _l
                elif loss_type == "hinge":
                    _l = loss_ctx._dpo_loss_hinge(  # type: ignore[attr-defined]
                        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
                    ).mean() * weight
                    extra_info["dpo_hinge_loss"] = _l.detach()
                    loss = loss + _l
                elif loss_type == "ipo":
                    _l = loss_ctx._dpo_loss_ipo(  # type: ignore[attr-defined]
                        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
                    ).mean() * weight
                    extra_info["dpo_ipo_loss"] = _l.detach()
                    loss = loss + _l
                elif loss_type == "bco_pair":
                    _l = loss_ctx._bco_pair_loss(  # type: ignore[attr-defined]
                        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
                    ).mean() * weight
                    extra_info["bco_pair_loss"] = _l.detach()
                    loss = loss + _l
                elif loss_type == "nca_pair":
                    _l = loss_ctx._nca_pair_loss(  # type: ignore[attr-defined]
                        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
                    ).mean() * weight
                    extra_info["nca_pair_loss"] = _l.detach()
                    loss = loss + _l
                elif loss_type == "sppo_hard":
                    _l = loss_ctx._sppo_hard_loss(  # type: ignore[attr-defined]
                        policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
                    ).mean() * weight
                    extra_info["sppo_hard_loss"] = _l.detach()
                    loss = loss + _l
                elif loss_type == "sft":
                    # SFT loss = cross-entropy = -logprob on chosen tokens
                    _l = (-all_logprobs * chosen_mask.float() * loss_weight).sum() * weight
                    if self.sp_mesh.size() > 1:
                        dist.all_reduce(_l, op=dist.ReduceOp.SUM, group=self.sp_mesh.get_group())
                    extra_info["sft_loss"] = _l.detach()
                    loss = loss + _l
                else:
                    raise ValueError(f"Unsupported loss_type: {loss_type}")

            total_loss = total_loss + loss

            # Collect metrics
            for k, v in extra_info.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                if isinstance(v, torch.Tensor):
                    all_metrics[k].append(v.detach())
                else:
                    all_metrics[k].append(v)

        # Average loss over batch
        total_loss = total_loss / len(batch)

        # Backward pass
        total_loss.backward()

        # Gradient accumulation
        if (self._cur_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.train_engine.optimizer.step()
            self.lr_scheduler.step()
            self.train_engine.optimizer.zero_grad()

        # Prepare metrics
        metrics = {"loss": total_loss.item()}
        for k, v_list in all_metrics.items():
            if isinstance(v_list[0], torch.Tensor):
                metrics[k] = torch.stack(v_list).mean().item()
            elif isinstance(v_list[0], (int, float)):
                metrics[k] = sum(v_list) / len(v_list)

        return metrics

    def fit(self):
        """Run the DPO training loop."""
        self.logger.info("Starting DPO training")

        for epoch in range(self.config.total_epochs):
            self._cur_epoch = epoch

            # Set epoch for distributed sampler
            if hasattr(self._dataloader, "set_epoch"):
                self._dataloader.set_epoch(epoch)
            elif hasattr(self._dataloader.sampler, "set_epoch"):
                self._dataloader.sampler.set_epoch(epoch)

            self.logger.info(f"Epoch {epoch + 1}/{self.config.total_epochs}")

            # Training loop
            self.train_engine.model.train()
            epoch_metrics = []

            progress_bar = tqdm(
                self._dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=get_rank() != 0,
            )

            for batch_idx, batch in enumerate(progress_bar):
                metrics = self._train_step(batch)
                epoch_metrics.append(metrics)
                self._cur_step += 1

                # Logging
                if self._cur_step % self.config.log_interval == 0:
                    avg_metrics = self._average_metrics(epoch_metrics[-self.config.log_interval:])
                    avg_metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                    # Format lr with scientific notation, other metrics with 4 decimal places
                    log_parts = []
                    for k, v in avg_metrics.items():
                        if k == "lr":
                            log_parts.append(f"{k}={v:.2e}")
                        else:
                            log_parts.append(f"{k}={v:.4f}")
                    log_str = f"Step {self._cur_step}: " + ", ".join(log_parts)
                    self.logger.info(log_str)
                    progress_bar.set_postfix(avg_metrics)

                # Save checkpoint
                if self.config.save_interval and self._cur_step % self.config.save_interval == 0:
                    self._save_checkpoint()

                # Evaluation
                if (
                    self.config.eval_interval
                    and self._cur_step % self.config.eval_interval == 0
                    and self.eval_dataloader is not None
                ):
                    eval_metrics = self._evaluate()
                    self.logger.info(f"Evaluation: {eval_metrics}")

                # Garbage collection for memory management (same as SFT Trainer)
                if self._cur_step % 50 == 0:
                    gc.collect()

            # End of epoch
            avg_epoch_metrics = self._average_metrics(epoch_metrics)
            self.logger.info(f"Epoch {epoch + 1} average metrics: {avg_epoch_metrics}")

            # Save at end of epoch
            self._save_checkpoint()

        self.logger.info("Training completed!")

    def _evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        self.train_engine.model.eval()
        eval_metrics = []

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=get_rank() != 0):
                metrics = self._eval_step(batch)
                eval_metrics.append(metrics)

        self.train_engine.model.train()
        return self._average_metrics(eval_metrics)

    def _eval_step(self, batch: List[DPOColateItem]) -> Dict[str, float]:
        """Perform a single evaluation step."""
        total_loss = torch.tensor(0.0, device=DEVICE)
        all_metrics = {}

        for item in batch:
            chosen_seq_ctx = item["chosen_seq_ctx"].to(DEVICE)
            rejected_seq_ctx = item["rejected_seq_ctx"].to(DEVICE)
            chosen_labels = item["chosen_shifted_labels"].to(DEVICE)
            rejected_labels = item["rejected_shifted_labels"].to(DEVICE)

            # Apply sequence parallel split if enabled
            if self.sp_mesh.size() > 1:
                sp_size = self.sp_mesh.size()
                chosen_seq_ctx = chosen_seq_ctx.split(sequence_parallel_mesh=self.sp_mesh)
                rejected_seq_ctx = rejected_seq_ctx.split(sequence_parallel_mesh=self.sp_mesh)
                chosen_labels = pad_to_multiple_of(chosen_labels, -100, sp_size, dim=1)
                chosen_labels = split_for_sequence_parallel(chosen_labels, dim=1, sp_mesh=self.sp_mesh)
                rejected_labels = pad_to_multiple_of(rejected_labels, -100, sp_size, dim=1)
                rejected_labels = split_for_sequence_parallel(rejected_labels, dim=1, sp_mesh=self.sp_mesh)

            ref_chosen_logprobs = item.get("ref_chosen_logps")
            ref_rejected_logprobs = item.get("ref_rejected_logps")

            if ref_chosen_logprobs is None and self.ref_engine is not None:
                ref_chosen_logprobs, ref_rejected_logprobs = self._compute_ref_logprobs(
                    chosen_seq_ctx, rejected_seq_ctx, chosen_labels, rejected_labels
                )

            dpo_input = DPOLossContextInputItem(
                chosen_shifted_labels=chosen_labels,
                rejected_shifted_labels=rejected_labels,
                ref_chosen_logprobs=ref_chosen_logprobs,
                ref_rejected_logprobs=ref_rejected_logprobs,
            )

            loss_kwargs_list = DPOLossContext.build_batches_loss_kwargs(
                [dpo_input], self.config.loss_cfg
            )

            chosen_output = self.train_engine.forward_only(chosen_seq_ctx)
            rejected_output = self.train_engine.forward_only(rejected_seq_ctx)

            hidden_states = torch.cat([chosen_output.hidden_states, rejected_output.hidden_states], dim=1)

            lm_head = self.train_engine.model.get_output_embeddings()
            head_weight = lm_head.weight
            head_bias = getattr(lm_head, "bias", None)

            loss_ctx = DPOLossContext(self.config.loss_cfg, loss_kwargs_list[0])
            loss, (_, extra_info) = loss_ctx.loss_fn(
                hidden_states, head_weight, head_bias, loss_kwargs_list[0]
            )

            total_loss = total_loss + loss

            for k, v in extra_info.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                if isinstance(v, torch.Tensor):
                    all_metrics[k].append(v.detach())
                else:
                    all_metrics[k].append(v)

        total_loss = total_loss / len(batch)

        metrics = {"eval_loss": total_loss.item()}
        for k, v_list in all_metrics.items():
            if isinstance(v_list[0], torch.Tensor):
                metrics[f"eval_{k}"] = torch.stack(v_list).mean().item()

        return metrics

    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics over a list."""
        if not metrics_list:
            return {}

        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            avg_metrics[key] = sum(values) / len(values)

        return avg_metrics

    def _save_checkpoint(self):
        """Save model checkpoint."""
        save_path = self.exp_dir / f"checkpoint-{self._cur_step}"
        self.logger.info(f"Saving checkpoint to {save_path}")

        if get_rank() == 0:
            save_path.mkdir(parents=True, exist_ok=True)

        # Synchronize before saving
        if dist.is_initialized():
            dist.barrier()

        # Save model
        self.train_engine.save_hf(str(save_path))

        # Save tokenizer
        if get_rank() == 0:
            if isinstance(self.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                self.tokenizer.save_pretrained(str(save_path))

        # Update meta
        if get_rank() == 0:
            self._meta.latest_exp.hf_checkpoint_list.append(str(save_path))

            # Save meta
            meta_path = self._work_dir / self.META_PATH
            with meta_path.open("w") as f:
                f.write(self._meta.model_dump_json(indent=2))

        if dist.is_initialized():
            dist.barrier()

    def _init_logger(self, work_dir: Path):
        """Initialize logger."""
        return get_logger(log_dir=work_dir, tag="DPOTrainer")

    def _set_deterministic(self):
        """Set deterministic mode."""
        if XTUNER_DETERMINISTIC:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _set_random_seed(self, seed: int):
        """Set random seed."""
        set_random_seed(seed)

    def _init_xtuner_meta(self, work_dir: Path, resume: bool) -> XTunerMeta:
        """Initialize experiment metadata.
        
        This follows the same pattern as RLTrainer._init_xtuner_meta.
        """
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)

        meta_path = work_dir / self.META_PATH
        if not meta_path.exists():
            meta = XTunerMeta(exps=[])
            with open(meta_path, "w") as f:
                f.write(meta.model_dump_json(indent=2))

        meta = cast(XTunerMeta, XTunerMeta.model_validate(load(meta_path, file_format="json")))

        resume = resume and bool(meta.exps)

        if resume:
            # Resume from existing experiment
            latest_exp = meta.exps[-1]
            latest_exp_history = latest_exp.history[-1]

            begin = cast(int, latest_exp_history.get("end") or latest_exp_history["begin"])
            exp_dir = Path(latest_exp.exp_dir)
            git_dir = exp_dir / f"git-info-begin-{begin}"

            if not git_dir.exists():
                git_dir.mkdir(parents=True, exist_ok=True)

            staged_path, unstaged_path = git_dir / "staged.diff", git_dir / "unstaged.diff"

            commit = record_git_info(staged_path, unstaged_path)
            git_info = GitInfo(
                commit=commit,
                staged=str(staged_path),
                unstaged=str(unstaged_path),
            )

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            new_exp_history = ExpHistory(
                begin=begin,
                timestamp=timestamp,
                git_info=git_info,
            )
            latest_exp.history.append(new_exp_history)
        else:
            # Start new experiment
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            exp_dir = work_dir / timestamp
            git_dir = Path(f"{exp_dir}/git-info-begin-0")

            if not git_dir.exists():
                git_dir.mkdir(parents=True, exist_ok=True)

            staged_path, unstaged_path = git_dir / "staged.diff", git_dir / "unstaged.diff"
            commit = record_git_info(staged_path, unstaged_path)
            git_info = GitInfo(
                commit=commit,
                staged=str(staged_path),
                unstaged=str(unstaged_path),
            )

            new_history = ExpHistory(
                begin=0,
                timestamp=timestamp,
                git_info=git_info,
            )
            new_exp = ExpInfo(history=[new_history], exp_dir=str(exp_dir))
            meta.exps.append(new_exp)

        return meta

    @property
    def work_dir(self) -> Path:
        return self._work_dir

    @property
    def exp_dir(self) -> Path:
        return Path(self._meta.latest_exp.exp_dir)

    @property
    def cur_step(self) -> int:
        return self._cur_step

    @property
    def cur_epoch(self) -> int:
        return self._cur_epoch
# [XTuner][2026-01-12 07:36:21][WARNING] Failed to process inputs: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'images', using text-only