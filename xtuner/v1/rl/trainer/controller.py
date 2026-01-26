import os
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

import ray
import torch

from xtuner.v1.rl.utils import free_object_refs
from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import get_logger

from .pack import RLDataPacker
from .worker import TrainingWorker, WorkerInputItem, WorkerLogItem


TRAIN_RAY_GET_TIMEOUT = os.getenv("XTUNER_TRAIN_RAY_GET_TIMEOUT", 5 * 3600)  # default 5 hours


class TrainingLogInfo(TypedDict):
    worker_log_infos: list[WorkerLogItem]
    padding_tokens: int
    pack_time: float
    train_time: float


def _summarize_process_group_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "ranks=0"

    count_key = next(
        (key for key in ("suspended", "resumed", "destroyed", "reloaded") if key in results[0]),
        "count",
    )
    counts = [result.get(count_key, 0) for result in results]
    count_summary = f"{counts[0]} on all ranks" if len(set(counts)) == 1 else f"by_rank={counts}"
    skipped_counts = [result.get("skipped", 0) for result in results]
    result_errors = [error for result in results for error in result.get("errors", [])]
    summary = f"ranks={len(results)}, {count_key}={count_summary}"
    if any(skipped_counts):
        skipped_summary = (
            f"{skipped_counts[0]} on all ranks" if len(set(skipped_counts)) == 1 else f"by_rank={skipped_counts}"
        )
        summary += f", skipped={skipped_summary}"
    if result_errors:
        summary += f", errors={len(result_errors)}, first_error={result_errors[0]}"
    return summary


class TrainingController:
    def __init__(self, workers: list[TrainingWorker]) -> None:
        self.workers = workers
        refs = [
            self.workers[0].get_model_cfg.remote(),
            self.workers[0].get_worker_cfg.remote(),
            self.workers[0].get_data_replicate_size.remote(),
        ]
        self.model_cfg, self.worker_cfg, self.data_replicate_size = ray.get(refs)
        self.worker_dp_ranks = ray.get([worker.get_dp_rank.remote() for worker in self.workers])
        self.pack_max_length = self.worker_cfg.pack_max_length
        self.pack_strategy = self.worker_cfg.pack_strategy
        self.data_packer = RLDataPacker(
            pack_max_length=self.pack_max_length,
            world_size=len(self.workers),
            data_replicate_size=self.data_replicate_size,
            optimizer_steps=self.worker_cfg.optimizer_steps,
            pack_strategy=self.pack_strategy,
            model_cfg=self.model_cfg,
            worker_log_dir=self.worker_cfg.log_dir,
        )
        log_dir = self.worker_cfg.log_dir
        self.log_dir = Path(log_dir) if log_dir is not None else None
        if self.log_dir is not None:
            self.logger = get_logger(log_dir=self.log_dir, tag="TrainingController")
        else:
            self.logger = get_logger()

    def fit(
        self,
        data_batches: list[WorkerInputItem],
        pack_max_length: int,
        rollout_idx: int,
    ) -> TrainingLogInfo:
        if pack_max_length != self.pack_max_length:
            raise ValueError(
                f"pack_max_length {pack_max_length} does not match worker config {self.pack_max_length}"
            )

        start_time = time.perf_counter()
        packed_data_batches, padding_tokens_num = self.data_packer.pack(data_batches)
        pack_end_time = time.perf_counter()

        handles = []
        data_batch_refs: dict[int, ray.ObjectRef] = {}
        for worker_idx, worker in enumerate(self.workers):
            dp_rank = self.worker_dp_ranks[worker_idx]
            if dp_rank not in data_batch_refs:
                data_batch_refs[dp_rank] = ray.put(packed_data_batches[dp_rank])
            handles.append(
                worker.fit.remote(  # type: ignore[attr-defined]
                    data_batches=data_batch_refs[dp_rank],
                    rollout_idx=rollout_idx,
                )
            )

        try:
            worker_log_infos = ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
            train_end_time = time.perf_counter()
        finally:
            free_pixel_value_refs: list[ray.ObjectRef] = []
            for dp_batches in packed_data_batches:
                for step_batches in dp_batches:
                    for data in step_batches:
                        pixel_values = data["seq_ctx"].pixel_values
                        if isinstance(pixel_values, list):
                            free_pixel_value_refs.extend(pixel_values)
            if free_pixel_value_refs:
                free_object_refs(free_pixel_value_refs)
            del data_batch_refs
            del packed_data_batches

        return {
            "worker_log_infos": worker_log_infos,
            "pack_time": pack_end_time - start_time,
            "train_time": train_end_time - pack_end_time,
            "padding_tokens": padding_tokens_num,
        }

    def offload(self, target: Literal["model", "optimizer", "all"] = "all"):
        if target == "model":
            ray.get([worker.offload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "optimizer":
            ray.get([worker.offload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "all":
            ray.get([worker.offload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
            ray.get([worker.offload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        return

    def onload(self, target: Literal["model", "optimizer", "all"] = "all"):
        """Onload the model or optimizer of the training workers."""
        if target == "model":
            ray.get([worker.onload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "optimizer":
            ray.get([worker.onload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "all":
            ray.get([worker.onload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
            ray.get([worker.onload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        return

    def bind_rollout_weight_update(
        self,
        *,
        targets,
        rollout_config,
    ):
        ray.get(
            [
                worker.bind_rollout_weight_update.remote(
                    targets=targets,
                    rollout_config=rollout_config,
                )
                for worker in self.workers
            ]
        )

    def weight_update(self, **kwargs):
        """Update the weights from the training workers."""
        handles = [worker.weight_update.remote(**kwargs) for worker in self.workers]
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    def suspend_train_nccl_process_groups(self):
        """Suspend train-side NCCL process groups after weight sync."""
        handles = [
            worker.suspend_train_nccl_process_groups.remote()  # type: ignore[attr-defined]
            for worker in self.workers
        ]
        results = ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        self.logger.info(f"Suspended train NCCL process groups: {_summarize_process_group_results(results)}")
        return results

    def resume_train_nccl_process_groups(self):
        """Resume train-side NCCL process groups before training."""
        handles = [
            worker.resume_train_nccl_process_groups.remote()  # type: ignore[attr-defined]
            for worker in self.workers
        ]
        results = ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        self.logger.info(f"Resumed train NCCL process groups: {_summarize_process_group_results(results)}")
        return results

    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        handles = [worker.save_hf.remote(hf_dir, save_dtype) for worker in self.workers]  # type: ignore
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    def resume(self, load_checkpoint_cfg: LoadCheckpointConfig):
        """Resume the training workers from the checkpoint."""
        handles = [worker.resume.remote(load_checkpoint_cfg) for worker in self.workers]  # type: ignore
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    def save(self, dcp_dir: str, no_save_optimizer: bool = False):
        """Save the DCP checkpoint of the training workers."""
        handles = [worker.save.remote(dcp_dir, no_save_optimizer) for worker in self.workers]  # type: ignore
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return
