import os
import time
from pathlib import Path
from typing import Literal

import ray
import torch
from ray.actor import ActorProxy
from typing_extensions import TypedDict

from xtuner.v1.train.trainer import LoadCheckpointConfig
from xtuner.v1.utils import get_logger, ray_method

from .pack import DataBatchPacker
from .worker import TrainingWorker, WorkerInputItem, WorkerLogItem


TRAIN_RAY_GET_TIMEOUT = os.getenv("XTUNER_TRAIN_RAY_GET_TIMEOUT", 5 * 3600)  # default 5 hours


class TrainingLogInfo(TypedDict):
    worker_log_infos: list[WorkerLogItem]
    padding_tokens: int
    pack_time: float
    train_time: float


class RawTrainingController:
    def __init__(self, workers: list[TrainingWorker]) -> None:
        self.workers = workers
        refs = [
            self.workers[0].get_model_cfg.remote(),
            self.workers[0].get_worker_cfg.remote(),
            self.workers[0].get_data_replicate_size.remote(),
        ]
        self.model_cfg, self.worker_cfg, self.data_replicate_size = ray.get(refs)
        dp_ranks_handle = [worker.get_dp_rank.remote() for worker in self.workers]
        self.worker_dp_ranks = ray.get(dp_ranks_handle)
        self.pack_max_length = self.worker_cfg.pack_max_length
        self.pack_strategy = self.worker_cfg.pack_strategy
        self.data_packer = DataBatchPacker(
            pack_max_length=self.pack_max_length,
            world_size=len(self.workers),
            data_replicate_size=self.data_replicate_size,
            optimizer_steps=self.worker_cfg.optimizer_steps,
            pack_strategy=self.pack_strategy,
            worker_log_dir=self.worker_cfg.log_dir,
        )
        log_dir = self.worker_cfg.log_dir
        self.log_dir = None
        if log_dir is not None:
            self.log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
            self.logger = get_logger(log_dir=self.log_dir, tag="TrainingController")
        else:
            self.logger = get_logger()

    @ray_method
    def fit(
        self,
        data_batches: list[WorkerInputItem],
        rollout_idx: int,
    ) -> TrainingLogInfo:
        start_time = time.perf_counter()
        packed_data_batches, padding_tokens_num = self.data_packer.pack(data_batches)
        pack_end_time = time.perf_counter()
        handles = []
        for worker_idx, worker in enumerate(self.workers):
            dp_rank = self.worker_dp_ranks[worker_idx]
            handles.append(
                worker.fit.remote(  # type: ignore[attr-defined]
                    data_batches=packed_data_batches[dp_rank],
                    rollout_idx=rollout_idx,
                )
            )
        train_end_time = time.perf_counter()
        worker_log_infos = ray.get(handles)
        train_log_info: TrainingLogInfo = {
            "worker_log_infos": worker_log_infos,
            "pack_time": pack_end_time - start_time,
            "train_time": train_end_time - pack_end_time,
            "padding_tokens": padding_tokens_num,
        }
        return train_log_info

    @ray_method
    def offload(self, target: Literal["model", "optimizer", "all"] = "all"):
        if target == "model":
            ray.get([worker.offload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "optimizer":
            ray.get([worker.offload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        elif target == "all":
            ray.get([worker.offload_model.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
            ray.get([worker.offload_optimizer.remote() for worker in self.workers], timeout=TRAIN_RAY_GET_TIMEOUT)  # type: ignore
        return

    @ray_method
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

    @ray_method
    def update_rollout_info(self, info_dict):
        ray.get([worker.update_rollout_info.remote(**info_dict) for worker in self.workers])  # type: ignore[attr-defined]

    @ray_method
    def update_weights(self):
        """Update the weights of the training workers."""
        handles = [worker.update_weights.remote() for worker in self.workers]
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    @ray_method
    def save_hf(self, hf_dir: str, save_dtype: torch.dtype = torch.bfloat16):
        handles = [worker.save_hf.remote(hf_dir, save_dtype) for worker in self.workers]  # type: ignore
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    @ray_method
    def resume(self, load_checkpoint_cfg: LoadCheckpointConfig):
        """Resume the training workers from the checkpoint."""
        handles = [worker.resume.remote(load_checkpoint_cfg) for worker in self.workers]  # type: ignore
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    @ray_method
    def save(self, dcp_dir: str, no_save_optimizer: bool = False):
        """Save the DCP checkpoint of the training workers."""
        handles = [worker.save.remote(dcp_dir, no_save_optimizer) for worker in self.workers]  # type: ignore
        ray.get(handles, timeout=TRAIN_RAY_GET_TIMEOUT)
        return

    @ray_method
    def ready(self) -> bool:
        return True


TrainingController = ray.remote(RawTrainingController)
TrainingControllerProxy = ActorProxy[RawTrainingController]
