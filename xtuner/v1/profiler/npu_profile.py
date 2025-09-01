import pickle
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch_npu
from torch import distributed as dist

from xtuner.v1.utils import get_logger


logger = get_logger()


class MemoryProfiler:
    def __init__(self, profile_dir: Path):
        torch.npu.memory._record_memory_history()  # type: ignore
        self.profile_dir = profile_dir

    def step(self, exit_ctx: bool = False):
        if dist.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if exit_ctx:
            output_dir = self.profile_dir.with_name(self.profile_dir.name + "_exit")
        else:
            output_dir = self.profile_dir

        output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Dumping memory snapshot to {output_dir}")
        begin = time.monotonic()
        with open(self.profile_dir / f"rank{rank}_memory_snapshot.pickle", "wb") as output:
            pickle.dump(torch.npu.memory._snapshot(), output)  # type: ignore
        logger.info(f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds")


@contextmanager
def profilling_memory(profile_dir: Path):
    profiler = MemoryProfiler(profile_dir)
    yield
    try:
        profiler.step(exit_ctx=False)
    except torch.OutOfMemoryError:
        profiler.step(exit_ctx=True)


@contextmanager
def profilling_time(profile_dir: Path):
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     export_type=[
    #         torch_npu.profiler.ExportType.Text,
    #         torch_npu.profiler.ExportType.Db
    #         ],
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    #     msprof_tx=False,
    #     aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    #     l2_cache=False,
    #     op_attr=False,
    #     data_simplification=False,
    #     record_op_args=False,
    #     gc_detect_threshold=None
    # )

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
    )

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        # schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
        experimental_config=experimental_config,
    ) as prof:
        yield

        prof.step()
