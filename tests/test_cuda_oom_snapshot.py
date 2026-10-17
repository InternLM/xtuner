import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from xtuner.v1.profiler import cuda_profile


def test_oom_observer_saves_rank_snapshot(tmp_path, monkeypatch):
    observers = []
    history = []
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setattr(torch.cuda.memory, "_record_memory_history", lambda **kwargs: history.append(kwargs))
    monkeypatch.setattr(
        torch.cuda.memory,
        "_dump_snapshot",
        lambda filename: Path(filename).write_bytes(pickle.dumps({"device_traces": [[{"action": "oom"}]]})),
    )
    monkeypatch.setattr(torch._C, "_cuda_attach_out_of_memory_observer", observers.append)

    cuda_profile.record_memory_history_on_oom(tmp_path)
    assert history == [{"max_entries": 100_000, "stacks": "python"}]
    assert len(observers) == 1

    observers[0](0, 1024, 2048, 512)
    snapshot = pickle.loads((tmp_path / "rank7_memory_snapshot.pickle").read_bytes())
    assert snapshot["device_traces"][0][0]["action"] == "oom"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_oom_after_sampled_memory_step_is_saved(tmp_path):
    # A child process keeps the private CUDA observer out of the rest of the test suite.
    script = """
import os
import sys
from pathlib import Path

import torch

from xtuner.v1.profiler.cuda_profile import profiling_memory, record_memory_history_on_oom

os.environ["RANK"] = "0"
os.environ["PROFILE_RANKS"] = "all"
output = Path(sys.argv[1])
torch.cuda.set_device(0)
record_memory_history_on_oom(output / "oom")
with profiling_memory(output / "step-0", keep_oom_history=True):
    torch.empty(1024, device="cuda")

total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(0.01)
try:
    torch.empty(int(total * 0.02), dtype=torch.uint8, device="cuda")
except torch.OutOfMemoryError:
    pass
else:
    raise AssertionError("Expected a CUDA allocator OOM")

try:
    with profiling_memory(output / "step-1", keep_oom_history=True):
        torch.empty(int(total * 0.02), dtype=torch.uint8, device="cuda")
except torch.OutOfMemoryError:
    pass
else:
    raise AssertionError("Expected an OOM in the sampled step")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "step-0" / "rank0_memory_snapshot.pickle").is_file()
    assert (tmp_path / "step-1_exit" / "rank0_memory_snapshot.pickle").is_file()
    snapshot = pickle.loads((tmp_path / "oom" / "rank0_memory_snapshot.pickle").read_bytes())
    assert any(event.get("action") == "oom" for trace in snapshot["device_traces"] for event in trace)
