"""CPU-only checks for GPU-local NUMA plans, NVML queries, and coordination."""

import ast
import ctypes
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xtuner.v1.utils import numa_affinity as affinity


def make_plans(cpu_counts, gpu_nodes):
    cpus_by_node = {node: set(range(node * 100, node * 100 + count)) for node, count in cpu_counts.items()}
    return {
        rank: {
            "gpu_uuid": f"GPU-test-{rank}",
            "numa_id": node,
            "cpu_ids": cpus_by_node.get(node, set()).copy(),
            "mems_allowed": set(cpu_counts),
        }
        for rank, node in enumerate(gpu_nodes)
    }


@pytest.fixture
def setup_plan(monkeypatch):
    def setup(plan, rank_ids=None):
        monkeypatch.setenv("XTUNER_DISABLE_NUMA", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", str(len(plan)))
        monkeypatch.setattr("xtuner.v1.utils.device.get_device", lambda: "cuda")
        monkeypatch.setattr(os, "uname", lambda: SimpleNamespace(nodename="this-host"))
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: len(plan))
        monkeypatch.setattr(affinity, "_get_rank_numa_plan", lambda initial_cpu_affinity: deepcopy(plan.get(0)))
        reports = [
            {
                "host": "this-host",
                "rank": rank,
                "local_rank": rank if rank_ids is None else rank_ids[rank],
                "plan": binding,
                "error": None,
                "skip_reason": None,
            }
            for rank, binding in plan.items()
        ]
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_object",
            lambda output, report: output.__setitem__(slice(None), deepcopy(reports)),
        )

    return setup


class TestNumaBindingPlan:
    @pytest.mark.parametrize(
        ("cpu_counts", "gpu_nodes"),
        [
            ({0: 4, 1: 44}, [0] * 4 + [1] * 4),
            ({0: 16, 1: 48}, [0] * 4 + [1] * 4),
            ({0: 3, 1: 5}, [0] * 3 + [1] * 5),
            ({0: 4, 1: 3}, [0] * 4 + [1] * 3),
            ({0: 4, 1: 44}, [1]),
            ({0: 0, 1: 44}, [1]),
            ({0: 0, 1: 44}, [1, 1]),
            ({2: 2, 5: 3}, [5, 2, 5]),
        ],
    )
    def test_gpu_local_plan(self, setup_plan, cpu_counts, gpu_nodes):
        expected = make_plans(cpu_counts, gpu_nodes)
        setup_plan(expected)
        assert affinity.plan_numa_binding(set(range(4)) | set(range(100, 104))) == expected

    @pytest.mark.parametrize(
        ("cpu_counts", "gpu_nodes"),
        [
            ({0: 3, 1: 44}, [0] * 4 + [1] * 4),
            ({0: 2, 1: 64}, [0] * 4 + [1] * 4),
            ({0: 1, 1: 44}, [0] * 4 + [1] * 4),
            ({0: 3, 1: 44}, [0] * 4 + [1] * 3),
            ({0: 0, 1: 44}, [0]),
            ({0: 1}, [0, 0]),
        ],
    )
    def test_infeasible_plan_does_not_move_ranks_to_remote_nodes(self, setup_plan, cpu_counts, gpu_nodes):
        setup_plan(make_plans(cpu_counts, gpu_nodes))
        with pytest.raises(ValueError, match="GPU-local"):
            affinity.plan_numa_binding(set(range(4)) | set(range(100, 104)))

    def test_multiple_gpu_local_nodes_are_rejected(self, setup_collection):
        context = setup_collection(0, topology_reports())
        context.gpu_query.return_value = ({0, 100}, {0, 1})
        with pytest.raises(ValueError, match="must identify one GPU-local NUMA node"):
            affinity.plan_numa_binding(set(range(4)) | set(range(100, 104)))

    def test_each_rank_must_have_enough_cpus_for_all_ranks_on_its_node(self, setup_plan):
        plan = make_plans({0: 2}, [0, 0])
        plan[1]["cpu_ids"] = {0}
        setup_plan(plan)
        with pytest.raises(ValueError, match="local rank 1 has 1 available CPUs on node 0 for 2 local ranks"):
            affinity.plan_numa_binding(set(range(4)) | set(range(100, 104)))

    def test_shared_cpu_is_not_counted_as_capacity_twice(self, setup_plan):
        plan = make_plans({0: 2}, [0, 0])
        for entry in plan.values():
            entry["cpu_ids"] = {0}
        setup_plan(plan)
        with pytest.raises(ValueError, match="capacity is insufficient"):
            affinity.plan_numa_binding(set(range(4)) | set(range(100, 104)))

    def test_disallowed_memory_node_is_rejected(self, setup_collection):
        context = setup_collection(0, topology_reports())
        context.memory_query.return_value = [1]
        with pytest.raises(ValueError, match="no accessible GPU-local"):
            affinity.plan_numa_binding(set(range(4)) | set(range(100, 104)))

    def test_cpu_affinity_is_intersected_with_job_permissions(self, setup_collection):
        context = setup_collection(0, topology_reports())
        context.gpu_query.return_value = ({1, 2, 99}, {0})
        assert affinity.plan_numa_binding(set(range(4)) | set(range(100, 104)))[0]["cpu_ids"] == {1, 2}

    @pytest.mark.parametrize("ranks", [[], [0, 0], [1], [0, 2]])
    def test_invalid_local_rank_ids_are_rejected(self, setup_plan, ranks):
        setup_plan(make_plans({0: 4}, [0] * len(ranks)), ranks)
        with pytest.raises(ValueError, match="exactly once"):
            affinity.plan_numa_binding(set(range(4)) | set(range(100, 104)))


@pytest.fixture
def nvml(monkeypatch):
    word_bits = ctypes.sizeof(ctypes.c_ulonglong) * 8

    def handle_by_uuid(uuid):
        ctypes.c_char_p(uuid)
        return "handle-by-uuid"

    library = SimpleNamespace(
        nvmlInit=Mock(),
        nvmlShutdown=Mock(),
        nvmlDeviceGetHandleByUUID=Mock(side_effect=handle_by_uuid),
        nvmlDeviceGetCpuAffinity=Mock(return_value=[1, 0, 0, 1]),
        nvmlDeviceGetPciInfo=Mock(return_value=SimpleNamespace(busId=b"00000000:AB:00.0")),
    )
    monkeypatch.setitem(sys.modules, "pynvml", library)
    files = {
        Path("/sys/devices/system/cpu/possible"): f"0-{4 * word_bits - 1}",
        Path("/sys/bus/pci/devices/0000:ab:00.0/numa_node"): "1\n",
    }
    read_text = Mock(side_effect=lambda path: files[path])
    monkeypatch.setattr(affinity.Path, "read_text", lambda path: read_text(path))
    return library, read_text, word_bits, files


class TestGpuAffinityQuery:
    def test_mask_covers_host_cpus_even_when_job_only_has_low_cpu_ids(self, nvml):
        library, read_text, word_bits, _ = nvml
        cpus, nodes = affinity._get_gpu_affinity("GPU-actual-device")
        assert read_text.call_args_list == [
            ((Path("/sys/devices/system/cpu/possible"),), {}),
            ((Path("/sys/bus/pci/devices/0000:ab:00.0/numa_node"),), {}),
        ]
        library.nvmlDeviceGetHandleByUUID.assert_called_once_with(b"GPU-actual-device")
        library.nvmlDeviceGetCpuAffinity.assert_called_once_with("handle-by-uuid", 4)
        library.nvmlDeviceGetPciInfo.assert_called_once_with("handle-by-uuid")
        assert cpus == {0, 3 * word_bits}
        assert nodes == {1}
        library.nvmlShutdown.assert_called_once_with()

    def test_shutdown_runs_when_query_fails(self, nvml):
        library, _, _, _ = nvml
        library.nvmlDeviceGetCpuAffinity.side_effect = RuntimeError("NVML failure")
        with pytest.raises(RuntimeError, match="NVML failure"):
            affinity._get_gpu_affinity("GPU-actual-device")
        library.nvmlShutdown.assert_called_once_with()

    def test_unknown_pci_numa_node_prevents_binding(self, nvml, setup_collection):
        library, _, _, files = nvml
        files[Path("/sys/bus/pci/devices/0000:ab:00.0/numa_node")] = "-1\n"
        cpus, nodes = affinity._get_gpu_affinity("GPU-actual-device")
        assert nodes == {-1}
        context = setup_collection(0, topology_reports())
        context.gpu_query.return_value = (cpus, nodes)
        with pytest.raises(ValueError, match="no accessible GPU-local"):
            affinity.plan_numa_binding(set(range(4)) | set(range(100, 104)))
        library.nvmlShutdown.assert_called_once_with()

    def test_missing_pci_numa_node_is_rejected(self, nvml):
        library, read_text, _, _ = nvml
        read_text.side_effect = ["0-255", FileNotFoundError("numa_node is not mounted")]
        with pytest.raises(FileNotFoundError, match="numa_node is not mounted"):
            affinity._get_gpu_affinity("GPU-actual-device")
        library.nvmlShutdown.assert_called_once_with()

    def test_pci_domain_keeps_nonzero_value(self, nvml):
        library, _, _, files = nvml
        library.nvmlDeviceGetPciInfo.return_value.busId = b"00000002:AB:00.1"
        files[Path("/sys/bus/pci/devices/0002:ab:00.1/numa_node")] = "2"
        assert affinity._get_gpu_affinity("GPU-actual-device")[1] == {2}


@pytest.fixture
def setup_collection(monkeypatch):
    def setup(rank, reports, query_error=None):
        binding = make_plans({0: 4, 1: 4}, [0, 1])[rank]
        monkeypatch.setenv("XTUNER_DISABLE_NUMA", "0")
        monkeypatch.setattr("xtuner.v1.utils.device.get_device", lambda: "cuda")
        monkeypatch.setenv("LOCAL_RANK", str(rank))
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
        monkeypatch.setattr(os, "uname", lambda: SimpleNamespace(nodename="this-host"))
        cpu_query = Mock(return_value={0: set(range(4)), 1: set(range(100, 104))})
        monkeypatch.setattr(affinity, "get_available_cpus_by_numa_node", cpu_query)
        memory_query = Mock(return_value=[0, 1])
        monkeypatch.setitem(
            sys.modules,
            "numa",
            SimpleNamespace(
                memory=SimpleNamespace(
                    get_allocation_allowed_nodes=memory_query,
                )
            ),
        )
        device_query = Mock(return_value=SimpleNamespace(uuid="GPU-current-device"))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
        monkeypatch.setattr(torch.cuda, "get_device_properties", device_query)
        gpu_query = Mock(return_value=(binding["cpu_ids"], {binding["numa_id"]}), side_effect=query_error)
        monkeypatch.setattr(affinity, "_get_gpu_affinity", gpu_query)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: len(reports))

        def gather(output, report):
            output[:] = deepcopy(reports)
            output[rank] = report

        gather_mock = Mock(side_effect=gather)
        monkeypatch.setattr(torch.distributed, "all_gather_object", gather_mock)
        return SimpleNamespace(
            gather=gather_mock,
            cpu_query=cpu_query,
            memory_query=memory_query,
            device_query=device_query,
            gpu_query=gpu_query,
        )

    return setup


def topology_reports():
    return [
        {"host": "this-host", "rank": rank, "local_rank": rank, "plan": binding, "error": None, "skip_reason": None}
        for rank, binding in make_plans({0: 4, 1: 4}, [0, 1]).items()
    ]


class TestNumaTopologyCoordination:
    @pytest.mark.parametrize("reason", ["XTUNER_DISABLE_NUMA=1", "current device is not cuda"])
    @pytest.mark.parametrize("rank", [0, 1])
    def test_rank_configuration_mismatch_still_participates_in_collective(
        self, setup_collection, monkeypatch, rank, reason
    ):
        reports = topology_reports()
        reports[1]["skip_reason"] = reason
        reports[1]["plan"] = None
        context = setup_collection(rank, reports)
        if rank == 1:
            if reason == "XTUNER_DISABLE_NUMA=1":
                monkeypatch.setenv("XTUNER_DISABLE_NUMA", "1")
            else:
                monkeypatch.setattr("xtuner.v1.utils.device.get_device", lambda: "cpu")
        with pytest.raises(ValueError, match=reason):
            affinity.plan_numa_binding(set(range(4)) | set(range(100, 104)))
        context.gather.assert_called_once()
        if rank == 1:
            context.cpu_query.assert_not_called()
            context.memory_query.assert_not_called()
            context.device_query.assert_not_called()
            context.gpu_query.assert_not_called()

    @pytest.mark.parametrize("rank", [0, 1])
    def test_local_query_failure_is_exchanged_before_skipping(self, setup_collection, rank):
        reports = topology_reports()
        reports[1]["error"] = "rank 1: RuntimeError: NVML failed"
        reports[1]["plan"] = None
        context = setup_collection(rank, reports, RuntimeError("NVML failed") if rank == 1 else None)
        with pytest.raises(ValueError, match="NVML failed"):
            affinity.plan_numa_binding(set(range(4)) | set(range(100, 104)))
        context.gather.assert_called_once()

    @pytest.mark.parametrize("remote_status", ["skip", "error"])
    def test_other_hosts_can_skip_without_preventing_local_binding(self, setup_collection, remote_status):
        reports = topology_reports()
        reports.append(
            {
                "host": "other-host",
                "rank": 2,
                "local_rank": None,
                "plan": None,
                "skip_reason": "disabled" if remote_status == "skip" else None,
                "error": "NVML failed" if remote_status == "error" else None,
            }
        )
        context = setup_collection(0, reports)
        assert len(affinity.plan_numa_binding(set(range(4)) | set(range(100, 104)))) == 2
        context.gather.assert_called_once()
        context.device_query.assert_called_once_with(3)
        context.gpu_query.assert_called_once_with("GPU-current-device")

    @pytest.mark.parametrize(
        ("disabled", "device"),
        [("1", "cuda"), ("0", "cpu"), ("0", "cuda")],
    )
    def test_trainer_never_returns_before_coordination(self, disabled, device):
        path = Path(__file__).parents[2] / "xtuner/v1/train/trainer.py"
        tree = ast.parse(path.read_text())
        trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
        method = next(
            node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_try_bind_numa"
        )
        build_plan = Mock(side_effect=ValueError("host skips binding"))
        namespace = {
            "os": SimpleNamespace(getenv=lambda *_: disabled, sched_getaffinity=lambda _: {0, 1}),
            "DEVICE": device,
            "plan_numa_binding": build_plan,
            "logger": Mock(),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        namespace["_try_bind_numa"](SimpleNamespace(rank=0))
        build_plan.assert_called_once_with({0, 1})
