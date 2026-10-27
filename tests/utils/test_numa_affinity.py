"""NUMA 绑定的公开行为测试。

TestNumaBindingDiscovery
    test_binds_selected_gpu_with_rank_permissions
    test_skips_disabled_or_noncuda
    test_skips_inaccessible_gpu_local_resources
    test_skips_unavailable_driver_information
    test_skips_incomplete_other_gpu_information
TestNumaCapacity
    test_checks_available_cpu_capacity_on_every_gpu_node
    test_allows_equal_cpu_gpu_counts_and_uneven_layout
TestNumaBindingApplication
    test_memory_failure_restores_initial_cpu_affinity
    test_cpu_failure_does_not_bind_memory
"""

import os
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from xtuner.v1.utils import device as device_utils
from xtuner.v1.utils import numa_affinity as affinity


@pytest.fixture
def numa_environment(monkeypatch, tmp_path):
    def setup(cpus_by_node=None, gpus=None, selected_uuid="GPU-A", allowed_cpus=None, device_index=0):
        if cpus_by_node is None:
            cpus_by_node = {0: {0, 1}}
        if gpus is None:
            gpus = {"GPU-A": ("0000:ab:00.0", 0)}
        if allowed_cpus is None:
            allowed_cpus = set().union(*cpus_by_node.values())

        # 重置设备探测缓存，真实执行项目内get_device；只替换外部硬件接口。
        monkeypatch.setattr(device_utils, "_DEVICE", None)
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
        monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cuda"))
        monkeypatch.setenv("XTUNER_DISABLE_NUMA", "0")
        monkeypatch.setitem(sys.modules, "pynvml", None)

        get_affinity = Mock(return_value=set(allowed_cpus))
        set_affinity = Mock()
        monkeypatch.setattr(os, "sched_getaffinity", get_affinity)
        monkeypatch.setattr(os, "sched_setaffinity", set_affinity)

        cpu_query = Mock(side_effect=lambda node: sorted(cpus_by_node.get(node, set())))
        memory_query = Mock(return_value=list(cpus_by_node))
        membind = Mock()
        monkeypatch.setitem(
            sys.modules,
            "numa",
            SimpleNamespace(
                info=SimpleNamespace(node_to_cpus=cpu_query),
                memory=SimpleNamespace(get_allocation_allowed_nodes=memory_query, set_membind_nodes=membind),
            ),
        )
        # PyTorch UUID字符串不带驱动procfs使用的GPU-前缀。
        device_query = Mock(return_value=SimpleNamespace(uuid=selected_uuid.removeprefix("GPU-")))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: device_index)
        monkeypatch.setattr(torch.cuda, "get_device_properties", device_query)

        # 仅重定向系统路径；glob/read_text以及项目内拓扑、解析和绑定流程都真实执行。
        def system_path(path):
            return tmp_path / path.lstrip("/")

        monkeypatch.setattr(affinity, "Path", system_path)
        for node in cpus_by_node:
            system_path(f"/sys/devices/system/node/node{node}").mkdir(parents=True)
        information = {}
        numa_nodes = {}
        for gpu_uuid, (pci_bus_id, node) in gpus.items():
            info_path = system_path(f"/proc/driver/nvidia/gpus/{pci_bus_id}/information")
            info_path.parent.mkdir(parents=True)
            info_path.write_text(f"Model: Test GPU\nGPU UUID: {gpu_uuid}\nBus Location: {pci_bus_id}\n")
            node_path = system_path(f"/sys/bus/pci/devices/{pci_bus_id}/numa_node")
            node_path.parent.mkdir(parents=True)
            node_path.write_text(f"{node}\n")
            information[gpu_uuid] = info_path
            numa_nodes[gpu_uuid] = node_path

        return SimpleNamespace(
            set_affinity=set_affinity,
            cpu_query=cpu_query,
            memory_query=memory_query,
            membind=membind,
            device_query=device_query,
            information=information,
            numa_nodes=numa_nodes,
            proc_dir=system_path("/proc/driver/nvidia/gpus"),
        )

    return setup


class TestNumaBindingDiscovery:
    """按当前GPU身份和rank权限发现目标资源，异常信息只导致跳过绑定。"""

    def test_binds_selected_gpu_with_rank_permissions(self, numa_environment):
        # 验证非零PCI域、高位CPU、GPU枚举顺序及rank权限共同作用下仍准确绑定。
        gpu_uuid = "GPU-21d706f7-e141-bbe7-18f9-00499abcb68a"
        context = numa_environment(
            cpus_by_node={0: {0, 1}, 1: set(range(48, 96)) | set(range(144, 192))},
            gpus={"GPU-other": ("0000:ab:00.0", 0), gpu_uuid: ("0002:cd:00.1", 1)},
            selected_uuid=gpu_uuid,
            allowed_cpus={0, 1, 144, 145},
            device_index=1,
        )
        affinity.try_bind_numa(9)
        context.device_query.assert_called_once_with(1)
        context.set_affinity.assert_called_once_with(0, {144, 145})
        context.membind.assert_called_once_with(1)

    @pytest.mark.parametrize("reason", ["disabled", "noncuda"])
    def test_skips_disabled_or_noncuda(self, numa_environment, monkeypatch, reason):
        # 验证显式禁用或没有CUDA时，跳过硬件拓扑查询与绑定。
        context = numa_environment()
        if reason == "disabled":
            monkeypatch.setenv("XTUNER_DISABLE_NUMA", "1")
        else:
            monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
        affinity.try_bind_numa(0)
        context.cpu_query.assert_not_called()
        context.device_query.assert_not_called()
        context.set_affinity.assert_not_called()
        context.membind.assert_not_called()

    @pytest.mark.parametrize("resource", ["cpu", "memory", "unknown-node"])
    def test_skips_inaccessible_gpu_local_resources(self, numa_environment, resource):
        # 验证CPU交集为空、内存节点无权限或PCI节点未知时不改绑其他节点。
        context = numa_environment(allowed_cpus={2, 3} if resource == "cpu" else {0, 1})
        if resource == "memory":
            context.memory_query.return_value = [1]
        elif resource == "unknown-node":
            context.numa_nodes["GPU-A"].write_text("-1\n")
        affinity.try_bind_numa(0)
        context.set_affinity.assert_not_called()
        context.membind.assert_not_called()

    @pytest.mark.parametrize("failure", ["missing-procfs", "unreadable-file", "unknown-device"])
    def test_skips_unavailable_driver_information(self, numa_environment, failure):
        # 验证驱动目录缺失、信息无法读取或当前GPU不在拓扑中时安全跳过。
        context = numa_environment()
        if failure == "missing-procfs":
            shutil.rmtree(context.proc_dir)
        elif failure == "unreadable-file":
            context.information["GPU-A"].unlink()
            context.information["GPU-A"].mkdir()  # read_text将抛出真实的OSError子类。
        else:
            context.device_query.return_value.uuid = "missing"
        affinity.try_bind_numa(0)
        context.set_affinity.assert_not_called()
        context.membind.assert_not_called()

    @pytest.mark.parametrize("information", ["", "Model: Other GPU\n", "GPU UUID:  \t\n"])
    def test_skips_incomplete_other_gpu_information(self, numa_environment, information):
        # 验证当前GPU有效时，另一GPU缺失或空UUID也不能被漏计后继续绑定。
        context = numa_environment(gpus={"GPU-A": ("0000:ab:00.0", 0), "GPU-B": ("0000:cd:00.0", 0)})
        context.information["GPU-B"].write_text(information)
        affinity.try_bind_numa(0)
        context.set_affinity.assert_not_called()
        context.membind.assert_not_called()


class TestNumaCapacity:
    """按每个GPU节点与当前rank权限交集内的CPU数量判断容量。"""

    @pytest.mark.parametrize("device_index", [0, 4], ids=["node0-rank", "node1-rank"])
    @pytest.mark.parametrize(("node0_count", "node1_count"), [(3, 44), (2, 64), (1, 44), (0, 44), (4, 44)])
    def test_checks_available_cpu_capacity_on_every_gpu_node(
        self, numa_environment, device_index, node0_count, node1_count
    ):
        # 验证硬件CPU充足但权限内CPU不足时，两节点rank均跳过；4/44的相等边界可绑定。
        allowed_cpus = set(range(node0_count)) | set(range(1000, 1000 + node1_count))
        gpu_nodes = {f"GPU-{index}": (f"0000:{0x50 + index:02x}:00.0", index // 4) for index in range(8)}
        context = numa_environment(
            cpus_by_node={0: set(range(96)), 1: set(range(1000, 1096))},
            gpus=gpu_nodes,
            selected_uuid=f"GPU-{device_index}",
            allowed_cpus=allowed_cpus,
            device_index=device_index,
        )
        affinity.try_bind_numa(device_index)
        if node0_count < 4:
            context.set_affinity.assert_not_called()
            context.membind.assert_not_called()
        else:
            node = device_index // 4
            expected_cpus = set(range(4)) if node == 0 else set(range(1000, 1044))
            context.set_affinity.assert_called_once_with(0, expected_cpus)
            context.membind.assert_called_once_with(node)

    def test_allows_equal_cpu_gpu_counts_and_uneven_layout(self, numa_environment):
        # 验证CPU/GPU数相等和不均匀布局可绑定，且没有GPU的空节点不参与门禁。
        context = numa_environment(
            cpus_by_node={0: {0, 1}, 1: {100}, 2: set()},
            gpus={"GPU-A": ("0000:ab:00.0", 0), "GPU-B": ("0000:cd:00.0", 0), "GPU-C": ("0000:ef:00.0", 1)},
            allowed_cpus={0, 1, 100},
        )
        affinity.try_bind_numa(0)
        context.set_affinity.assert_called_once_with(0, {0, 1})
        context.membind.assert_called_once_with(0)


class TestNumaBindingApplication:
    """CPU与内存绑定的执行顺序和失败回滚。"""

    def test_memory_failure_restores_initial_cpu_affinity(self, numa_environment):
        # 验证CPU已绑定而内存绑定失败时，恢复修改前完整的CPU集合。
        context = numa_environment(cpus_by_node={0: {0, 1}, 1: {2, 3}})
        context.membind.side_effect = RuntimeError("membind failed")
        affinity.try_bind_numa(0)
        assert context.set_affinity.call_args_list == [call(0, {0, 1}), call(0, {0, 1, 2, 3})]
        context.membind.assert_called_once_with(0)

    def test_cpu_failure_does_not_bind_memory(self, numa_environment):
        # 验证CPU绑定失败时，不继续修改内存策略或错误地尝试CPU回滚。
        context = numa_environment()
        context.set_affinity.side_effect = OSError("affinity failed")
        affinity.try_bind_numa(0)
        context.set_affinity.assert_called_once_with(0, {0, 1})
        context.membind.assert_not_called()
