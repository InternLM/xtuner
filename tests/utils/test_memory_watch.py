"""内存监控行为测试。

TestMonitorSwitchAndSetup
    test_disabled_monitor_does_not_create_output
    test_none_work_dir_falls_back_to_cwd
    test_invalid_interval_disables_monitor_without_interrupting_training
    test_output_setup_failure_does_not_interrupt_training
TestMonitorLifecycle
    test_enabled_monitor_writes_start_and_stop
    test_training_exception_is_preserved_and_monitor_stops
    test_long_interval_still_stops_promptly
    test_shutdown_timeout_does_not_depend_on_sampling_interval
TestMemoryOutput
    test_short_write_is_retried
    test_synthetic_sources_cover_record_and_cgroup_versions
"""

import json
import os
import threading
import time
from pathlib import Path

import pytest

from xtuner.v1.utils.memory_watch import MemoryEvent, _MemoryWatch, monitor_rank_memory


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _enable_monitor(monkeypatch, *, interval="10", rank="0", local_rank="1"):
    monkeypatch.setenv("XTUNER_MEMORY_WATCH_ENABLED", "1")
    monkeypatch.setenv("XTUNER_MEMORY_WATCH_INTERVAL", interval)
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("LOCAL_RANK", local_rank)


def _prepare_synthetic_sources(tmp_path, version):
    proc_self = tmp_path / "proc/self"
    cgroup_root = tmp_path / "cgroup"
    node_root = tmp_path / "node"
    membership = "0::/work/job\n" if version == 2 else "5:memory:/work/job\n6:cpuset:/work/job\n"
    _write(proc_self / "cgroup", membership)
    _write(proc_self / "stat", "123 (name with space) " + " ".join(["S", *(["0"] * 18), "456"]))
    _write(
        proc_self / "status",
        "VmRSS:\t12 kB\nVmHWM:\t20 kB\nRssAnon:\t8 kB\nCpus_allowed_list:\t2-3\nMems_allowed_list:\t1\n",
    )
    _write(
        proc_self / "numa_maps",
        "00400000 bind:1 anon=3 N0=1 N1=2 kernelpagesize_kB=4\n00800000 default N1=1 kernelpagesize_kB=2048\n",
    )
    _write(proc_self / "smaps_rollup", "Rss: 12 kB\nPss: 10 kB\n")
    _write(proc_self / "mountinfo", "36 25 0:32 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n")
    _write(node_root / "node1/cpulist", "2-3\n")
    _write(node_root / "node1/meminfo", "Node 1 MemTotal: 100 kB\nNode 1 MemFree: 30 kB\n")

    if version == 2:
        cgroup = cgroup_root / "work/job"
        events = cgroup / "memory.events"
        _write(events, "oom_kill 2\nmax 1\n")
        _write(cgroup / "memory.current", "4096\n")
        _write(cgroup / "memory.max", "8192\n")
        _write(cgroup / "memory.numa_stat", "anon N0=4096 N1=8192\n")
        _write(cgroup / "cpuset.cpus.effective", "2-3\n")
    else:
        cgroup = cgroup_root / "memory/work/job"
        events = cgroup / "memory.oom_control"
        _write(cgroup / "memory.usage_in_bytes", "4096\n")
        _write(cgroup / "memory.limit_in_bytes", "8192\n")
        _write(cgroup / "memory.numa_stat", "total=4 N0=1 N1=3\n")
        _write(events, "oom_kill 2\nunder_oom 0\n")
        _write(cgroup_root / "cpuset/work/job/cpuset.cpus", "2-3\n")
    _write(cgroup / "memory.stat", "anon 200\nfile 100\n")
    return proc_self, cgroup_root, node_root, membership, events


class TestMonitorSwitchAndSetup:
    """监控开关、参数归一化与初始化降级。"""

    def test_disabled_monitor_does_not_create_output(self, tmp_path, monkeypatch):
        # 开关关闭时，监控不得创建目录或文件。
        monkeypatch.setenv("XTUNER_MEMORY_WATCH_ENABLED", "0")
        with monitor_rank_memory(tmp_path):
            assert not (tmp_path / "memory").exists()
        assert not (tmp_path / "memory").exists()

    def test_none_work_dir_falls_back_to_cwd(self, tmp_path, monkeypatch):
        # 未配置 work_dir 时，监控应写入 cwd/work_dir，且不能阻断训练。
        monkeypatch.chdir(tmp_path)
        _enable_monitor(monkeypatch)

        with monitor_rank_memory(None):
            pass

        records = _records(tmp_path / "work_dir/memory/rank_0.jsonl")
        assert [record["event"] for record in records] == ["start", "stop"]

    @pytest.mark.parametrize("interval", ["invalid", "0", "nan"])
    def test_invalid_interval_disables_monitor_without_interrupting_training(
        self, tmp_path, monkeypatch, capsys, interval
    ):
        # 非法采样间隔应降级为不监控，训练体仍正常执行。
        _enable_monitor(monkeypatch, interval=interval)
        entered = False

        with monitor_rank_memory(tmp_path):
            entered = True

        assert entered
        assert not (tmp_path / "memory").exists()
        assert "[XTuner memory watch] disabled" in capsys.readouterr().err

    def test_output_setup_failure_does_not_interrupt_training(self, tmp_path, monkeypatch, capsys):
        # 输出目录初始化失败时，监控应告警并放行训练。
        _enable_monitor(monkeypatch)
        (tmp_path / "memory").write_text("not a directory")
        entered = False

        with monitor_rank_memory(tmp_path):
            entered = True

        assert entered
        assert (tmp_path / "memory").is_file()
        assert "[XTuner memory watch] disabled" in capsys.readouterr().err


class TestMonitorLifecycle:
    """监控启停、异常传播与线程退出。"""

    @pytest.mark.parametrize("local_rank", ["0", "1"])
    def test_enabled_monitor_writes_start_and_stop(self, tmp_path, monkeypatch, local_rank):
        # 启用监控后应写完整起止记录，且仅 local rank 0 写 host 文件。
        _enable_monitor(monkeypatch, rank=local_rank, local_rank=local_rank)

        with monitor_rank_memory(tmp_path):
            pass

        records = _records(tmp_path / "memory" / f"rank_{local_rank}.jsonl")
        assert [record["event"] for record in records] == ["start", "stop"]
        assert all(record["rank"] == local_rank and "VmRSS" in record["process"] for record in records)
        assert all(record["host"] == os.uname().nodename for record in records)
        assert all("cgroup" not in record and "host_nodes" not in record for record in records)

        host_output = tmp_path / "memory" / f"host_{os.uname().nodename}.jsonl"
        if local_rank == "0":
            host_records = _records(host_output)
            assert [record["event"] for record in host_records] == ["start", "stop"]
            assert all(record["host"] == os.uname().nodename for record in host_records)
            assert all("cgroup" in record and "host_nodes" in record for record in host_records)
        else:
            assert not host_output.exists()

    def test_training_exception_is_preserved_and_monitor_stops(self, tmp_path, monkeypatch):
        # 训练体异常应原样传播，同时监控仍写入 stop 记录。
        _enable_monitor(monkeypatch)

        with pytest.raises(ValueError, match="training failed"):
            with monitor_rank_memory(tmp_path):
                raise ValueError("training failed")

        assert [row["event"] for row in _records(tmp_path / "memory/rank_0.jsonl")] == ["start", "stop"]

    def test_long_interval_still_stops_promptly(self, tmp_path, monkeypatch):
        # 长采样间隔不应阻塞上下文退出，起止记录仍完整。
        _enable_monitor(monkeypatch, interval="10")

        with monitor_rank_memory(tmp_path):
            pass

        records = _records(tmp_path / "memory/rank_0.jsonl")
        assert [record["event"] for record in records] == ["start", "stop"]

    def test_shutdown_timeout_does_not_depend_on_sampling_interval(self, tmp_path, monkeypatch):
        # 即使用户配置长采样间隔，线程退出等待也应使用固定短超时。
        _enable_monitor(monkeypatch, interval="3600")
        join_timeouts = []
        real_join = threading.Thread.join

        def record_join(thread, timeout=None):
            join_timeouts.append(timeout)
            return real_join(thread, timeout)

        monkeypatch.setattr(threading.Thread, "join", record_join)
        with monitor_rank_memory(tmp_path):
            pass

        assert join_timeouts == [2.0]


class TestMemoryOutput:
    """JSONL 完整性、数据作用域与合成 proc/cgroup 输入。"""

    def test_short_write_is_retried(self, tmp_path, monkeypatch):
        # 操作系统短写时应继续写满，最终每行 JSON 均可解析。
        _enable_monitor(monkeypatch, interval="0.01")
        real_write = os.write
        short_write_done = False

        def short_write_once(fd, payload):
            nonlocal short_write_done
            data = bytes(payload)
            if not short_write_done:
                short_write_done = True
                midpoint = len(data) // 2
                return real_write(fd, data[:midpoint])
            return real_write(fd, data)

        monkeypatch.setattr("xtuner.v1.utils.memory_watch.os.write", short_write_once)
        with monitor_rank_memory(tmp_path):
            time.sleep(0.05)

        records = _records(tmp_path / "memory/rank_0.jsonl")
        events = [record["event"] for record in records]
        assert short_write_done
        assert events[0] == "start" and events[-1] == "stop"
        assert "periodic" in events

    @pytest.mark.parametrize("version", [1, 2])
    def test_synthetic_sources_cover_record_and_cgroup_versions(self, tmp_path, version):
        # 合成 cgroup v1/v2 输入应生成正确的 rank/host 记录与 OOM 增量。
        proc_self, cgroup_root, node_root, membership, events = _prepare_synthetic_sources(tmp_path, version)
        output = tmp_path / "rank.jsonl"
        watcher = _MemoryWatch(
            output,
            include_cgroup=True,
            interval_s=10,
            proc_self=proc_self,
            cgroup_root=cgroup_root,
            node_root=node_root,
        )

        with watcher:
            watcher._sample(MemoryEvent.PERIODIC)
            _write(events, "oom_kill 3\n")

        rank_records = _records(output)
        host_records = _records(tmp_path / f"host_{os.uname().nodename}.jsonl")
        assert [record["event"] for record in rank_records] == ["start", "periodic", "stop"]
        assert [record["event"] for record in host_records] == ["start", "periodic", "stop"]
        assert all(record["host"] == os.uname().nodename for record in rank_records + host_records)
        assert all("cgroup" not in record and "host_nodes" not in record for record in rank_records)

        start, periodic, _ = rank_records
        host_start, _, host_stop = host_records
        assert start["pid_starttime_ticks"] == 456
        assert start["cgroup_membership"] == membership.splitlines()
        assert start["process"]["VmRSS"] == 12 * 1024
        assert start["process"]["Mems_allowed_list"] == "1"
        assert start["smaps_rollup"]["Pss"] == 10 * 1024
        assert start["numa_maps"]["resident_pages_by_node"] == {"N0": 1, "N1": 3}
        assert start["numa_maps"]["resident_bytes_by_node"]["N1"] == 2 * 4096 + 2 * 1024 * 1024
        assert start["numa_maps"]["policy_mappings"] == {"bind:1": 1, "default": 1}
        assert "numa_maps" not in periodic
        assert "smaps_rollup" not in periodic

        assert host_start["host_node_cpus"] == {"node1": "2-3"}
        assert host_start["host_nodes"]["node1"]["MemFree"] == 30 * 1024
        assert host_start["cgroup"]["version"] == version
        assert host_start["cgroup"]["memory.stat"] == {"anon": 200, "file": 100}
        assert host_start["cgroup"]["memory.current" if version == 2 else "memory.usage_in_bytes"] == "4096"
        assert host_start["cgroup"]["memory.numa_stat_unit"] == ("bytes" if version == 2 else "pages")
        assert host_start["cgroup"]["memory.numa_stat"]["anon" if version == 2 else "total"]["N1"] == (
            8192 if version == 2 else 3
        )
        assert host_start["cgroup"]["cpuset.cpus.effective" if version == 2 else "cpuset.cpus"] == "2-3"
        assert host_stop["oom_kill_delta"] == 1
