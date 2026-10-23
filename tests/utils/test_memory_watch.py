import json
from pathlib import Path

import pytest

from xtuner.v1.utils.memory_watch import (
    _cgroup_dir,
    _MemoryWatch,
    _read_cgroup,
    _read_node_meminfo,
    _read_numa_maps,
    _read_smaps_rollup,
    _read_status,
    monitor_rank_memory,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled_monitor_does_not_create_output(tmp_path, monkeypatch):
    monkeypatch.setenv("XTUNER_MEMORY_WATCH_ENABLED", "0")
    with monitor_rank_memory(tmp_path):
        assert not (tmp_path / "memory").exists()
    assert not (tmp_path / "memory").exists()


@pytest.mark.parametrize("local_rank", ["0", "1"])
def test_enabled_monitor_writes_start_and_stop(tmp_path, monkeypatch, local_rank):
    monkeypatch.setenv("XTUNER_MEMORY_WATCH_ENABLED", "1")
    monkeypatch.setenv("XTUNER_MEMORY_WATCH_INTERVAL", "10")
    monkeypatch.setenv("RANK", local_rank)
    monkeypatch.setenv("LOCAL_RANK", local_rank)

    with monitor_rank_memory(tmp_path):
        pass

    records = _records(tmp_path / "memory" / f"rank_{local_rank}.jsonl")
    assert [record["event"] for record in records] == ["start", "stop"]
    assert all(record["rank"] == local_rank and "VmRSS" in record["process"] for record in records)
    if local_rank == "0":
        assert all("cgroup" in record and "host_nodes" in record for record in records)
    else:
        assert all("cgroup" not in record and "host_nodes" not in record for record in records)


@pytest.mark.parametrize("interval", ["invalid", "0", "-1", "nan", "inf"])
def test_invalid_interval_disables_monitor_without_interrupting_training(tmp_path, monkeypatch, capsys, interval):
    monkeypatch.setenv("XTUNER_MEMORY_WATCH_ENABLED", "1")
    monkeypatch.setenv("XTUNER_MEMORY_WATCH_INTERVAL", interval)
    entered = False

    with monitor_rank_memory(tmp_path):
        entered = True

    assert entered
    assert not (tmp_path / "memory").exists()
    assert "[XTuner memory watch] disabled" in capsys.readouterr().err


def test_output_setup_failure_does_not_interrupt_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XTUNER_MEMORY_WATCH_ENABLED", "1")
    (tmp_path / "memory").write_text("not a directory")
    entered = False

    with monitor_rank_memory(tmp_path):
        entered = True

    assert entered
    assert (tmp_path / "memory").is_file()
    assert "[XTuner memory watch] disabled" in capsys.readouterr().err


def test_cgroup_discovery_failure_does_not_interrupt_training(tmp_path, monkeypatch, capsys):
    def fail_discovery(*_args):
        raise PermissionError("cgroup unavailable")

    monkeypatch.setenv("XTUNER_MEMORY_WATCH_ENABLED", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr("xtuner.v1.utils.memory_watch._cgroup_dir", fail_discovery)

    with monitor_rank_memory(tmp_path):
        pass

    assert not (tmp_path / "memory").exists()
    assert "cgroup unavailable" in capsys.readouterr().err


def test_training_exception_is_preserved_and_monitor_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("XTUNER_MEMORY_WATCH_ENABLED", "1")
    monkeypatch.setenv("XTUNER_MEMORY_WATCH_INTERVAL", "10")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "1")

    with pytest.raises(ValueError, match="training failed"):
        with monitor_rank_memory(tmp_path):
            raise ValueError("training failed")

    assert [row["event"] for row in _records(tmp_path / "memory/rank_0.jsonl")] == ["start", "stop"]


def test_long_interval_still_joins_sampling_thread(tmp_path):
    watcher = _MemoryWatch(tmp_path / "rank.jsonl", include_cgroup=False, interval_s=10)

    with watcher:
        pass

    assert not watcher._thread.is_alive()
    assert [record["event"] for record in _records(tmp_path / "rank.jsonl")] == ["start", "stop"]


@pytest.mark.parametrize("version", [1, 2])
def test_synthetic_sources_cover_record_and_cgroup_versions(tmp_path, version):
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

    assert _read_status(proc_self / "status")["VmRSS"] == 12 * 1024
    assert _read_smaps_rollup(proc_self / "smaps_rollup")["Pss"] == 10 * 1024
    assert _read_numa_maps(proc_self / "numa_maps")["resident_pages_by_node"] == {"N0": 1, "N1": 3}
    assert _read_node_meminfo(node_root)["node1"]["MemFree"] == 30 * 1024
    assert _cgroup_dir(cgroup_root, proc_self / "cgroup") == cgroup
    parsed_cgroup = _read_cgroup(cgroup, root=cgroup_root, membership=proc_self / "cgroup")
    assert parsed_cgroup["version"] == version
    assert parsed_cgroup["memory.numa_stat_unit"] == ("bytes" if version == 2 else "pages")

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
        _write(events, "oom_kill 3\n")

    start, stop = _records(output)
    assert start["pid_starttime_ticks"] == 456
    assert start["cgroup_membership"] == membership.splitlines()
    assert start["host_node_cpus"] == {"node1": "2-3"}
    assert start["process"]["VmRSS"] == 12 * 1024
    assert start["process"]["Mems_allowed_list"] == "1"
    assert start["smaps_rollup"]["Pss"] == 10 * 1024
    assert start["numa_maps"]["resident_pages_by_node"] == {"N0": 1, "N1": 3}
    assert start["numa_maps"]["resident_bytes_by_node"]["N1"] == 2 * 4096 + 2 * 1024 * 1024
    assert start["numa_maps"]["policy_mappings"] == {"bind:1": 1, "default": 1}
    assert start["host_nodes"]["node1"]["MemFree"] == 30 * 1024
    assert start["cgroup"]["version"] == version
    assert start["cgroup"]["memory.stat"] == {"anon": 200, "file": 100}
    assert start["cgroup"]["memory.current" if version == 2 else "memory.usage_in_bytes"] == "4096"
    assert start["cgroup"]["memory.numa_stat_unit"] == ("bytes" if version == 2 else "pages")
    assert start["cgroup"]["memory.numa_stat"]["anon" if version == 2 else "total"]["N1"] == (
        8192 if version == 2 else 3
    )
    assert start["cgroup"]["cpuset.cpus.effective" if version == 2 else "cpuset.cpus"] == "2-3"
    assert stop["oom_kill_delta"] == 1
