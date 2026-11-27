#!/usr/bin/env python3
"""CC8 tree monitor: 1s sampling of every process in the workload tree.

Per process: Pss/Uss/Rss/HWM (kB), full cmdline (never truncated), ppid/pgrp, starttime.
Tree totals use PSS (shared pages counted fractionally; never sum RSS across processes).
First sample touches READY_FLAG before the launcher starts the load. Threshold breach
writes ABORT_FLAG (protection abort — never labeled OOM). Adapted from CC3's
monitor_cc3.py (6 successful locked rounds) with USS + full-cmdline additions.
"""
import argparse, json, os, subprocess, time


def read_rollup(pid):
    d = {}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                parts = line.split()
                if parts and parts[0] in ("Pss:", "Private_Clean:", "Private_Dirty:", "Shared_Clean:", "Shared_Dirty:"):
                    d[parts[0].rstrip(":")] = int(parts[1])
        # USS = Private_Clean + Private_Dirty (smaps_rollup has no direct USS field)
        if "Private_Clean" in d or "Private_Dirty" in d:
            d["Uss"] = d.get("Private_Clean", 0) + d.get("Private_Dirty", 0)
    except OSError:
        return None
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(("VmRSS", "VmHWM", "VmSwap")):
                    k, v = line.split(":", 1)
                    d[k] = int(v.strip().split()[0])
    except OSError:
        return None
    return d


def read_start_ticks(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().rsplit(") ", 1)[1].split()
        return int(parts[19])
    except (OSError, IndexError, ValueError):
        return None


def full_cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def tree_pids(root_pid):
    procs = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                parts = f.read().rsplit(") ", 1)[1].split()
            procs[int(pid)] = (int(parts[1]), int(parts[2]))  # ppid, pgrp
        except (OSError, IndexError, ValueError):
            continue
    keep = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, pgrp) in procs.items():
            if pid not in keep and (ppid in keep or pgrp == root_pid):
                keep.add(pid)
                changed = True
    return keep


def host_stats():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(("MemAvailable", "MemTotal")):
                k, v = line.split(":", 1)
                out[k] = int(v.strip().split()[0])  # kB
    try:
        st = os.statvfs("/dev/shm")
        out["shm_avail_kb"] = st.f_bavail * st.f_frsize // 1024
    except OSError:
        pass
    for cur, lim in (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                     ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            c = int(open(cur).read().strip())
            l = open(lim).read().strip()
            out["cgroup_usage_kb"] = c // 1024
            out["cgroup_limit_kb"] = 0 if l in ("max",) or int(l) >= (1 << 62) else int(l) // 1024
            break
        except (OSError, ValueError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root-pid", required=True, type=int)
    ap.add_argument("--ready", required=True)
    ap.add_argument("--abort", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--pss-limit-gib", type=float, default=16.0)
    ap.add_argument("--host-avail-limit-gib", type=float, default=256.0)
    ap.add_argument("--shm-limit-gib", type=float, default=64.0)
    args = ap.parse_args()

    limits = {"tree_pss_gib": args.pss_limit_gib, "host_avail_gib": args.host_avail_limit_gib,
              "shm_gib": args.shm_limit_gib}
    with open(args.out, "a") as out:
        n = 0
        while True:
            rec = {"ts": time.time(), "n": n, "limits": limits}
            pids = tree_pids(args.root_pid) - {args.root_pid}
            procs = []
            total_pss = total_uss = 0
            for pid in sorted(pids):
                r = read_rollup(pid)
                if r is None:
                    continue  # exited between listing and read
                procs.append({"pid": pid, "pss_kb": r.get("Pss"),
                              "uss_kb": r.get("Uss"), "rss_kb": r.get("VmRSS"),
                              "hwm_kb": r.get("VmHWM"), "start_ticks": read_start_ticks(pid),
                              "cmdline": full_cmdline(pid)})
                total_pss += r.get("Pss") or 0
                total_uss += r.get("Uss") or 0
            rec["procs"] = procs
            rec["tree_pss_kb"] = total_pss
            rec["tree_uss_kb"] = total_uss
            rec["n_procs"] = len(procs)
            rec.update(host_stats())

            breach = None
            if total_pss > args.pss_limit_gib * (1 << 30) / 1024:
                breach = f"tree_pss {total_pss/1048576:.1f}GiB > {args.pss_limit_gib}GiB"
            elif rec.get("MemAvailable", 1 << 62) < args.host_avail_limit_gib * (1 << 20):
                breach = f"host_avail {rec['MemAvailable']/1048576:.1f}GiB < {args.host_avail_limit_gib}GiB"
            elif rec.get("shm_avail_kb", 1 << 62) < args.shm_limit_gib * (1 << 20):
                breach = f"shm_avail {rec['shm_avail_kb']/1048576:.1f}GiB < {args.shm_limit_gib}GiB"
            if breach:
                rec["abort_reason"] = breach
                out.write(json.dumps(rec) + "\n")
                out.flush()
                with open(args.abort, "w") as f:
                    f.write(breach)
                return
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if n == 0:
                with open(args.ready, "w") as f:
                    f.write(str(time.time()))
            n += 1
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
