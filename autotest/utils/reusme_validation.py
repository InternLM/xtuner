import json
import math
import os
import sys


def get_latest_subdir(work_dir):
    dirs = [
        d
        for d in os.listdir(work_dir)
        if os.path.isdir(os.path.join(work_dir, d)) and len(d) == 14 and d.isdigit()
    ]
    if not dirs:
        return None
    latest = max(dirs, key=lambda d: os.path.getmtime(os.path.join(work_dir, d)))
    return os.path.join(work_dir, latest)


def extract_value(file, center_step, metrics):
    window_steps = list(range(center_step + 1, center_step + 4))
    want = frozenset(window_steps)
    by_step = {s: {m: [] for m in metrics} for s in window_steps}
    total_lines = 0
    with open(file, encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            obj = json.loads(line)
            s = obj.get("step")
            if s not in want:
                continue
            row = by_step[s]
            for m in metrics:
                row[m].append(obj[m] if m in obj else None)
    return total_lines, window_steps, by_step


def verify_window(path, center_step, metrics):
    _, window_steps, by_step = extract_value(path, center_step, metrics)
    missing_steps = {}
    missing_keys = []
    not_equal = {}

    for m in metrics:
        miss = [s for s in window_steps if not by_step[s][m]]
        if miss:
            missing_steps[m] = miss
            continue

        bad_step = None
        for s in window_steps:
            vals = by_step[s][m]
            if any(v is None for v in vals):
                bad_step = s
                break
        if bad_step is not None:
            missing_keys.append((m, bad_step))
            continue

        for s in window_steps:
            vals = by_step[s][m]
            if len(vals) > 1:
                first = vals[0]
                if any(not math.isclose(v, first, rel_tol=1e-6, abs_tol=0.0) for v in vals[1:]):
                    not_equal.setdefault(m, []).append((s, list(vals)))

    check_result = not (missing_steps or missing_keys or not_equal)
    if not check_result:
        if missing_steps:
            print("Missing step data (no records for this step):", file=sys.stderr)
            for m, steps in missing_steps.items():
                print(f"  {m}: step {steps}", file=sys.stderr)
        if missing_keys:
            print("Missing key (metric absent in tracker line, value is None):", file=sys.stderr)
            for m, s in missing_keys:
                print(f"  {m}: step {s}", file=sys.stderr)
        if not_equal:
            print("Inconsistent metric values across duplicate records at the same step:", file=sys.stderr)
            for m, pairs in not_equal.items():
                parts = []
                for s, vals in pairs:
                    parts.append(f"step {s}: {vals}")
                print(f"  {m}: " + "; ".join(parts), file=sys.stderr)
    return check_result

if __name__ == "__main__":
    base_dir = f"{sys.argv[1]}/{os.environ['GITHUB_RUN_ID']}/{sys.argv[2]}/{sys.argv[3]}"
    real_dir = get_latest_subdir(base_dir)
    tracker = os.path.join(real_dir, "logs/exp_tracking/rank0/tracker.jsonl")
    center_step = int(sys.argv[4])
    metrics = sys.argv[5].split(',')
    assert verify_window(tracker, center_step, metrics)
