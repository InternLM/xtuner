"""Turn the JSON written by ``run_decoupled_ep_fsdp_numerics.py`` into markdown tables."""

import argparse
import json
from pathlib import Path


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--ref", default=None, help="reference mode name (default: first mode)")
    args = parser.parse_args()
    data = json.loads(Path(args.json_path).read_text())
    results = {r["mode"]["name"]: r for r in data["results"]}
    names = list(results)
    ref = args.ref or names[0]
    steps = len(results[ref]["losses"])

    print("## Modes\n")
    print("| mode | ep | decouple | hsdp_sharding_size |")
    print("|---|---|---|---|")
    for n in names:
        m = results[n]["mode"]
        print(f"| {n} | {m['ep']} | {m['decouple']} | {m['hsdp']} |")

    print("\n## Loss curve (reduced_llm_loss)\n")
    print("| step | " + " | ".join(names) + " | " + " | ".join(f"rel({n} vs {ref})" for n in names if n != ref) + " |")
    print("|---|" + "---|" * (2 * len(names) - 1))
    for s in list(range(0, steps, 5)) + [steps - 1]:
        row = [f"{results[n]['losses'][s]:.6f}" for n in names]
        diffs = [f"{rel(results[n]['losses'][s], results[ref]['losses'][s]):.2e}" for n in names if n != ref]
        print(f"| {s} | " + " | ".join(row) + " | " + " | ".join(diffs) + " |")
    print("\n| pair | max rel loss diff over all steps | mean rel loss diff |")
    print("|---|---|---|")
    for n in names:
        if n == ref:
            continue
        d = [rel(a, b) for a, b in zip(results[n]["losses"], results[ref]["losses"])]
        print(f"| {n} vs {ref} | {max(d):.2e} | {sum(d) / len(d):.2e} |")

    print("\n## Total grad norm (pre-clip)\n")
    print("| step | " + " | ".join(names) + " | " + " | ".join(f"rel({n} vs {ref})" for n in names if n != ref) + " |")
    print("|---|" + "---|" * (2 * len(names) - 1))
    for s in list(range(0, steps, 5)) + [steps - 1]:
        row = [f"{results[n]['grad_norms'][s]:.6f}" for n in names]
        diffs = [f"{rel(results[n]['grad_norms'][s], results[ref]['grad_norms'][s]):.2e}" for n in names if n != ref]
        print(f"| {s} | " + " | ".join(row) + " | " + " | ".join(diffs) + " |")

    print("\n## Per-parameter grad norms (max / mean relative diff vs reference)\n")
    print("| step | pair | dense params | expert params | worst param |")
    print("|---|---|---|---|---|")
    for s in results[ref]["param_grad_norms"]:
        ref_norms = results[ref]["param_grad_norms"][s]
        for n in names:
            if n == ref:
                continue
            norms = results[n]["param_grad_norms"][s]
            dense, expert, worst = [], [], (0.0, "")
            for k, v in ref_norms.items():
                d = rel(norms[k], v)
                (expert if ".experts" in k else dense).append(d)
                if d > worst[0]:
                    worst = (d, k)
            print(
                f"| {s} | {n} vs {ref} | max {max(dense):.2e} / mean {sum(dense) / len(dense):.2e} "
                f"| max {max(expert):.2e} / mean {sum(expert) / len(expert):.2e} | {worst[1]} ({worst[0]:.2e}) |"
            )

    print("\n## Per-rank memory (MiB) and step time\n")
    keys = [
        "dense_param_mib",
        "expert_param_mib",
        "allocated_after_load_mib",
        "allocated_after_step0_mib",
        "peak_allocated_mib",
        "peak_reserved_mib",
    ]
    print("| mode | " + " | ".join(keys) + " | step time (ms) |")
    print("|---|" + "---|" * (len(keys) + 1))
    for n in names:
        mem = results[n]["memory"]
        print(f"| {n} | " + " | ".join(f"{mem[k]:.1f}" for k in keys) + f" | {results[n]['step_time_s'] * 1000:.1f} |")


if __name__ == "__main__":
    main()
