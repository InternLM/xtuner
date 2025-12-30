import json
import logging
from statistics import mean
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def extract_value(file, metrics):
    metric_all = {metric: [] for metric in metrics}
    total_step = 0
    with open(file) as f:
        for line in f:
            line = json.loads(line)
            for metric in metrics:
                metric_all[metric].append(line[metric])
            total_step += 1

    return total_step, metric_all

def plot_all(case_name, check_metric, base_metrics, cur_metrics, output_root: Path):
    metric_list = list(check_metric.keys())
    n_plots = len(metric_list)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n_plots:
            x_base = np.arange(len(base_metrics[metric_list[i]]))
            x_current = np.arange(len(cur_metrics[metric_list[i]]))
            ax.plot(
                x_base,
                base_metrics[metric_list[i]],
                "r--",
                label="Base",
                marker="x",
                markersize=4,
            )
            ax.plot(
                x_current,
                cur_metrics[metric_list[i]],
                "b-",
                label="Current",
                marker="o",
                markersize=4,
            )
            ax.set_title(f"{metric_list[i].replace('/', '_')}_comparison")
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)
        else:
            ax.axis("off")
    fig.suptitle(f"{case_name}_metrics_comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_root / f"{case_name}_comparison.png")
    plt.close()


def check_result(base_path, cur_path, check_metric):
    fail_metric = {}
    check_metric = check_metric
    metric_list = list(check_metric.keys())
    base_steps, base_metrics = extract_value(base_path, metric_list)
    cur_steps, cur_metrics = extract_value(cur_path, metric_list)
    assert cur_steps == base_steps, (
        f"current steps is not equal to base steps, current steps: {cur_steps}, base steps: {base_steps}"
    )

    output_path = Path(f"../{os.environ['GITHUB_RUN_ID']}")
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"debug tmp *****{{output_path.exists()},{output_path.absolute()}")
    plot_all(case_name, check_metric, base_metrics, cur_metrics, output_path)

    for metric, threshold in check_metric.items():
        max_error = 0.0
        max_error_idx = 0
        check_flag = True
        if metric == "runtime_info/tgs":
            if cur_steps > 10:
                max_error = abs(mean(base_metrics[metric][10:-1]) - mean(cur_metrics[metric][10:-1])) / (
                    mean(base_metrics[metric][10:-1])
                )
                if max_error > threshold:
                    mean_base_metrics = f"{mean(base_metrics[metric][10:-1]):.6f}"
                    mean_cur_metrics = f"{mean(cur_metrics[metric][10:-1]):.6f}"
                    fail_metric[metric] = (
                        f"{metric} relative error bigger than {threshold} after 10 step, baseline: {mean_base_metrics}, now: {mean_cur_metrics}, relative error: {max_error}"
                    )
                    check_flag = False
                else:
                    check_flag = True
            else:
                logger.warning("It's meaningless to compare tgs because of the small steps.")
                check_flag = False
        else:
            for idx, (old, cur) in enumerate(zip(base_metrics[metric], cur_metrics[metric])):
                relative_error = round(abs(old - cur) / abs(old), 2)
                # update max_error
                if relative_error > max_error:
                    max_error = relative_error
                    max_error_idx = idx
                if relative_error > threshold:
                    baseline_old = f"{old:.6f}"
                    baseline_cur = f"{cur:.6f}"

                    fail_metric[metric] = (
                        f"{metric} relative error bigger than {threshold} in {idx} steps, baseline: {baseline_old}, now: {baseline_cur}, relative error: {relative_error}"
                    )
                    check_flag = False
                    break
        if check_flag:
            logger.info(f"✓ {metric} check pass，the most relative error is {max_error:.2%} in {max_error_idx} step.")
    result = not fail_metric
    return result, f"Some metric check failed,{fail_metric}"

if __name__ == "__main__":
    print(check_result("qwen3-sft", "./base/tracker.jsonl", "./current/tracker.jsonl",{"grad_norm":0.000001,"loss/reduced_llm_loss":0.000001,"lr":0,"memory/max_memory_GB":0.2,"runtime_info/tgs":0.05,"runtime_info/text_tokens":0}))
