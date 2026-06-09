import json
import logging

import numpy as np
from utils.metric_report import publish_comparison_report


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
                if metric in line:
                    metric_all[metric].append(line[metric])
            total_step += 1

    return total_step, metric_all


def check_result(case_name, base_path, cur_path, check_metric):
    fail_metric = {}
    metric_list = list(check_metric.keys())
    base_steps, base_metrics = extract_value(base_path, metric_list)
    cur_steps, cur_metrics = extract_value(cur_path, metric_list)
    assert cur_steps == base_steps, (
        f"current steps is not equal to base steps, current steps: {cur_steps}, base steps: {base_steps}"
    )

    publish_comparison_report(case_name, check_metric, base_metrics, cur_metrics, base_path, cur_path)

    for metric, threshold in check_metric.items():
        max_error = 0.0
        max_error_idx = 0
        check_flag = True
        if metric == "runtime_info/tgs":
            if cur_steps > 10:
                relative_errors = abs(np.array(base_metrics[metric][10:-1]) - np.array(cur_metrics[metric][10:-1])) / (
                    np.array(base_metrics[metric][10:-1])
                )
                max_error = np.percentile(relative_errors, 80)
                if max_error > threshold:
                    fail_metric[metric] = (
                        f"{metric} relative error bigger than {threshold} after 10 step, baseline: {base_metrics[metric][10:-1]}, now: {cur_metrics[metric][10:-1]}, relative error: {relative_errors}"
                    )
                    check_flag = False
                else:
                    check_flag = True
            else:
                logger.warning("It's meaningless to compare tgs because of the small steps.")
                check_flag = False
        else:
            for idx, (old, cur) in enumerate(zip(base_metrics[metric], cur_metrics[metric])):
                if abs(old) < 1e-10:
                    relative_error = float("inf") if abs(cur) > 1e-10 else 0.0
                else:
                    relative_error = round(abs(old - cur) / abs(old), 2)
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
    if result:
        return result, "All metrics check passed."
    return result, f"Some metric check failed: {fail_metric}"


def check_rl_result(case_name, base_path, cur_path, assert_info):
    fail_metric = {}
    check_metrics_list = assert_info["check_metrics"]

    metric_list = [item["metric"] for item in check_metrics_list]

    base_steps, base_metrics = extract_value(base_path, metric_list)
    cur_steps, cur_metrics = extract_value(cur_path, metric_list)

    assert cur_steps == base_steps, (
        f"current steps is not equal to base steps, current steps: {cur_steps}, base steps: {base_steps}"
    )

    check_metric_dict = {item["metric"]: item["threshold"] for item in check_metrics_list}
    publish_comparison_report(case_name, check_metric_dict, base_metrics, cur_metrics, base_path, cur_path)

    for config in check_metrics_list:
        metric = config["metric"]
        threshold = config["threshold"]
        method = config["method"]
        operator = config["operator"]

        max_error = 0.0
        max_error_idx = 0
        check_flag = True

        for idx, (base_val, cur_val) in enumerate(zip(base_metrics[metric], cur_metrics[metric])):
            if method == "absolute":
                error = round(abs(cur_val - base_val), 5)
            elif method == "relative":
                if abs(base_val) < 1e-10:
                    error = float("inf") if abs(cur_val) > 1e-10 else 0.0
                else:
                    error = round(abs(cur_val - base_val) / abs(base_val), 5)
            else:
                raise ValueError(f"Unknown method: {method}")

            if error > max_error:
                max_error = error
                max_error_idx = idx

            if operator == "<":
                if not (error < threshold):
                    fail_metric[metric] = (
                        f"{metric} error {error:.6f} not less than threshold {threshold} "
                        f"(method: {method}, operator: {operator}) at step {idx}, "
                        f"baseline: {base_val:.6f}, current: {cur_val:.6f}"
                    )
                    check_flag = False
                    break
            elif operator == "<=":
                if not (error <= threshold):
                    fail_metric[metric] = (
                        f"{metric} error {error:.6f} not less than or equal to threshold {threshold} "
                        f"(method: {method}, operator: {operator}) at step {idx}, "
                        f"baseline: {base_val:.6f}, current: {cur_val:.6f}"
                    )
                    check_flag = False
                    break
            else:
                raise ValueError(f"Unknown operator: {operator}")

        if check_flag:
            logger.info(
                f"✓ {metric} check passed, max error is {max_error:.6f} at step {max_error_idx} "
                f"(method: {method}, operator: {operator})"
            )

    result = not bool(fail_metric)
    if result:
        return result, "All metrics check passed."
    return result, f"Some metric check failed: {fail_metric}"


if __name__ == "__main__":
    print(
        check_result(
            "qwen3-sft",
            "./base/tracker.jsonl",
            "./current/tracker.jsonl",
            {
                "grad_norm": 0.000001,
                "loss/reduced_llm_loss": 0.000001,
                "lr": 0,
                "memory/max_memory_GB": 0.2,
                "runtime_info/tgs": 0.05,
                "runtime_info/text_tokens": 0,
            },
        )
    )
