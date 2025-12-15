import json
import logging
from statistics import mean


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


def check_result(base_path, cur_path, check_metric):
    fail_metric = {}
    check_metric = check_metric
    metric_list = list(check_metric.keys())
    base_steps, base_metrics = extract_value(base_path, metric_list)
    cur_steps, cur_metrics = extract_value(cur_path, metric_list)
    assert cur_steps == base_steps, (
        f"current steps is not equal to base steps, current steps: {cur_steps}, base steps: {base_steps}"
    )

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
