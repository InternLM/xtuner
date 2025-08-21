import argparse
import ast
import json
import logging
import re
from statistics import mean


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def extract_var_value(file, var):
    with open(file, encoding="utf-8") as f:
        source_code = f.read()
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var:
                    try:
                        value = ast.literal_eval(node.value)
                        return value
                    except (ValueError, SyntaxError):
                        raise ValueError(f"Failed to parse {var} value.")
    raise NameError(f"Couldn't find {var}.")


def extract_data_from_log(logfile):
    pattern_str = r"\[XTuner\].*Step.*lr:\s(\d+.\d*)\s.*text_tokens:\s(\d+.\d*)\s.*reduced_llm_loss:\s(\d+.\d*)\s.*max_memory:\s(\d+.\d*)\s*GB\s.*grad_norm:\s(\d+.\d*)\s.*(?<!e2e_)tgs:\s(\d+.\d*)"
    compiled_pattern = re.compile(pattern_str)

    cur_lr = []
    cur_reduced_llm = []
    cur_grad_norm = []
    cur_max_memory = []
    cur_text_tokens = []
    cur_tgs = []

    with open(logfile) as f:
        for data in f:
            if match := compiled_pattern.search(data):
                cur_lr.append(float(match.group(1)))
                cur_text_tokens.append(float(match.group(2)))
                cur_reduced_llm.append(float(match.group(3)))
                cur_max_memory.append(float(match.group(4)))
                cur_grad_norm.append(float(match.group(5)))
                cur_tgs.append(float(match.group(6)))
    return (
        cur_lr,
        cur_text_tokens,
        cur_reduced_llm,
        cur_max_memory,
        cur_grad_norm,
        cur_tgs,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", help="Current rank0 log of training task")
    parser.add_argument("base_py", help="Python file with baseline")
    parser.add_argument("--check_metric", default="all", help="Check metric,eg,{'loss': 0.01,'memory': 0.1}")
    return parser.parse_args()


def main():
    args = parse_args()
    all_metric = ["lr", "text_tokens", "reduced_llm_loss", "max_memory", "grad_norm", "tgs"]
    fail_metric = {}
    if args.check_metric == "all":
        check_metric = {
            "lr": 0,
            "text_tokens": 0,
            "reduced_llm_loss": 0.01,
            "max_memory": 0.2,
            "grad_norm": 0.02,
            "tgs": 50,
        }
    else:
        check_metric = json.loads(args.check_metric)
    (
        cur_lr,
        cur_text_tokens,
        cur_reduced_llm_loss,
        cur_max_memory,
        cur_grad_norm,
        cur_tgs,
    ) = extract_data_from_log(args.log_path)
    for metric, threshold in check_metric.items():
        if metric not in all_metric:
            logger.warning(f"don't support {metric}!")
            continue
        metric_value = extract_var_value(args.base_py, metric)
        cur_metric = locals().get(f"cur_{metric}")
        assert len(metric_value) == len(cur_metric), (
            f"Current steps is {len(cur_metric)} which not equals baseline steps {len(metric_value)}. Check failed!"
        )
        logger.info(f"{metric} baseline is {metric_value}")
        logger.info(f"Current {metric} is {cur_metric}")
        max_error = 0.0
        max_error_idx = 0
        check_flag = True
        if metric == "tgs":
            if len(metric_value) > 10:
                max_error = abs(mean(metric_value[10:-1]) - mean(cur_metric[10:-1]))
                if max_error > 50:
                    fail_metric[metric] = (
                        f"{metric} relative error bigger than {threshold} after 10 step, baseline: {mean(metric_value[10:-1]):.6f}, now: {mean(cur_metric[10:-1]):.6f}, relative error: {max_error}"
                    )
                    check_flag = False
                else:
                    check_flag = True
            else:
                logger.warning("It's meaningless to compare tgs because of the small steps.")
                check_flag = False
        else:
            for idx, (old, cur) in enumerate(zip(metric_value, cur_metric)):
                relative_error = abs(old - cur) / abs(old)
                # update max_error
                if relative_error > max_error:
                    max_error = relative_error
                    max_error_idx = idx
                if relative_error > threshold:
                    fail_metric[metric] = (
                        f"{metric} relative error bigger than {threshold:.2%} in {idx} steps, baseline: {old:.6f}, now: {cur:.6f}, relative error: {relative_error:.2%}"
                    )
                    check_flag = False
                    break
        if check_flag:
            logger.info(f"✓ {metric} check pass，the most relative error is {max_error:.2%} in {max_error_idx} step.")
    assert fail_metric is None, f"Some metric check failed,{fail_metric}"


if __name__ == "__main__":
    main()
