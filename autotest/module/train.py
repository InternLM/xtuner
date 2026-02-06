import os

from utils.check_metric import check_result, check_rl_result
from utils.run_cmd import run_cmd


class Train:
    def get_cmd(config):
        print(config)
        config_path = config.get("parameters").get("config")
        train_type = config.get("type")
        nproc_per_node = config.get("resource", {}).get("gpus_per_task", 8)
        if train_type in ["sft", "rl"]:
            model_config = config.get("parameters", {}).get("model", None)
            config_path = config.get("parameters", {}).get("config", None)
            dataset_path = config.get("parameters", {}).get("dataset", None)
            chat_template = config.get("parameters", {}).get("chat_template", None)
            current_dir = config.get("current_dir", "xtuner")
            work_dir = "/".join(
                [
                    config.get("base_path").get("base_output_path"),
                    config.get("run_id"),
                    config.get("case_name"),
                    train_type,
                ]
            )

            if train_type == "sft":
                command = (
                    f"cd {current_dir}; pwd; pip install -e .[all]; pip install more-itertools; export GITHUB_RUN_ID={config.get('run_id')}; "
                    + f"torchrun --nproc-per-node {nproc_per_node} --master_addr=${{MASTER_ADDR}} --master_port=${{MASTER_PORT}} --nnodes=${{WORLD_SIZE}} --node_rank=${{RANK}} "
                    + f"xtuner/v1/train/cli/{train_type}.py"
                )
                if config_path:
                    output_path = model_config = config.get("parameters", {}).get("output_path", ".")
                    if output_path == ".":
                        command += f" --config {config_path}; mkdir -p {work_dir}; mv {output_path}/.xtuner {work_dir}; mv {output_path}/202* {work_dir}"
                    else:
                        command += f" --config {config_path}"
                else:
                    if model_config:
                        command += f" --model-cfg {model_config}"
                    if chat_template:
                        command += f" --chat_template {chat_template}"
                    if dataset_path:
                        command += f" --dataset {dataset_path}"
                    command += f" --work_dir {work_dir}"

                config["work_dir"] = work_dir
                return command, config
            elif train_type == "rl":
                infer_type = config.get("parameters", {}).get("infer_backend", "lmdeploy")
                config["work_dir"] = work_dir
                command = (
                    f"cd {current_dir}; pwd; pip install -e .[all]; export GITHUB_RUN_ID={config.get('run_id')}; export WORK_DIR={work_dir}; "
                    + f"bash -x examples/v1/scripts/run_rl.sh {config_path} {infer_type} ${{MODEL_PATH}} ${{DATA_PATH}} ${{EVAL_DATA_PATH}}"
                )
                return command, config
        else:
            return "", config

    def validate(config):
        work_dir = config.get("work_dir", None)
        base_path = os.path.join(
            config.get("base_path").get("base_baseline_path"), config.get("assert_info", {}).get("base_metric", None)
        )
        train_type = config.get("type")
        if train_type == 'sft':
            cur_path = os.path.join(get_latest_subdir(work_dir), "logs/exp_tracking/rank0/tracker.jsonl")
            check_metrics = config.get("assert_info", {}).get("check_metrics", {})
            return check_result(config["case_name"], base_path, cur_path, check_metrics)
        elif train_type == 'rl':
            cur_path = os.path.join(get_latest_subdir(work_dir), "exp_tracking/tracker.jsonl")
            check_metrics = config.get("assert_info", {})
            return check_rl_result(config["case_name"], base_path, cur_path, check_metrics)
        else:
            print("Unknown type: {train_type}")
            return False

    def pre_action(config=None):
        action_info = config.get("pre_action", None)
        if action_info:
            action_cmd = action_info.get("command", None)
            if action_cmd:
                run_cmd(action_cmd)

    def post_action(config=None):
        return True, config


def get_latest_subdir(work_dir):
    dirs = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d)) and len(d) == 14 and d.isdigit()]

    if not dirs:
        return None
    latest = max(dirs, key=lambda d: os.path.getmtime(os.path.join(work_dir, d)))
    return os.path.join(work_dir, latest)
