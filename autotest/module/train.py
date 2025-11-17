import os

from utils.check_metric import check_result


class Train:
    def get_cmd(config):
        print(config)
        config_path = config.get("parameters").get("config")
        train_type = config.get("type")
        if train_type in ["sft", "rl"]:
            model_config = config.get("parameters", {}).get("model", None)
            config_path = config.get("parameters", {}).get("config", None)
            dataset_path = config.get("parameters", {}).get("dataset", None)
            chat_template = config.get("parameters", {}).get("chat_template", None)
            work_dir = "/".join(
                [
                    config.get("base_path").get("base_output_path"),
                    config.get("run_id"),
                    config.get("case_name"),
                    train_type,
                ]
            )

            command = f"cd xtuner; pwd; torchrun xtuner/v1/train/cli/{train_type}.py"
            if config_path:
                # os.makedirs(work_dir, exist_ok=True)
                command += f" --config {config_path}; mkdir -p {work_dir}; cp -r 202* {work_dir}"
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
        else:
            return "", config

    def validate(config):
        work_dir = config.get("work_dir", None)
        base_path = config.get("assert_info", {}).get("base_metric", None)
        cur_path = os.path.join(get_latest_subdir(work_dir), "logs/exp_tracking/rank0/tracker.jsonl")
        check_metrics = config.get("assert_info", {}).get("check_metrics", {})
        return check_result(base_path, cur_path, check_metrics)


def get_latest_subdir(work_dir):
    dirs = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]

    if not dirs:
        return None
    latest = max(dirs, key=lambda d: os.path.getmtime(os.path.join(work_dir, d)))
    return os.path.join(work_dir, latest)
