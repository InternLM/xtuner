class Train:
    def get_cmd(config):
        print(config)
        config_path = config.get("parameters").get("config")
        if config.get("type") == "sft":
            model_config = config.get("parameters", {}).get("model", None)
            config_path = config.get("parameters", {}).get("config", None)
            dataset_path = config.get("parameters", {}).get("dataset", None)
            chat_template = config.get("parameters", {}).get("chat_template", None)
            work_dir = "/".join(
                [
                    config.get("base_path").get("base_output_path"),
                    config.get("run_id"),
                    config.get("case_name"),
                    config.get("type"),
                ]
            )

            command = "cd xtuner; pwd; torchrun xtuner/v1/train/cli/sft.py"
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
            command = 'echo "Not implemented"; exit 1'
            return command, config

    def validate(config):
        return True, "train validate executed"
