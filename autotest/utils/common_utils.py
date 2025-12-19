import os

import yaml


CONFIG_FILE = "autotest/config.yaml"


def dict_merge(default, override):
    if not isinstance(default, dict) or not isinstance(override, dict):
        if override is None:
            return default
        return override
    merge_result = dict()
    for key in set(default.keys() | override.keys()):
        merge_result[key] = dict_merge(default.get(key, None), override.get(key, None))
    return merge_result


def get_config():
    # Use device-specific config file if DEVICE environment variable is set
    device = os.environ.get("DEVICE", "")
    if device:
        device_config_path = f"autotest/config-{device}.yaml"
        if os.path.exists(device_config_path):
            config_path = device_config_path
        else:
            config_path = CONFIG_FILE
    else:
        config_path = CONFIG_FILE

    with open(config_path) as f:
        env_config = yaml.load(f.read(), Loader=yaml.SafeLoader)

    default_config = env_config["default_config"]
    case_config = env_config["case"]

    for case, steps in case_config.items():
        steps_config = []
        for step in steps:
            step_type = step["type"]
            if step["type"] in ["pre_train", "rl", "sft"]:
                step_type = "train"

            default_step_config = default_config.get(step_type, {})
            steps_config.append(dict_merge(default_step_config, step))
        case_config[case] = steps_config

    return env_config


def get_case_list(case_type: str = "all"):
    config = get_config()
    case_list = config["case"]

    if case_type == "all":
        return case_list.keys()
    else:
        filtered_cases = []
        for case in case_list:
            filter_type_set = {x.get("type") for x in case_list[case] if x.get("type") not in ["eval", "infer"]}
            if case_type in filter_type_set and len(filter_type_set) == 1:
                filtered_cases.append(case)
        return filtered_cases
