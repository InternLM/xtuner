import os
import re

import yaml


CONFIG_FILE = "autotest/config.yaml"

DEFAULT_CLUSTERX_PARTITION = "llmrazor_gpu"
DEFAULT_CLUSTERX_PROJECT_NAME = "ailab-llmrazor"


def dict_merge(default, override):
    if not isinstance(default, dict) or not isinstance(override, dict):
        if override is None:
            return default
        return override
    merge_result = dict()
    for key in set(default.keys() | override.keys()):
        merge_result[key] = dict_merge(default.get(key, None), override.get(key, None))
    return merge_result


def _resolve_clusterx_config(env_config: dict) -> dict:
    """Resolve clusterx partition/project_name from config and env overrides."""
    clusterx_cfg = dict(env_config.get("clusterx") or {})
    clusterx_cfg.setdefault("partition", DEFAULT_CLUSTERX_PARTITION)
    clusterx_cfg.setdefault("project_name", DEFAULT_CLUSTERX_PROJECT_NAME)

    partition_override = os.environ.get("CI_ETE_CLUSTERX_PARTITION", "").strip()
    project_override = os.environ.get("CI_ETE_CLUSTERX_PROJECT_NAME", "").strip()
    if partition_override:
        clusterx_cfg["partition"] = partition_override
    if project_override:
        clusterx_cfg["project_name"] = project_override
    return clusterx_cfg


def _merge_step_clusterx(step: dict, clusterx_cfg: dict) -> dict:
    """Merge global/per-step/per-resource clusterx submit options."""
    merged = dict(clusterx_cfg)
    merged.update(step.get("clusterx") or {})

    resource = step.get("resource") or {}
    if isinstance(resource.get("clusterx"), dict):
        merged.update(resource["clusterx"])
    for key in ("partition", "project_name"):
        if resource.get(key):
            merged[key] = resource[key]
    return merged


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def strip_xtuner_editable_install(pip_package: str) -> str:
    """Remove editable xtuner installs; keep other pip commands in the chain."""
    segments = [seg.strip() for seg in pip_package.split(";") if seg.strip()]
    kept = [seg for seg in segments if not re.match(r"^pip\s+install\s+-e\s+\.", seg, re.IGNORECASE)]
    return "; ".join(kept) if kept else "true"


def _resolve_skip_xtuner_install(env_config: dict, step: dict) -> bool:
    if _env_flag("CI_ETE_SKIP_XTUNER_INSTALL"):
        return True
    resource = step.get("resource") or {}
    if "skip_xtuner_install" in resource:
        return bool(resource["skip_xtuner_install"])
    if "skip_xtuner_install" in step:
        return bool(step["skip_xtuner_install"])
    return bool(env_config.get("skip_xtuner_install"))


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
    registry = os.environ.get("CI_NPU_IMAGE_REGISTRY") if device == "npu" else os.environ.get("CI_GPU_IMAGE_REGISTRY")
    train_image_override = os.environ.get("CI_ETE_TRAIN_IMAGE", "").strip()
    clusterx_cfg = _resolve_clusterx_config(env_config)
    case_config = env_config["case"]

    for case, steps in case_config.items():
        steps_config = []
        for step in steps:
            step_type = step["type"]
            if step["type"] in ["pre_train", "rl", "sft"]:
                step_type = "train"

            default_step_config = default_config.get(step_type, {})
            merged = dict_merge(default_step_config, step)
            r = merged.get("resource")
            if train_image_override and step_type == "train":
                r["image"] = train_image_override.lstrip("/")
            r["image"] = f"{registry}/{r['image']}"
            if step_type == "train" and _resolve_skip_xtuner_install(env_config, merged):
                pip_package = r.get("pip_package")
                if pip_package:
                    r["pip_package"] = strip_xtuner_editable_install(str(pip_package))
            merged["clusterx"] = _merge_step_clusterx(merged, clusterx_cfg)
            steps_config.append(merged)
        case_config[case] = steps_config

    env_config["clusterx"] = clusterx_cfg
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
