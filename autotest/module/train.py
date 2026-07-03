import json
import os
import shutil
from typing import Any

from utils.check_metric import check_result, check_rl_result
from utils.run_cmd import run_cmd


FIRST_RUN_TRACKER_SNAPSHOT = "first_run_tracker.jsonl"


class Train:
    def get_cmd(config):
        print(config)
        config_path = config.get("parameters").get("config")
        train_type = config.get("type")
        nproc_per_node = config.get("resource", {}).get("gpus_per_task", 8)
        pip_package = config.get("resource", {}).get("pip_package", "ls")
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
            config["work_dir"] = work_dir

            # this patch is for torch 2.9.1 Conv3d memory issue fix
            cudnn_patch = (
                "TORCH_VERSION=$(python -c 'import torch;print(torch.__version__.split(chr(43))[0])'); "
                "if [[ $TORCH_VERSION == 2.9.1 ]]; then pip install nvidia-cudnn-cu12==9.15.1.9; fi; "
            )

            if train_type == "sft":
                command = (
                    f"cd {current_dir}; pwd; {pip_package}; export GITHUB_RUN_ID={config.get('run_id')}; export WORK_DIR={work_dir}; "
                    + cudnn_patch
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

                return command, config
            elif train_type == "rl":
                infer_type = config.get("parameters", {}).get("infer_backend", "lmdeploy")
                accelerator = config.get("parameters", {}).get("accelerator", "GPU")
                command = (
                    f"cd {current_dir}; pwd; {pip_package}; export GITHUB_RUN_ID={config.get('run_id')}; export WORK_DIR={work_dir}; "
                    + cudnn_patch
                    + f"bash -x autotest/utils/ci_run_rl.sh {accelerator} {infer_type} {config_path} ${{MODEL_PATH}} ${{DATA_PATH}} ${{EVAL_DATA_PATH}}"
                )
                return command, config
        else:
            return "", config

    def validate(config):
        work_dir = config.get("work_dir", None)
        base_metric = config.get("assert_info", {}).get("base_metric", None)
        base_path = os.path.join(config.get("base_path").get("base_baseline_path"), base_metric)
        train_type = config.get("type")
        case_name = config["case_name"]
        phase = config.get("phase")
        context = config.get("context", {})

        run_id = config.get("run_id")

        cur_path = resolve_tracker_path(
            work_dir, train_type, phase, context=context, run_id=run_id, case_name=case_name
        )

        if train_type == "sft":
            check_metrics = config.get("assert_info", {}).get("check_metrics", {})
            result = check_result(case_name, base_path, cur_path, check_metrics, phase=phase)
        elif train_type == "rl":
            check_metrics = config.get("assert_info", {})
            result = check_rl_result(case_name, base_path, cur_path, check_metrics, phase=phase)
        else:
            print("Unknown type: {train_type}")
            return False

        snapshot_first_run_tracker(run_id, case_name, train_type, phase, cur_path, context=context)
        return result

    def pre_action(config=None):
        action_info = config.get("pre_action", None)
        if action_info:
            action_cmd = action_info.get("command", None)
            if action_cmd:
                run_cmd(action_cmd)

    def post_action(config=None):
        action_info = config.get("post_action", None)
        if action_info:
            action_cmd = action_info.get("command", None)
            if action_cmd:
                run_cmd(action_cmd)


def list_timestamp_subdirs(work_dir: str) -> list[str]:
    return sorted(
        name
        for name in os.listdir(work_dir)
        if os.path.isdir(os.path.join(work_dir, name)) and len(name) == 14 and name.isdigit()
    )


def _tracker_relpath(train_type: str) -> str:
    if train_type == "sft":
        return "logs/exp_tracking/rank0/tracker.jsonl"
    return "logs/exp_tracking/tracker.jsonl"


def _tracker_path(exp_dir: str | None, train_type: str) -> str:
    return os.path.join(exp_dir, _tracker_relpath(train_type))


def _snapshot_path(run_id: str | None, case_name: str, train_type: str) -> str:
    snapshot_dir = os.path.join(os.getcwd(), str(run_id or "0"), ".snapshots", case_name, train_type)
    return os.path.join(snapshot_dir, FIRST_RUN_TRACKER_SNAPSHOT)


def _write_first_run_segment(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    seen_steps: set[Any] = set()
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            step = json.loads(line).get("step")
            if step in seen_steps:
                break
            seen_steps.add(step)
            fout.write(line if line.endswith("\n") else f"{line}\n")


def _has_duplicate_steps(tracker_path: str) -> bool:
    steps: list[Any] = []
    with open(tracker_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                steps.append(json.loads(line).get("step"))
    return len(steps) != len(set(steps))


def resolve_tracker_path(
    work_dir: str,
    train_type: str,
    phase: str | None,
    context: dict[str, Any] | None = None,
    run_id: str | None = None,
    case_name: str | None = None,
) -> str:
    context = context or {}
    snapshot = context.get("first_run_tracker")
    if snapshot is None and case_name is not None:
        snapshot = _snapshot_path(run_id, case_name, train_type)

    if phase == "first":
        if snapshot and os.path.isfile(snapshot):
            return snapshot

        subdirs = list_timestamp_subdirs(work_dir)
        if len(subdirs) > 1:
            exp_dir = os.path.join(work_dir, subdirs[0])
        else:
            exp_dir = os.path.join(work_dir, subdirs[-1]) if subdirs else None
        live_tracker = _tracker_path(exp_dir, train_type)

        if snapshot and os.path.isfile(live_tracker) and _has_duplicate_steps(live_tracker):
            _write_first_run_segment(live_tracker, snapshot)
            if os.path.isfile(snapshot) and os.path.getsize(snapshot) > 0:
                return snapshot
        return live_tracker

    subdirs = list_timestamp_subdirs(work_dir)
    exp_dir = os.path.join(work_dir, subdirs[-1]) if subdirs else None
    return _tracker_path(exp_dir, train_type)


def snapshot_first_run_tracker(
    run_id: str | None,
    case_name: str,
    train_type: str,
    phase: str | None,
    cur_path: str,
    context: dict[str, Any] | None = None,
) -> None:
    if phase != "first" or not os.path.isfile(cur_path):
        return
    snapshot = _snapshot_path(run_id, case_name, train_type)
    os.makedirs(os.path.dirname(snapshot), exist_ok=True)
    if cur_path != snapshot:
        shutil.copy2(cur_path, snapshot)
    if context is not None:
        context["first_run_tracker"] = snapshot
