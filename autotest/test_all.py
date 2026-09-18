import os

import pytest
from cluster.clusterx import ClusterTaskExecutor
from module.get_module import TestHandler
from utils.common_utils import get_case_list


@pytest.fixture
def task_executor():
    return ClusterTaskExecutor()


handler = TestHandler()


@pytest.mark.parametrize("case", get_case_list("pre_train"))
@pytest.mark.pre_train
def test_pretrain(config, case, task_executor):
    run_all_cases(config, case, task_executor)


@pytest.mark.parametrize("case", get_case_list("rl"))
@pytest.mark.rl
def test_rl(config, case, task_executor):
    run_all_cases(config, case, task_executor)


@pytest.mark.parametrize("case", get_case_list("sft"))
@pytest.mark.sft
def test_sft(config, case, task_executor):
    run_all_cases(config, case, task_executor)


@pytest.mark.parametrize("case", get_case_list())
@pytest.mark.all
def test_all(config, case, task_executor):
    run_all_cases(config, case, task_executor)


def _is_resume_case(case_config) -> bool:
    return any(step.get("phase") == "resume" for step in case_config)


def _step_label(step_config) -> str:
    phase = step_config.get("phase")
    if phase:
        return f"{step_config['case_name']}[{phase}]"
    return f"{step_config['case_name']}[{step_config.get('type')}]"


def run_all_cases(config, case_name, task_executor) -> None:
    case_config = config["case"].get(case_name)
    base_path_config = config["base_path"]
    current_dir = os.getcwd()
    context = {}
    continue_after_first_validation_fail = _is_resume_case(case_config)
    step_failures: list[str] = []

    for step_config in case_config:
        step_config["case_name"] = case_name
        step_config["run_id"] = config.get("run_id")
        step_config["current_dir"] = current_dir
        step_config["base_path"] = base_path_config
        step_config["context"] = context

        failure = exec_step_test(
            step_config,
            task_executor,
            context,
            continue_after_first_validation_fail=continue_after_first_validation_fail,
        )
        if failure:
            step_failures.append(failure)
            if step_config.get("phase") == "first" and " task failed:" in failure:
                break

    if step_failures:
        pytest.fail("\n".join(step_failures))


def exec_step_test(
    step_config,
    task_executor,
    context,
    *,
    continue_after_first_validation_fail: bool = False,
):
    label = _step_label(step_config)

    handler.pre_action(step_config.get("type"), step_config)

    command, step_config = handler.get_cmd(step_config.get("type"), step_config)
    step_config["command"] = command

    task_result, task_info = task_executor.execute_task(step_config)
    if not task_result:
        return f"{label} task failed: {task_info}"

    result, info = handler.validate(step_config.get("type"), step_config)
    if not result:
        msg = f"{label} validation failed: {info}"
        if continue_after_first_validation_fail and step_config.get("phase") == "first":
            print(f"WARNING: {msg}; continuing resume case")
            return msg
        return msg

    handler.post_action(step_config.get("type"), step_config)
    return None
