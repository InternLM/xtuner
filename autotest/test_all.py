import os

import pytest
from cluster.clusterx import ClusterTaskExecutor
from module.get_module import TestHandler
from utils.common_utils import get_case_list


@pytest.fixture
def task_executor():
    return ClusterTaskExecutor()


handler = TestHandler()


@pytest.mark.parametrize("case", get_case_list(type="pre_train"))
@pytest.mark.pre_train
def test_pretrain(config, case, task_executor):
    run_all_cases(config, case, task_executor)


@pytest.mark.parametrize("case", get_case_list(type="rl"))
@pytest.mark.rl
def test_rl(config, case, task_executor):
    run_all_cases(config, case, task_executor)


@pytest.mark.parametrize("case", get_case_list(type="sft"))
@pytest.mark.sft
def test_sft(config, case, task_executor):
    run_all_cases(config, case, task_executor)


@pytest.mark.parametrize("case", get_case_list())
@pytest.mark.all
def test_all(config, case, task_executor):
    run_all_cases(config, case, task_executor)


def run_all_cases(config, case_name, task_executor) -> None:
    case_config = config["case"].get(case_name)
    base_path_config = config["base_path"]
    current_dir = os.getcwd()
    context = {}

    for step_config in case_config:
        step_config["case_name"] = case_name
        step_config["run_id"] = config.get("run_id")
        step_config["current_dir"] = current_dir
        step_config["base_path"] = base_path_config
        step_config["context"] = context

        exec_step_test(step_config, task_executor, context)


def exec_step_test(step_config, task_executor, context):
    # pre action
    command, step_config = handler.pre_action(step_config.get("type"), step_config)
    step_config["command"] = command

    # get cmd
    command, step_config = handler.get_cmd(step_config.get("type"), step_config)
    step_config["command"] = command

    # run task
    task_result, task_info = task_executor.execute_task(step_config)
    assert task_result, task_info

    # verify task result
    result, info = handler.validate(step_config.get("type"), step_config)
    assert result, info

    # post action
    result, info = handler.post_action(step_config.get("type"), step_config)
    assert result, info
