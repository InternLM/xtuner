import pytest
from utils.common_utils import get_config


CONFIG_FILE = "autotest/config.yaml"


@pytest.fixture(scope="session")
def config(request):
    config = get_config()
    if request:
        run_id = request.config.getoption("--run_id")

    config["run_id"] = run_id
    return config


def pytest_addoption(parser):
    parser.addoption("--run_id", action="store", default="", help="github run_id")
