import os
import sys

import pytest


sys.path.insert(0, "autotest")

from utils.common_utils import _resolve_train_image, strip_xtuner_editable_install


@pytest.mark.parametrize(
    ("pip_package", "expected"),
    [
        ("pip install -e .[all]; pip install more-itertools", "pip install more-itertools"),
        (
            "FLASH_MLA=1 pip install -v /path;pip install -e .[all]; pip install pytest-xdist",
            "FLASH_MLA=1 pip install -v /path; pip install pytest-xdist",
        ),
        ("pip install -e .", "true"),
        ("pip install -e '.[all]'", "true"),
        ("pip install --editable .[all]", "true"),
        ("pip install -e ./", "true"),
        (
            "pip install -e ./some_other_pkg; pip install more-itertools",
            "pip install -e ./some_other_pkg; pip install more-itertools",
        ),
    ],
)
def test_strip_xtuner_editable_install(pip_package, expected):
    assert strip_xtuner_editable_install(pip_package) == expected


@pytest.mark.parametrize(
    ("image", "registry", "expected"),
    [
        ("ailab-llmrazor/xtuner:pt29_latest", "registry.example.com", "registry.example.com/ailab-llmrazor/xtuner:pt29_latest"),
        (
            "registry.example.com/ailab-llmrazor/xtuner:tag",
            "other.registry.com",
            "registry.example.com/ailab-llmrazor/xtuner:tag",
        ),
        ("ailab-llmrazor/xtuner:pt29_latest", None, "ailab-llmrazor/xtuner:pt29_latest"),
        ("localhost/xtuner:tag", "registry.example.com", "localhost/xtuner:tag"),
        (
            "registry:5000/ailab-llmrazor/xtuner:tag",
            "other.registry.com",
            "registry:5000/ailab-llmrazor/xtuner:tag",
        ),
    ],
)
def test_resolve_train_image(image, registry, expected):
    assert _resolve_train_image(image, registry) == expected


def test_get_config_train_image_override(monkeypatch):
    monkeypatch.setenv("CI_GPU_IMAGE_REGISTRY", "registry.example.com")
    monkeypatch.setenv("CI_ETE_TRAIN_IMAGE", "ailab-llmrazor/xtuner:custom")

    from utils.common_utils import get_config

    config = get_config()
    image = config["case"]["glm5-2-sft-30B"][0]["resource"]["image"]
    assert image == "registry.example.com/ailab-llmrazor/xtuner:custom"


def test_get_config_full_train_image_override(monkeypatch):
    full = "registry.example.com/ailab-llmrazor/xtuner:custom"
    monkeypatch.setenv("CI_GPU_IMAGE_REGISTRY", "other.registry.com")
    monkeypatch.setenv("CI_ETE_TRAIN_IMAGE", full)

    from utils.common_utils import get_config

    config = get_config()
    image = config["case"]["glm5-2-sft-30B"][0]["resource"]["image"]
    assert image == full
