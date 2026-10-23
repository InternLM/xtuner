"""Regression tests for the rollout-teacher MOPD e2e config env parsing.

The launcher (launch_teacher_utils.sh) clears ``XTUNER_RL_NUM_WORKERS`` to an
empty string while discovering Teachers, so the config must treat an empty
value as "unset" instead of crashing on ``int("")``.
"""

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from xtuner.v1.utils.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "recipe/on_policy_distillation/config/rl_qwen3_vl_2b_sampled_token_mopd_rollout_teacher.py"


def _stub_glm5_next_for_old_transformers() -> None:
    """The config transitively imports glm53_vl_tokenize_fn, which needs
    ``transformers.models.glm5_next``. Stub it only when the installed
    transformers predates glm5_next; the parsing under test never touches it.
    """
    try:
        import transformers.models.glm5_next  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    pkg = ModuleType("transformers.models.glm5_next")
    sys.modules["transformers.models.glm5_next"] = pkg
    for name in ("image_processing_glm5_next", "video_processing_glm5_next"):
        mod = ModuleType(f"transformers.models.glm5_next.{name}")
        mod.smart_resize = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        sys.modules[f"transformers.models.glm5_next.{name}"] = mod


_stub_glm5_next_for_old_transformers()


@pytest.fixture
def config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    meta_file = tmp_path / "train_meta.json"
    meta_file.write_text(json.dumps({"openai/gsm8k": {"annotation": "dummy.jsonl"}}))
    monkeypatch.setenv("WORK_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_PATH", "/dummy/student")
    monkeypatch.setenv("GSM8K_TEACHER_MODEL_PATH", "/dummy/gsm8k_teacher")
    monkeypatch.setenv("GEO3K_TEACHER_MODEL_PATH", "/dummy/geo3k_teacher")
    monkeypatch.setenv("DATA_PATH", str(meta_file))
    monkeypatch.setenv("EVAL_DATA_PATH", "")
    return monkeypatch


class TestOPDRolloutTeacherConfig:
    def test_empty_num_workers_env_falls_back_to_default(self, config_env: pytest.MonkeyPatch) -> None:
        config_env.setenv("XTUNER_RL_NUM_WORKERS", "")
        cfg = Config.fromfile(CONFIG_PATH)
        assert cfg["resources"].num_workers == 6

    def test_num_workers_env_override(self, config_env: pytest.MonkeyPatch) -> None:
        config_env.setenv("XTUNER_RL_NUM_WORKERS", "8")
        cfg = Config.fromfile(CONFIG_PATH)
        assert cfg["resources"].num_workers == 8
