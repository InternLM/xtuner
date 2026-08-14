import importlib
import os
import sys
from pathlib import Path

# Activate the Triton autotune pin installed by `xtuner.v1.__init__` (gated by this env
# var). The pin must run before any module imports `fla`; see the patch's docstring in
# `xtuner/v1/__init__.py` for why it lives there rather than here in the conftest.
os.environ.setdefault("XTUNER_DETERMINISTIC", "true")

# Trigger that patch in the pytest parent process. Each `MultiProcessTestCase` child is a
# fresh `multiprocessing.spawn` Python that re-imports the test class top-level, which pulls
# `xtuner.v1.*` and runs the same `xtuner/v1/__init__.py` block — so the patch is installed
# in every process as long as `XTUNER_DETERMINISTIC=true` was inherited via the env.
import xtuner.v1  # noqa: E402,F401

from huggingface_hub import constants  # noqa: E402

from xtuner._testing.patch_rollout_config import patch_rollout_config_dist_port_base


def _patch_parametrize_namespace_scan() -> None:
    """Avoid lazy imports mutating module dictionaries during collection."""
    parametrize_module = importlib.import_module("parametrize.parametrize")
    original = parametrize_module._find_possible_decorators

    def find_possible_decorators(namespace, search_in_modules=True):
        return original(dict(namespace), search_in_modules=search_in_modules)

    parametrize_module._find_possible_decorators = find_possible_decorators


def _isolate_pytest_compiler_cache() -> None:
    """Give the pytest parent process its own Triton/Inductor cache dirs.

    Distributed ranks already isolate via ``DeterministicDDPTestCase``; this
    covers non-DDP tests and the parent process so serial full-suite runs do not
    corrupt a shared ``/tmp/.triton`` under torch 2.12 / triton 3.7.
    """
    import tempfile

    base = os.environ.get("TRITON_CACHE_DIR") or os.path.join(tempfile.gettempdir(), ".triton")
    leaf = os.path.basename(base)
    if leaf.startswith("pytest_p") or leaf.startswith("r"):
        cache_dir = base
    else:
        cache_dir = os.path.join(base, f"pytest_p{os.getpid()}")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    inductor_dir = cache_dir + "_inductor"
    os.makedirs(inductor_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_dir


_patch_parametrize_namespace_scan()
_isolate_pytest_compiler_cache()


_HF_DYNAMIC_MODULE_PREFIX = "transformers_modules"
_HF_PATCH_MODULES_CACHE_PREFIX = "modules_cache"


def _is_hf_dynamic_module_root(path: str) -> bool:
    path_obj = Path(path)
    hf_home = Path(constants.HF_HOME)
    return path_obj.parent == hf_home and (
        path_obj.name == "modules" or path_obj.name.startswith(_HF_PATCH_MODULES_CACHE_PREFIX)
    )


def _cleanup_hf_dynamic_modules() -> None:
    for module_name in tuple(sys.modules):
        if module_name == _HF_DYNAMIC_MODULE_PREFIX or module_name.startswith(f"{_HF_DYNAMIC_MODULE_PREFIX}."):
            sys.modules.pop(module_name, None)

    removed_paths = [path for path in sys.path if _is_hf_dynamic_module_root(path)]
    sys.path[:] = [path for path in sys.path if not _is_hf_dynamic_module_root(path)]
    for path in removed_paths:
        sys.path_importer_cache.pop(path, None)

    if _is_hf_dynamic_module_root(os.environ.get("HF_MODULES_CACHE", "")):
        os.environ.pop("HF_MODULES_CACHE", None)

    default_modules_cache = os.path.join(constants.HF_HOME, "modules")
    for module_name in ("transformers.utils.hub", "transformers.utils", "transformers.dynamic_module_utils"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "HF_MODULES_CACHE"):
            module.HF_MODULES_CACHE = default_modules_cache

    importlib.invalidate_caches()


def pytest_runtest_setup(item):
    item_path_obj = getattr(item, "path", None)
    if item_path_obj is None:
        item_path_obj = item.fspath
    item_path = Path(str(item_path_obj))
    if item_path.parent.name == "rl":
        patch_rollout_config_dist_port_base()
    _cleanup_hf_dynamic_modules()
