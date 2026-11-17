"""Actual sandbox package import without Lagent; no training backends needed.

Only parent training packages and unrelated Trainer dependencies are replaced.
The sandbox package initializer and lifecycle modules are executed unchanged.
"""

from __future__ import annotations

import ast
import importlib
import importlib.abc
import logging
import os
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "xtuner.v1.rl.agent_loop.sandbox_agent_loop"
for name in ("xtuner", "xtuner.v1", "xtuner.v1.rl", "xtuner.v1.rl.agent_loop"):
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT / name.replace(".", "/"))]
    sys.modules[name] = module
for key in list(os.environ):
    if key.startswith("XTUNER_SANDBOX_"):
        os.environ.pop(key)


class MissingLagent(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "lagent" or fullname.startswith("lagent."):
            raise ModuleNotFoundError("No module named 'lagent'", name="lagent")


sys.meta_path.insert(0, MissingLagent())


def trainer_method(cls_name, method_name, scope):
    source = ROOT / "xtuner/v1/train/rl_trainer.py"
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), scope)
    return scope[method_name]


class ImportCompatibility(unittest.TestCase):
    def test_cleanup_import_and_no_config_entrypoints(self):
        package = importlib.import_module(PACKAGE)
        self.assertTrue(package.__file__.endswith("sandbox_agent_loop/__init__.py"))
        cleanup = importlib.import_module(PACKAGE + ".cleanup")
        cleanup.start_cleanup_supervisor()
        self.assertTrue(cleanup.request_stop_from_environment())
        with cleanup.guard_ray_shutdown():
            pass
        cleanup.finish_training(training_outcome="succeeded")
        self.assertNotIn("lagent", sys.modules)
        self.assertIn("Runner", dir(package))
        self.assertIs(package.SandboxSpec, importlib.import_module(PACKAGE + ".schemas").SandboxSpec)
        with self.assertRaises(ModuleNotFoundError):
            _ = package.Runner
        self.assertNotIn("Runner", vars(package), "A failed lazy export must not be cached")
        self.assertIs(package.run_trial_process, importlib.import_module(PACKAGE + ".trial_process").run_trial_process)

    def test_trainer_common_init_reaches_training_configuration_without_lagent(self):
        events = []
        utils = types.ModuleType("xtuner.v1.rl.utils")
        utils.register_cleanup = lambda: events.append("register")
        sys.modules[utils.__name__] = utils
        scope = {
            "DistillationTrainerAdapter": lambda config: object(),
            "check_fa3": lambda: None,
            "_migrate_legacy_is_valid_sample_fn": lambda cfg, logger: None,
            "get_writer": lambda **kw: object(),
        }
        init = trainer_method("BaseRLTrainer", "_init_common", scope)
        trainer = types.SimpleNamespace(logger=logging.getLogger("import-test"), _EXP_TRACKING_PATH="tracker")
        for name in (
            "_init_work_dir_and_meta",
            "_init_load_source",
            "_init_save_config",
            "_init_trace",
            "_save_runtime_environment",
            "_init_train_state",
            "_init_train_worker_config",
            "_init_rollout_config",
            "_ensure_rollout_proxy_config",
            "_init_runtime_flags",
        ):
            setattr(trainer, name, lambda *args: None)
        trainer._init_logger = lambda *args: Path("unused")
        config = types.SimpleNamespace(
            distillation_config=None,
            advantage_estimator_config=types.SimpleNamespace(build=lambda: "estimator"),
            exp_tracker="none",
        )
        init(trainer, config, meta_path="unused", logger_tag="test")
        self.assertEqual(events, ["register"])
        self.assertEqual(trainer._advantage_estimator, "estimator")
        self.assertNotIn("lagent", sys.modules)

    def test_both_trainer_fit_exit_wrappers_without_lagent(self):
        for cls in ("RLColocateTrainer", "RLDisaggregatedTrainer"):
            fit = trainer_method(cls, "fit", {"sys": sys})
            for failure in (None, ValueError("fit failed"), KeyboardInterrupt()):

                def train():
                    if failure is not None:
                        raise failure
                    return "done"

                trainer = types.SimpleNamespace(_fit_with_cleanup=train, logger=logging.getLogger("test"))
                if failure is None:
                    self.assertEqual(fit(trainer), "done")
                else:
                    with self.assertRaises(type(failure)) as caught:
                        fit(trainer)
                    self.assertIs(caught.exception, failure)
        self.assertNotIn("lagent", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
