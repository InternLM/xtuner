"""Exercise cleanup integration interfaces without Ray, Lagent or real resources."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "xtuner.v1.rl.agent_loop.sandbox_agent_loop"
for name in ("xtuner", "xtuner.v1", "xtuner.v1.rl", "xtuner.v1.rl.agent_loop"):
    module = types.ModuleType(name)
    module.__path__ = [str(ROOT / name.replace(".", "/"))]
    sys.modules[name] = module

cleanup = importlib.import_module(PACKAGE + ".cleanup")
control = importlib.import_module(PACKAGE + ".run_control")


class RunControlIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.logger = Mock(spec=logging.Logger)

    def test_worker_projection_hides_unrelated_settings_and_peer_policy(self):
        self.assertEqual(control.worker_runtime_env(), {"env_vars": {}})
        settings = {
            "XTUNER_SANDBOX_CLEANUP_DIR": "/shared/run",
            "XTUNER_SANDBOX_PROVIDER_KEY": "backend",
            "XTUNER_SANDBOX_CLEANUP_EXIT_TIMEOUT": "42",
            "XTUNER_SANDBOX_CLEANUP_FACTORY": "trusted:resolve",
            "XTUNER_SANDBOX_RECOVERY_CONCURRENCY": "3",
            "XTUNER_SANDBOX_RECOVERY_ATTEMPTS": "4",
            "XTUNER_SANDBOX_RECOVERY_BACKOFF": "2",
            "XTUNER_SANDBOX_RECOVERY_MAX_BACKOFF": "30",
            "XTUNER_SANDBOX_RECOVERY_ATTEMPT_TIMEOUT": "10",
            "XTUNER_SANDBOX_RECOVERY_BUDGET": "90",
            "XTUNER_SANDBOX_TRIAL_TIMEOUT": "7200",
            "XTUNER_SANDBOX_TRIAL_STOP_TIMEOUT": "60",
            "XTUNER_SANDBOX_TRIAL_RECOVERY_TIMEOUT": "20",
        }
        os.environ.update(settings)
        os.environ.update(
            XTUNER_SANDBOX_CLEANUP_PEERS="0",
            XTUNER_SANDBOX_PRIVATE_TOKEN="must-not-be-forwarded",
            OTHER_ENVIRONMENT="unrelated",
        )
        self.assertEqual(control.worker_runtime_env(), {"env_vars": settings})
        os.environ["XTUNER_SANDBOX_CLEANUP_PEERS"] = "1"
        self.assertEqual(
            control.worker_runtime_env(),
            {"env_vars": {**settings, "XTUNER_SANDBOX_CLEANUP_PEERS": "1"}},
        )

    def test_peer_registration_requires_explicit_opt_in(self):
        with patch.object(cleanup, "start_cleanup_supervisor") as start:
            control.register_peer_if_configured()
            os.environ["XTUNER_SANDBOX_CLEANUP_PEERS"] = "0"
            control.register_peer_if_configured()
            start.assert_not_called()
            os.environ["XTUNER_SANDBOX_CLEANUP_PEERS"] = "1"
            control.register_peer_if_configured()
            start.assert_called_once_with(peer=True)

    def test_training_registration_validates_before_signal_registration(self):
        events = []
        utils = types.ModuleType("xtuner.v1.rl.utils")
        utils.register_cleanup = lambda: events.append("signals")
        with (
            patch.dict(sys.modules, {utils.__name__: utils}),
            patch.object(cleanup, "start_cleanup_supervisor", side_effect=lambda: events.append("validate")),
        ):
            control.register_training_cleanup()
        self.assertEqual(events, ["validate", "signals"])
        events.clear()
        with (
            patch.dict(sys.modules, {utils.__name__: utils}),
            patch.object(
                cleanup, "start_cleanup_supervisor", side_effect=ValueError("invalid recovery configuration")
            ),
        ):
            with self.assertRaises(ValueError):
                control.register_training_cleanup()
        self.assertEqual(events, [])

    def test_training_scope_classifies_outcomes_and_preserves_original_error(self):
        for error, outcome in (
            (None, "succeeded"),
            (ValueError("training failed"), "failed"),
            (KeyboardInterrupt(), "interrupted"),
            (asyncio.CancelledError(), "interrupted"),
        ):
            with self.subTest(outcome=outcome, error=type(error).__name__):
                with patch.object(cleanup, "cleanup_from_environment", return_value=True) as recover:
                    if error is None:
                        with control.training_scope():
                            pass
                    else:
                        with self.assertRaises(type(error)) as caught:
                            with control.training_scope():
                                raise error
                        self.assertIs(caught.exception, error)
                    recover.assert_called_once_with(finalize=True, training_outcome=outcome)

    def test_training_scope_does_not_classify_an_outer_handled_exception(self):
        with patch.object(cleanup, "finish_training") as finish:
            try:
                raise ValueError("already handled by caller")
            except ValueError:
                with control.training_scope():
                    pass
        finish.assert_called_once_with("succeeded")

    def test_training_scope_preserves_business_error_when_cleanup_fails(self):
        original = ValueError("training failed")
        with patch.object(cleanup, "cleanup_from_environment", side_effect=RuntimeError("recovery unavailable")):
            with self.assertRaises(ValueError) as caught:
                with control.training_scope():
                    raise original
            self.assertIs(caught.exception, original)
            with self.assertRaises(cleanup.CleanupError):
                with control.training_scope():
                    pass

    def test_stop_before_drain_logs_pending_admission(self):
        events = []
        with patch.object(
            cleanup, "request_stop_from_environment", side_effect=lambda: events.append("stop") or False
        ):
            control.stop_before_drain(self.logger)
            events.append("drain")
        self.assertEqual(events, ["stop", "drain"])
        self.logger.error.assert_called_once()

    def test_ray_shutdown_scope_orders_stop_guard_drain_and_recovery(self):
        for fails in (False, True):
            with self.subTest(drain_fails=fails):
                events = []
                original = RuntimeError("Ray drain failed")

                @contextmanager
                def guard(**kwargs):
                    self.assertEqual(kwargs, {"training_outcome": "interrupted"})
                    events.append("guard-start")
                    try:
                        yield
                    finally:
                        events.append("guard-end")

                def recover(**kwargs):
                    self.assertEqual(kwargs, {"finalize": True, "training_outcome": "interrupted"})
                    events.append("recover")
                    raise OSError("recovery failed")

                with (
                    patch.object(
                        cleanup, "request_stop_from_environment", side_effect=lambda: events.append("stop") or True
                    ),
                    patch.object(cleanup, "guard_ray_shutdown", guard),
                    patch.object(cleanup, "cleanup_from_environment", recover),
                ):
                    if fails:
                        with self.assertRaises(RuntimeError) as caught:
                            with control.ray_shutdown_scope("interrupted", self.logger):
                                events.append("drain")
                                raise original
                        self.assertIs(caught.exception, original)
                    else:
                        with control.ray_shutdown_scope("interrupted", self.logger):
                            events.append("drain")
                self.assertEqual(events, ["stop", "guard-start", "drain", "guard-end", "recover"])
                self.logger.error.assert_called_with("Sandbox cleanup failed (OSError); retain the run journal")

    def test_disabled_scopes_do_not_launch_processes_or_import_lagent(self):
        with patch.object(cleanup.subprocess, "Popen", side_effect=AssertionError("unexpected process launch")):
            control.register_peer_if_configured()
            with control.training_scope():
                pass
            with control.ray_shutdown_scope(None, self.logger):
                pass
        self.logger.error.assert_not_called()
        self.assertNotIn("lagent", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
