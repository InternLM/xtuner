import unittest
from unittest.mock import MagicMock, patch

from xtuner.v1.rl.rollout.proxy_manager import RolloutProxyManager
from xtuner.v1.rl.utils.misc import check_chat_completions


class ProxyProbeTests(unittest.TestCase):
    def test_configured_health_probe_uses_validated_budget_and_stable_request_session(self):
        captured = []

        def post(url, payload):
            captured.append((url, payload))
            return {"choices": [{"message": {"content": "pong"}}]}

        with patch("xtuner.v1.rl.utils.misc._post", side_effect=post):
            self.assertTrue(check_chat_completions("http://proxy/v1", "model", max_tokens=1024, session_id="health"))
        self.assertEqual(captured[0][1]["max_tokens"], 1024)
        self.assertEqual(captured[0][1]["session_id"], "health")

    def test_turn_probe_never_retries_model_and_releases_trace_on_success_or_failure(self):
        for success in (True, False):
            manager = RolloutProxyManager.__new__(RolloutProxyManager)
            manager._turn_resume = True
            manager._model_name = "m"
            manager._routed_proxy_url = "http://proxy"
            manager._logger = MagicMock()
            store = MagicMock()
            with (
                patch("xtuner.v1.rl.rollout.proxy_manager.check_chat_completions", return_value=success) as check,
                patch("xtuner.v1.rl.rollout.trace_store.get_existing_store", return_value=store),
                patch("xtuner.v1.rl.rollout.proxy_manager.ray.get"),
            ):
                if success:
                    manager._validate_routed_proxy()
                else:
                    with self.assertRaises(RuntimeError):
                        manager._validate_routed_proxy()
            self.assertEqual(check.call_count, 1)
            store.release_sessions.remote.assert_called_once_with([check.call_args.kwargs["session_id"]])
