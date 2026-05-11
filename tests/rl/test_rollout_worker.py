import asyncio
import unittest
from unittest.mock import MagicMock

from xtuner.v1.rl.rollout.sglang import SGLangWorker


class TestSGLangWorker(unittest.TestCase):
    def test_continue_generation_clears_abort_flag(self):
        worker = SGLangWorker.__new__(SGLangWorker)
        worker.receive_abort_request = asyncio.Event()
        worker.receive_abort_request.set()
        worker._make_request = MagicMock(return_value="ok")

        result = worker.continue_generation()

        self.assertEqual(result, "ok")
        self.assertFalse(worker.receive_abort_request.is_set())
        worker._make_request.assert_called_once_with("continue_generation")


if __name__ == "__main__":
    unittest.main()
