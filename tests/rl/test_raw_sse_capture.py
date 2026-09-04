import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xtuner.v1.rl.rollout import session_server


class TestRawSSECapture(unittest.TestCase):
    def test_capture_is_byte_exact_and_bounded(self):
        raw = b'event: message\ndata: {"type":"content_block_delta","delta":{"text":"ok"}}\n\ndata: [DONE]\n\n'
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"XTUNER_RAW_SSE_DIR": directory, "XTUNER_RAW_SSE_MAX_FILES": "1"},
            clear=False,
        ), mock.patch.object(session_server, "_raw_sse_capture_counts", {}):
            session_server._capture_raw_sse(raw, fmt="anthropic", upstream_status=200, request_path="v1/messages")
            session_server._capture_raw_sse(
                b"different", fmt="anthropic", upstream_status=200, request_path="v1/messages"
            )

            captures = list(Path(directory).glob("*.sse"))
            metadata = list(Path(directory).glob("*.meta.json"))
            self.assertEqual(len(captures), 1)
            self.assertEqual(len(metadata), 1)
            self.assertEqual(captures[0].read_bytes(), raw)
            details = json.loads(metadata[0].read_text(encoding="utf-8"))
            self.assertEqual(
                {key for key in details if key not in {"captured_at_ns", "pid"}},
                {"byte_count", "format", "request_path", "upstream_status"},
            )
            self.assertEqual(details["byte_count"], len(raw))
            self.assertEqual(details["format"], "anthropic")
            self.assertEqual(details["request_path"], "v1/messages")
            self.assertEqual(details["upstream_status"], 200)
            self.assertEqual(captures[0].stat().st_mode & 0o777, 0o600)
            self.assertEqual(metadata[0].stat().st_mode & 0o777, 0o600)
