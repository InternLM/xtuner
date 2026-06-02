import os
import sys
from pathlib import Path


RL_TEST_ROOT = Path(__file__).resolve().parents[1]

if str(RL_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_TEST_ROOT))

# Fast RL tests start their own local Ray clusters. Disable Ray's uv runtime-env
# hook by default because it can inspect sandbox parent processes and fail before
# the test logic runs.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
