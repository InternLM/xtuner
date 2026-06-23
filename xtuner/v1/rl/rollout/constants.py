"""Rollout-wide constants.

Producer strategies control the real rollout fan-out with their scheduled
pending task targets. These large caps keep Ray actors and HTTP client pools
from introducing extra queues before requests reach the inference engine.
"""

# Per-rollout-worker HTTP connection cap. httpx does not pre-open these
# connections; this only keeps its connection pool from becoming the normal
# rollout queue. The inference engine should own request scheduling.
ROLLOUT_HTTP_MAX_CONNECTIONS = 100_000

# Ray actor method concurrency cap for rollout generate paths. This is
# intentionally far above expected producer fan-out so RolloutController and
# RolloutWorker actors do not become the queueing point.
ROLLOUT_RAY_GENERATE_MAX_CONCURRENCY = 100_000_000

# Agent-loop actors should also avoid adding an independent queue in front of
# rollout generation. Producer pending limits remain the effective fan-out
# control.
AGENT_LOOP_RAY_GENERATE_MAX_CONCURRENCY = ROLLOUT_RAY_GENERATE_MAX_CONCURRENCY
