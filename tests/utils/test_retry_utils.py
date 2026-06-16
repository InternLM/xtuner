import unittest


class TestSyncRetryPolicies(unittest.TestCase):
    def test_retry_trace_store_bootstrap_uses_incremental_wait_and_reraises_last_value_error(self):
        from xtuner.v1.utils.retry_utils import retry_trace_store_bootstrap

        attempts = 0
        sleep_seconds: list[float] = []

        def attempt_once():
            nonlocal attempts
            attempts += 1
            raise ValueError(f"attempt {attempts}")

        retryer = retry_trace_store_bootstrap(
            attempts=3,
            wait_start_seconds=0.2,
            wait_increment_seconds=0.2,
            wait_max_seconds=2.0,
            sleep=lambda seconds: sleep_seconds.append(seconds),
        )

        with self.assertRaisesRegex(ValueError, "attempt 3"):
            retryer(attempt_once)

        self.assertEqual(attempts, 3)
        self.assertEqual(sleep_seconds, [0.2, 0.4])


class TestAsyncRetryPolicies(unittest.IsolatedAsyncioTestCase):
    async def test_retry_rollout_request_uses_configured_attempts_and_fixed_wait(self):
        from xtuner.v1.utils.retry_utils import retry_rollout_request

        class RetryableRolloutError(Exception):
            pass

        attempts = 0
        sleep_seconds: list[float] = []
        before_retry_attempts: list[int] = []

        async def attempt_once():
            nonlocal attempts
            attempts += 1
            raise RetryableRolloutError(f"attempt {attempts}")

        async def sleep(seconds: float):
            sleep_seconds.append(seconds)

        retryer = retry_rollout_request(
            attempts=3,
            retry_exceptions=(RetryableRolloutError,),
            wait_seconds=0.1,
            before_retry=lambda retry_state: before_retry_attempts.append(retry_state.attempt_number),
            sleep=sleep,
        )

        with self.assertRaisesRegex(RetryableRolloutError, "attempt 3"):
            await retryer(attempt_once)

        self.assertEqual(attempts, 3)
        self.assertEqual(sleep_seconds, [0.1, 0.1])
        self.assertEqual(before_retry_attempts, [1, 2])

    async def test_retry_sandbox_acquire_uses_policy_waits_for_unhealthy_and_create_failures(self):
        from xtuner.v1.utils.retry_utils import retry_sandbox_acquire

        class UnhealthySandboxError(Exception):
            pass

        class SandboxError(Exception):
            pass

        attempts = 0
        sleep_seconds: list[float] = []

        async def attempt_once():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise UnhealthySandboxError("unhealthy")
            raise SandboxError(f"attempt {attempts}")

        async def sleep(seconds: float):
            sleep_seconds.append(seconds)

        retryer = retry_sandbox_acquire(
            attempts=3,
            unhealthy_exceptions=(UnhealthySandboxError,),
            create_wait_multiplier=2.0,
            create_wait_min_seconds=2.0,
            create_wait_max_seconds=8.0,
            sleep=sleep,
        )

        with self.assertRaisesRegex(SandboxError, "attempt 3"):
            await retryer(attempt_once)

        self.assertEqual(attempts, 3)
        self.assertEqual(sleep_seconds, [0.0, 4.0])

    async def test_poll_sandbox_health_returns_last_result_after_timeout(self):
        from tenacity import stop_after_attempt

        from xtuner.v1.utils.retry_utils import poll_sandbox_health

        attempts = 0
        sleep_seconds: list[float] = []

        async def poll_once():
            nonlocal attempts
            attempts += 1
            return False

        async def sleep(seconds: float):
            sleep_seconds.append(seconds)

        retryer = poll_sandbox_health(
            stop_strategy=stop_after_attempt(3),
            wait_seconds=0.1,
            sleep=sleep,
        )

        result = await retryer(poll_once)

        self.assertFalse(result)
        self.assertEqual(attempts, 3)
        self.assertEqual(sleep_seconds, [0.1, 0.1])
