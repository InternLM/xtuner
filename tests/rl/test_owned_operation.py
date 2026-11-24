"""await_owned_operation 的取消协议测试。

Good Tests:
- 通过真实 asyncio 取消验证：等待者取消后，被拥有的操作仍会实际结束。
- 只观察返回值、异常和副作用完成标志。

Bad Tests:
- 不断言内部 Task 状态或 shield 的实现细节。
"""

import asyncio
import time
import unittest

from xtuner.v1.rl.utils import await_owned_operation


class TestAwaitOwnedOperation(unittest.IsolatedAsyncioTestCase):
    async def test_returns_the_operation_result(self):
        async def operation() -> str:
            return "done"

        self.assertEqual(await await_owned_operation(operation()), "done")

    async def test_cancellation_waits_for_the_owned_operation_to_finish(self):
        finished = []

        def blocking_work() -> str:
            time.sleep(0.05)
            finished.append(True)
            return "done"

        waiter = asyncio.create_task(await_owned_operation(asyncio.to_thread(blocking_work)))
        await asyncio.sleep(0)
        waiter.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertEqual(finished, [True])

    async def test_operation_failure_propagates_to_the_caller(self):
        async def failing() -> None:
            raise RuntimeError("operation failed")

        with self.assertRaisesRegex(RuntimeError, "operation failed"):
            await await_owned_operation(failing())


if __name__ == "__main__":
    unittest.main()
