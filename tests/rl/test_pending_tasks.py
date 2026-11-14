"""_PendingTasks 的并发协议测试。

Good Tests:
- 把 _PendingTasks 视为 AsyncProduceStrategy 内部的深模块契约，直接验证 claim/schedule/cancel 的并发语义。
- 只观察公开方法返回值和 count()，不读取内部 set/lock。
- 测试说明 pending task 在 produce_batch 与 pause_produce 竞争时不会重复 claim 或丢失。

Bad Tests:
- 不测试 AsyncProduceStrategy 如何循环调用 _PendingTasks。
- 不断言 `_tasks`、`_lock` 等内部结构，也不绑定具体 asyncio.wait 的实现细节。

本文件主要覆盖的 public 行为:
- done task 只能被 claim 一次。
- schedule_one 会尊重 abort 信号和 max_pending 上限。
- cancel_all 清空 pending 后，后续 wait/claim 不会再次取到同一 task。
- wait snapshot 后会二次 claim，避免并发路径重复处理同一结果。
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from xtuner.v1.rl.agent_loop_manager.produce_utils import _PendingTasks, pause_pending_tasks


class TestPendingTasks(unittest.IsolatedAsyncioTestCase):
    async def test_pause_pending_tasks_drains_local_set_and_pending_container(self):
        # 验证共卡本地 set 和非共卡 _PendingTasks 都复用同一套 pause/drain 协议。
        for pending_tasks in (set(), _PendingTasks()):
            ctx = MagicMock()
            ctx.task_name = "test_pause_helper"
            ctx.agent_loop.pause = AsyncMock(return_value=None)
            claimed_results: list[str] = []

            async def done():
                return "done"

            if isinstance(pending_tasks, set):
                pending_tasks.add(asyncio.create_task(done()))
            else:

                async def spawn_one():
                    return asyncio.create_task(done())

                await pending_tasks.schedule_one(
                    max_pending=1,
                    should_abort=lambda: False,
                    spawn_one=spawn_one,
                )

            async def put_claimed_task(task: asyncio.Task) -> None:
                claimed_results.append(task.result())

            pause_time_s = await pause_pending_tasks(
                pending_tasks=pending_tasks,
                ctx=ctx,
                put_claimed_task=put_claimed_task,
            )

            self.assertGreaterEqual(pause_time_s, 0.0)
            self.assertEqual(claimed_results, ["done"])
            self.assertEqual(len(pending_tasks) if isinstance(pending_tasks, set) else pending_tasks.count(), 0)
            self.assertGreaterEqual(ctx.agent_loop.pause.await_count, 1)

    async def test_claim_ready_returns_each_done_task_once(self):
        # 验证 done task 被 claim 后会从 pending 集合移除，避免 producer/pause 重复处理同一结果。
        pending_tasks = _PendingTasks()

        async def spawn_one():
            async def done():
                return "done"

            return asyncio.create_task(done())

        scheduled = await pending_tasks.schedule_one(
            max_pending=1,
            should_abort=lambda: False,
            spawn_one=spawn_one,
        )
        self.assertTrue(scheduled)
        self.assertEqual(pending_tasks.count(), 1)

        await asyncio.sleep(0)
        claimed = await pending_tasks.claim_ready()
        self.assertEqual(len(claimed), 1)
        self.assertEqual(await pending_tasks.claim_ready(), set())
        self.assertEqual(pending_tasks.count(), 0)

    async def test_schedule_one_respects_abort_signal_and_pending_limit(self):
        # 验证调度新 task 时原子检查 abort 信号和 max_pending，pause 后不会继续新增请求。
        pending_tasks = _PendingTasks()
        spawn_count = 0

        async def spawn_one():
            nonlocal spawn_count
            spawn_count += 1

            async def wait_forever():
                await asyncio.Event().wait()

            return asyncio.create_task(wait_forever())

        self.assertFalse(
            await pending_tasks.schedule_one(max_pending=0, should_abort=lambda: False, spawn_one=spawn_one)
        )
        self.assertFalse(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: True, spawn_one=spawn_one)
        )
        self.assertTrue(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        self.assertFalse(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        self.assertEqual(spawn_count, 1)

        self.assertEqual(await pending_tasks.cancel_all(), 1)
        self.assertEqual(pending_tasks.count(), 0)

    async def test_cancel_all_clears_pending_before_later_wait_claims(self):
        # 验证 cancel 会先清空 pending 集合，后续 wait/claim 不会重新拿到已取消 task。
        pending_tasks = _PendingTasks()

        async def spawn_one():
            async def wait_forever():
                await asyncio.Event().wait()

            return asyncio.create_task(wait_forever())

        self.assertTrue(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        self.assertEqual(await pending_tasks.cancel_all(), 1)
        self.assertEqual(await pending_tasks.wait_and_claim(timeout_s=0), set())
        self.assertEqual(pending_tasks.count(), 0)

    async def test_close_cancels_pending_tasks_once_and_waits_for_cleanup(self):
        # close 必须等待原任务的 finally 清理真正结束，而不是发出 cancel 就返回。
        pending_tasks = _PendingTasks()
        cleaned_up = asyncio.Event()

        async def spawn_one():
            async def wait_and_clean():
                try:
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(0.01)
                    cleaned_up.set()

            return asyncio.create_task(wait_and_clean())

        self.assertTrue(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        await pending_tasks.close()

        self.assertTrue(cleaned_up.is_set())
        self.assertEqual(pending_tasks.count(), 0)
        self.assertFalse(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )

    async def test_close_reports_cleanup_errors(self):
        pending_tasks = _PendingTasks()

        async def spawn_one():
            async def fail_while_closing():
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise RuntimeError("pool cleanup failed") from None

            return asyncio.create_task(fail_while_closing())

        self.assertTrue(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        await asyncio.sleep(0)

        with self.assertRaisesRegex(RuntimeError, "pool cleanup failed"):
            await pending_tasks.close()

    async def test_close_is_idempotent(self):
        pending_tasks = _PendingTasks()

        async def spawn_one():
            async def wait_forever():
                await asyncio.Event().wait()

            return asyncio.create_task(wait_forever())

        self.assertTrue(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        await pending_tasks.close()
        await pending_tasks.close()
        self.assertEqual(pending_tasks.count(), 0)

    async def test_wait_and_claim_ignores_tasks_claimed_by_another_path(self):
        # 验证 wait 使用快照后仍会二次 claim，避免 pause 与 produce_batch 同时 wait 时重复处理。
        pending_tasks = _PendingTasks()
        release_task = asyncio.Event()

        async def spawn_one():
            async def wait_until_released():
                await release_task.wait()
                return "done"

            return asyncio.create_task(wait_until_released())

        self.assertTrue(
            await pending_tasks.schedule_one(max_pending=1, should_abort=lambda: False, spawn_one=spawn_one)
        )
        waiter = asyncio.create_task(pending_tasks.wait_and_claim(timeout_s=1))
        await asyncio.sleep(0)

        self.assertEqual(await pending_tasks.cancel_all(), 1)
        release_task.set()
        self.assertEqual(await waiter, set())
        self.assertEqual(pending_tasks.count(), 0)
