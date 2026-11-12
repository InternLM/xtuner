import asyncio,unittest
from types import SimpleNamespace as N
from unittest.mock import patch,AsyncMock
from xtuner.v1.rl.agent_loop_manager.colocate_keepalive import LoopThread,ColocateKeepaliveManager
from xtuner.v1.rl.agent_loop_manager.produce_utils import _PendingTasks
class Tests(unittest.IsolatedAsyncioTestCase):
 async def test_cleanup_deadline_keeps_original_owner_then_retry(self):
  owner=LoopThread()
  async def setup():
   ready=asyncio.Event();finish=asyncio.Event();calls=[]
   async def work():
    ready.set()
    try:await asyncio.Event().wait()
    finally:calls.append('finally');await finish.wait()
   task=asyncio.create_task(work(),name='sandbox-fixture:owned-tail');await ready.wait();task.cancel();await asyncio.sleep(0)
   inner=N(_finalizing_tasks={task},task_runners=[],release_final_leftovers=AsyncMock())
   return inner,finish,calls
  inner,finish,calls=owner.submit(setup()).result();manager=ColocateKeepaliveManager(inner,owner);wait=asyncio.wait
  async def bounded(tasks,**kwargs):return await wait(tasks,timeout=.01)
  with patch('asyncio.wait',new=bounded):
   with self.assertRaisesRegex(RuntimeError,'owned-tail'):await manager.close()
  self.assertTrue(owner.thread.is_alive());self.assertEqual(calls,['finally']);self.assertFalse(manager._closed)
  owner.loop.call_soon_threadsafe(finish.set)
  await manager.close();self.assertFalse(owner.thread.is_alive());self.assertEqual(calls,['finally'])
 async def test_error_does_not_skip_other_cleanup(self):
  owner=LoopThread()
  async def setup():
   finished=asyncio.Event();p=_PendingTasks()
   async def regular():
    try:await asyncio.Event().wait()
    finally:finished.set()
   task=asyncio.create_task(regular(),name='other-agent');await asyncio.sleep(0)
   async def spawn():return task
   await p.schedule_one(max_pending=1,should_abort=lambda:False,spawn_one=spawn)
   async def bad():raise ValueError('cleanup failure')
   failed=asyncio.create_task(bad(),name='broken-tail');await asyncio.sleep(0)
   return N(_finalizing_tasks={failed},task_runners=[N(task_name='other',produce_strategy=N(_pending_tasks=p))],release_final_leftovers=AsyncMock()),finished
  inner,finished=owner.submit(setup()).result();m=ColocateKeepaliveManager(inner,owner)
  with self.assertRaisesRegex(RuntimeError,'broken-tail'):await m.close()
  self.assertTrue(finished.is_set());self.assertFalse(owner.thread.is_alive());inner.release_final_leftovers.assert_awaited_once()
if __name__=='__main__':unittest.main(verbosity=2)
