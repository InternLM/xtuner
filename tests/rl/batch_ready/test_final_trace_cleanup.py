import asyncio,unittest
from unittest.mock import AsyncMock,patch
from .test_full_manager import Fixture
class Tests(unittest.IsolatedAsyncioTestCase):
 async def test_only_unselected_pending_and_ready_released(self):
  f=Fixture()
  try:
   first=await f.produce(1);f.tool.set();second=await f.produce(2)
   tail=await f.sample()
   async def spawn():
    task=asyncio.create_task(f.generate_group(tail),name='owned-tail:3');f.strategy._pending_tasks.track_group(task,tail);return task
   await f.strategy._pending_tasks.schedule_one(max_pending=2,should_abort=lambda:False,spawn_one=spawn);await asyncio.sleep(0)
   # One ready group is selected and another remains unselected in replay.
   from xtuner.v1.data_proto.rl_data import Status
   for _ in range(2):
    ready=await f.sample()
    for k,item in enumerate(ready):
     item.status=Status.COMPLETED;item.reward={'score':k};item.response_ids=[2];item.response_model_steps=[1];item.num_tokens=2
    await f.buffer.put(ready,'task')
   final=await f.produce(3,True);self.assertEqual(final.rollout_states[0][0].group_id,4)
   f.cleanup.set();await asyncio.gather(*f.m._finalizing_tasks,return_exceptions=True)
   release=AsyncMock()
   with patch('xtuner.v1.rl.rollout.trace_store.release_and_discard_rollout_groups',new=release):await f.m.release_final_leftovers()
   groups=release.call_args.args[0];self.assertEqual({x.group_id for g in groups for x in g},{3,5})
   self.assertEqual(len(f.buffer),0)
  finally:await f.close()
if __name__=='__main__':unittest.main(verbosity=2)
