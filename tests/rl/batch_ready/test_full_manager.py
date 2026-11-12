"""Actual manager+producer+buffer+policy+gate, scripted CPU Agent/backend only."""
import asyncio,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as N
from xtuner.v1.data_proto.rl_data import RolloutState,Status
from xtuner.v1.rl.replay_buffer import ReplayBuffer,FIFOReplayPolicy,NaiveStorage
from xtuner.v1.rl.agent_loop_manager.agent_loop_manager import AgentLoopManager
from xtuner.v1.rl.agent_loop_manager.producer import AsyncProduceStrategy,default_should_continue_fn
from xtuner.v1.rl.agent_loop_manager.produce_utils import _TaskRunner
from .test_manager import Controller,Logger,CANDIDATE
from xtuner.v1.rl.agent_loop_manager.agent_turn_pause_policy import AgentTurnPausePolicy
from xtuner.v1.rl.rollout.request_gate import SettlementReceipt
EVENTS=[]
class Fixture:
 def __init__(self):
  self.control=Controller();self.tool=asyncio.Event();self.cleanup=asyncio.Event();self.cancelled=asyncio.Event();self.counter=0;self.history=[]
  self.buffer=ReplayBuffer(FIFOReplayPolicy(),NaiveStorage())
  self.strategy=AsyncProduceStrategy(over_sample_threshold=1,enable_partial_rollout=False,tail_batch_trigger_size=-1,max_pending_tasks=2,max_staleness=2,max_token_staleness=None,sync_weights_interval=1,should_continue_fn=default_should_continue_fn);self.strategy.retain_live_tasks=True
  self.m=object.__new__(AgentLoopManager);self.m.name='real-control';self.m.logger=Logger();self.m.task_names=['task'];self.m.replay_buffer=self.buffer;self.m._rollout_controller=self.control;self.m._update_policy='batch_ready';self.m._checkpoint_mode='model_export'
  self.m._keepalive=AgentTurnPausePolicy(freeze_timeout=1,drain_timeout=1,quiesce_timeout=.06)
  self.m.task_runners=[_TaskRunner(task_name='task',agent_loop=self,produce_strategy=self.strategy,sampler=self)]
 async def sample(self,**kw):
  self.counter+=1;g=self.counter
  return [RolloutState(group_id=g,rollout_id=g*10+k,session_id=g*10+k,message=[],prompt_ids=[1],extra_fields={}) for k in range(2)]
 async def generate_group(self,group,**kw):
  g=group[0].group_id
  try:
   if g==2:await self.tool.wait()
   if g==3:await asyncio.Event().wait()
   for k,item in enumerate(group):
    async with self.control.gate.request(str(item.session_id)) as lease:
     item.status=Status.COMPLETED;item.reward={'score':k};item.response_ids=[2];item.response_model_steps=[lease.model_step];item.num_tokens=2;item.set_training_fields(input_ids=[1,2],labels=[-100,2],logprobs=[0.,-.2])
     lease.settlement=SettlementReceipt(True,True,'trace_committed');self.history.append((g,item.rollout_id,lease.model_step))
   return group
  finally:
   if g==3:self.cancelled.set();await self.cleanup.wait()
 async def produce(self,step,final=False):
  kw=dict(model_step=step-1,quiesce=True)
  if CANDIDATE:kw['final']=final
  return await self.m.produce_batch(1,step,**kw)
 async def close(self):
  self.cleanup.set();await self.strategy._pending_tasks.cancel_all()
  fs=getattr(self.m,'_finalizing_tasks',set())
  if fs:await asyncio.gather(*fs,return_exceptions=True)
class Tests(unittest.IsolatedAsyncioTestCase):
 async def test_native_selection_and_same_episode(self):
  f=Fixture()
  try:
   if not CANDIDATE:
    with self.assertRaises(TimeoutError):await f.produce(1)
    self.assertEqual(f.strategy.pending_task_count(),1);EVENTS.append({'baseline_real_producer_quiesce_blocked':True});return
   one=await f.produce(1);self.assertEqual(one.rollout_states[0][0].group_id,1);self.assertEqual(f.strategy.pending_task_count(),1)
   with tempfile.TemporaryDirectory() as td:await f.m.save(td,1)
   f.tool.set()
   two=await f.produce(2);self.assertEqual(two.rollout_states[0][0].group_id,2);self.assertEqual([x[2] for x in f.history if x[0]==2],[1,1])
   self.assertEqual(f.counter,2) # ready tail consumed without any new admission
   tail_group=await f.sample()
   async def spawn():
    task=asyncio.create_task(f.generate_group(tail_group),name="tool-tail:group3")
    f.strategy._pending_tasks.track_group(task,tail_group)
    return task
   await f.strategy._pending_tasks.schedule_one(max_pending=2,should_abort=lambda:False,spawn_one=spawn)
   await asyncio.sleep(0)
   group=await f.sample()
   # Fixture completed before this final window; no model call behind sealed gate.
   for k,item in enumerate(group):
    item.status=Status.COMPLETED;item.reward={'score':k};item.response_ids=[2];item.response_model_steps=[1];item.num_tokens=2;item.set_training_fields(input_ids=[1,2],labels=[-100,2],logprobs=[0.,-.2])
   await f.buffer.put(group,'task',model_step=1,current_train_step=3,stale_threshold=3)
   three=await f.produce(3,True);self.assertEqual(three.rollout_states[0][0].group_id,4)
   await f.cancelled.wait();self.assertFalse(f.cleanup.is_set());self.assertEqual(f.strategy.pending_task_count(),0)
   EVENTS.append({'selected_group_ids':[1,2,4],'history':f.history,'cancelled_tail':3,'cleanup_not_waited':True,'actual_manager_producer_buffer':True})
  finally:await f.close()
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 (Path(os.environ['RESULT_DIR'])/('FULL_CPU_'+os.environ['CASE']+'.json')).write_text(json.dumps({'passed':r.wasSuccessful(),'events':EVENTS},indent=2));raise SystemExit(not r.wasSuccessful())
