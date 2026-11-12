"""Actual imported manager/policy/gate/pending; CPU buffer/Agent/backend boundaries.
No CUDA, sandbox, tokenizer or reward-quality claim. Run once for each source.
"""
import asyncio,json,os,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import AsyncMock,patch
import xtuner.v1.rl.agent_loop_manager.agent_loop_manager as mod
from xtuner.v1.rl.agent_loop_manager.produce_utils import _PendingTasks,ProduceBatchResult,manager_state_path
from xtuner.v1.rl.agent_loop_manager.agent_turn_pause_policy import AgentTurnPausePolicy
from xtuner.v1.rl.rollout.request_gate import RequestGate,SettlementReceipt
CASE=os.environ.get('CASE','candidate');CANDIDATE=CASE=='candidate';RESULTS=[]
class Logger:
 def info(self,*a):pass
 def warning(self,*a):pass
class Remote:
 def __init__(self,fn):self.remote=fn
class Controller:
 def __init__(self):
  self.gate=RequestGate(wait_timeout=5);self.epoch=0;self.events=[]
  for name in ('prepare_pause','pause_generation','continue_generation','resume_requests','assert_request_gates_healthy'):
   setattr(self,name,Remote(getattr(self,'_'+name)))
 async def _prepare_pause(self,**kw):
  self.epoch+=1;await self.gate.seal(self.epoch);self.events.append('sealed');await self.gate.drain(kw['drain_timeout']);self.events.append('drained');return self.epoch
 async def _pause_generation(self):self.events.append('sleep_complete')
 async def _continue_generation(self):self.events.append('onload')
 async def _resume_requests(self,epoch,step,**kwargs):await self.gate.install_version(epoch,step);await self.gate.reopen(epoch);self.events.append(f'published:{step}')
 async def _assert_request_gates_healthy(self):
  if self.gate.snapshot()['failures']:raise RuntimeError('unknown gate failure')
class Harness:
 def __init__(self,policy='batch_ready'):
  self.owner=_PendingTasks();self.control=Controller();self.ready=[];self.selected=[];self.tool=asyncio.Event();self.cleanup=asyncio.Event();self.started=asyncio.Event();self.cancelled=asyncio.Event();self.cleaned=asyncio.Event();self.turns=['committed:0'];self.events=[]
  self.m=object.__new__(mod.AgentLoopManager);self.m.name='cpu';self.m.logger=Logger();self.m.task_names=['task'];self.m.replay_buffer=N();self.m._rollout_controller=self.control
  self.m._keepalive=AgentTurnPausePolicy(freeze_timeout=1,drain_timeout=.3,quiesce_timeout=.06)
  self.m._update_policy=policy;self.m._checkpoint_mode='model_export'
  async def produce(ctx):
   self.events.append('produce')
   done=await self.owner.claim_ready()
   for f in done:self.ready.append(f.result())
   if not self.ready:self.ready.append(['fresh'])
  self.strategy=N(_pending_tasks=self.owner,produce_batch=produce,pending_task_count=self.owner.count)
  self.sampler=N(save=lambda p:None,resume=lambda p:None)
  self.task=N(task_name='task',produce_strategy=self.strategy,sampler=self.sampler,agent_loop=N(),stale_threshold=2,token_stale_threshold=None,expired_groups_retryable=False)
  self.m.task_runners=[self.task]
 async def tail(self,cleanup_block=False):
  self.started.set()
  try:
   await self.tool.wait()
   async with self.control.gate.request('same-episode') as lease:
    self.turns.append(f'committed:{lease.model_step}');lease.settlement=SettlementReceipt(True,True,'completed')
   return ['same-episode']
  finally:
   self.cancelled.set()
   if cleanup_block:await self.cleanup.wait()
   self.cleaned.set()
 async def add_tail(self,cleanup_block=False):
  f=asyncio.create_task(self.tail(cleanup_block));await self.owner.schedule_one(max_pending=10,should_abort=lambda:False,spawn_one=self.spawn(f));await self.started.wait();return f
 @staticmethod
 def spawn(f):
  async def get():return f
  return get
 async def take(self,**kw):
  n=kw['task_batch_sizes']['task'];groups=self.ready[:n];del self.ready[:n];self.selected+=groups;self.events.append('selected');return ProduceBatchResult(rollout_states=groups)
 async def produce(self,step=1,final=False,quiesce=True):
  with patch.object(mod,'allocate_task_batch_sizes',return_value={'task':1}),patch.object(mod,'refresh_for_all_tasks',new=AsyncMock()),patch.object(mod,'take_train_batch',new=self.take):
   kwargs=dict(model_step=step-1,quiesce=quiesce)
   if CANDIDATE:kwargs['final']=final
   return await self.m.produce_batch(1,step,**kwargs)
 async def close(self):
  self.cleanup.set();await self.owner.cancel_all()
  extra=getattr(self.m,'_finalizing_tasks',set())
  if extra:await asyncio.gather(*extra,return_exceptions=True)
class Tests(unittest.IsolatedAsyncioTestCase):
 async def test_checkpoint_live_tail(self):
  h=Harness();f=await h.add_tail()
  try:
   if not CANDIDATE:
    with self.assertRaises(TimeoutError):await h.produce()
    self.assertFalse(f.done());RESULTS.append({'case':'checkpoint_long_tail','baseline_block_reproduced':True});return
   await h.produce();self.assertFalse(f.done());self.assertEqual(h.owner.count(),1)
   with tempfile.TemporaryDirectory() as td:
    await h.m.save(td,1);s=json.loads(manager_state_path(Path(td)).read_text());self.assertFalse(s['resumable']);self.assertFalse(s['sampler_saved'])
    with self.assertRaisesRegex(RuntimeError,'cannot restore'):await h.m.resume(td)
   h.tool.set();await asyncio.sleep(0);self.assertFalse(f.done()) # waiting at sealed gate
   await h.produce(step=2,quiesce=False)
   await asyncio.wait_for(f,1)
   # The same live coroutine, same committed history; no tool replay.
   self.assertEqual(h.turns,['committed:0','committed:1'])
   await h.produce(step=3,quiesce=False)
   self.assertIn(['same-episode'],h.selected);self.assertEqual(h.selected.count(['same-episode']),1)
   RESULTS.append({'case':'checkpoint_long_tail','same_episode_continued':True,'turns':h.turns,'events':h.control.events})
  finally:await h.close()
 async def test_final_does_not_wait_cleanup(self):
  h=Harness();f=await h.add_tail(True)
  try:
   if not CANDIDATE:
    with self.assertRaises(TimeoutError):await h.produce(final=True)
    RESULTS.append({'case':'final_tail','baseline_block_reproduced':True});return
   r=await h.produce(final=True);self.assertEqual(len(r.rollout_states),1)
   await h.cancelled.wait();self.assertFalse(h.cleaned.is_set());self.assertEqual(h.owner.count(),0)
   self.assertIn(f,h.m._finalizing_tasks);self.assertEqual(h.control.gate.snapshot()['active'],[])
   h.events.append('train_can_start');self.assertFalse(h.cleanup.is_set())
   with self.assertRaisesRegex(RuntimeError,'final batch'):await h.produce(step=2)
   RESULTS.append({'case':'final_tail','training_boundary_before_cleanup':True,'selected':r.rollout_states})
  finally:await h.close()
 async def test_incomplete_never_cancels_tail(self):
  if not CANDIDATE:return
  h=Harness();f=await h.add_tail()
  async def incomplete(**kw):return ProduceBatchResult(rollout_states=[['one']]) # target2
  try:
   with patch.object(mod,'allocate_task_batch_sizes',return_value={'task':2}),patch.object(mod,'refresh_for_all_tasks',new=AsyncMock()),patch.object(mod,'take_train_batch',new=incomplete):
    with self.assertRaisesRegex(RuntimeError,'full valid batch'):await h.m.produce_batch(2,1,model_step=0,final=True)
   self.assertFalse(f.done());self.assertEqual(h.owner.count(),1)
  finally:await h.close()
 async def test_final_waits_for_active_lease_before_cancelling_agent(self):
  if not CANDIDATE:return
  h=Harness();active=asyncio.Event();finish=asyncio.Event();cancelled=asyncio.Event()
  async def tail():
   try:
    async with h.control.gate.request('active-tail') as lease:
     active.set();await finish.wait();lease.settlement=SettlementReceipt(True,True,'trace_committed')
    await asyncio.Event().wait()
   finally:cancelled.set()
  f=asyncio.create_task(tail())
  async def spawn():return f
  await h.owner.schedule_one(max_pending=1,should_abort=lambda:False,spawn_one=spawn);await active.wait()
  produce=asyncio.create_task(h.produce(final=True))
  try:
   while 'sealed' not in h.control.events:await asyncio.sleep(0)
   self.assertFalse(produce.done());self.assertFalse(cancelled.is_set());self.assertEqual(h.owner.count(),1)
   finish.set();await asyncio.wait_for(produce,1);await cancelled.wait()
   self.assertEqual(h.control.gate.snapshot()['active'],[]);self.assertFalse(h.control.gate.snapshot()['failures'])
  finally:
   finish.set();await h.close()

 async def test_unknown_failure_still_blocks(self):
  h=Harness();h.control.gate._failures['unknown']='unsettled'
  with self.assertRaises(RuntimeError):await h.produce()
 async def test_full_mode_still_rejects_pending(self):
  h=Harness();h.m._checkpoint_mode='full';await h.add_tail()
  try:
   with tempfile.TemporaryDirectory() as td:
    with self.assertRaisesRegex(RuntimeError,'pending'):await h.m.save(td,0)
  finally:await h.close()
 async def test_default_policy_keeps_quiesce(self):
  h=Harness('quiescent');await h.add_tail()
  try:
   with self.assertRaises(TimeoutError):await h.produce()
  finally:await h.close()
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
 out=Path(os.environ['RESULT_DIR']);out.mkdir(parents=True,exist_ok=True);(out/f'CPU_{CASE}.json').write_text(json.dumps({'passed':r.wasSuccessful(),'tests':r.testsRun,'evidence':RESULTS,'source':mod.__file__},indent=2))
 raise SystemExit(not r.wasSuccessful())
