"""torchrun --standalone --nproc-per-node 4 ... --backend gloo --sp 2.
For manual GPU validation: eight workers, --backend nccl --sp 2 or --sp 4.
Exercises the exact media protocol used by the model, without loading a model.
"""
import argparse
from datetime import timedelta
import os
from types import SimpleNamespace
import torch
import torch.distributed as dist
from xtuner.v1.train import lazy_media_prefetch as m
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['gloo','nccl'],default='gloo');p.add_argument('--sp',type=int,default=2);a=p.parse_args()
rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE'])
assert world % a.sp == 0 and world > a.sp
if a.backend=='nccl':
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
else: device=torch.device('cpu')
dist.init_process_group(a.backend,timeout=timedelta(seconds=60))
group=None
for start in range(0,world,a.sp):
    g=dist.new_group(list(range(start,start+a.sp)),timeout=timedelta(seconds=60))
    if start<=rank<start+a.sp: group=g
media_dtype=torch.bfloat16 if os.environ.get('TEST_BF16') == '1' else torch.float32
leader=dist.get_rank(group)==0; gid=rank//a.sp; calls=[]
def decode(plan):
    calls.append(plan['value'])
    if plan.get('fail'): raise ValueError('injected decode failure')
    return torch.full((4,3),float(plan['value']),dtype=media_dtype),torch.tensor([[1,2,2]])
m.materialize_image_plan=decode
def raw():
    for step in range(4):
        # Different groups consume different modalities in the same step.
        plan=None if (step+gid)%2 else {'images':[{'grid':[1,2,2]}],'value':gid*10+step,'media_dtype':media_dtype}
        yield [{'seq_ctx':SimpleNamespace(media_plan=plan)}]
try:
    for step,batch in enumerate(m.lookahead_iter(raw(),decode_here=leader,count=4)):
        plan=batch[0]['seq_ctx'].media_plan
        if plan is not None:
            pixels,grid=m.shared_materialize(plan,group,device,3)
            assert torch.equal(pixels.cpu(),torch.full((4,3),float(gid*10+step)))
            assert grid.cpu().tolist()==[[1,2,2]]
    assert len(calls)==(2 if leader else 0)
    plan={'images':[{'grid':[1,2,2]}],'value':0,'fail':True,'media_dtype':media_dtype}
    m.kickoff_lazy_media([{'seq_ctx':SimpleNamespace(media_plan=plan)}],decode_here=leader)
    try: m.shared_materialize(plan,group,device,3)
    except RuntimeError as e: assert 'failed' in str(e)
    else: raise AssertionError('decode failure was swallowed')
    dist.barrier()
    if rank==0: print('MULTI-SP PROTOCOL PASS',flush=True)
finally: dist.destroy_process_group()
