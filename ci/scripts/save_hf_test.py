import argparse
import time
import torch
import torch.distributed as dist
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.config import FSDPConfig

from memory_profiler import profile

MB = 1024 ** 2

def get_args():
    p = argparse.ArgumentParser("Profile build/shard/save with @profile (RSS) and simple GPU stats")
    p.add_argument("hf_path", type=str, help="HF model path")
    p.add_argument("out", type=str, help="Output HF path")
    p.add_argument("--ep", type=int, default=1, help="expert parallel size")
    return p.parse_args()

def set_device_for_rank():
    if torch.cuda.is_available():
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.cuda.set_device(rank % torch.cuda.device_count())

def gpu_mem(label):
    if not torch.cuda.is_available():
        print(f"[GPU] {label}: no CUDA")
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / MB
    reserved = torch.cuda.memory_reserved() / MB
    peak = torch.cuda.max_memory_allocated() / MB
    print(f"[GPU] {label}: alloc={alloc:.2f}MB reserved={reserved:.2f}MB peak={peak:.2f}MB")

def build_model(hf_path: str):
    cfg = get_model_config_from_hf(hf_path)
    model = cfg.build()
    return model

def shard_model(model, ep: int):
    fsdp_cfg = FSDPConfig(ep_size=ep)
    model.fully_shard(fsdp_config=fsdp_cfg)
    return model

@profile
def save_model(model, out: str):
    model.save_hf(out)

def main():
    args = get_args()

    dist.init_process_group(backend="nccl")
    set_device_for_rank()

    t0 = time.perf_counter()
    gpu_mem("init")

    torch.cuda.reset_peak_memory_stats()
    model = build_model(args.hf_path)
    gpu_mem("after_build")

    torch.cuda.reset_peak_memory_stats()
    shard_model(model, args.ep)
    gpu_mem("after_shard")

    torch.cuda.reset_peak_memory_stats()
    save_model(model, args.out)
    gpu_mem("after_save")

    print(f"[TIME] total={time.perf_counter()-t0:.3f}s")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
