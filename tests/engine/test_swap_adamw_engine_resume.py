"""Engine-level SwapAdamW resume tests.

Coverage:
- TestSwapAdamWEngineResume
  - test_swap_adamw_engine_resume_matches_continuous_training
    Engine-level DCP resume with swap_optimizer=True: two engines built from the
    same seeded random init, engine1 trains one step, saves DCP, trains another
    step; engine2 loads the checkpoint (restoring hyperparameters) and trains
    one step. Compares model weights and optimizer moments (values, DTensor
    layout, device, dtype/shape against the owning parameter), the step counter,
    and that optimizer states stay on pinned host memory.
"""

import os
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig, FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.model.base import ModelItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.optim.swap_adamw import SwapAdamW
from xtuner.v1.utils.device import get_device

QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]
DEVICE = get_device()
# Deterministic training data: fixed token ids and labels for every step.
INPUT_IDS = [151644, 872, 198, 113948, 99553, 11, 53982, 9370, 1773, 11, 100158, 12068, 101765, 11, 104545, 11, 104591, 4850, 102021, 11, 105074, 1773, 30748, 11, 78398, 104592, 30, 151645, 151644, 77091, 198]
SEQ_LEN = 32


def _build_engine(lr: float) -> TrainEngine:
    moe_cfg = get_model_config_from_hf(Path(QWEN3_MOE_PATH))
    moe_cfg.dispatcher = None
    moe_cfg.ep_size = 1
    moe_cfg.compile_cfg = False
    moe_cfg.num_hidden_layers = 2
    optim_cfg = AdamWConfig(lr=lr, swap_optimizer=True)
    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1)
    return TrainEngine(model_cfg=moe_cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg)


def _run_one_train_step(engine: TrainEngine) -> None:
    input_ids = torch.tensor([INPUT_IDS[:SEQ_LEN]], dtype=torch.long)
    labels = input_ids.clone()
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device=DEVICE)
    loss_cfg = CELossConfig()
    loss_ctx = loss_cfg.build(data={"shifted_labels": labels.to(DEVICE)}, sp_mesh=None)
    loss_ctx_list = loss_cfg.loss_ctx_cls.build_batches([loss_ctx])
    engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx_list[0]})]
    engine.train_step(engine_input)
    grad_norm = engine.clip_grad_norm()
    engine.step_optimizer(grad_norm)


def _state_key_name(key: str) -> bool:
    return key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "step")


def _gpu_mem(tag: str) -> None:
    # 显存观测点:打印当前已分配显存,用于观察 init/load 是否在 GPU 上多放了一份优化器状态
    if torch.cuda.is_available():
        print(
            f"[mem rank{dist.get_rank()}] {tag}: "
            f"allocated={torch.cuda.memory_allocated() / 1024**2:.1f}MB, "
            f"reserved={torch.cuda.memory_reserved() / 1024**2:.1f}MB",
            flush=True,
        )


def _assert_single_state_copy(optimizer: SwapAdamW, tag: str) -> None:
    # 行为:self.state 与 _param_to_cpu_states_map 必须指向同一块底层 buffer,
    # 否则 step 更新一套、state_dict/save_dcp 保存另一套,checkpoint 会静默失真。
    # 注意 DTensor 参数下 state 侧存 DTensor、map 侧存其 local tensor,
    # 两者是不同对象但必须共享 storage,故比较 data_ptr 而非对象身份。
    for param, cpu_state in optimizer._param_to_cpu_states_map.items():
        state_entry = optimizer.state[param]
        for key, cpu_tensor in cpu_state.items():
            if cpu_tensor is None:
                assert state_entry.get(key) is None, f"{tag}: {key} None mismatch"
                continue
            state_tensor = optimizer._to_local_tensor(state_entry[key])
            assert (
                state_tensor.data_ptr() == optimizer._to_local_tensor(cpu_tensor).data_ptr()
            ), f"{tag}: {key} storage diverged"


class TestSwapAdamWEngineResume(DeterministicDDPTestCase):
    """Engine-level resume equivalence for SwapAdamW (swap_optimizer=True)."""

    def test_swap_adamw_engine_resume_matches_continuous_training(self) -> None:
        # 行为:SwapAdamW 引擎级 DCP resume 后继续训练,数值/布局/超参与连续训练一致
        pg = self.create_pg("cuda")
        os.environ["LOCAL_RANK"] = str(dist.get_rank())
        torch.accelerator.set_device_index(int(dist.get_rank()))
        lr1, lr2 = 1e-4, 1e-3

        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            dist.broadcast_object_list(syncdir, src=0)
            weights_dir = Path(syncdir[0]) / "weights"

            torch.manual_seed(1234)
            _gpu_mem("engine1 before build")
            engine = _build_engine(lr1)
            _gpu_mem("engine1 after build")
            engine.init_model_weights()
            _gpu_mem("engine1 after init_model_weights")
            assert isinstance(engine.optimizer, SwapAdamW)

            _run_one_train_step(engine)
            dist.barrier()
            engine.save_dcp(weights_dir=weights_dir)
            dist.barrier()
            time.sleep(1)
            _run_one_train_step(engine)
            _assert_single_state_copy(engine.optimizer, "engine1 after step2")

            torch.manual_seed(1234)
            _gpu_mem("engine2 before build")
            engine2 = _build_engine(lr2)
            _gpu_mem("engine2 after build")
            engine2.init_model_weights()
            _gpu_mem("engine2 after init_model_weights")
            _gpu_mem("engine2 before load_dcp")
            engine2.load_dcp(weights_dir=weights_dir)
            _gpu_mem("engine2 after load_dcp")
            # 超参恢复:加载后 lr 应回到 engine1 保存时的值,而不是 engine2 的初始 lr2
            for group1, group2 in zip(engine.optimizer.param_groups, engine2.optimizer.param_groups):
                assert group2["lr"] == group1["lr"]
                assert group2["lr"] == lr1
            _run_one_train_step(engine2)
            # resume + 一步训练后,self.state 与 swap 别名表必须仍指向同一批 buffer
            _assert_single_state_copy(engine2.optimizer, "engine2 after step2")
            _gpu_mem("engine2 after step2")

            # optimizer state:数值、DTensor 布局、CPU/pinned、dtype/shape 须与所属参数一致
            # (先于权重比较执行:权重差异往往是 moment/step 差异的下游结果,先打印上游证据)
            opt_state = engine.optimizer.state_dict()["state"]
            opt_state2 = engine2.optimizer.state_dict()["state"]
            assert len(opt_state) == len(opt_state2) != 0
            # state_dict() 的参数 id 跨所有 group 全局连续编号,须用全局递增编号重建映射
            id_map = {
                old_id: param
                for old_id, param in enumerate(
                    p for group in engine2.optimizer.param_groups for p in group["params"]
                )
            }
            for param_id, cur_state in opt_state.items():
                cur_state2 = opt_state2[param_id]
                assert set(cur_state) == set(cur_state2)
                param = id_map[param_id]
                # amsgrad=False 时 max_exp_avg_sq 为 None 占位,只比较非 None 条目
                assert {k for k, v in cur_state.items() if v is not None} == {
                    k for k, v in cur_state2.items() if v is not None
                }
                for key, val in cur_state.items():
                    if not _state_key_name(key) or val is None:
                        continue
                    val2 = cur_state2[key]
                    if key == "step":
                        assert val2.item() == 2, "step counter must reach 2 after resume + 1 step"
                        continue
                    assert isinstance(val, DTensor) == isinstance(val2, DTensor), f"layout mismatch in {key}"
                    if isinstance(val, DTensor):
                        assert val.placements == val2.placements, f"placements mismatch in {key}"
                    local, local2 = (
                        (v._local_tensor if isinstance(v, DTensor) else v) for v in (val, val2)
                    )
                    # moment 的 dtype/shape 必须与所属参数一致,防止退化为局部 shard 或错误精度
                    assert local.dtype == param.dtype, f"{key} dtype diverges from its parameter"
                    assert val.shape == param.shape, f"{key} shape diverges from its parameter"
                    assert local2.device.type == "cpu", f"{key} must stay on host memory"
                    if torch.cuda.is_available():
                        assert local2.is_pinned(), f"{key} must stay pinned"
                    # TODO(debug): 定位 moment 恢复差异,定位后移除
                    if not torch.equal(local2, local):
                        diff = (local.float() - local2.float()).abs().max()
                        print(
                            f"[dbg rank{dist.get_rank()}] param_id={param_id} {key}: max_diff={diff}, "
                            f"dtype={local.dtype}, is_dt={isinstance(val, DTensor)}",
                            flush=True,
                        )
                    torch.testing.assert_close(local2, local, msg=f"moment mismatch in {key}")

            # 模型权重:数值、dtype、shape、device 逐参数对齐
            state_dict, state_dict2 = engine.model.state_dict(), engine2.model.state_dict()
            assert set(state_dict) == set(state_dict2)
            for key, val in state_dict.items():
                val2 = state_dict2[key]
                assert val.dtype == val2.dtype, f"dtype mismatch in {key}"
                assert val.shape == val2.shape, f"shape mismatch in {key}"
                assert isinstance(val, DTensor) == isinstance(val2, DTensor), f"layout mismatch in {key}"
                if isinstance(val, DTensor):
                    assert val.placements == val2.placements, f"placements mismatch in {key}"
                local, local2 = (
                    (v._local_tensor if isinstance(v, DTensor) else v) for v in (val, val2)
                )
                assert local.device == local2.device, f"device mismatch in {key}"
                # TODO(debug): 定位 resume 后个别权重不一致的根因,定位后移除
                if not torch.equal(local, local2):
                    diff = (local.float() - local2.float()).abs().max()
                    print(
                        f"[dbg rank{dist.get_rank()}] weight {key}: max_diff={diff}, dtype={local.dtype}, "
                        f"is_dt={isinstance(val, DTensor)}, placements={getattr(val, 'placements', None)}, "
                        f"shape={local.shape}",
                        flush=True,
                    )
                torch.testing.assert_close(local, local2, msg=f"weight mismatch in {key}")

        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "1"))

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False
