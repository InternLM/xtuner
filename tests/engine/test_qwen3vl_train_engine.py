import os
import parametrize
import torch
import torch.distributed as dist
from xtuner._testing import DeterministicDDPTestCase
from transformers import AutoTokenizer

from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.compose.qwen3_vl import Qwen3VLDense4BConfig
from xtuner.v1.model.base import ModelItem
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem
from xtuner.v1.config import FSDPConfig, LRConfig, AdamWConfig
from xtuner.v1.engine.vision_compose_train_engine import VisionComposeTrainEngine
from torch.optim.lr_scheduler import LambdaLR
from xtuner.v1.utils.device import get_device
from xtuner.v1.utils.test_utils import init_data_mesh
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig


QWEN3_VL_DENSE_PATH = os.environ["QWEN3_VL_DENSE_PATH"]
DEVICE = get_device()


class TestQwen3VLEngine(DeterministicDDPTestCase):
    @parametrize.parametrize(
        "device,tp_size,sp_size",
        [
            ("cuda", 1, 1),
            ("cuda", 1, 2),
        ],
    )
    def test_dense_engine_train(self, device, tp_size, sp_size):
        pg = self.create_pg(device)

        dense_cfg = Qwen3VLDense4BConfig(compile_cfg=False)
        optim_cfg: AdamWConfig = AdamWConfig()
        lr_cfg: LRConfig = LRConfig()
        fsdp_cfg: FSDPConfig = FSDPConfig(
            torch_compile=True,
            cpu_offload=False,
            tp_size=tp_size
        )
        engine = VisionComposeTrainEngine(
            model_cfg=dense_cfg, optim_cfg=optim_cfg, fsdp_cfg=fsdp_cfg
        )
        engine.from_hf(hf_path=QWEN3_VL_DENSE_PATH)

        total_steps = 1000
        warmup_steps = total_steps * lr_cfg.warmup_ratio

        def warmup_fn(x):
            return x / warmup_steps if x < warmup_steps else 1

        lr_scheduler = LambdaLR(engine.optimizer, warmup_fn)

        tokenizer = AutoTokenizer.from_pretrained(QWEN3_VL_DENSE_PATH)
        tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_VL_DENSE_PATH, add_vision_id=True).build(
            tokenizer)

        raw_data = {"id": 3, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {
            "url": "tests/resource/mscoco_twocat_000000039769.jpg", "image_wh": [640, 480]}}, {"type": "image_url",
                                                                                               "image_url": {
                                                                                                   "url": "tests/resource/mscoco_dog_000000319154.jpg",
                                                                                                   "image_wh": [375,
                                                                                                                500]}},
                                                                       {"type": "text",
                                                                        "text": "<IMG_CONTEXT>\n<IMG_CONTEXT>\n请描述下第二幅图片中的狗是什么颜色？"}]},
                                          {"role": "assistant", "content": "图片中的狗是棕色的。"}]}
        tokenized_data = tokenize_fn(raw_data)
        input_ids = torch.tensor(tokenized_data['input_ids'])[None].to(device)
        labels = torch.tensor(tokenized_data['labels'])[None].to(device)
        pixel_values = tokenized_data['pixel_values'].to(device)
        image_grid_thw = tokenized_data['image_grid_thw'].to(device)
        position_ids = tokenized_data['position_ids'].to(device)

        loss_cfg = CELossConfig()

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = labels[:, 1:]
        if position_ids is not None:
            position_ids = position_ids[..., :-1]

        sp_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)
            sp_mesh = data_mesh["sp"]

        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to(device),))
        seq_ctx.image_grid_thw = image_grid_thw
        seq_ctx.pixel_values = pixel_values
        if position_ids is not None:
            seq_ctx.position_ids = position_ids
        seq_ctx.to(device)
        loss_ctx_input = CELossContextInputItem(shifted_labels=shifted_labels)
        loss_ctx_input = loss_ctx_input.to(device)

        if sp_size > 1:
            seq_ctx = seq_ctx.split(sp_mesh)
            loss_ctx_input = loss_ctx_input.sp_split(sp_mesh)

        seq_ctx_list = [seq_ctx]
        loss_ctx_input_list: list[CELossContextInputItem] = [loss_ctx_input]

        LossContext = loss_cfg.loss_ctx_cls
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list,
            loss_cfg,
        )
        loss_kwargs = batches_loss_kwargs[0]
        loss_ctx = LossContext(loss_cfg, loss_kwargs)
        seq_ctx = seq_ctx_list[0]

        losses = []
        for _ in range(5):
            engine_input = [ModelItem(seq_ctx=seq_ctx, loss_ctx=loss_ctx)]
            loss_log, _ = engine.train_step(engine_input)
            grad_norm = engine.clip_grad_norm()
            engine.step_optimizer(grad_norm)
            lr_scheduler.step()
            losses.append(loss_log["reduced_llm_loss"])
        losses_ref = [2.57, 2.57, 2.58, 2.53, 2.43]
        for loss, loss_ref in zip(losses, losses_ref):
            self.assertTrue(abs(loss - loss_ref) < 0.02, f"loss={loss}, loss_ref={loss_ref}, diff={abs(loss - loss_ref)}")

        torch.cuda.empty_cache()
        try:
            dist.destroy_process_group(pg)
        except:
            pass

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))

    @property
    def destroy_pg_upon_exit(self) -> bool:
        return False
