import os
from packaging import version
import parametrize
import torch
from xtuner._testing import patch_hf_rms_norm, DeterministicDDPTestCase
from transformers import AutoTokenizer, AutoModelForImageTextToText
import torch.distributed as dist
import tempfile
from pathlib import Path
import json
from safetensors import safe_open
from xtuner.v1.model import Qwen3VLMoE30BA3Config, Qwen3VLDense4BConfig
from xtuner.v1.model.compose.qwen3_vl.modeling_vision import init_world_mesh
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.config import FSDPConfig
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.utils.test_utils import init_data_mesh
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)

QWEN3_VL_MOE_PATH = os.environ["QWEN3_VL_MOE_PATH"]
QWEN3_VL_DENSE_PATH = os.environ["QWEN3_VL_DENSE_PATH"]
VIDEO_ROOT = os.environ["VIDEO_ROOT"]


class TestQwen3VL(DeterministicDDPTestCase):

    # 在没有 sp 情况下，可以实现和 hf loss 完全一致
    # 在开启 sp 后，纯文本的 loss 会差 0.01，非常不合理。其余模态 loss 完全对齐
    # TODO(hha) 可能是一个隐患，后续要排查
    def _test_all(self, hf_model, qwen3vl_model, type, device, sp_size, tol):
        if type == 'image':
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
            input_ids = torch.tensor(tokenized_data['input_ids'])[None].cuda()
            labels = torch.tensor(tokenized_data['labels'])[None].cuda()
            pixel_values = tokenized_data['pixel_values'].cuda()
            image_grid_thw = tokenized_data['image_grid_thw'].cuda()
            position_ids = tokenized_data['position_ids'].cuda()
        elif type == 'video':
            tokenizer = AutoTokenizer.from_pretrained(QWEN3_VL_DENSE_PATH)
            tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_VL_DENSE_PATH, rand_video_max_frames=14,
                                                  add_vision_id=True).build(tokenizer)

            raw_data = {"id": 9, "messages": [{"role": "user", "content": [{"type": "video_url",
                                                                            "video_url": {"url": "tennis_frames_4fps/",
                                                                                          "image_wh": [1280, 720],
                                                                                          "origin_video_length": 182,
                                                                                          "origin_fps": 30.0,
                                                                                          "processed_video_length": 23,
                                                                                          "processed_fps": 4}},
                                                                           {"type": "video_url",
                                                                            "video_url": {"url": "tennis_frames_2fps/",
                                                                                          "image_wh": [1280, 720],
                                                                                          "origin_video_length": 182,
                                                                                          "origin_fps": 30.0,
                                                                                          "processed_video_length": 13,
                                                                                          "processed_fps": 2}},
                                                                           {"type": "text",
                                                                            "text": "<VIDEO_CONTEXT><VIDEO_CONTEXT>两个视频中都在做什么？"}]},
                                              {"role": "assistant", "content": "打网球"}]}

            tokenized_data = tokenize_fn(raw_data, media_root=VIDEO_ROOT)
            input_ids = torch.tensor(tokenized_data['input_ids'])[None].cuda()
            labels = torch.tensor(tokenized_data['labels'])[None].cuda()
            pixel_values = tokenized_data['pixel_values'].cuda()
            image_grid_thw = tokenized_data['image_grid_thw'].cuda()
            position_ids = tokenized_data['position_ids'].cuda()
        else:
            tokenizer = AutoTokenizer.from_pretrained(QWEN3_VL_DENSE_PATH)
            input_ids = tokenizer(f"今天天气不错，是学习的好日子。请听题： 1+1 等于多少？",
                                  return_tensors="pt").input_ids.to(device)
            labels = input_ids.clone()
            pixel_values = None
            image_grid_thw = None
            position_ids = None

        hf_model.to(device)
        with torch.no_grad():
            if type == 'video':
                output = hf_model(
                    input_ids=input_ids,
                    labels=labels,
                    pixel_values_videos=pixel_values,
                    video_grid_thw=image_grid_thw,
                    position_ids=position_ids,
                )
            else:
                output = hf_model(
                    input_ids=input_ids,
                    labels=labels,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    position_ids=position_ids,
                )
        expected_loss = output.loss
        dist.all_reduce(expected_loss.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)

        hf_model.to('cpu')
        torch.cuda.empty_cache()

        loss_cfg = CELossConfig()

        shift_input_ids = input_ids[:, :-1]
        shifted_labels = labels[:, 1:]
        if position_ids is not None:
            position_ids = position_ids[..., :-1]

        sp_mesh = None
        if sp_size > 1:
            data_mesh = init_data_mesh(device, sp_size=sp_size)
            sp_mesh = data_mesh["sp"]

        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
        seq_ctx.image_grid_thw = image_grid_thw
        seq_ctx.pixel_values = pixel_values
        if position_ids is not None:
            seq_ctx.position_ids = position_ids
        seq_ctx.to('cuda')
        loss_ctx_input = CELossContextInputItem(shifted_labels=shifted_labels)
        loss_ctx_input = loss_ctx_input.to('cuda')

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

        qwen3vl_model.to(device)
        with torch.no_grad():
            output = qwen3vl_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        qwen3vl_model.to('cpu')
        torch.cuda.empty_cache()
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device,sp_size,tol",
        [
            ("cuda", 1, 1e-2),
            ("cuda", 2, 1e-2)
        ],
    )
    def test_qwen3vl_run(self, device, sp_size, tol):
        self.create_pg(device)
        maybe_compile.clear_compile_targets()

        hf_model = AutoModelForImageTextToText.from_pretrained(
            QWEN3_VL_DENSE_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu"
        ).eval()
        patch_hf_rms_norm(hf_model)

        with torch.device("meta"):
            model_cfg = Qwen3VLDense4BConfig()
            qwen3vl_model = model_cfg.build().to(torch.bfloat16)

        qwen3vl_model.from_hf(QWEN3_VL_DENSE_PATH)
        qwen3vl_model.eval()
        qwen3vl_model.to('cpu')

        self._test_all(hf_model, qwen3vl_model, 'text', device, sp_size, tol)
        self._test_all(hf_model, qwen3vl_model, 'image', device, sp_size, tol)
        self._test_all(hf_model, qwen3vl_model, 'video', device, sp_size, tol)

    @parametrize.parametrize(
        "device,sp_size,compile, tol",
        [
            ("cuda", 1, False, 1e-2),
            ("cuda", 2, False, 1e-2)
        ],
    )
    def test_fsdp_qwen3_run(self, device, sp_size, compile, tol):
        self.create_pg(device)
        if compile is False:
            maybe_compile.clear_compile_targets()

        hf_model = AutoModelForImageTextToText.from_pretrained(
            QWEN3_VL_DENSE_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu"
        ).eval()
        patch_hf_rms_norm(hf_model)

        with torch.device("meta"):
            model_cfg = Qwen3VLDense4BConfig()
            qwen3vl_model = model_cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            cpu_offload=False,
            torch_compile=compile
        )

        qwen3vl_model.language_model.fully_shard(fsdp_config=fsdp_config)

        # 非常神奇，一旦开了这个，image 和 video 的单测就过不了。
        # qwen3vl_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
        # 将整个 vit 打包为一个大的 FSDP module 就完全一致。实际跑发现对每一层进行 FSDP 切分会导致每次计算有细微差异
        fsdp_mesh = init_world_mesh()
        mp_policy = MixedPrecisionPolicy(param_dtype=fsdp_config.param_dtype)
        fully_shard(
            qwen3vl_model.vision_tower,
            mesh=fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True
        )
        qwen3vl_model.vision_tower.fsdp_mesh = fsdp_mesh
        qwen3vl_model.vision_tower.fsdp_config = fsdp_config

        qwen3vl_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
        qwen3vl_model.fully_shard(fsdp_config=fsdp_config)

        qwen3vl_model.from_hf(QWEN3_VL_DENSE_PATH)
        qwen3vl_model.eval()
        qwen3vl_model.to('cpu')
        self._test_all(hf_model, qwen3vl_model, 'text', device, sp_size, tol)
        self._test_all(hf_model, qwen3vl_model, 'image', device, sp_size, tol)
        self._test_all(hf_model, qwen3vl_model, 'video', device, sp_size, tol)

    @parametrize.parametrize(
        "device,tp_size",
        [
            ("cuda", 1),
        ],
    )
    def test_save_hf(self, device, tp_size):
        self.create_pg(device)
        with torch.device("meta"):
            model_cfg = Qwen3VLMoE30BA3Config()
            qwen3vl_model = model_cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            tp_size=tp_size,
            cpu_offload=False,
        )

        cache_save_fh = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            dist.broadcast_object_list(syncdir, src=0)
            tmpdir = Path(syncdir[0])
            qwen3vl_model.language_model.fully_shard(fsdp_config=fsdp_config)
            qwen3vl_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
            qwen3vl_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
            qwen3vl_model.fully_shard(fsdp_config=fsdp_config)
            qwen3vl_model.from_hf(QWEN3_VL_MOE_PATH)
            qwen3vl_model.save_hf(tmpdir)

            origin_hf_path = Path(QWEN3_VL_MOE_PATH)
            origin_index_path = origin_hf_path / "model.safetensors.index.json"
            saved_index_path = tmpdir / "model.safetensors.index.json"

            # Test saved hf tensor value match the origin hf tensor value
            if dist.get_rank() == 0:
                with open(origin_index_path, "r") as f:
                    origin_index = json.load(f)
                with open(saved_index_path, "r") as f:
                    saved_index = json.load(f)

                for key in origin_index["weight_map"].keys():
                    origin_safetensor_name = origin_index["weight_map"][key]
                    saved_safetensor_name = saved_index["weight_map"][key]

                    origin_sf_fh_name = str(origin_hf_path / origin_safetensor_name)
                    expected_sf_fh_name = str(tmpdir / saved_safetensor_name)

                    if origin_safetensor_name not in cache_save_fh:
                        cache_save_fh[origin_safetensor_name] = safe_open(origin_sf_fh_name, framework="pt")
                    if saved_safetensor_name not in cache_save_fh:
                        cache_save_fh[saved_safetensor_name] = safe_open(expected_sf_fh_name, framework="pt")

                    origin_fh = cache_save_fh[origin_safetensor_name]
                    saved_fh = cache_save_fh[saved_safetensor_name]

                    origin_tensor = origin_fh.get_tensor(key)
                    saved_tensor = saved_fh.get_tensor(key)
                    self.assertTrue(torch.equal(origin_tensor, saved_tensor))

                # Test the tensor number in safetensors match the tensor number in model index
                safetensor_keys = []
                for safetensor_path in tmpdir.glob("*.safetensors"):
                    fh = cache_save_fh[safetensor_path.name]
                    safetensor_keys.extend(fh.keys())
                    safetensor_keys.sort()
                model_index_keys = list(saved_index["weight_map"].keys())
                model_index_keys.sort()

                self.assertListEqual(safetensor_keys, model_index_keys)
        dist.barrier()

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
