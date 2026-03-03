import os
import unittest
import parametrize
import torch
from packaging.version import Version
from transformers import __version__ as transformers_version
from xtuner._testing import DeterministicDDPTestCase
from transformers import AutoTokenizer
import torch.distributed as dist
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.utils.test_utils import init_data_mesh
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model.compose.qwen3_vl.modeling_vision import init_world_mesh


QWEN3_VL_MOE_PATH = os.environ["QWEN3_5_MOE_PATH"]
VIDEO_ROOT = os.environ["VIDEO_ROOT"]

@unittest.skipIf(
    Version(transformers_version) < Version("5.2.0"),
    f"transformers >= 5.2.0 is required, but got {transformers_version}"
)
class TestQwen3_5_VL(DeterministicDDPTestCase):

    def _forward(self, model, type, device, sp_size):
        if type == 'image':
            tokenizer = AutoTokenizer.from_pretrained(QWEN3_VL_MOE_PATH)
            tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_VL_MOE_PATH, add_vision_id=True).build(
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
            tokenizer = AutoTokenizer.from_pretrained(QWEN3_VL_MOE_PATH)
            tokenize_fn = Qwen3VLTokenizeFnConfig(processor_path=QWEN3_VL_MOE_PATH, rand_video_max_frames=14,
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
            rank = dist.get_rank()
            tokenizer = AutoTokenizer.from_pretrained(QWEN3_VL_MOE_PATH)
            if sp_size == 1:
                input_ids = tokenizer(f"今天天气不错，是学习的好日子。请听题： 1+1 等于多少？",
                                      return_tensors="pt").input_ids.to(device)
            else:
                input_ids = tokenizer(f"今天天气不错，是学习的好日子。请听题： 1+{rank} 等于多少？",
                                      return_tensors="pt").input_ids.to(device)
            labels = input_ids.clone()
            pixel_values = None
            image_grid_thw = None
            position_ids = None
        
        is_hf_model = isinstance(model, Qwen3_5MoeForConditionalGeneration)

        if is_hf_model:
            with torch.no_grad():
                if type == 'video':
                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                        pixel_values_videos=pixel_values,
                        video_grid_thw=image_grid_thw,
                        position_ids=position_ids,
                        use_cache = False
                    )
                else:
                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        position_ids=position_ids,
                        use_cache = False
                    )
            return output.loss
        else:
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
            if sp_size > 1:
                seq_ctx = seq_ctx.split(sp_mesh)

            seq_ctx_list = [seq_ctx]
            LossContext = loss_cfg.loss_ctx_cls
            loss_ctx = loss_cfg.build(shifted_labels=shifted_labels, sp_mesh=sp_mesh)
            loss_ctx_list = [loss_ctx]
            loss_ctx_list = LossContext.build_batches(loss_ctx_list)
            loss_ctx = loss_ctx_list[0]
            seq_ctx = seq_ctx_list[0]

            with torch.no_grad():
                output = model(
                    seq_ctx=seq_ctx,
                    loss_ctx=loss_ctx,
                )
            loss = output["loss"]
            return loss

    @parametrize.parametrize(
        "device,sp_size,tol",
        [
            ("cuda", 1, 1e-2),
        ],
    )
    def test_qwen3_5_vl_run(self, device, sp_size, tol):
        self.create_pg(device)
        
        from transformers import Qwen3_5MoeForConditionalGeneration
        
        hf_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
                    QWEN3_VL_MOE_PATH,
                    dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="cuda",
                    trust_remote_code=True
                ).eval()
        # Cannot understand, but must accept. Once there is no this code, it will appear cuda access illegal memory error in multi-GPU
        torch.distributed.barrier()

        loss_hf_text = self._forward(hf_model, type='text', device=device, sp_size=sp_size)
        loss_hf_image = self._forward(hf_model, type='image', device=device, sp_size=sp_size)
        # loss_hf_video = self._forward(hf_model, type='video', device=device, sp_size=sp_size)

        del hf_model
        torch.cuda.empty_cache()
 
        with torch.device("meta"):
            model_cfg = Qwen3_5_VLMoE35BA3Config(compile_cfg=False)
            qwen3vl_model = model_cfg.build().to(torch.bfloat16)

        qwen3vl_model.from_hf(QWEN3_VL_MOE_PATH)
        qwen3vl_model.eval()

        loss_xtuner_text = self._forward(qwen3vl_model, type='text',device=device, sp_size=sp_size)
        loss_xtuner_image = self._forward(qwen3vl_model, type='image',device=device, sp_size=sp_size)
        loss_xtuner_video = self._forward(qwen3vl_model, type='video',device=device, sp_size=sp_size)

        self.assertTrue(torch.allclose(loss_xtuner_text, loss_hf_text.to(loss_xtuner_text.dtype), atol=tol, rtol=tol))
        self.assertTrue(torch.allclose(loss_xtuner_image, loss_hf_image.to(loss_xtuner_image.dtype), atol=tol, rtol=tol))
        # self.assertTrue(torch.allclose(loss_xtuner_video, loss_hf_video.to(loss_xtuner_video.dtype), atol=tol, rtol=tol))
        
        del qwen3vl_model
        torch.cuda.empty_cache()

        # test fsdp
        with torch.device("meta"):
            model_cfg = Qwen3_5_VLMoE35BA3Config(compile_cfg=False)
            qwen3vl_model = model_cfg.build().to(torch.bfloat16)
        
        fsdp_config = FSDPConfig(cpu_offload=False)
        fsdp_mesh = init_world_mesh()
        qwen3vl_model.vision_tower.fsdp_mesh = fsdp_mesh
        qwen3vl_model.vision_tower.fsdp_config = fsdp_config
        qwen3vl_model.fully_shard(fsdp_config=fsdp_config)
        qwen3vl_model.from_hf(QWEN3_VL_MOE_PATH)
        qwen3vl_model.eval()
        
        loss_xtuner_text_fsdp = self._forward(qwen3vl_model, type='text',device=device, sp_size=sp_size)
        loss_xtuner_image_fsdp = self._forward(qwen3vl_model, type='image',device=device, sp_size=sp_size)
        loss_xtuner_video_fsdp = self._forward(qwen3vl_model, type='video',device=device, sp_size=sp_size)
        self.assertTrue(torch.allclose(loss_xtuner_text_fsdp, loss_xtuner_text, atol=tol, rtol=tol))
        self.assertTrue(torch.allclose(loss_xtuner_image_fsdp, loss_xtuner_image, atol=tol, rtol=tol))
        self.assertTrue(torch.allclose(loss_xtuner_video_fsdp, loss_xtuner_video, atol=tol, rtol=tol))

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "4"))
