import os

import parametrize
import torch
from torch.testing._internal.common_distributed import DistributedTestBase
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from xtuner.v1.model.interns1 import InternS1Config, InternS1VisionConfig, InternS1ProjectorConfig
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.config import FSDPConfig
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.engine.utils import cal_global_grad_tokens
from xtuner.v1.loss import CELossContext


# Intern-S1 30B A3
INTERNS1_MOE_PATH = os.environ["INTERNS1_MOE_PATH"]


class TestInternS1(DistributedTestBase):
    @parametrize.parametrize(
        "device,dispatcher,ep_size,compile,tol",
        [
            ("cuda", "deepep", 8, False, 1e-2),
            ("cuda", "all2all", 8, False, 1e-2),
            ("cuda", None, 1, False, 1e-2),
            ("cuda", "deepep", 8, True, 4e-2),  # TODO: This test is flaky, need to fix it
        ],
    )
    def test_interns1_text_run(self, device, dispatcher, ep_size, compile, tol):
        self.create_pg(device)
        if not compile:
            maybe_compile.clear_compile_targets()

        hf_config = AutoConfig.from_pretrained(
            INTERNS1_MOE_PATH,
            trust_remote_code=True,
        )
        hf_config.text_config.attn_implementation = "flash_attention_2"
        hf_config.vision_config.attn_implementation = "flash_attention_2"
        hf_model = AutoModelForCausalLM.from_pretrained(
            INTERNS1_MOE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda"
        ).eval()  # avoid open drop_path

        tokenizer = AutoTokenizer.from_pretrained(INTERNS1_MOE_PATH, trust_remote_code=True)
        input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            vision_cfg = InternS1VisionConfig()
            projector_cfg = InternS1ProjectorConfig()
            llm_cfg = Qwen3MoE30BA3Config(vocab_size=152967)
            llm_cfg.dispatcher = dispatcher
            llm_cfg.ep_size = ep_size
            model_cfg = InternS1Config(vision_config=vision_cfg, text_config=llm_cfg, projector_config=projector_cfg)
            interns1_model = model_cfg.build().to(torch.bfloat16)

        seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids, ))
        seq_ctx, shifted_labels = seq_ctx.shift_with_labels(labels=input_ids)
        seq_ctx.to('cuda')
        global_grad_tokens = cal_global_grad_tokens([shifted_labels])
        loss_ctx = CELossContext()
        loss_ctx = loss_ctx.build_forward_item(seq_ctx, shifted_labels,
                                               grad_accumulation_steps=1,
                                               global_grad_tokens=global_grad_tokens)
        interns1_model.from_hf(INTERNS1_MOE_PATH)
        interns1_model.eval()  # avoid open drop_path

        with torch.no_grad():
            output = interns1_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=tol, rtol=tol))

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", "all2all", 4),
            ("cuda", "all2all", 8),
            ("cuda", None, 1),
        ],
    )
    def test_fsdp_accuracy(self, device, dispatcher, ep_size):
        self.create_pg(device)
        maybe_compile.clear_compile_targets()
        hf_config = AutoConfig.from_pretrained(
            INTERNS1_MOE_PATH,
            trust_remote_code=True,
        )
        hf_config.text_config.attn_implementation = "flash_attention_2"
        hf_config.vision_config.attn_implementation = "flash_attention_2"
        hf_model = AutoModelForCausalLM.from_pretrained(
            INTERNS1_MOE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda"
        ).eval()  # avoid open drop_path

        tokenizer = AutoTokenizer.from_pretrained(INTERNS1_MOE_PATH, trust_remote_code=True)
        input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            output = hf_model(
                input_ids=input_ids,
                labels=input_ids.clone(),
            )
        expected_loss = output.loss

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            vision_cfg = InternS1VisionConfig()
            projector_cfg = InternS1ProjectorConfig()
            llm_cfg = Qwen3MoE30BA3Config(vocab_size=152967)
            llm_cfg.dispatcher = dispatcher
            llm_cfg.ep_size = ep_size
            model_cfg = InternS1Config(vision_config=vision_cfg, text_config=llm_cfg, projector_config=projector_cfg)
            interns1_model = model_cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            ep_size=ep_size,
            cpu_offload=False,
        )

        seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids, ))
        seq_ctx, shifted_labels = seq_ctx.shift_with_labels(labels=input_ids)
        seq_ctx.to('cuda')
        global_grad_tokens = cal_global_grad_tokens([shifted_labels])
        loss_ctx = CELossContext()
        loss_ctx = loss_ctx.build_forward_item(seq_ctx, shifted_labels,
                                               grad_accumulation_steps=1,
                                               global_grad_tokens=global_grad_tokens)

        interns1_model.language_model.fully_shard(fsdp_config=fsdp_config)
        interns1_model.vision_tower.fully_shard(fsdp_config=fsdp_config)
        interns1_model.multi_modal_projector.fully_shard(fsdp_config=fsdp_config)
        interns1_model.fully_shard(fsdp_config=fsdp_config)

        interns1_model.from_hf(INTERNS1_MOE_PATH)
        interns1_model.eval()  # avoid open drop_path

        with torch.no_grad():
            output = interns1_model(
                seq_ctx=seq_ctx,
                loss_ctx=loss_ctx,
            )
        loss = output["loss"]
        self.assertTrue(torch.allclose(loss, expected_loss.to(loss.dtype), atol=1e-2, rtol=1e-2))

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", None, 1),
            ("cuda", "all2all", 4),
            ("cuda", "all2all", 8),
        ],
    )
    def test_save_hf(self, device, dispatcher, ep_size):
        pass

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
