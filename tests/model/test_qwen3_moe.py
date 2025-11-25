import os
import json

import parametrize
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import tempfile
from pathlib import Path
from safetensors import safe_open

from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.model.moe.moe import SequenceContext
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.config import FSDPConfig
from xtuner.v1.utils.compile import maybe_compile
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem
from xtuner._testing import patch_hf_rms_norm, DeterministicDDPTestCase
from xtuner.v1.model import get_model_config_from_hf, Qwen3MoEConfig


# Qwen3 30B A3
QWEN3_MOE_PATH = os.environ["QWEN3_MOE_PATH"]
QWEN3_MOE_FOPE_PATH = os.environ["QWEN3_MOE_FOPE_PATH"]


class TestQwen3MoE(DeterministicDDPTestCase):
    def prepare(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir.cleanup()

    @parametrize.parametrize(
        "device,dispatcher,ep_size,compile,tol,loss_class",
        [
            # ("cuda", "deepep", 8, False, 1e-2, "cross_entropy"),
            ("cuda", "all2all", 8, False, 1e-2, "cross_entropy"),
            ("cuda", None, 1, False, 1e-2, "cross_entropy"),
            # ("cuda", "deepep", 8, True, 4e-2, "cross_entropy"),  # TODO: This test is flaky, need to fix it
            ("cuda", None, 1, False, 1e-2, "chunk_cross_entropy"),
        ],
    )
    def test_qwen3_moe_run(self, device, dispatcher, ep_size, compile, tol, loss_class):
        os.environ["TRITON_CACHE_DIR"] = str(Path(self.temp_dir.name) / "triton_cache")
        self.create_pg(device)
        if not compile:
            maybe_compile.clear_compile_targets()

        hf_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MOE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda"
        )
        patch_hf_rms_norm(hf_model)

        text_list = [
            "数据应该像山间的清泉，自然地流向它该去的地方",
            "当异常来临时，就像秋风中飘落的叶子， 应该被温柔地接住，而不是粗暴地丢弃",
            "当函数被调用时，它应该像春天的第一缕阳光，温柔地唤醒沉睡的数据结构",
            "就像老树拥抱归巢的鸟儿，内存管理应该给予每个对象足够的安全感",
        ]
        expected_losses = []
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MOE_PATH, trust_remote_code=True)
        for text in text_list:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                output = hf_model(
                    input_ids=input_ids,
                    labels=input_ids.clone(),
                )
            expected_loss = output.loss
            expected_losses.append(expected_loss)

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            cfg = Qwen3MoE30BA3Config()
            cfg.dispatcher = dispatcher
            cfg.ep_size = ep_size
            qwen_model = cfg.build().to(torch.bfloat16)
        qwen_model.from_hf(QWEN3_MOE_PATH)

        losses = []

        for text in text_list:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            shift_input_ids = input_ids[:, :-1]
            shifted_labels = input_ids[:, 1:]
            seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
            loss_cfg = CELossConfig()
            seq_ctx_list = [seq_ctx]
            loss_ctx_input_list: list[CELossContextInputItem] = [CELossContextInputItem(shifted_labels=shifted_labels)]
            LossContext = loss_cfg.loss_ctx_cls
            batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
                loss_ctx_input_list, 
                loss_cfg,
            )
            loss_kwargs = batches_loss_kwargs[0]
            loss_ctx = LossContext(loss_cfg, loss_kwargs)
            seq_ctx = seq_ctx_list[0]

            with torch.no_grad():
                output = qwen_model(
                    seq_ctx=seq_ctx,
                    loss_ctx=loss_ctx,
                )
            loss = output["loss"]
            losses.append(loss)
        self._check_loss_curve( losses=torch.tensor(losses), losses_ref=torch.tensor(expected_losses), sim_tol=tol, rtol=tol,)

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

        hf_model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MOE_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda"
        )
        patch_hf_rms_norm(hf_model)

        text_list = [
            "数据应该像山间的清泉，自然地流向它该去的地方",
            "当异常来临时，就像秋风中飘落的叶子， 应该被温柔地接住，而不是粗暴地丢弃",
            "当函数被调用时，它应该像春天的第一缕阳光，温柔地唤醒沉睡的数据结构",
            "就像老树拥抱归巢的鸟儿，内存管理应该给予每个对象足够的安全感",
        ]
        expected_losses = []
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MOE_PATH, trust_remote_code=True)
        for text in text_list:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                output = hf_model(
                    input_ids=input_ids,
                    labels=input_ids.clone(),
                )
            expected_loss = output.loss
            expected_losses.append(expected_loss)

        del hf_model
        torch.cuda.empty_cache()

        with torch.device("meta"):
            cfg = Qwen3MoE30BA3Config()
            cfg.ep_size = ep_size
            cfg.dispatcher = dispatcher
            qwen_model = cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            ep_size=ep_size,
            cpu_offload=False,
        )
        qwen_model.fully_shard(fsdp_config=fsdp_config)
        qwen_model.from_hf(QWEN3_MOE_PATH)

        losses = []

        for text in text_list:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            shift_input_ids = input_ids[:, :-1]
            shifted_labels = input_ids[:, 1:]
            seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
            loss_cfg = CELossConfig()
            seq_ctx_list = [seq_ctx]
            loss_ctx_input_list: list[CELossContextInputItem] = [CELossContextInputItem(shifted_labels=shifted_labels)]
            LossContext = loss_cfg.loss_ctx_cls
            batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
                loss_ctx_input_list,
                loss_cfg,
            )
            loss_kwargs = batches_loss_kwargs[0]
            loss_ctx = LossContext(loss_cfg, loss_kwargs)
            seq_ctx = seq_ctx_list[0]

            with torch.no_grad():
                output = qwen_model(
                    seq_ctx=seq_ctx,
                    loss_ctx=loss_ctx,
                )
            loss = output["loss"]
            losses.append(loss)

        self._check_loss_curve(losses=torch.tensor(losses), losses_ref=torch.tensor(expected_losses), sim_tol=1e-2, rtol=1e-2)

    @parametrize.parametrize(
        "use_sliding_window, max_window_layers, sliding_window",
        [
            (False, 6, 1024),
            (True, 6, 1024),
            (True, 4, 2048),
        ],
    )
    def test_sliding_windows(self, use_sliding_window, max_window_layers, sliding_window):
        self.create_pg('cuda')
        with torch.device("meta"):
            num_hidden_layers = 6
            attention = MHAConfig(num_attention_heads=32,
                                  num_key_value_heads=4,
                                  head_dim=128,
                                  qk_norm=True,
                                  sliding_window=sliding_window)
            cfg = Qwen3MoE30BA3Config(num_hidden_layers=num_hidden_layers,
                                      use_sliding_window=use_sliding_window,
                                      max_window_layers=max_window_layers,
                                      attention=attention)
            qwen_model = cfg.build().to(torch.bfloat16)
        loss_cfg = CELossConfig()

        if use_sliding_window is False or max_window_layers >= num_hidden_layers:
            expected_sliding_window_size_list = [(-1, -1) for _ in range(num_hidden_layers)]
        else:
            expected_sliding_window_size_list = [(-1, -1) for _ in range(max_window_layers)]
            expected_sliding_window_size_list += [(sliding_window, sliding_window) for _ in
                                                  range(num_hidden_layers - max_window_layers)]

        model_sliding_window_size_list = []
        for layer in qwen_model.layers.values():
            model_sliding_window_size_list.append(layer.self_attn.window_size)

        self.assertListEqual(model_sliding_window_size_list, expected_sliding_window_size_list)

        # test forward
        if use_sliding_window is True:
            with torch.device("meta"):
                num_hidden_layers = 6
                attention = MHAConfig(num_attention_heads=32,
                                      num_key_value_heads=4,
                                      head_dim=128,
                                      qk_norm=True,
                                      sliding_window=sliding_window)
                cfg = Qwen3MoE30BA3Config(num_hidden_layers=num_hidden_layers,
                                          use_sliding_window=use_sliding_window,
                                          max_window_layers=max_window_layers,
                                          attention=attention)
                qwen_model = cfg.build().to(torch.bfloat16)

            fsdp_config = FSDPConfig()
            tokenizer = AutoTokenizer.from_pretrained(QWEN3_MOE_PATH, trust_remote_code=True)
            input_ids = tokenizer("吃葡萄不吐葡萄皮", return_tensors="pt").input_ids.to("cuda")
            shift_input_ids = input_ids[:, :-1]
            shifted_labels = input_ids[:, 1:]
            seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids.to('cuda'),))
            seq_ctx_list = [seq_ctx]
            loss_ctx_input_list: list[CELossContextInputItem] = [CELossContextInputItem(shifted_labels=shifted_labels)]
            LossContext = loss_cfg.loss_ctx_cls
            batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
                loss_ctx_input_list, 
                loss_cfg,
            )
            loss_kwargs = batches_loss_kwargs[0]
            loss_ctx = LossContext(loss_cfg, loss_kwargs)
            seq_ctx = seq_ctx_list[0]
            qwen_model.fully_shard(fsdp_config=fsdp_config)
            qwen_model.from_hf(QWEN3_MOE_PATH, strict=False)

            with torch.no_grad():
                output = qwen_model(
                    seq_ctx=seq_ctx,
                    loss_ctx=loss_ctx,
                )
            assert "loss" in output

    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", None, 1),
            ("cuda", "all2all", 4),
            ("cuda", "all2all", 8),
        ],
    )
    def test_save_hf(self, device, dispatcher, ep_size):
        self.create_pg(device)
        with torch.device("meta"):
            cfg = Qwen3MoE30BA3Config()
            cfg.dispatcher = dispatcher
            cfg.ep_size = ep_size
            qwen_model = cfg.build().to(torch.bfloat16)

        fsdp_config = FSDPConfig(
            ep_size=ep_size,
            cpu_offload=False,
        )

        cache_save_fh = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            dist.broadcast_object_list(syncdir, src=0)
            tmpdir = Path(syncdir[0])
            qwen_model.fully_shard(fsdp_config=fsdp_config)
            qwen_model.from_hf(QWEN3_MOE_PATH)
            qwen_model.save_hf(tmpdir)

            origin_hf_path = Path(QWEN3_MOE_PATH)
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
    
    @parametrize.parametrize(
        "device,dispatcher,ep_size",
        [
            ("cuda", None, 1),
            ("cuda", "all2all", 4),
            ("cuda", "all2all", 8),
        ],
    )
    def test_save_hf_fope(self, device, dispatcher, ep_size):
        self.create_pg(device)
        with tempfile.TemporaryDirectory() as tmpdir:
            load_from = Path(QWEN3_MOE_FOPE_PATH)
            # 1. create 
            qwen_model_fope = create_model_from_hf(load_from, dispatcher, ep_size)
            # 2. operate 
            syncdir = [tmpdir]
            dist.broadcast_object_list(syncdir, src=0)
            tmpdir = Path(syncdir[0])
            qwen_model_fope.save_hf(tmpdir)
            # 3. check
            origin_index_path = load_from / "model.safetensors.index.json"
            saved_index_path = tmpdir / "model.safetensors.index.json"

            if dist.get_rank() == 0:
                with open(origin_index_path, "r") as f:
                    origin_index = json.load(f)
                with open(saved_index_path, "r") as f:
                    saved_index = json.load(f)
                # check rotary_emb.sin_coef and rotary_emb.cos_coef are saved
                assert 'model.rotary_emb.sin_coef' in saved_index["weight_map"]
                assert 'model.rotary_emb.cos_coef' in saved_index["weight_map"]

                # check all saved tensors equal to the origin tensors
                assert len(origin_index["weight_map"]) == len(saved_index["weight_map"])
                cache_save_fh = {}
                for key in origin_index["weight_map"].keys():
                    origin_safetensor_name = origin_index["weight_map"][key]
                    saved_safetensor_name = saved_index["weight_map"][key]

                    origin_sf_fh_name = str(load_from / origin_safetensor_name)
                    expected_sf_fh_name = str(tmpdir / saved_safetensor_name)

                    if origin_safetensor_name not in cache_save_fh:
                        cache_save_fh[origin_safetensor_name] = safe_open(origin_sf_fh_name, framework="pt")
                    if saved_safetensor_name not in cache_save_fh:
                        cache_save_fh[saved_safetensor_name] = safe_open(expected_sf_fh_name, framework="pt")

                    origin_fh = cache_save_fh[origin_safetensor_name]
                    saved_fh = cache_save_fh[saved_safetensor_name]

                    origin_tensor = origin_fh.get_tensor(key)
                    saved_tensor = saved_fh.get_tensor(key)
                    self.assertTrue(torch.equal(origin_tensor, saved_tensor), f"tensor {key} is not equal")

                # check the tensors saved in *.safetensors match the tensors in model_index.json
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


def create_model_from_hf(load_from: Path, dispatcher: str, ep_size: int):
    with torch.device("meta"):
        cfg : Qwen3MoEConfig = get_model_config_from_hf(load_from)
        cfg.dispatcher = dispatcher
        cfg.ep_size = ep_size
        qwen_model = cfg.build()

    fsdp_config = FSDPConfig(
        ep_size=ep_size,
        cpu_offload=False,
    )
    qwen_model.fully_shard(fsdp_config=fsdp_config)
    qwen_model.from_hf(load_from)
    return qwen_model
