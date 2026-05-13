from contextlib import ExitStack
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import FSDPConfig
from xtuner.v1.model import Qwen3VLMoE30BA3Config


QWEN3_VL_MOE_PATH = os.environ["QWEN3_VL_MOE_PATH"]


def _set_hf_for_compose_submodules(model, hf_path: str | Path) -> None:
    model.set_hf(hf_path)
    for submodule_name in ("language_model", "vision_tower", "multi_modal_projector"):
        submodule = getattr(model, submodule_name, None)
        if submodule is not None and hasattr(submodule, "set_hf"):
            submodule.set_hf(hf_path)


def _load_weight_map(hf_dir: Path) -> dict[str, str]:
    with open(hf_dir / "model.safetensors.index.json", "r") as f:
        return json.load(f)["weight_map"]


def _load_safetensors_keys(hf_dir: Path) -> list[str]:
    keys: list[str] = []
    for safetensors_path in hf_dir.glob("*.safetensors"):
        with safe_open(str(safetensors_path), framework="pt") as f:
            keys.extend(f.keys())
    return sorted(keys)


def _assert_hf_tensor_equal(test_case, origin_hf_dir: Path, saved_hf_dir: Path) -> None:
    origin_weight_map = _load_weight_map(origin_hf_dir)
    saved_weight_map = _load_weight_map(saved_hf_dir)
    file_handles = {}
    with ExitStack() as stack:
        for key, origin_safetensor_name in origin_weight_map.items():
            saved_safetensor_name = saved_weight_map[key]

            if origin_safetensor_name not in file_handles:
                file_handles[origin_safetensor_name] = stack.enter_context(
                    safe_open(str(origin_hf_dir / origin_safetensor_name), framework="pt")
                )
            if saved_safetensor_name not in file_handles:
                file_handles[saved_safetensor_name] = stack.enter_context(
                    safe_open(str(saved_hf_dir / saved_safetensor_name), framework="pt")
                )

            origin_tensor = file_handles[origin_safetensor_name].get_tensor(key)
            saved_tensor = file_handles[saved_safetensor_name].get_tensor(key)
            test_case.assertTrue(torch.equal(origin_tensor, saved_tensor), f"tensor {key} is not equal")


class TestQwen3VLAsyncHF(DeterministicDDPTestCase):
    def _build_sharded_model(self, ep_size: int):
        with torch.device("meta"):
            model_cfg = Qwen3VLMoE30BA3Config(compile_cfg=False)
            model_cfg.text_config.ep_size = ep_size
            model = model_cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

        fsdp_config = FSDPConfig(
            torch_compile=False,
            cpu_offload=False,
            ep_size=ep_size,
        )
        model.fully_shard(fsdp_config=fsdp_config)
        return model

    def _assert_async_hf_structure(self, origin_hf_dir: Path, async_hf_dir: Path) -> None:
        self.assertTrue(async_hf_dir.exists())
        self.assertTrue((async_hf_dir / "model.safetensors.index.json").exists())

        origin_weight_map = _load_weight_map(origin_hf_dir)
        async_weight_map = _load_weight_map(async_hf_dir)

        self.assertListEqual(sorted(origin_weight_map), sorted(async_weight_map))

        async_files = set(async_weight_map.values())
        self.assertTrue(any(name.startswith("model-language-") for name in async_files))
        self.assertTrue(any(name.startswith("model-vision-") for name in async_files))
        self.assertTrue(any(name.startswith("model-projector-") for name in async_files))

        self.assertListEqual(sorted(async_weight_map), _load_safetensors_keys(async_hf_dir))

        unexpected_async_files = [
            path
            for path in async_hf_dir.iterdir()
            if path.name.endswith(".incomplete")
            or path.name.startswith("async-hf-writer-status-")
            or path.name.startswith(".async-hf-save-file-")
            or path.name.startswith(".async-hf-cleanup-")
        ]
        self.assertListEqual(unexpected_async_files, [])

    def test_base_compose_model_async_save_hf(self):
        self.create_pg("cuda")
        if self.world_size < 8:
            raise unittest.SkipTest("Qwen3-VL MoE 30B async HF save requires 8 GPUs in this test.")

        ep_size = int(os.environ.get("XTUNER_TEST_EP_SIZE", str(self.world_size)))
        if ep_size > self.world_size:
            raise RuntimeError(f"XTUNER_TEST_EP_SIZE={ep_size} exceeds world_size={self.world_size}")

        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            if self.world_size > 1:
                dist.broadcast_object_list(syncdir, src=0)
            tmpdir = Path(syncdir[0])
            async_hf_dir = tmpdir / "async-hf"
            origin_hf_dir = Path(QWEN3_VL_MOE_PATH)

            model = self._build_sharded_model(ep_size)
            _set_hf_for_compose_submodules(model, QWEN3_VL_MOE_PATH)

            finalized = model.async_save_hf(async_hf_dir)
            self.assertIsNone(finalized)
            finalized = model.wait_async_hf()
            self.assertEqual(finalized, async_hf_dir)

            dist.barrier()

            if dist.get_rank() == 0:
                self._assert_async_hf_structure(origin_hf_dir, async_hf_dir)

            dist.barrier()
            del model
            torch.cuda.empty_cache()

    def test_base_compose_model_async_save_hf_value_equal(self):
        self.create_pg("cuda")
        if self.world_size < 8:
            raise unittest.SkipTest("Qwen3-VL MoE 30B async HF save requires 8 GPUs in this test.")

        ep_size = int(os.environ.get("XTUNER_TEST_EP_SIZE", str(self.world_size)))
        if ep_size > self.world_size:
            raise RuntimeError(f"XTUNER_TEST_EP_SIZE={ep_size} exceeds world_size={self.world_size}")

        with tempfile.TemporaryDirectory() as tmpdir:
            syncdir = [tmpdir]
            if self.world_size > 1:
                dist.broadcast_object_list(syncdir, src=0)
            tmpdir = Path(syncdir[0])
            async_hf_dir = tmpdir / "async-hf-value-equal"
            origin_hf_dir = Path(QWEN3_VL_MOE_PATH)

            model = self._build_sharded_model(ep_size)
            model.from_hf(QWEN3_VL_MOE_PATH)

            finalized = model.async_save_hf(async_hf_dir)
            self.assertIsNone(finalized)
            finalized = model.wait_async_hf()
            self.assertEqual(finalized, async_hf_dir)

            dist.barrier()

            if dist.get_rank() == 0:
                self._assert_async_hf_structure(origin_hf_dir, async_hf_dir)
                _assert_hf_tensor_equal(self, origin_hf_dir, async_hf_dir)

            dist.barrier()
            del model
            torch.cuda.empty_cache()

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "8"))
