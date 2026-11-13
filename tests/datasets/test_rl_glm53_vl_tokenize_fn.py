"""Real-processor RL prompt production, including image/video modality IDs."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import ray
from PIL import Image

from transformers import AutoTokenizer
from xtuner.v1.data_proto.rl_data import Status
from xtuner.v1.datasets.rl_tokenize_fn import RLGlm53VLTokenizeFnConfig
from xtuner.v1.model.compose.glm53 import Glm53BaseConfig
from xtuner.v1.rl.agent_loop.agent_loop import AgentLoop
from xtuner.v1.rl.agent_loop_manager.sampler import put_to_ray
from xtuner.v1.rl.distillation import DistillationConfig, DistillationTrainerAdapter, TrainTeacherConfig
from xtuner.v1.rl.distillation.train_teacher_manager import TrainTeacherManager
from xtuner.v1.rl.loss import DistillationLossConfig
from xtuner.v1.rl.trainer.controller import TrainingController
from xtuner.v1.rl.trainer.pack import RankPlan
from xtuner.v1.rl.trainer.train_batch import TrainBatch, resolve_mm_inputs


@pytest.fixture(scope="module")
def tokenizer():
    model_path = os.environ.get("GLM_5_3_FLASH_PATH")
    if not model_path:
        pytest.skip("Set GLM_5_3_FLASH_PATH to test the real GLM53 processor")
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


class TestRLGlm53VLTokenize:
    @pytest.mark.parametrize("media", ["text", "image", "video"])
    def test_real_producer_to_teacher(self, media: str, tokenizer, tmp_path: Path, monkeypatch) -> None:
        image_path = tmp_path / "image.png"
        Image.new("RGB", (112, 112), (20, 50, 80)).save(image_path)
        content = [{"type": "text", "text": "Describe this."}]
        if media == "image":
            content.insert(0, {"type": "image", "image": {"url": image_path.name, "image_wh": [112, 112]}})
        elif media == "video":
            frames = tmp_path / "frames"
            frames.mkdir()
            for i in range(4):
                Image.new("RGB", (112, 112), (i * 30, 20, 50)).save(frames / f"{i:08d}.jpg")
            content.insert(
                0,
                {
                    "type": "video",
                    "video": {"url": "frames/", "image_wh": [112, 112], "origin_video_length": 4, "origin_fps": 2.0},
                },
            )
        item = {"prompt": [{"role": "user", "content": content}], "data_source": "vl", "extra_info": {"tag": 1}}
        original = copy.deepcopy(item)
        fn = RLGlm53VLTokenizeFnConfig(processor_path=os.environ["GLM_5_3_FLASH_PATH"]).build(tokenizer)
        fn.set_state("cache")
        cached = fn(item, media_root=str(tmp_path))
        fn.set_state("runtime")
        state = fn(item, media_root=str(tmp_path))
        assert item == original
        prompt = state.extra_fields["train_prompt_ids"]
        assert state.num_tokens == cached["num_tokens"] == len(prompt)
        mm_ids = state.mm_info["mm_token_type_ids"]
        assert mm_ids.shape == (1, len(prompt))
        expected_modality = {"text": 0, "image": 1, "video": 2}[media]
        assert int(mm_ids.max()) == expected_modality
        if media != "text":
            grid = state.mm_info["image_grid_thw" if media == "image" else "video_grid_thw"]
            assert (mm_ids == expected_modality).sum() == np.prod(grid, axis=1).sum() // fn.merge_unit
        state.status = Status.COMPLETED
        state.response = "fixed response"
        state.response_ids = [10, 11, 12]
        state.logprobs = [-0.1] * 3
        state.reward = {"score": 1.0}
        loop = SimpleNamespace(logger=Mock(), tokenizer=tokenizer)
        assert AgentLoop.canonicalize_train_fields(loop, [state]) == [state]
        assert state.status == Status.COMPLETED
        assert state.train_meta.prompt_length == len(prompt)
        assert state.num_tokens == len(prompt) + 2
        state.extra_fields["origin_data_source"] = "vl"
        adapter = DistillationTrainerAdapter(
            DistillationConfig(
                loss_config=DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"}),
                teachers=[
                    TrainTeacherConfig(
                        name="teacher",
                        model_path=os.environ["GLM_5_3_FLASH_PATH"],
                        model_cfg=Glm53BaseConfig.from_hf(os.environ["GLM_5_3_FLASH_PATH"]),
                    )
                ],
                data_source_teacher_map={"vl": "teacher"},
            )
        )
        # Real ObjectRefs with an in-memory store exercise the current worker materialization path.
        store = {}

        def put(value):
            ref = ray.ObjectRef.from_random()
            store[ref] = value
            return ref

        monkeypatch.setattr(ray, "put", put)
        monkeypatch.setattr(ray, "get", lambda refs: [store[ref] for ref in refs])
        put_to_ray(state)
        prepared = TrainingController([], distillation=adapter)._prepare_rollout_items([[state]], rollout_idx=0)
        batch = TrainBatch(
            prepared["rollout_items"],
            RankPlan(global_indices=(0,), steps=(((0,),),)),
            prepared["batch_attr"],
            adapter,
            pack_max_length=len(prompt) + 5,
            model_cfg=SimpleNamespace(),
            teacher_loss_keys=["teacher_indices"],
            distillation_config=None,
        )
        packed_batch = batch.materialize((0,)).full_data
        packed = packed_batch["seq_ctx"]
        assert packed.input_ids.tolist() == [prompt + [10, 11, 0, 0, 0]]
        resolve_mm_inputs(packed)
        # Each worker stage must be able to re-materialize the same pristine source refs.
        repeated = batch.materialize((0,)).full_data["seq_ctx"]
        resolve_mm_inputs(repeated)
        for field in ("pixel_values", "pixel_values_videos"):
            values = getattr(packed, field)
            if values is not None:
                np.testing.assert_array_equal(getattr(repeated, field).numpy(), values.numpy())
        assert packed_batch["teacher_indices"][0, -3:].tolist() == [-1] * 3
        teacher = TrainTeacherManager.construct_teacher_seq_ctx(packed, is_composed=True)
        assert teacher.mm_token_type_ids.shape == teacher.input_ids.shape
        np.testing.assert_array_equal(teacher.mm_token_type_ids[0, : len(prompt)], mm_ids[0])
        assert teacher.mm_token_type_ids[0, -5:].tolist() == [0] * 5
        if media == "image":
            assert state.prompt_ids.count(fn.processor.image_token_id) == 1
            assert state.extra_fields["image_data"] == [str(image_path)]
        if media == "video":
            assert state.prompt_ids.count(fn.processor.video_token_id) == 1
            assert state.extra_fields["video_data"] == [str(tmp_path / "frames") + "/"]

    def test_cache_filters_full_prompt_instead_of_truncating(self, tokenizer) -> None:
        config = RLGlm53VLTokenizeFnConfig(processor_path=os.environ["GLM_5_3_FLASH_PATH"], max_length=2)
        fn = config.build(tokenizer)
        item = {"prompt": [{"role": "user", "content": "A long enough prompt."}], "data_source": "text"}
        fn.set_state("cache")
        assert fn(item)["num_tokens"] == 0
        fn.set_state("runtime")
        with pytest.raises(ValueError, match="exceeds max_length"):
            fn(item)
