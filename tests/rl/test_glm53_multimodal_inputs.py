"""GLM53 RL media must survive sampler, packing, materialization, and teacher construction."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import ray
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.data_proto.rl_data import RolloutState
from xtuner.v1.rl.agent_loop_manager.sampler import put_to_ray
from xtuner.v1.rl.distillation.rollout_teacher_manager import RolloutTeacherClient
from xtuner.v1.rl.distillation.train_teacher_manager import TrainTeacherManager
from xtuner.v1.rl.distillation.trainer_adapter import DistillationTrainerAdapter
from xtuner.v1.rl.rollout.worker import RolloutWorker
from xtuner.v1.rl.trainer.pack import RankPlan
from xtuner.v1.rl.trainer.train_batch import TrainBatch, get_train_seq_ctx, resolve_mm_inputs
from xtuner.v1.rl.utils.misc import validate_image_only_rollout
from xtuner.v1.train.rl_trainer import BaseRLTrainer


class TestMultimodalTrainingInputs:
    @pytest.mark.parametrize("response_length", [1, 3])
    def test_train_batch_uses_expanded_prompt_length_with_masked_response(self, response_length) -> None:
        prompt = [10, 99, 99, 11]
        response = [20, 21, 22][:response_length]
        labels = [-100] * len(prompt) + response
        if response_length == 3:
            labels[-2] = -100  # A response mask hole must not extend the prompt metadata.
        state = RolloutState(
            message=[],
            prompt_ids=[10, 99, 11],
            response_ids=response,
            extra_fields={"train_prompt_ids": prompt},
            mm_info={"mm_token_type_ids": np.array([[0, 1, 1, 0]])},
        )
        state.set_training_fields(input_ids=prompt + response, labels=labels, logprobs=None)
        batch = TrainBatch(
            [state],
            RankPlan(global_indices=(0,), steps=(((0,),),)),
            {"rollout_idx": 0, "use_3d_position_ids": False, "pack_loss_keys": [], "has_routed_experts": False},
            DistillationTrainerAdapter(None),
            pack_max_length=9,
            model_cfg=SimpleNamespace(),
            teacher_loss_keys=[],
            distillation_config=None,
        )
        batch.set_advantages([1.0])
        packed = batch.materialize((0,)).full_data
        assert packed["seq_ctx"].mm_token_type_ids.tolist() == [[0, 1, 1, 0, 0, 0, 0, 0, 0]]
        assert packed["shifted_labels"][0, : len(labels) - 1].tolist() == labels[1:]
        assert packed["advantages"][0, len(labels) - 1 :].eq(0).all()

    @pytest.mark.parametrize("response_length", [0, 1, 3])
    def test_response_and_pack_padding_are_text(self, response_length: int) -> None:
        info = {"mm_token_type_ids": np.array([[0, 1, 2, 0]])}
        ctx = get_train_seq_ctx(
            torch.arange(4 + response_length).view(1, -1), multimodal_train_info=info, prompt_length=4
        )
        pad = SequenceContext.from_input_ids((torch.zeros(1, 2, dtype=torch.long),), device="cpu")
        pad.num_padding = 2
        packed = SequenceContext.cat([ctx, pad])
        teacher = TrainTeacherManager.construct_teacher_seq_ctx(packed, is_composed=True)
        assert teacher.mm_token_type_ids.tolist() == [[0, 1, 2, 0] + [0] * (response_length + 2)]
        assert teacher.mm_token_type_ids.shape == teacher.input_ids.shape
        assert teacher.cu_seq_lens_q_cpu is packed.cu_seq_lens_q_cpu
        np.testing.assert_array_equal(info["mm_token_type_ids"], [[0, 1, 2, 0]])

    @pytest.mark.parametrize("shape", [(4,), (1, 3), (1, 5), (2, 4)])
    def test_rejects_misaligned_prompt_metadata(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="training prompt shape"):
            get_train_seq_ctx(
                torch.arange(6).view(1, -1),
                multimodal_train_info={"mm_token_type_ids": np.zeros(shape)},
                prompt_length=4,
            )

    def test_media_refs_are_shared_resolved_and_freed(self, monkeypatch) -> None:
        # Real ObjectRef values with a fake object store avoid launching a Ray cluster in a unit test.
        store = {}

        def put(value):
            ref = ray.ObjectRef.from_random()
            store[ref] = value
            return ref

        monkeypatch.setattr(ray, "put", put)
        monkeypatch.setattr(ray, "get", lambda refs: [store[ref] for ref in refs])
        image = np.full((4, 3), 11, dtype=np.float32)
        video = np.full((8, 3), 22, dtype=np.float32)
        state = RolloutState(
            message=[],
            mm_info={
                "pixel_values": image,
                "image_grid_thw": np.array([[1, 2, 2]]),
                "pixel_values_videos": video,
                "video_grid_thw": np.array([[2, 2, 2]]),
                "mm_token_type_ids": np.array([[1, 2, 2, 0]]),
                "num_img_tokens": [4, 4, 4],
            },
        )
        put_to_ray(state)
        refs = [state.mm_info[name] for name in ("pixel_values", "pixel_values_videos")]
        put_to_ray(state)
        assert len(store) == 2  # No nested ObjectRefs if the sampler is called twice.
        ctx = get_train_seq_ctx(torch.arange(4).view(1, -1), multimodal_train_info=state.mm_info)
        packed = SequenceContext.cat([ctx, ctx])
        worker_ctx = packed.copy()
        resolve_mm_inputs(worker_ctx)
        teacher = TrainTeacherManager.construct_teacher_seq_ctx(worker_ctx, is_composed=True)
        torch.testing.assert_close(teacher.pixel_values, torch.from_numpy(np.concatenate([image, image])))
        torch.testing.assert_close(teacher.pixel_values_videos, torch.from_numpy(np.concatenate([video, video])))
        assert teacher.mm_token_type_ids.tolist() == [[1, 2, 2, 0] * 2]
        assert teacher.video_grid_thw.tolist() == [[2, 2, 2], [2, 2, 2]]
        with patch("xtuner.v1.rl.utils.ray_utils.ray._private.internal_api.free") as free:
            BaseRLTrainer._release_train_batch_resources(
                SimpleNamespace(_release_trace_sessions=Mock(return_value=set())), [[state, state]]
            )
        assert set(free.call_args.args[0]) == set(refs)
        assert len(free.call_args.args[0]) == 2
        assert packed.pixel_values == [refs[0], refs[0]]

    @pytest.mark.parametrize("field", ["pixel_values", "pixel_values_videos"])
    def test_resolve_rejects_payloads_outside_ref_list_contract(self, field: str) -> None:
        # 生产契约与旧实现一致：packing 后媒体字段是 list[ObjectRef]；直收 ndarray/单 ref 一律拒绝。
        ctx = SequenceContext.from_input_ids((torch.ones(1, 2, dtype=torch.long),), device="cpu")
        setattr(ctx, field, np.ones((4, 3)))
        with pytest.raises(AssertionError, match="list of Ray ObjectRef"):
            resolve_mm_inputs(ctx)


class TestOnlineVideoGuard:
    @pytest.mark.parametrize("source", ["video_data", "video_pixels"])
    def test_rejects_video_before_backend_payload_construction(self, source: str) -> None:
        state = RolloutState(message=[])
        if source == "video_data":
            state.extra_fields["video_data"] = ["video.mp4"]
        else:
            state.mm_info = {"pixel_values_videos": np.ones((4, 3))}
        backend_payload = Mock()
        worker = SimpleNamespace(_get_request_payload=backend_payload)
        with pytest.raises(NotImplementedError, match="training replay only"):
            RolloutWorker._prepare_request_payload(worker, state, request_max_tokens=3)
        backend_payload.assert_not_called()

    def test_rollout_teacher_rejects_video_before_scoring_or_http(self) -> None:
        client = RolloutTeacherClient.__new__(RolloutTeacherClient)
        client._prepare_scoring_input = Mock()
        state = RolloutState(message=[], extra_fields={"video_data": ["video.mp4"]})
        with pytest.raises(NotImplementedError, match="training replay only"):
            asyncio.run(client.compute_logprobs(state))
        client._prepare_scoring_input.assert_not_called()

    def test_image_and_text_rollouts_are_unchanged(self) -> None:
        for info in (None, {"pixel_values": np.ones((4, 3))}):
            state = RolloutState(message=[], mm_info=info)
            validate_image_only_rollout(state)
