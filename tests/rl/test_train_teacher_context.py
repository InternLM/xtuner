"""Teacher reconstruction must preserve the composed model's visual input contract."""

from unittest.mock import Mock

import pytest
import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.rl.distillation.train_teacher_manager import TrainTeacherManager


class TestTrainTeacherContext:
    @pytest.mark.parametrize("media", ["text", "image", "video", "image_and_video"])
    @pytest.mark.parametrize("is_composed", [False, True])
    def test_teacher_preserves_or_drops_visual_fields(self, media: str, is_composed: bool) -> None:
        student = SequenceContext.from_input_ids((torch.arange(6).view(1, -1),), device="cpu")
        student.num_padding = 1
        if media != "text":
            student.position_ids = torch.arange(6).view(1, 1, 6).expand(3, 1, 6).clone()
            student.mm_token_type_ids = torch.zeros(1, 6, dtype=torch.long)
        if media in ("image", "image_and_video"):
            student.pixel_values = torch.full((4, 3), 10.0)
            student.image_grid_thw = torch.tensor([[1, 2, 2]])
            student.num_img_tokens = [[1]]
            student.mm_token_type_ids[0, 1] = 1
        if media in ("video", "image_and_video"):
            student.pixel_values_videos = torch.full((8, 3), 20.0)
            student.video_grid_thw = torch.tensor([[2, 2, 2]])
            student.mm_token_type_ids[0, 2:4] = 2
        # Teacher must recompute its own embeddings and never reuse rollout routing metadata.
        student.inputs_embeds = torch.ones(1, 6, 4)
        student.deepstack_visual_embeds = [torch.ones(1, 4)]
        student.visual_pos_masks = torch.ones(1, 6, dtype=torch.bool)
        student.rollout_routed_experts = torch.ones(1, 6, 1, dtype=torch.long)
        student.offload_rollout_routed_experts = True
        original_fields = {
            name: getattr(student, name)
            for name in (
                "pixel_values",
                "image_grid_thw",
                "pixel_values_videos",
                "video_grid_thw",
                "mm_token_type_ids",
                "num_img_tokens",
            )
        }
        original_positions = student.position_ids

        teacher = TrainTeacherManager.construct_teacher_seq_ctx(student, is_composed=is_composed)

        assert teacher is not student
        for name, value in original_fields.items():
            assert getattr(teacher, name) is (value if is_composed else None)
            assert getattr(student, name) is value
        assert student.position_ids is original_positions
        if is_composed:
            assert teacher.position_ids is original_positions
        else:
            assert teacher.position_ids.ndim == 2
            torch.testing.assert_close(teacher.position_ids, torch.arange(6).view(1, -1))
        assert teacher.input_ids is student.input_ids
        assert teacher.cu_seq_lens_q is student.cu_seq_lens_q
        assert teacher.cu_seq_lens_k is student.cu_seq_lens_k
        assert teacher.cu_seq_lens_q_cpu is student.cu_seq_lens_q_cpu
        assert teacher.num_padding == 1
        assert teacher.inputs_embeds is None
        assert teacher.deepstack_visual_embeds is None
        assert teacher.visual_pos_masks is None
        assert teacher.rollout_routed_experts is None
        assert not teacher.offload_rollout_routed_experts

    @pytest.mark.parametrize("is_composed", [False, True])
    def test_teacher_preserves_cpu_offsets_after_device_transfer(self, is_composed: bool) -> None:
        student = SequenceContext.from_input_ids((torch.arange(6).view(1, -1),), device="cpu")
        offsets_cpu = student.cu_seq_lens_q_cpu
        # Meta exercises the constructor's non-CPU branch without launching CUDA.
        # Keep explicit 2D positions so reconstruction does not read any meta scalars.
        student.to("meta")
        assert student.cu_seq_lens_q.device.type == "meta"
        assert student.cu_seq_lens_q_cpu is offsets_cpu

        teacher = TrainTeacherManager.construct_teacher_seq_ctx(student, is_composed=is_composed)

        assert teacher.cu_seq_lens_q is student.cu_seq_lens_q
        assert teacher.cu_seq_lens_k is student.cu_seq_lens_k
        assert teacher.cu_seq_lens_q_cpu is offsets_cpu
        assert teacher.cu_seq_lens_q_cpu.device.type == "cpu"
        assert teacher.position_ids is student.position_ids
        assert teacher.input_ids is student.input_ids

    def test_composed_teacher_preserves_shard_reconstruction_metadata(self) -> None:
        raw_ids = torch.arange(12).view(1, -1)
        raw_embeds = torch.ones(1, 12, 4)
        mesh = Mock()
        mesh.get_local_rank.return_value = 1
        student = SequenceContext.from_input_ids((raw_ids,), device="cpu").copy(
            input_ids=raw_ids[:, 6:],
            position_ids=torch.arange(6, 12).view(1, -1),
            sequence_parallel_mesh=mesh,
            raw_input_ids=raw_ids,
            raw_inputs_embeds=raw_embeds,
            shard_start=6,
            shard_size=6,
            mm_token_type_ids=torch.tensor([[0, 1, 0, 0, 0, 0]]),
        )

        teacher = TrainTeacherManager.construct_teacher_seq_ctx(student, is_composed=True)

        assert teacher.sequence_parallel_mesh is mesh
        assert teacher.sp_rank == 1
        assert teacher._raw_input_ids is raw_ids
        assert teacher._raw_inputs_embeds is None
        assert teacher._shard_start == 6
        assert teacher._shard_size == 6
        assert teacher.mm_token_type_ids is student.mm_token_type_ids
        assert student._raw_inputs_embeds is raw_embeds
