"""LMDeploy incremental R3 preserves full history, independently of partial rollout."""

import asyncio
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest

from xtuner.v1.data_proto.rl_data import RolloutState, SampleParams, Status
from xtuner.v1.rl.rollout.lmdeploy import LMDeployWorker
from xtuner.v1.rl.rollout.sglang import SGLangWorker
from xtuner.v1.rl.rollout.utils import PartialRolloutHandler, merge_incremental_routed_experts, routed_expert_row_count
from xtuner.v1.rl.rollout.worker import RolloutWorker


class _ObjectRef:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def __await__(self):
        async def resolve() -> np.ndarray:
            return self.value

        return resolve().__await__()


def _state() -> RolloutState:
    return RolloutState(
        message=[],
        prompt_ids=[1, 2],
        tokens=[1, 2],
        sample_params=SampleParams(max_tokens=8),
    )


def _worker(worker_type: type = LMDeployWorker, *, partial: bool = True, capture: bool = True):
    worker = worker_type.__new__(worker_type)
    worker.supports_routed_experts_start_len = worker_type is LMDeployWorker
    worker.enable_return_routed_experts = capture
    worker.enable_partial_rollout = partial
    worker.model_name = "test"
    worker.partial_rollout_handler = PartialRolloutHandler()
    worker.receive_abort_request = threading.Event()
    worker.logger = MagicMock()
    worker._decode_routed_experts = AsyncMock()
    return worker


def _response(prompt_tokens: int, ids: list[int], *, finish: str = "abort", routes: str | None = "store-key"):
    return httpx.Response(
        200,
        json={
            "text": "text" if ids else "",
            "output_ids": ids,
            "meta_info": {
                "finish_reason": {"type": finish},
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(ids),
                "output_token_logprobs": [[-0.1, token] for token in ids],
                "routed_experts": routes,
            },
        },
    )


class TestIncrementalRoutedExperts:
    @pytest.mark.parametrize("worker_type", [LMDeployWorker])
    def test_first_request_has_no_crop(self, worker_type: type) -> None:
        worker = _worker(worker_type)
        state, payload = worker._prepare_request_payload(_state(), 8)
        assert payload["input_ids"] == [1, 2]
        assert "routed_experts_start_len" not in payload
        assert "routed_experts_start_len" not in payload.get("sampling_params", {})
        assert worker._get_routed_experts_start_len(state) == 0

    @pytest.mark.parametrize("worker_type", [LMDeployWorker])
    @pytest.mark.parametrize("history_count", [1, 2, 5])
    def test_resume_skips_only_retained_routes(self, worker_type: type, history_count: int) -> None:
        worker = _worker(worker_type)
        state = _state()
        state.response_ids = list(range(10, 10 + history_count))
        state.routed_experts = np.zeros((2 + history_count - 1, 2, 2), dtype=np.uint16)
        state.status = Status.ABORTED
        _, payload = worker._prepare_request_payload(state, 8)
        # Keep the last generated token: it has not been processed yet.
        assert payload["routed_experts_start_len"] == len(payload["input_ids"]) - 1
        assert payload["routed_experts_start_len"] == len(state.routed_experts)

    @pytest.mark.parametrize("worker_type", [LMDeployWorker])
    @pytest.mark.parametrize("disabled", ["partial", "capture", "request", "missing_history", "no_response"])
    def test_non_continuation_does_not_crop(self, worker_type: type, disabled: str) -> None:
        worker = _worker(worker_type, partial=disabled != "partial", capture=disabled != "capture")
        state = _state()
        state.response_ids = [3, 4] if disabled != "no_response" else []
        state.routed_experts = np.zeros((3, 2, 2), dtype=np.uint16) if disabled != "missing_history" else None
        if disabled == "request":
            state.sample_params.return_routed_experts = False
        assert "routed_experts_start_len" not in worker._get_request_payload(state)

    def test_unsupported_backend_keeps_legacy_mode(self) -> None:
        worker = _worker(RolloutWorker)
        state = _state()
        state.response_ids = [3, 4]
        state.routed_experts = np.zeros((3, 2, 2), dtype=np.uint16)
        assert worker._get_routed_experts_start_len(state) == 0

    @pytest.mark.parametrize("worker_type", [LMDeployWorker])
    def test_multiple_rounds_are_byte_exact_to_legacy_and_free_refs(self, worker_type: type) -> None:
        async def run() -> None:
            worker = _worker(worker_type)
            state = _state()
            expected = None
            # Captured prefixes deliberately differ across rounds: preserve prior routes,
            # just as the legacy full-transfer + prefix-discard implementation does.
            for turn, ids in enumerate(([3, 4], [5], [6, 7])):
                _, payload = worker._prepare_request_payload(state, 8)
                prompt_len = len(payload["input_ids"])
                start = payload.get("routed_experts_start_len", 0)
                full = np.arange((prompt_len + len(ids) - 1) * 4, dtype=np.uint16).reshape(-1, 2, 2)
                full = full + 100 * turn
                history_ref = state.routed_experts
                current_ref = _ObjectRef(full[start:])
                worker._decode_routed_experts.return_value = current_ref
                free_before = free.call_count
                state = await worker._safe_handle_response(state, _response(prompt_len, list(ids)))
                if expected is None:
                    expected = full
                    assert free.call_count == free_before
                else:
                    expected = np.concatenate([expected, full[len(expected) :]], axis=0)
                    free.assert_any_call([history_ref])
                    free.assert_any_call([current_ref])
                    assert free.call_count == free_before + 2
                np.testing.assert_array_equal(state.routed_experts.value, expected)
                assert len(expected) == len(state.prompt_ids) + len(state.response_ids) - 1
            state.input_ids = [1, 2, 3, 4, 5, 6, 7]
            state.labels = [-100, -100, 3, 4, 5, 6, 7]
            state.logprobs = None

        with (
            patch("xtuner.v1.rl.rollout.worker.ray.ObjectRef", _ObjectRef),
            patch("xtuner.v1.rl.rollout.utils.RayObjectRef", _ObjectRef),
            patch("xtuner.v1.rl.rollout.utils.ray.put", side_effect=_ObjectRef),
            patch("xtuner.v1.rl.rollout.utils.free_object_refs") as free,
        ):
            asyncio.run(run())

    @pytest.mark.parametrize("has_empty_routes", [False, True])
    def test_abort_without_new_tokens_preserves_full_history(self, has_empty_routes: bool) -> None:
        async def run() -> None:
            worker = _worker()
            state = _state()
            state.response_ids = [3, 4]
            state.logprobs = [-0.1, -0.1]
            history = np.arange(12, dtype=np.uint16).reshape(3, 2, 2)
            state.routed_experts = _ObjectRef(history)
            worker._prepare_request_payload(state, 8)
            worker._decode_routed_experts.return_value = _ObjectRef(history[:0])
            result = await worker._safe_handle_response(
                state, _response(4, [], routes="store-key" if has_empty_routes else None)
            )
            np.testing.assert_array_equal(result.routed_experts.value, history)
            assert result.response_ids == [3, 4]
            assert result.status == Status.ABORTED

        with (
            patch("xtuner.v1.rl.rollout.worker.ray.ObjectRef", _ObjectRef),
            patch("xtuner.v1.rl.rollout.utils.RayObjectRef", _ObjectRef),
            patch("xtuner.v1.rl.rollout.utils.ray.put", side_effect=_ObjectRef),
            patch("xtuner.v1.rl.rollout.utils.free_object_refs"),
        ):
            asyncio.run(run())

    def test_retry_discards_history_and_restarts_full_transfer(self) -> None:
        worker = _worker()
        state = _state()
        state.response_ids = [3, 4]
        state.routed_experts = np.zeros((3, 2, 2), dtype=np.uint16)
        state.status = Status.FAILED
        state.extra_fields["_r3_retained_num_rows"] = 3
        state, payload = worker._prepare_request_payload(state, 8, discard_response=True)
        assert payload["input_ids"] == [1, 2]
        assert "routed_experts_start_len" not in payload
        assert state.routed_experts is None
        assert not state.response_ids
        assert state.sample_params.max_tokens == 8
        assert "_r3_retained_num_rows" not in state.extra_fields

    def test_old_server_ignoring_start_is_rejected_and_ref_freed(self) -> None:
        async def run() -> None:
            worker = _worker()
            state = _state()
            state.response_ids = [3, 4]
            history = _ObjectRef(np.zeros((3, 2, 2), dtype=np.uint16))
            state.routed_experts = history
            wrong = _ObjectRef(np.zeros((4, 2, 2), dtype=np.uint16))
            worker._decode_routed_experts.return_value = wrong
            result = await worker._safe_handle_response(state, _response(4, [5]))
            assert result.status == Status.FAILED
            assert "check server routed_experts_start_len support" in result.error_msg
            free.assert_called_once_with([wrong])
            assert state.routed_experts is history

        with (
            patch("xtuner.v1.rl.rollout.worker.ray.ObjectRef", _ObjectRef),
            patch("xtuner.v1.rl.rollout.utils.free_object_refs") as free,
            patch("xtuner.v1.rl.rollout.utils.RayObjectRef", _ObjectRef),
        ):
            asyncio.run(run())

    @pytest.mark.parametrize("start", [1, 4])
    def test_gap_or_arbitrary_crop_is_rejected(self, start: int) -> None:
        state = _state()
        state.response_ids = [3, 4]
        state.routed_experts = np.zeros((3, 2, 2), dtype=np.uint16)
        with pytest.raises(ValueError, match="offset does not match"):
            asyncio.run(
                PartialRolloutHandler().postprocess(
                    state,
                    response="new",
                    response_ids=[5],
                    logprobs=[-0.1],
                    routed_experts=np.zeros((1, 2, 2), dtype=np.uint16),
                    routed_experts_start_len=start,
                    finish_reason="abort",
                    status=Status.ABORTED,
                    prompt_tokens=4,
                    completion_tokens=1,
                )
            )

    def test_initial_crop_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires retained history"):
            asyncio.run(
                PartialRolloutHandler().postprocess(
                    _state(),
                    response="new",
                    response_ids=[3],
                    logprobs=[-0.1],
                    routed_experts=np.zeros((1, 2, 2), dtype=np.uint16),
                    routed_experts_start_len=1,
                    finish_reason="abort",
                    status=Status.ABORTED,
                    prompt_tokens=2,
                    completion_tokens=1,
                )
            )


class TestImageExpandedRoutedExpertsStart:
    @pytest.mark.parametrize("worker_type", [LMDeployWorker])
    @pytest.mark.parametrize("expanded_prompt_len", [18, 80, 158])
    def test_offset_uses_expanded_training_prompt(self, worker_type: type, expanded_prompt_len: int) -> None:
        worker = _worker(worker_type)
        state = _state()
        state.prompt_ids = [11, 99, 12]
        expanded = [11] + [99] * (expanded_prompt_len - 2) + [12]
        state.extra_fields = {"image_data": ["data:image/png;base64,fixture"], "train_prompt_ids": expanded}
        state.response_ids = [13, 14]
        state.routed_experts = np.zeros((expanded_prompt_len + 1, 2, 2), dtype=np.uint16)
        _, payload = worker._prepare_request_payload(state, 8)
        assert len(payload["input_ids"]) == 5
        assert payload["routed_experts_start_len"] == expanded_prompt_len + 1
        assert payload["routed_experts_start_len"] == len(state.routed_experts)
        assert payload["image_data"] == state.extra_fields["image_data"]
        assert state.extra_fields["train_prompt_ids"] == expanded


class TestNonPartialIncrementalRoutedExperts:
    @pytest.mark.parametrize("with_tool_tokens", [False, True])
    @pytest.mark.parametrize("image_token_counts", [[], [16], [256], [16, 64]])
    def test_three_turns_merge_routes_but_not_responses(
        self, with_tool_tokens: bool, image_token_counts: list[int]
    ) -> None:
        async def run():
            worker = _worker(partial=False)
            state = _state()
            if image_token_counts:
                wire, expanded = [1], [1]
                for index, count in enumerate(image_token_counts):
                    wire.extend([99, index + 2])
                    expanded.extend([99] * count + [index + 2])
                state.prompt_ids = wire
                state.tokens = list(wire)
                state.extra_fields = {
                    "train_prompt_ids": expanded,
                    "image_data": ["fixture"] * len(image_token_counts),
                }
            expansion = sum(count - 1 for count in image_token_counts)
            expected = None
            tokens = list(state.tokens)
            for turn in range(3):
                state.tokens = list(tokens)
                state.status = Status.INIT
                _, payload = worker._prepare_request_payload(state, 2)
                start = payload.get("routed_experts_start_len", 0)
                assert start == (0 if expected is None else len(expected))
                ids = [10 + turn * 2, 11 + turn * 2]
                prompt_len = len(tokens) + expansion
                full = np.arange((prompt_len + 1) * 4, dtype=np.uint16).reshape(-1, 2, 2) + turn * 100
                worker._decode_routed_experts.return_value = _ObjectRef(full[start:])
                state = await worker._safe_handle_response(state, _response(prompt_len, ids, finish="length"))
                expected = full if expected is None else np.concatenate([expected, full[len(expected) :]])
                np.testing.assert_array_equal(state.routed_experts.value, expected)
                assert state.extra_fields["_r3_retained_num_rows"] == len(expected)
                if image_token_counts:
                    assert payload["image_data"] == ["fixture"] * len(image_token_counts)
                    assert state.extra_fields["train_prompt_ids"] == expanded
                assert state.response_ids == ids  # Not cumulative: agent owns turn assembly.
                assert len(state.logprobs) == 2
                tokens.extend(ids)
                if with_tool_tokens:
                    tokens.extend([80, 81])
            assert not worker.enable_partial_rollout

        with (
            patch("xtuner.v1.rl.rollout.worker.ray.ObjectRef", _ObjectRef),
            patch("xtuner.v1.rl.rollout.utils.RayObjectRef", _ObjectRef),
            patch("xtuner.v1.rl.rollout.utils.ray.put", side_effect=_ObjectRef),
            patch("xtuner.v1.rl.rollout.utils.free_object_refs") as free,
        ):
            asyncio.run(run())
            assert free.call_count == 4

    @pytest.mark.parametrize("disabled", ["missing_history", "missing_count", "r3_disabled"])
    def test_missing_history_or_disabled_capture_does_not_crop(self, disabled: str) -> None:
        worker = _worker(partial=False)
        state = _state()
        state.tokens = [1, 2, 3, 4, 99]
        state.extra_fields["_r3_retained_num_rows"] = 3
        state.routed_experts = np.zeros((3, 2, 2), dtype=np.uint16)
        if disabled == "missing_history":
            state.routed_experts = None
        elif disabled == "missing_count":
            state.extra_fields.pop("_r3_retained_num_rows")
        else:
            state.sample_params.return_routed_experts = False
        assert worker._get_routed_experts_start_len(state) == 0

    def test_expiry_reset_invalidates_retained_routes_but_keeps_images(self) -> None:
        worker = _worker(partial=False)
        state = _state()
        state.tokens = [1, 2, 3, 4]
        state.response_ids = [3, 4]
        state.routed_experts = np.zeros((3, 2, 2), dtype=np.uint16)
        state.extra_fields = {"_r3_retained_num_rows": 3, "image_data": ["fixture"]}
        state.reset_response()  # ReplayBuffer uses this same reset for expired samples.
        assert state.tokens == state.prompt_ids
        assert state.extra_fields["image_data"] == ["fixture"]
        assert worker._get_routed_experts_start_len(state) == 0

    def test_sglang_incremental_is_not_enabled(self) -> None:
        assert not SGLangWorker.supports_routed_experts_start_len

    @pytest.mark.parametrize("has_routes", [False, True])
    def test_empty_abort_restarts_without_incremental_history(self, has_routes: bool) -> None:
        async def run():
            worker = _worker(partial=False)
            state = _state()
            state.tokens = [1, 2, 3, 4, 80, 81]
            state.extra_fields["_r3_retained_num_rows"] = 3
            state.routed_experts = _ObjectRef(np.zeros((3, 2, 2), dtype=np.uint16))
            worker._decode_routed_experts.return_value = _ObjectRef(np.zeros((0, 2, 2), dtype=np.uint16))
            state = await worker._safe_handle_response(state, _response(6, [], routes="key" if has_routes else None))
            assert state.status == Status.ABORTED
            assert state.routed_experts is None
            assert "_r3_retained_num_rows" not in state.extra_fields
            _, payload = worker._prepare_request_payload(state, 8)
            assert payload["input_ids"] == [1, 2]
            assert "routed_experts_start_len" not in payload

        with (
            patch("xtuner.v1.rl.rollout.worker.ray.ObjectRef", _ObjectRef),
            patch("xtuner.v1.rl.rollout.worker.free_object_refs") as free,
        ):
            asyncio.run(run())
            assert free.call_count == 1 + int(has_routes)


@pytest.mark.parametrize("tokens, expected", [(0, 0), (1, 0), (4, 3), (128, 127)])
def test_routed_expert_row_count(tokens, expected):
    assert routed_expert_row_count(tokens) == expected


@pytest.mark.parametrize("supported, capture", [(False, True), (False, False), (True, True)])
def test_lmdeploy_capability_is_checked_before_worker_startup(supported, capture):
    @dataclass
    class LegacyGenerationConfig:
        max_new_tokens: int = 32

    @dataclass
    class OffsetGenerationConfig:
        routed_experts_start_len: int = 0

    config = SimpleNamespace(
        enable_return_routed_experts=capture, tokenizer_path="test", api_key="test", model_name="test"
    )

    def parent_init(self, *args):
        self.config = config
        self.endpoints = {}

    generation_type = OffsetGenerationConfig if supported else LegacyGenerationConfig
    with (
        patch("lmdeploy.GenerationConfig", generation_type),
        patch.object(RolloutWorker, "__init__", parent_init),
        patch("xtuner.v1.rl.rollout.lmdeploy.AutoTokenizer.from_pretrained"),
    ):
        if capture and not supported:
            with pytest.raises(ValueError, match="R3 collection requires LMDeploy"):
                LMDeployWorker(config, 0, "127.0.0.1", 1, 1)
        else:
            worker = LMDeployWorker(config, 0, "127.0.0.1", 1, 1)
            assert worker.supports_routed_experts_start_len is supported


def test_full_transfer_passes_through_and_releases_old_history():
    old = _ObjectRef(np.zeros((3, 2, 2), dtype=np.uint16))
    current = _ObjectRef(np.ones((5, 2, 2), dtype=np.uint16))
    with (
        patch("xtuner.v1.rl.rollout.utils.RayObjectRef", _ObjectRef),
        patch("xtuner.v1.rl.rollout.utils.free_object_refs") as free,
    ):
        result = asyncio.run(merge_incremental_routed_experts(old, current, start_len=0, total_len=5))
        assert result is current
        free.assert_called_once_with([old])


def test_nonpartial_invalid_suffix_fails_only_the_sample():
    async def run():
        worker = _worker(partial=False)
        state = _state()
        state.tokens = [1, 2, 3, 4]
        state.extra_fields["_r3_retained_num_rows"] = 3
        state.routed_experts = _ObjectRef(np.zeros((3, 2, 2), dtype=np.uint16))
        worker._decode_routed_experts.return_value = _ObjectRef(np.zeros((4, 2, 2), dtype=np.uint16))
        failed = await worker._safe_handle_response(state, _response(4, [5], finish="length"))
        assert failed.status == Status.FAILED
        assert "Incremental R3 length mismatch" in failed.error_msg
        # A sibling sample still completes on the same worker.
        worker._decode_routed_experts.return_value = _ObjectRef(np.zeros((2, 2, 2), dtype=np.uint16))
        completed = await worker._safe_handle_response(_state(), _response(2, [5], finish="length"))
        assert completed.status == Status.COMPLETED

    with (
        patch("xtuner.v1.rl.rollout.worker.ray.ObjectRef", _ObjectRef),
        patch("xtuner.v1.rl.rollout.utils.RayObjectRef", _ObjectRef),
        patch("xtuner.v1.rl.rollout.utils.free_object_refs"),
    ):
        asyncio.run(run())


def test_native_generate_contains_route_failure_and_preserves_diagnostic():
    async def run():
        from xtuner.v1.utils.httpx_utils import HttpRequestResult

        worker = _worker(partial=False)
        worker.server_url = "http://test"
        worker.endpoints = {"generate": "generate"}
        worker.eos_token = []
        worker.config = SimpleNamespace(api_key="test", max_retry_per_sample=0)
        state = _state()
        state.tokens = [1, 2, 3, 4]
        state.extra_fields["_r3_retained_num_rows"] = 3
        state.routed_experts = np.zeros((3, 2, 2), dtype=np.uint16)
        worker._decode_routed_experts.return_value = _ObjectRef(np.zeros((4, 2, 2), dtype=np.uint16))
        worker._safe_post_request = AsyncMock(
            return_value=HttpRequestResult(response=_response(4, [5], finish="length"))
        )
        result = await worker.generate(state)
        assert result.status == Status.FAILED
        assert "Incremental R3 length mismatch" in result.error_msg
        assert result.routed_experts is None  # Native generate finally invalidates failed data.

    with (
        patch("xtuner.v1.rl.rollout.worker.ray.ObjectRef", _ObjectRef),
        patch("xtuner.v1.rl.rollout.utils.RayObjectRef", _ObjectRef),
        patch("xtuner.v1.rl.rollout.utils.free_object_refs"),
    ):
        asyncio.run(run())
