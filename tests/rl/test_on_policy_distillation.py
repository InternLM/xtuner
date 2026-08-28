import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import torch

from recipe.on_policy_distillation.build_teacher_server_commands import (
    build_teacher_launch_server_commands,
)
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.rl.distillation import RolloutTeacherClient, RolloutTeacherConfig
from xtuner.v1.rl.loss import DistillationLossConfig
from xtuner.v1.rl.trainer.controller import TrainingController


class TestDistillationRecipeConfig(unittest.TestCase):
    def test_teacher_launcher_reads_distillation_config(self) -> None:
        config_source = """
from xtuner.v1.rl.distillation import (
    DistillationConfig,
    RolloutTeacherConfig,
    RolloutTeacherLaunchConfig,
)
from xtuner.v1.rl.loss import DistillationLossConfig

loss_cfg = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
distillation_config = DistillationConfig(
    loss_config=loss_cfg,
    teachers=[
        RolloutTeacherConfig(
            name="teacher",
            launch_config=RolloutTeacherLaunchConfig(
                model_path="/models/teacher",
                num_workers=1,
                server_port=13141,
            ),
        )
    ],
    data_source_teacher_map={"math": "teacher"},
)
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "distillation_config.py"
            config_path.write_text(config_source)
            with patch.dict(
                "os.environ",
                {
                    "NODE_COUNT": "1",
                    "NODE_RANK": "0",
                    "PROC_PER_NODE": "2",
                    "WORKER_ALL_SOCKET_ADDRS": "127.0.0.1",
                },
            ):
                endpoint_map, student_num_workers, student_local_num_workers, records = (
                    build_teacher_launch_server_commands(str(config_path), "lmdeploy")
                )

        self.assertEqual(endpoint_map, {"teacher": ["http://127.0.0.1:13141"]})
        self.assertEqual(student_num_workers, 1)
        self.assertEqual(student_local_num_workers, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][:4], ["teacher[0]", "0", "1", "http://127.0.0.1:13141"])
        self.assertIn("lmdeploy", records[0][7:])


class TestRolloutTeacherClient(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _response(payload: dict) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("POST", "http://teacher/generate"),
            json=payload,
        )

    def _build_topk_client(self, *, max_retry_per_sample: int = 0) -> RolloutTeacherClient:
        loss_config = DistillationLossConfig(
            policy_loss_cfg={"loss_type": "vanilla"},
            loss_mode="forward_kl_topk",
            use_policy_gradient=False,
            top_k=2,
        )
        with patch.dict(
            "os.environ",
            {"XTUNER_USE_LMDEPLOY": "1", "XTUNER_USE_SGLANG": "0", "XTUNER_USE_VLLM": "0"},
        ):
            client = RolloutTeacherClient(
                RolloutTeacherConfig(
                    name="teacher",
                    endpoints=["http://teacher"],
                    max_retry_per_sample=max_retry_per_sample,
                ),
                loss_config,
            )
        self.addAsyncCleanup(client._client.aclose)
        return client

    @staticmethod
    def _state() -> RolloutState:
        return RolloutState(
            group_id=1,
            message=[],
            prompt_ids=[10, 11, 12],
            response_ids=[13, 14],
            status=Status.COMPLETED,
            extra_fields={"origin_data_source": "math"},
        )

    async def test_compute_sampled_token_logprobs_uses_current_interface(self) -> None:
        response = httpx.Response(
            200,
            request=httpx.Request("POST", "http://teacher/generate"),
            json={
                "meta_info": {
                    "prompt_tokens": 5,
                    "input_token_logprobs": [
                        [-0.1, 11],
                        [-0.2, 12],
                        [-0.3, 13],
                        [-0.4, 14],
                    ],
                }
            },
        )
        loss_config = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        with patch.dict(
            "os.environ",
            {"XTUNER_USE_LMDEPLOY": "1", "XTUNER_USE_SGLANG": "0", "XTUNER_USE_VLLM": "0"},
        ):
            client = RolloutTeacherClient(
                RolloutTeacherConfig(name="teacher", endpoints=["http://teacher"]),
                loss_config,
            )
        self.addAsyncCleanup(client._client.aclose)
        client._client.post = AsyncMock(return_value=response)
        state = RolloutState(
            group_id=1,
            message=[],
            prompt_ids=[10, 11, 12],
            response_ids=[13, 14],
            status=Status.COMPLETED,
            extra_fields={"origin_data_source": "math"},
        )

        result = await client.compute_logprobs(state)

        self.assertEqual(result.status, Status.COMPLETED)
        self.assertEqual(result.teacher_tokens, [13, 14])
        self.assertEqual(result.teacher_logprobs, [-0.3, -0.4])
        self.assertIn("teacher_score_time_s", result.extra_fields)

    async def test_malformed_sampled_response_becomes_failed_state(self) -> None:
        response = self._response(
            {
                "meta_info": {
                    "input_token_logprobs": [
                        [-0.1, 11],
                        [-0.2, 12],
                        [-0.3, 13],
                        [-0.4, 14],
                    ]
                }
            }
        )
        loss_config = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        with patch.dict(
            "os.environ",
            {"XTUNER_USE_LMDEPLOY": "1", "XTUNER_USE_SGLANG": "0", "XTUNER_USE_VLLM": "0"},
        ):
            client = RolloutTeacherClient(
                RolloutTeacherConfig(
                    name="teacher",
                    endpoints=["http://teacher"],
                    max_retry_per_sample=0,
                ),
                loss_config,
            )
        self.addAsyncCleanup(client._client.aclose)
        client._client.post = AsyncMock(return_value=response)

        result = await client.compute_logprobs(self._state())

        self.assertEqual(result.status, Status.FAILED)
        self.assertIn("prompt_tokens", result.error_msg or "")

    async def test_sampled_response_rejects_non_numeric_logprob(self) -> None:
        response = self._response(
            {
                "meta_info": {
                    "prompt_tokens": 5,
                    "input_token_logprobs": [
                        [-0.1, 11],
                        [-0.2, 12],
                        [True, 13],
                        [-0.4, 14],
                    ],
                }
            }
        )
        loss_config = DistillationLossConfig(policy_loss_cfg={"loss_type": "vanilla"})
        with patch.dict(
            "os.environ",
            {"XTUNER_USE_LMDEPLOY": "1", "XTUNER_USE_SGLANG": "0", "XTUNER_USE_VLLM": "0"},
        ):
            client = RolloutTeacherClient(
                RolloutTeacherConfig(
                    name="teacher",
                    endpoints=["http://teacher"],
                    max_retry_per_sample=0,
                ),
                loss_config,
            )
        self.addAsyncCleanup(client._client.aclose)
        client._client.post = AsyncMock(return_value=response)

        result = await client.compute_logprobs(self._state())

        self.assertEqual(result.status, Status.FAILED)
        self.assertIn("non-numeric logprob", result.error_msg or "")

    async def test_malformed_topk_responses_become_failed_states(self) -> None:
        valid_rows = [
            [[-0.1, 1], [-0.2, 2]],
            [[-0.3, 3], [-0.4, 4]],
            [[-0.5, 5], [-0.6, 6]],
            [[-0.7, 7], [-0.8, 8]],
        ]
        malformed_payloads = {
            "missing_meta_info": {},
            "missing_topk_field": {"meta_info": {"prompt_tokens": 5}},
            "wrong_topk_type": {"meta_info": {"prompt_tokens": 5, "input_top_logprobs": "invalid"}},
            "wrong_row_count": {"meta_info": {"prompt_tokens": 5, "input_top_logprobs": valid_rows[:-1]}},
            "ragged_k": {"meta_info": {"prompt_tokens": 5, "input_top_logprobs": [*valid_rows[:-1], [[-0.7, 7]]]}},
            "invalid_token_id": {
                "meta_info": {
                    "prompt_tokens": 5,
                    "input_top_logprobs": [*valid_rows[:-1], [[-0.7, "7"], [-0.8, 8]]],
                }
            },
        }

        for case_name, payload in malformed_payloads.items():
            with self.subTest(case=case_name):
                client = self._build_topk_client()
                client._client.post = AsyncMock(return_value=self._response(payload))

                result = await client.compute_logprobs(self._state())

                self.assertEqual(result.status, Status.FAILED)
                self.assertIsNone(result.teacher_tokens)
                self.assertIsNone(result.teacher_logprobs)
                self.assertIn("last_error=", result.error_msg or "")
                client._client.post.assert_awaited_once()

        client = self._build_topk_client()
        non_finite_response = httpx.Response(
            200,
            request=httpx.Request("POST", "http://teacher/generate"),
            headers={"content-type": "application/json"},
            content=(
                b'{"meta_info":{"prompt_tokens":5,"input_top_logprobs":'
                b"[[[-0.1,1],[-0.2,2]],[[-0.3,3],[-0.4,4]],"
                b"[[0.5,5],[-0.6,6]],[[NaN,7],[-0.8,8]]]}}"
            ),
        )
        client._client.post = AsyncMock(return_value=non_finite_response)

        result = await client.compute_logprobs(self._state())

        self.assertEqual(result.status, Status.FAILED)
        self.assertIn("NaN or Inf", result.error_msg or "")

    async def test_invalid_topk_response_is_retried_before_success(self) -> None:
        invalid_response = self._response({"meta_info": {"prompt_tokens": 5}})
        valid_response = self._response(
            {
                "meta_info": {
                    "prompt_tokens": 5,
                    "input_top_logprobs": [
                        [[-0.1, 1], [-0.2, 2]],
                        [[-0.3, 3], [-0.4, 4]],
                        [[-0.5, 5], [-0.6, 6]],
                        [[-0.7, 7], [-0.8, 8]],
                    ],
                }
            }
        )
        client = self._build_topk_client(max_retry_per_sample=1)
        client._client.post = AsyncMock(side_effect=[invalid_response, valid_response])

        with patch("xtuner.v1.rl.distillation.rollout_teacher_client.asyncio.sleep", new=AsyncMock()):
            result = await client.compute_logprobs(self._state())

        self.assertEqual(result.status, Status.COMPLETED)
        self.assertEqual(result.teacher_tokens, [[5, 6], [7, 8]])
        self.assertEqual(result.teacher_logprobs, [[-0.5, -0.6], [-0.7, -0.8]])
        self.assertEqual(client._client.post.await_count, 2)
        request_payload = client._client.post.await_args.kwargs["json"]
        self.assertEqual(request_payload["input_ids"], [10, 11, 12, 13, 14])
        self.assertEqual(request_payload["top_logprobs_num"], 2)


class TestTopKTrainingController(unittest.TestCase):
    def test_packs_targets_along_sequence_dimension(self) -> None:
        controller = TrainingController(workers=[])
        first = {
            "seq_ctx": SequenceContext.from_input_ids((torch.tensor([[1, 2]]),), device="cpu"),
            "shifted_labels": torch.tensor([[-100, 2]]),
            "advantage": [0.0, 0.0],
            "rollout_logprobs": torch.zeros(1, 2),
            "teacher_logprobs": torch.tensor([[[-0.1, -0.2], [-0.3, -0.4]]]),
            "target_token_ids": torch.tensor([[[1, 2], [3, 4]]]),
        }
        second = {
            "seq_ctx": SequenceContext.from_input_ids((torch.tensor([[3]]),), device="cpu"),
            "shifted_labels": torch.tensor([[3]]),
            "advantage": [0.0],
            "rollout_logprobs": torch.zeros(1, 1),
            "teacher_logprobs": torch.tensor([[[-0.5, -0.6]]]),
            "target_token_ids": torch.tensor([[[5, 6]]]),
        }

        packed = controller._packing([first, second], pack_max_length=4, language_cfg=None)

        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0]["teacher_logprobs"].shape, (1, 4, 2))
        self.assertEqual(packed[0]["target_token_ids"].shape, (1, 4, 2))
        torch.testing.assert_close(packed[0]["teacher_logprobs"][0, 3], torch.zeros(2))
        torch.testing.assert_close(packed[0]["target_token_ids"][0, 3], torch.zeros(2, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
