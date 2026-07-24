"""DenseDecoderLayer 多 micro-batch 行为测试。

TestDenseDecoderLayerMicroBatch
    test_batched_inputs_match_independent_forwards: 等长 micro-batch 的输出与梯度等价于独立调用。
"""

from copy import deepcopy

import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.attention import DSAMLAConfig
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer


def _build_dense_dsa_layer() -> DenseDecoderLayer:
    return DenseDecoderLayer(
        hidden_size=4,
        intermediate_size=8,
        hidden_act="silu",
        attention_config=DSAMLAConfig(
            num_attention_heads=2,
            head_dim=2,
            kv_lora_rank=3,
            q_lora_rank=4,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=3,
            index_topk=4,
            index_head_dim=4,
            index_n_heads=2,
            indexer_types=["full"],
            sparse_mla_backend="torch",
        ),
    )


def _build_inputs() -> tuple[
    list[torch.Tensor],
    list[tuple[torch.Tensor, torch.Tensor]],
    list[SequenceContext],
]:
    hidden_states = [
        torch.randn(1, 4, 4, requires_grad=True),
        torch.randn(1, 4, 4, requires_grad=True),
    ]
    position_embeddings = [
        (torch.ones(1, 4, 2), torch.zeros(1, 4, 2)),
        (torch.ones(1, 4, 2), torch.zeros(1, 4, 2)),
    ]
    seq_ctx = [
        SequenceContext.from_input_ids((torch.tensor([[1, 2, 3, 4]]),), device="cpu"),
        SequenceContext.from_input_ids((torch.tensor([[5, 6, 7, 8]]),), device="cpu"),
    ]
    return hidden_states, position_embeddings, seq_ctx


class TestDenseDecoderLayerMicroBatch:
    def test_batched_inputs_match_independent_forwards(self):
        # 验证一次多输入调用与逐 micro-batch 调用产生相同输出、输入梯度和参数梯度。
        torch.manual_seed(0)
        layer = _build_dense_dsa_layer()
        reference_layer = deepcopy(layer)
        hidden_states, position_embeddings, seq_ctx = _build_inputs()
        reference_hidden_states = [hidden.detach().clone().requires_grad_() for hidden in hidden_states]
        reference_seq_ctx = [
            SequenceContext.from_input_ids((ctx.input_ids.detach().clone(),), device="cpu")  # type: ignore[arg-type]
            for ctx in seq_ctx
        ]

        outputs = layer(
            *hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
        )
        reference_outputs = tuple(
            reference_layer(
                hidden,
                position_embeddings=position_embedding,
                seq_ctx=context,
            )
            for hidden, position_embedding, context in zip(
                reference_hidden_states,
                position_embeddings,
                reference_seq_ctx,
            )
        )

        assert isinstance(outputs, tuple)
        for output, reference_output in zip(outputs, reference_outputs):
            torch.testing.assert_close(output, reference_output)

        sum(output.sum() for output in outputs).backward()
        sum(output.sum() for output in reference_outputs).backward()

        for hidden, reference_hidden in zip(hidden_states, reference_hidden_states):
            torch.testing.assert_close(hidden.grad, reference_hidden.grad)
        for parameter, reference_parameter in zip(layer.parameters(), reference_layer.parameters()):
            torch.testing.assert_close(parameter.grad, reference_parameter.grad)
