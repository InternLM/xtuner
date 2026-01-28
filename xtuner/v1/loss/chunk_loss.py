from typing import Any, Callable

import torch
from torch.autograd import grad


class ChunkLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        head_weight: torch.Tensor,
        head_bias: torch.Tensor | None,
        loss_forward: Callable,
        loss_kwargs_chunks: list[Any],
        chunk_size: int,
    ):
        if head_bias is not None:
            raise NotImplementedError("head_bias is not supported in ChunkLoss")

        device = hidden_states.device
        accumulated_loss = torch.tensor(0.0, device=device)
        grad_inputs = torch.empty_like(hidden_states)
        grad_weight = torch.zeros_like(head_weight)

        grad_inputs_chunks = torch.split(grad_inputs, chunk_size, dim=1)
        hidden_states_chunks = torch.split(hidden_states, chunk_size, dim=1)

        from xtuner.v1.model.utils import ModelForwardExtraLogInfo

        chunked_extra_info = ModelForwardExtraLogInfo()
        for i in range(len(hidden_states_chunks)):
            hidden_states_chunk = hidden_states_chunks[i]
            grad_inputs_chunk = grad_inputs_chunks[i]

            # (chunk_grad_input, chunk_grad_weight), (chunk_loss, (_, extra_info)) = torch.func.grad_and_value(
            #     loss_forward, argnums=(0, 1), has_aux=True
            # )(hidden_states_chunk, head_weight, None, loss_kwargs_chunks[i])

            with torch.enable_grad():
                hidden_states_chunk.requires_grad_()
                chunk_loss, (_, extra_info) = loss_forward(
                    hidden_states_chunk, head_weight, None, loss_kwargs_chunks[i]
                )
                chunk_grad_input, chunk_grad_weight = grad(
                    chunk_loss, (hidden_states_chunk, head_weight), allow_unused=True
                )

            accumulated_loss.add_(chunk_loss)
            grad_inputs_chunk.copy_(chunk_grad_input)
            grad_weight.add_(chunk_grad_weight)

            chunked_extra_info.append(extra_info)

        ctx.save_for_backward(grad_inputs, grad_weight)
        return accumulated_loss, chunked_extra_info

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        if torch.ne(grad_output[0], torch.tensor(1.0, device=grad_output[0].device)):
            grad_input = grad_input * grad_output[0]
            grad_weight = grad_weight * grad_output[0]

        return grad_input, grad_weight, None, None, None, None
