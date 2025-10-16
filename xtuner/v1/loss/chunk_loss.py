from typing import Any, Callable

import torch


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

        chunked_extra_info: dict[str, torch.Tensor] = {}
        for i in range(len(hidden_states_chunks)):
            hidden_states_chunk = hidden_states_chunks[i]
            grad_inputs_chunk = grad_inputs_chunks[i]

            (chunk_grad_input, chunk_grad_weight), (chunk_loss, (_, extra_info)) = torch.func.grad_and_value(
                loss_forward, argnums=(0, 1), has_aux=True
            )(hidden_states_chunk, head_weight, None, loss_kwargs_chunks[i])

            accumulated_loss.add_(chunk_loss)
            grad_inputs_chunk.copy_(chunk_grad_input)
            grad_weight.add_(chunk_grad_weight)

            for k, v in extra_info.items():
                if v.dim() == 0:
                    v = v.unsqueeze(0)
                # 扩充一维，对第一维进行拼接，新tensor的shape为(n, ...), 其中n为chunk的数量, ...为v原本的shape
                v = v.unsqueeze(0)
                if k in chunked_extra_info:
                    last_value = chunked_extra_info[k]
                    new_value = torch.concat([last_value, v], dim=0)
                    chunked_extra_info[k] = new_value
                else:
                    chunked_extra_info[k] = v

        ctx.save_for_backward(grad_inputs, grad_weight)
        return accumulated_loss, chunked_extra_info

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        if torch.ne(grad_output[0], torch.tensor(1.0, device=grad_output[0].device)):
            grad_input = grad_input * grad_output[0]
            grad_weight = grad_weight * grad_output[0]

        return grad_input, grad_weight, None, None, None, None
