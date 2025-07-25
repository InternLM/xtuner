import torch
from mmengine.dist import all_reduce

from ..data_proto.loss_context import CELossForwardItem


def _chunk_loss(hidden_states_chunk, labels_chunk, head_weight, loss_fn, **kwargs):
    logits_chunk = hidden_states_chunk @ head_weight.t()
    return loss_fn(logits_chunk.float().view(-1, logits_chunk.shape[-1]), labels_chunk.view(-1), **kwargs)


def accumulate_chunk(hidden_states_chunk, labels_chunk, head_weight, loss_fn, **kwargs):
    (chunk_grad_input, chunk_grad_weight), chunk_loss = torch.func.grad_and_value(
        _chunk_loss, argnums=(0, 2), has_aux=False
    )(hidden_states_chunk, labels_chunk, head_weight, loss_fn, **kwargs)
    return chunk_loss, chunk_grad_input, chunk_grad_weight


class ChunkCELoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, head_weight, loss_froward_item: CELossForwardItem, head_bias, loss_fn):
        chunk_size = loss_froward_item.chunk_size
        labels = loss_froward_item.labels
        loss_reduction = loss_froward_item.loss_reduction
        loss_weight = loss_froward_item.loss_weight
        grad_accumulation_steps = loss_froward_item.grad_accumulation_steps

        device = hidden_states.device
        accumulated_loss = torch.tensor(0.0, device=device)
        grad_inputs = torch.empty_like(hidden_states)
        grad_weight = torch.zeros_like(head_weight)

        grad_inputs_chunks = torch.split(grad_inputs, chunk_size, dim=1)
        hidden_states_chunks = torch.split(hidden_states, chunk_size, dim=1)
        labels_chunks = torch.split(labels, chunk_size, dim=1)

        if loss_reduction == "global":
            global_loss_weight = torch.tensor(1.0, device=device)
        else:
            global_loss_weight = loss_weight.view(-1).to(device)
            global_loss_weight = global_loss_weight.sum()
            all_reduce(global_loss_weight, op="mean")

        if loss_weight.shape != ():
            loss_weight_chunks = torch.split(loss_weight, chunk_size, dim=1)
        else:
            loss_weight_chunks = [loss_weight] * len(hidden_states_chunks)  # type: ignore[assignment]

        for i in range(len(hidden_states_chunks)):
            hidden_states_chunk = hidden_states_chunks[i]
            labels_chunk = labels_chunks[i]
            loss_weight_chunk = loss_weight_chunks[i]
            grad_inputs_chunk = grad_inputs_chunks[i]

            # TODO(HHA): 这个地方暂时不够通用，因为不同的 loss 可能有不同的 chunk 逻辑和额外参数
            chunk_loss, chunk_grad_input, chunk_grad_weight = accumulate_chunk(
                hidden_states_chunk,
                labels_chunk,
                head_weight,
                loss_fn,
                loss_reduction=loss_reduction,
                loss_weight=loss_weight_chunk,
                global_loss_weight=global_loss_weight,
                grad_accumulation_steps=grad_accumulation_steps,
            )
            accumulated_loss.add_(chunk_loss)
            grad_inputs_chunk.copy_(chunk_grad_input)
            grad_weight.add_(chunk_grad_weight)

        ctx.save_for_backward(grad_inputs, grad_weight)
        return accumulated_loss, None

    @staticmethod
    def backward(ctx, *grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        if torch.ne(grad_output[0], torch.tensor(1.0, device=grad_output[0].device)):
            grad_input = grad_input * grad_output[0]
            grad_weight = grad_weight * grad_output[0]

        return grad_input, grad_weight, None, None, None


def chunk_ce_loss(logits, labels, loss_reduction, loss_weight, global_loss_weight, grad_accumulation_steps):
    if loss_reduction == "global":
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels) * loss_weight
    else:
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, labels)

        loss = loss * loss_weight
        loss = loss.sum() / (global_loss_weight + 1e-8)
        loss = loss / grad_accumulation_steps
    return loss
