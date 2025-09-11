import torch
import torch.nn as nn
import torch.nn.functional as F
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem, CELossContext
import time


hidden_states = torch.randn(32768, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
lm_head = nn.Linear(4096, 151936, bias=False).to(device="cuda", dtype=torch.bfloat16)
torch.cuda.reset_peak_memory_stats()
t1 = time.time()
logits = lm_head(hidden_states)
shifted_labels = torch.randint(0, 151936, (32768, ), device="cuda")
loss = F.cross_entropy(logits, shifted_labels)
loss.backward()
max_memory = torch.cuda.max_memory_allocated()
reserved_memory = torch.cuda.max_memory_reserved()
print(f"Eager mode Loss: {loss.item()}")
print(f"Eager mode hidden_states grad norm: {hidden_states.grad.norm().item()}")
print(f"Eager mode lm_head weight grad norm: {lm_head.weight.grad.norm().item()}")
print(f"Eager mode Max memory allocated: {max_memory / 1024**3:.2f} GB")
print(f"Eager mode Max memory reserved: {reserved_memory / 1024**3:.2f} GB")
print(f"Eager mode Time taken: {time.time() - t1:.2f} seconds")

del logits
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

shifted_labels = shifted_labels.unsqueeze(0)
hidden_states = hidden_states.unsqueeze(0)
hidden_states = hidden_states.clone().detach().requires_grad_(True)
lm_head.weight.grad = None
t1 = time.time()
loss_ctx_input_list = [CELossContextInputItem(shifted_labels=shifted_labels)]
loss_cfg = CELossConfig(mode='chunk', chunk_size=1024, loss_reduction="token")
batches_loss_kwargs = CELossContext.build_batches_loss_kwargs(loss_ctx_input_list, loss_cfg)
loss_ctx = CELossContext(loss_cfg, batches_loss_kwargs[0])
loss, _ = loss_ctx.forward(hidden_states, lm_head.weight)
loss.backward()
max_memory = torch.cuda.max_memory_allocated()
reserved_memory = torch.cuda.max_memory_reserved()
print(f"Chunk mode Loss: {loss.item()}")
print(f"Chunk mode hidden_states grad norm: {hidden_states.grad.norm().item()}")
print(f"Chunk mode lm_head weight grad norm: {lm_head.weight.grad.norm().item()}")
print(f"Chunk mode Max memory allocated: {max_memory / 1024**3:.2f} GB")
print(f"Chunk mode Max memory reserved: {reserved_memory / 1024**3:.2f} GB")
print(f"Chunk mode Time taken: {time.time() - t1:.2f} seconds")