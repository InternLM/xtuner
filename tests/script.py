import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func

print(flash_attn_varlen_func)  # 不应该是 None
