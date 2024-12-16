import torch
from torch import nn


def set_require_grad_param_to_fp32(model: nn.Module):

    def traverse(module: nn.Module):

        for m_name, child in module.named_children():

            all_require_grad = True
            for p_name, param in child.named_parameters():

                if not param.requires_grad:
                    all_require_grad = False
                    break

            if all_require_grad:
                child.to(torch.float32)

            traverse(child)

    traverse(model)
