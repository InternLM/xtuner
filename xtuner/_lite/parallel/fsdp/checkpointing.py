import random
from typing import Any, Tuple
import torch
import torch.nn as nn
from torch.utils.checkpoint import (
    _checkpoint_without_reentrant_generator,
    _DEFAULT_DETERMINISM_MODE,
)
from contextlib import nullcontext, contextmanager
from torch.distributed._composable.contract import contract


RECOMPUTE_MODULES = ('InternLM2DecoderLayer', 'CLIPEncoderLayer')


def checkpoint_check_fn(submodule, target=RECOMPUTE_MODULES, selective=1.0):
    ret = False
    if type(submodule).__name__ in target:
        if random.uniform(0, 1) < selective:
            ret = True
    return ret


@contextmanager
def _no_hook(module: nn.Module):
    r"""
    Disable hooks installed by checkpoint to avoid unintentional recursion
    during backward recomputation.
    """
    orig_enable_hook = checkpoint.state(module).enable_hook
    checkpoint.state(module).enable_hook = False
    try:
        yield
    finally:
        checkpoint.state(module).enable_hook = orig_enable_hook


# Support **kwargs
@contract()
def checkpoint(module: nn.Module) -> nn.Module:
    torch._C._log_api_usage_once("torch.distributed.checkpoint")

    def forward_pre_hook(module: nn.Module, *args) -> None:
        if checkpoint.state(module).enable_hook:
            def context_fns():
                return nullcontext(), _no_hook(module)

            checkpoint.state(
                module
            )._ac_generator = _checkpoint_without_reentrant_generator(
                module, True, context_fns, _DEFAULT_DETERMINISM_MODE, False, *args[0], **args[1]
            )
            next(checkpoint.state(module)._ac_generator)

    def forward_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
        if checkpoint.state(module).enable_hook:
            try:
                next(checkpoint.state(module)._ac_generator)
            except StopIteration:
                pass
            else:
                raise RuntimeError(
                    "Expected non-reentrant activation checkpoint generator to be exhausted, but it was not!"
                )

        #  Ensure that we no longer hold on to the generator. always_call=True helps ensure we
        # clear this even in the case of exception in fwd pass.
        checkpoint.state(module)._ac_generator = None

    checkpoint.state(module).enable_hook = True
    module.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)
    module.register_forward_hook(forward_hook, prepend=True, always_call=True)
    return module


if __name__ == '__main__':

    class MyModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(10, 10)
            self.l2 = nn.Linear(10, 10)

        def forward(self, x, b, a=4, c=4):
            print(b, a, c)
            return self.l2(self.l1(x))

    # from torch.distributed._composable.checkpoint_activation import checkpoint
    model = MyModel()
    checkpoint(model)  # apply activation checkpointing only to l1
    model(torch.zeros(2, 10), 2, a=5, c=6).sum().backward()
