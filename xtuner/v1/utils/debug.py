import torch


FOUND_NAN = False


def register_grad_hook(tensor: torch.Tensor, message):
    if (grad_fn := tensor.grad_fn) is not None:
        message = f"{tensor.grad_fn}: {message}"
        grad_fn.register_hook(get_grad_hook(message))


def get_grad_hook(message: str):
    def hook(g_in: tuple, g_out: tuple):
        global FOUND_NAN
        # torch.distributed.breakpoint()
        # if torch.distributed.get_rank() == 0:
        if FOUND_NAN:
            return

        for idx, i in enumerate(g_in):
            if isinstance(i, torch.Tensor) and i.isnan().any().item():
                FOUND_NAN = True
                print(f"{message} index {idx} of g_in has nan")

        for idx, o in enumerate(g_out):
            if isinstance(o, torch.Tensor) and o.isnan().any().item():
                FOUND_NAN = True
                print(f"{message} index {idx} of g_out has nan")

    return hook
