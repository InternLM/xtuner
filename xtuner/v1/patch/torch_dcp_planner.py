import torch.distributed.checkpoint.default_planner as torch_default_runner


def fake_validate_global_plan(*args, **kwargs):
    return True


def patch_default_save_plan():
    torch_default_runner._validate_global_plan = fake_validate_global_plan
