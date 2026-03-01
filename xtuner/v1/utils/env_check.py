from typing import Any, Callable, List

def check_torch_accelerator_available():
    """Check if PyTorch is installed and the torch accelerator is available.

    Returns:
        bool: True if PyTorch is installed and the torch accelerator is available, False otherwise.
    """
    try:
        import torch

        return torch.accelerator.is_available()
    except Exception:
        return False

def check_triton_available():
    """Check if Triton is installed.

    Returns:
        bool: True if Triton is installed, False otherwise.
    """
    import os

    if os.environ.get("XTUNER_USE_TRITON", "1") == "0":
        return False

    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False

def get_env_not_available_func(env_name_list: List[str]) -> Callable:
    """Get a function that raises an error indicating the environment is not
    available.

    Returns:
        function: A function that raises a RuntimeError when called.
    """

    def env_not_available_func(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"{' or '.join(env_name_list)} in environment is not available.")

    return env_not_available_func

def get_rollout_engine_version() -> dict:
    import os

    if os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "1":
        import lmdeploy

        version = lmdeploy.__version__
        return {"lmdeploy_version": version}
    elif os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
        import sglang

        version = sglang.__version__
        return {"sglang_version": version}
    elif os.environ.get("XTUNER_USE_VLLM", "0") == "1":
        import vllm

        version = vllm.__version__
        return {"vllm_version": version}
    else:
        return {}
