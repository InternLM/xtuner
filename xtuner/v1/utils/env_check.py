import os
import subprocess
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
    if os.environ.get("XTUNER_USE_LMDEPLOY", "0") == "1":
        import lmdeploy

        info = {"lmdeploy_version": str(getattr(lmdeploy, "__version__", "unknown"))}
        lmdeploy_dir = os.environ.get("LMDEPLOY_DIR")
        if not lmdeploy_dir:
            return info

        info["lmdeploy_dir"] = lmdeploy_dir
        try:
            repo_dir = subprocess.run(
                ["git", "-C", lmdeploy_dir, "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
            commit = subprocess.run(
                ["git", "-C", repo_dir, "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
            if commit:
                info["lmdeploy_commit"] = commit

            patch_parts = []
            tracked_patch = subprocess.run(
                ["git", "-C", repo_dir, "diff", "--binary", "HEAD", "--"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            if tracked_patch:
                patch_parts.append(tracked_patch)

            untracked_files = subprocess.run(
                ["git", "-C", repo_dir, "ls-files", "--others", "--exclude-standard"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.splitlines()
            for relative_path in untracked_files:
                untracked_patch = subprocess.run(
                    ["git", "-C", repo_dir, "diff", "--no-index", "--binary", "--", "/dev/null", relative_path],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                ).stdout
                if untracked_patch:
                    patch_parts.append(untracked_patch)

            if patch_parts:
                info["lmdeploy_patch"] = "\n".join(patch_parts)
        except (OSError, subprocess.SubprocessError):
            pass
        return info

    elif os.environ.get("XTUNER_USE_SGLANG", "0") == "1":
        import sglang

        version = str(getattr(sglang, "__version__", "unknown"))
        return {"sglang_version": version}
    elif os.environ.get("XTUNER_USE_VLLM", "0") == "1":
        import vllm

        version = str(getattr(vllm, "__version__", "unknown"))
        return {"vllm_version": version}
    else:
        return {}
