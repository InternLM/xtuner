import os
import torch
from contextlib import contextmanager
import io



def enable_full_determinism():
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    """
    #  Enable PyTorch deterministic mode. This potentially requires either the environment
    #  variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_deterministic_debug_mode(0)


class _CaptureIO(io.TextIOWrapper):
    def __init__(self) -> None:
        super().__init__(io.BytesIO(), encoding="UTF-8", newline="", write_through=True)

    def getvalue(self) -> str:
        assert isinstance(self.buffer, io.BytesIO)
        return self.buffer.getvalue().decode("UTF-8")

    def write(self, s: str) -> int:
        return super().write(s)


class LogCapture:
    def __init__(self, logger):
        self._logger = logger
        self._handle_id = None
        self._handle = _CaptureIO()


    def __enter__(self):
        self._handle_id = self._logger.add(self._handle)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._logger.remove(self._handle_id)
        except KeyboardInterrupt as e:
            raise e
        except:
            ...

    def get_output(self) -> str:
        self._handle.seek(0)
        out = self._handle.getvalue()
        self._handle.seek(0)
        self._handle.truncate(0)
        return out
