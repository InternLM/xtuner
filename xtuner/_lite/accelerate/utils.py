import time
from contextlib import contextmanager
from transformers.utils.import_utils import is_flash_attn_2_available
from xtuner._lite import get_device, get_logger, get_torch_device_module

logger = get_logger()




def npu_is_available():
    return get_device() == 'npu'


def varlen_attn_is_available():

    return is_flash_attn_2_available() or npu_is_available()


def lmdeploy_is_available():

    available = False
    try:
        import lmdeploy  # noqa: F401
        available = True
    except ImportError:
        available = False

    return available

def liger_kernel_is_available():

    available = False
    try:
        import liger_kernel  # noqa: F401
        available = True
    except ImportError:
        available = False

    return available


@contextmanager
def profile_time_and_memory(desc):

    torch_device = get_torch_device_module()
    start_t = time.time()
    torch_device.reset_peak_memory_stats()

    yield

    max_memory = torch_device.max_memory_allocated()
    cost_time = time.time() - start_t

    logger.success(f'{desc} Elapsed time {cost_time:.2f} seconds, '
                f'peak gpu memory {max_memory/1024**3:.1f}G')
