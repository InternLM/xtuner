import os
import subprocess

from xtuner.v1.utils.logger import get_logger


logger = get_logger()


def _get_optional_int_env(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default

    value = value.strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return int(value)


def get_async_save_cpu_priority() -> int | None:
    return _get_optional_int_env("XTUNER_ASYNC_SAVE_CPU_PRIORITY", 19)


def get_async_save_io_priority() -> int | None:
    return _get_optional_int_env("XTUNER_ASYNC_SAVE_IO_PRIORITY", 3)


def get_async_save_file_lock_slots() -> int:
    return max(0, int(os.environ.get("ASYNC_DCP_FILE_WRITE_LOCK_SLOTS", "1")))


def set_async_save_process_qos() -> None:
    set_process_qos(
        cpu_priority=get_async_save_cpu_priority(),
        io_priority=get_async_save_io_priority(),
    )


def set_process_qos(cpu_priority: int | None, io_priority: int | None) -> None:
    """Set CPU and I/O scheduling priority for the current process.

    Failures are logged but not fatal.
    """
    pid = os.getpid()

    if cpu_priority is not None and 0 <= cpu_priority <= 19:
        try:
            current_nice = os.nice(0)
            increment = cpu_priority - current_nice
            if increment == 0:
                logger.debug(f"PID {pid}: CPU nice already at target {cpu_priority}")
            elif increment < 0:
                logger.warning(
                    f"PID {pid}: Skipping CPU nice change from {current_nice} to {cpu_priority}; "
                    "lowering nice requires privilege"
                )
            else:
                new_nice = os.nice(increment)
                logger.debug(f"PID {pid}: Set CPU nice from {current_nice} to {new_nice} (target {cpu_priority})")
        except OSError as e:
            logger.warning(f"PID {pid}: Failed to set CPU priority: {e}")

    if io_priority is not None:
        try:
            subprocess.run(
                ["ionice", "-c", str(io_priority), "-p", str(pid)],
                check=True,
                capture_output=True,
            )
            logger.debug(f"PID {pid}: Set I/O priority class to {io_priority}")
        except (subprocess.CalledProcessError, OSError) as e:
            logger.warning(f"PID {pid}: Failed to set I/O priority: {e}")
