from torch.testing._internal.common_distributed import DistributedTestBase, MultiProcessTestCase, logger, TEST_SKIPS, c10d
import torch
import threading
import sys
import os
import unittest
import traceback
from .utils import enable_full_determinism
from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache
import torch.nn.functional as F



class DeterministicDDPTestCase(DistributedTestBase):
    def prepare(self):
        return

    def run_func(self, test_name):
        enable_full_determinism()
        monkey_patch_hf_modules_cache()
        self.prepare()
        return getattr(self, test_name)()

    def run_test(self, test_name: str, parent_pipe) -> None:
        # Start event listener thread.
        signal_recv_pipe, signal_send_pipe = torch.multiprocessing.Pipe(duplex=False)
        event_listener_thread = threading.Thread(
            target=MultiProcessTestCase._event_listener,
            args=(parent_pipe, signal_recv_pipe, self.rank),
            daemon=True,
        )
        event_listener_thread.start()
        if sys.platform != "win32" and sys.platform != "darwin":
            # Register signal handler to dump stack traces on FATALs.
            # Windows and MacOS do not support the signal handlers.
            torch._C._set_print_stack_traces_on_fatal_signal(True)
        # Show full C++ stacktraces when a Python error originating from C++ is raised.
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retrieving a corresponding test and executing it.
        try:
            self.run_func(test_name)
        except unittest.SkipTest as se:
            logger.info(
                "Process %s skipping test %s for following reason: %s", self.rank, test_name, str(se)
            )
            sys.exit(TEST_SKIPS["generic"].exit_code)
        except Exception:
            logger.error(
                "Caught exception: \n%s exiting "
                "process %s with exit code: %s",
                traceback.format_exc(), self.rank, MultiProcessTestCase.TEST_ERROR_EXIT_CODE
            )
            # Send error to parent process.
            parent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        finally:
            if signal_send_pipe is not None:
                signal_send_pipe.send(None)

            assert event_listener_thread is not None
            event_listener_thread.join()
            # Close pipe after done with test.
            parent_pipe.close()

        if self.destroy_pg_upon_exit:
            try:
                # Some tests do destroy the pgs, and destroy can't be called twice.
                # This avoids spewing warnings about improperly shutting down.
                c10d.destroy_process_group()
            except (AssertionError, ValueError):
                pass

    def _check_loss_curve(
        self,
        losses: torch.Tensor,
        losses_ref: torch.Tensor,
        sim_tol: float = 0.01,
        rtol: float=0.01,
    ):

        loss1_norm = F.normalize(losses, dim=0)
        loss2_norm = F.normalize(losses_ref, dim=0)

        similarity = torch.cosine_similarity(loss1_norm, loss2_norm, dim=0)
        if similarity <= 1 - sim_tol:
            raise AssertionError(
                f"Failed to check the similarity of loss! expected: {losses_ref}, got {losses}, Similarity: {similarity}")

        avg_relative_diff = ((losses - losses_ref) / losses_ref).abs().mean()
        if avg_relative_diff >= rtol:
            raise AssertionError(
                f"Failed to check relative error of loss, expected: {losses_ref}, got {losses}, Mean diff: {avg_relative_diff}")

