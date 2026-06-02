from torch.testing._internal.common_distributed import DistributedTestBase, MultiProcessTestCase, logger, TEST_SKIPS, c10d
import torch
import torch.distributed as dist
import threading
import sys
import os
import re
import contextlib
import inspect
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

    def create_pg(self, device):
        ret = super().create_pg(device)
        os.environ["LOCAL_RANK"] = str(dist.get_rank() % torch.cuda.device_count())
        return ret

    # ------------------------------------------------------------------ #
    # HuggingFace bitwise-parity helpers (shared by model parity tests)   #
    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def hf_impl(self):
        """Force XTuner ops onto the HF-exact eager path (``XTUNER_HF_IMPL``) for the duration of the
        block, restoring the previous value on exit. Build models *inside* the block so their
        attention modules pick the eager op at construction time."""
        prev = os.environ.get("XTUNER_HF_IMPL")
        os.environ["XTUNER_HF_IMPL"] = "true"
        try:
            yield
        finally:
            if prev is None:
                os.environ.pop("XTUNER_HF_IMPL", None)
            else:
                os.environ["XTUNER_HF_IMPL"] = prev

    @staticmethod
    def load_params_from_hf(module, loader, key_for=None) -> None:
        """Copy every parameter of ``module`` from a HF checkpoint.

        ``key_for`` selects each parameter's checkpoint key:

        * ``None`` (default): ``module`` must expose ``to_hf_key_list`` (any XTuner ``BaseModel``);
          ``module.to_hf_key_list(name)[0]`` is used. Lets a standalone XTuner tower load itself
          without the caller re-passing ``module`` in a lambda.
        * ``str``: treated as a prefix; ``key = key_for + name``. Convenient for an HF module
          whose checkpoint keys are ``"<prefix><param>"``.
        * ``Callable[[str], str]``: arbitrary mapping (used internally to apply ``hf_key_mapping``).

        ``loader`` (``HFCheckpointLoader``) reads only the safetensors shard holding each key, so a
        single layer can be loaded without materializing the full model."""
        if key_for is None:
            if not hasattr(module, "to_hf_key_list"):
                raise ValueError(
                    f"module of type {type(module).__name__} has no `to_hf_key_list`; pass `key_for=`."
                )
            get_key = lambda n: module.to_hf_key_list(n)[0]
        elif isinstance(key_for, str):
            prefix = key_for
            get_key = lambda n: prefix + n
        else:
            get_key = key_for
        for name, p in module.named_parameters():
            key = get_key(name)
            tensor = loader.load(key)
            assert tensor is not None, f"checkpoint key not found: {key}"
            p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))

    @staticmethod
    def xtuner_ckpt_key(model, param_name: str) -> str:
        """Resolve the HF checkpoint key for an XTuner parameter: ``to_hf_key_list`` plus the config's
        ``hf_key_mapping`` (the remap normally applied inside ``_init_load_spec``)."""
        key = model.to_hf_key_list(param_name)[0]
        for pattern, repl in (model.config.hf_key_mapping or {}).items():
            if re.search(pattern, key):
                return re.sub(pattern, repl, key)
        return key

    def materialize_submodule(self, model, submodule, loader, dtype=torch.bfloat16) -> None:
        """Materialize a single submodule of a meta-built model on CUDA and load its weights from
        the checkpoint. Works uniformly for XTuner ``BaseModel`` and HF ``PreTrainedModel``:

        * The submodule's path inside ``model`` is recovered by identity match against
          ``model.named_modules()``.
        * For XTuner the path is run through ``to_hf_key_list`` + the config's ``hf_key_mapping``;
          for HF the path is *already* the state_dict prefix so it is used directly, with HF's
          ``_tied_weights_keys`` honored (e.g. ``lm_head.weight`` redirected to the canonical
          ``model.language_model.embed_tokens.weight`` when ``lm_head`` is loaded standalone).

        Lets a single-layer test scale to any model size on both sides: build the parent model
        under ``torch.device("meta")``, then materialize only the submodules you actually touch
        (the tested layer, ``norm``, ``lm_head``)."""
        path = next((name for name, mod in model.named_modules() if mod is submodule), None)
        if path is None:
            raise ValueError(f"submodule of type {type(submodule).__name__} not found in model")
        prefix = f"{path}." if path else ""
        submodule.to_empty(device="cuda")
        submodule.to(dtype)
        # Rebuild RoPE ``inv_freq`` buffers that ``to_empty`` left as garbage. Scope by
        # ``inv_freq`` so we don't touch unrelated modules. Pull each rotary's ``__init__`` args
        # generically via ``inspect.signature`` + the matching stored attrs (the convention HF and
        # XTuner rotary modules follow), then re-instantiate on CPU and copy ``inv_freq`` back —
        # the init formula lives in the rotary class itself, whichever variant it is.
        for mod in submodule.modules():
            if not hasattr(mod, "inv_freq"):
                continue
            init_args: list | None = []
            for pname, param in inspect.signature(type(mod).__init__).parameters.items():
                if pname == "self":
                    continue
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    init_args = None
                    break
                if hasattr(mod, pname):
                    init_args.append(getattr(mod, pname))
                elif param.default is not inspect.Parameter.empty:
                    init_args.append(param.default)
                else:
                    # Required arg not recoverable from a stored attribute; skip.
                    init_args = None
                    break
            if init_args is None:
                continue
            with torch.device("cpu"):
                fresh = type(mod)(*init_args)
            mod.inv_freq.data = fresh.inv_freq.data.to(device=mod.inv_freq.device, dtype=mod.inv_freq.dtype)

        tied = getattr(model, "_tied_weights_keys", None) or {}
        is_xtuner = hasattr(model, "to_hf_key_list")

        def key_for_param(param_name: str) -> str:
            full = prefix + param_name
            # HF tied weights (e.g. lm_head.weight -> ...embed_tokens.weight) — the canonical key
            # is the one present in the checkpoint.
            if full in tied:
                return tied[full]
            # XTuner BaseModel sub-tower with its own `_hf_prefix` / `to_hf_key_list` (e.g.
            # `vision_tower`, `multi_modal_projector`): use that directly — the deployment prefix is
            # already baked in by the sub-tower, so no compose-level mapping is needed here.
            if hasattr(submodule, "to_hf_key_list"):
                return submodule.to_hf_key_list(param_name)[0]
            if is_xtuner:
                return self.xtuner_ckpt_key(model, full)
            # HF: state_dict key is exactly the named_parameters path.
            return full

        self.load_params_from_hf(submodule, loader, key_for=key_for_param)

