# Native frozen MTP (experimental)

Use `MTPConfig(num_layers=1, frozen=True)` to retain dedicated MTP parameters for native loading, saving and rollout weight publication while excluding them from actor training. The main model and its shared embedding/output head remain trainable. Frozen MTP does not execute its training forward or produce MTP training losses.

This is an alternative to the external `WorkerConfig.frozen_mtp` adapter proposed in XTuner #2289. Do not enable both routes. This standalone change does not depend on the Agent lifecycle changes in #2257.

For self-contained DCP checkpoints, explicitly set the outer model configuration `dcp_ignore_frozen_params=False`; its existing default can omit frozen parameters. Use the native training engine/Trainer saving and loading paths. An HF export or a worker model/optimizer checkpoint alone does not validate full Trainer cold restoration.

Qwen3.5 source checkpoints may contain fused expert tensors under `mtp.layers.*.mlp.experts.gate_up_proj` and `down_proj`. The loader resolves these into native per-expert matrices for unquantized BF16, FP16 and FP32 safetensors. Direct indexed keys retain priority. Other key families and bin checkpoints retain their previous behavior. Invalid fused shapes, expert indices and unsupported dtypes fail explicitly. Quantized fused-source aliases are not supported.

Inference speculation still requires the existing backend configuration: LMDeploy PyTorch `lmdeploy_speculative_algorithm="qwen3_5_mtp"` and a positive `lmdeploy_speculative_num_draft_tokens`. Freezing alone does not enable speculative decoding. Backend/topology support follows native weight publication; this change does not add a new transport.

The feature is experimental. Component GPU evidence and current integration validation are described in the PR, with exact source scopes. Short smoke tests do not establish learning quality, speedup or general distributed fault recovery.
