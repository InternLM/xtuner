# Framework anchors: where to inspect and probe

Most of an alignment investigation happens in **shared framework code**, not in the new model. Data
collation, sequence context, loss normalization, gradient-norm accounting, optimizer/scheduler order and
logging keep their entry points when a new model family is added. Start from this index, confirm each
symbol in the current checkouts, and then read only the model-specific part.

Entries name **symbols and search terms, not line numbers**. If a symbol is gone, search from the call
site in the active recipe/engine instead of guessing a sibling name; a same-package sibling can have a
different signature. These anchors locate measurement points; they do not assert that the two sides are
numerically equivalent.

## XTuner (this repository)

| Question | Entry point | What to check there |
|---|---|---|
| How is a batch collated; are labels shifted here? | `xtuner/v1/datasets/collator.py::sft_llm_collator` | Label shift/masking, packing and padding; a replay adapter must not shift pre-shifted labels again |
| How are document boundaries and positions represented? | `xtuner/v1/data_proto/sequence_context.py::SequenceContext` | Token counts, cumulative sequence lengths, padding accounting, position resets; reconstruct these fields when replaying packs |
| How is the loss normalized? | `xtuner/v1/loss/ce_loss.py::CELossConfig` | Reduction mode, chunking, the token-count denominator across microbatch/DP/CP, auxiliary terms. When cross-checking the other framework's fused/chunked CE, also compare the hidden-state gradient accumulation precision, not only the forward options |
| What is the per-update order? | `xtuner/v1/train/trainer.py::Trainer.fit` drives `train_step` → `clip_grad_norm` → `step_optimizer` → logging → `scheduler.step`; the per-step pieces live in `xtuner/v1/engine/train_engine.py::TrainEngine` | Confirm this order in the current revision: logging precedes `scheduler.step`, so the logged lr is the one that update applied. `step_optimizer` can skip an update on non-finite norms or `skip_grad_norm_threshold`, so a logged step is not always an applied update |
| Where does global pre-clip grad norm come from? | `TrainEngine.clip_grad_norm`, which first calls `model.scale_and_reduce_grad` (MoE override in `xtuner/v1/model/moe/moe.py`), then `xtuner/v1/utils/grad_norm.py::cal_grad_norm`, `xtuner/v1/utils/dtensor.py::cal_total_norm` and `group_tensors_by_device_mesh_and_placements` | `scale_and_reduce_grad` is the unscale/replica-mean step (loss-scale reversal, expert-parallel division, all-reduce of Replicate parameters); the pre-clip norm is only comparable after it. Check which process group covers Replicate params that FSDP does not manage — reducing over a sub-mesh silently drops other ranks' contributions. Then grouping by mesh/placement, accumulation dtype, which tensors enter the total; reuse this accounting before building a parallel measurement |
| Where can a probe read gradients without editing the engine? | `xtuner/v1/profiler/prober.py::BaseProber` / `ProberList`, enabled by `setup_prober_list` in `xtuner/v1/profiler/prober_utils.py` | `before_clip_grad_norm(model)` / `after_clip_grad_norm(model, grad_norm)` are engine-level hooks. Forward hooks wrap only the module types listed in `register_prober_list`, so check coverage on a new model; probers are selected by class-name list plus `profile_step`, and wrapping resets the dynamo cache |
| What does a working probe look like? | `tests/profiler/test_prober.py` | Existing setup and dump conventions; prefer them over a new instrumentation scheme |

## AutoModel (NVIDIA NeMo Automodel)

Paths are relative to that repository's checkout; verify them in the revision actually being run.

| Question | Entry point | What to check there |
|---|---|---|
| What is the per-update order? | `nemo_automodel/recipes/llm/train_ft.py` | The step calls `scale_grads_and_clip_grad_norm(...)`, then `opt.step()`, then `scheduler.step(1)`; logging uses `self.step_scheduler.step`. Derive the logged step/lr phase from this order rather than assuming both frameworks print the same phase |
| Where are gradient scaling, norm and clipping done? | `nemo_automodel/components/training/utils.py::scale_grads_and_clip_grad_norm` | Population, dtype, distributed reduction and clip timing; this is the reference for any extra per-group decomposition |
| How is the loss computed and normalized? | `nemo_automodel/components/loss/linear_ce_base.py::LinearCrossEntropy` and `nemo_automodel/components/loss/linear_ce.py` | Selected implementation and its options (for example the `cce_opts` dict passed to `linear_cross_entropy`). Copy the call-site import before inspecting a third-party signature, and confirm an option is both accepted and effective |
| How is the global supervised-token count obtained? | `nemo_automodel/components/loss/utils.py::_count_label_tokens` and the recipe's `num_label_tokens` all-reduce | The denominator used for global token-mean loss, including DP reduction and empty-supervision handling |
| How are batches and packed documents formed? | `nemo_automodel/components/datasets/loader.py::CollatorConfig` and the dataset actually configured | Label shift/masking and packed-row fields such as per-document `seq_lens`; the exact fields depend on the selected dataset, so read the configured one |
| Is there model- or objective-specific gradient scaling? | Search the active recipe for aux-loss scaling, e.g. `_set_moe_aux_loss_backward_scale` (present in the VLM/diffusion recipes; the current LLM `train_ft.py` has none) | Scaling applied before backward changes gradients without changing the logged loss; include it in the contract |

## Extend the index for a new model

1. Locate the model registration and forward path in each checkout. On the XTuner side start from
   `xtuner/v1/model/__init__.py::get_model_config_from_hf` (the `model_type` dispatch, including nested
   `text_config` handling for compose checkpoints). Note any override of loss, sequence
   context, auxiliary losses or gradient scaling; those are model-specific and are not covered by the
   shared anchors.
2. Check probe coverage before relying on it: engine-level gradient hooks usually still fire, while
   forward hooks driven by a module-type list may silently skip new module classes. Record which
   boundaries were actually observed.
3. Add the newly confirmed symbols to your run notes with their revision. Promote a generally useful,
   verified entry back into this table; keep one-off model details in the case notes instead.
4. If an expected hook does not exist on the other framework, instrument the equivalent boundary and
   verify that the instrumentation itself does not change accumulation, synchronization or timing.
5. When the XTuner-side checkout itself is under study (an older branch, a pending PR), add a
   **same-framework version-anchor arm**: a known-good newer snapshot on the same pack stream and config.
   If the arm under test deviates from the anchor by roughly the same amount as from the reference
   framework, the defect is version-internal rather than cross-framework. A function-level diff that
   isolates a single candidate, plus a one-variable transplant run, closes such cases faster than
   jumping straight to per-group gradient decomposition.
