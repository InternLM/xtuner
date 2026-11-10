# Retaining Agents across weight updates (experimental)

Ordinary rollout pauses can cancel pending Agent executions. The opt-in complete-turn path keeps the native pending tasks, sessions and sandboxes alive between model requests. Training and weight transfer still wait for every admitted model request to settle. This is an experimental fixed-topology, non-streaming path, not a guarantee that an arbitrary external Agent can resume from a checkpoint.

## Ownership and publication

The producer's existing `_PendingTasks` owns each live Agent task. The replay buffer owns training admission and consumption. Disaggregated sampling uses the native persistent loop. Colocated sampling uses a dedicated `LoopThread` to keep tools and HTTP clients progressing while the trainer performs synchronous GPU work; it does not run GPU inference during training and does not introduce a second pending queue.

Publication order is: freeze worker recovery, seal every entrypoint, drain admitted model requests, pause inference, train/synchronize, install the new version at every sealed entrypoint, clear backend pause flags, then reopen. Forwarding must certify both backend completion and terminal trace recording. Unknown backend/trace outcomes prevent publication; HTTP cancellation alone is insufficient. Only recognized pre-generation context-limit rejection can settle without a generated training trace.

Version install, backend resume and reopen share one monotonic control deadline, using the configured freeze timeout. This budget does not include training or weight transfer. Multi-entrypoint reopen is not atomic; a partial or late RPC may open a ready entrypoint while the controller fails and retains the barrier. Gate, HTTP and Trial waiting budgets remain finite and must cover the intended training pause.

## Required configuration

Start from a working native trainer/runner configuration; these fields are additions, not a standalone runnable recipe.

| Scope | Configuration |
| --- | --- |
| Rollout | `turn_resume_wait_timeout` positive; `health_check_failure_threshold=0` (fixed topology) |
| Sandbox loop | `require_turn_provenance=True`; preserve real session/request IDs from the Agent client |
| Disaggregated manager | `turn_resume_drain_timeout` positive; `turn_resume_freeze_timeout` and `turn_resume_quiesce_timeout` explicit |
| Colocated manager | `turn_keepalive=True`; set `turn_keepalive_drain_timeout`, `turn_keepalive_freeze_timeout`, `turn_keepalive_quiesce_timeout` |
| Evaluation/partial rollout | All online evaluation entrypoints disabled; token partial rollout disabled |
| Optional affinity | `rollout_config.session_affinity=True` and sandbox `retire_route_on_completion=True` |

Gates and the corresponding manager must be enabled together. Unsupported custom disaggregated strategies are rejected at manager construction unless they implement the declared turn-resume/quiesce capability. Colocated keepalive requires the native asynchronous producer. Ordinary configurations leave these fields disabled and preserve the native failed-attempt filter: discard failed members of mixed training groups and retain successful members. `min_valid_attempts` is a separate explicit recipe policy, default `None`; it does not change evaluation.

A client must return the server's real request ID and session ID with each recorded model response. A locally invented request ID or model-step counter cannot certify provenance. The current Lagent client integration used for experiments is on its v1 development line; its main branch does not contain the corresponding adapters/serving modules. That main-target prerequisite is unresolved. Do not treat installing Lagent main as a tested end-to-end recipe. The XT request-gate and controlled forwarding interfaces can be reviewed independently of that cross-repository integration.

## Trace, routing and cleanup

Each admitted request fixes its model version. Trace export retains historical per-segment logprobs and routed experts, plus complete token versions including tool tokens. Sandbox conversion maps the complete response suffix into the native training fields. Supervised labels and generated-token accounting remain separate from complete-history provenance.

Affinity binds a session to one concrete DP entrypoint, not merely an EP group. The binding survives training pauses and fails closed if that backend disappears. Terminal success, failure or cancellation retires the binding; tombstones reject late requests and have a finite router capacity. There is no automatic POST retry or reassignment. Affinity does not make old-weight KV valid after a weight update. Trace references retire according to training ownership separately from session routing; completion must not free R3 storage still borrowed by training.

Checkpoint quiescence stops new groups and lets existing Agents finish while gates are still open, then seals/drains and saves. A checkpoint does not serialize live tools, sandboxes or client coroutine stacks. Cold restore starts with closed admission until restored weights and the actual saved version are ready. HF warm-start and DCP optimizer/sampler restore are distinct operations. For diagnostics, collect actual `debug_rollout_N.pt` files before using train-only replay; ordinary rollout JSONL is not replay PT data.

## Verification and remaining work

The tests cover request leases and unsafe settlement, real localhost forwarding, trace provenance/R3 ownership, sampler restore, retained task identity, persistent-loop startup/cleanup, concrete backend affinity and terminal retirement. Some tests stub model/backend/teacher imports and RPCs; they do not demonstrate full-model GPU execution, distributed recovery or learning.

The main-adapted path still needs real Agent tool progress through synchronous training, online publication across an optimizer update, HF/DCP loading and resource-cleanup checks. Historical experiment success and the historical context-limit failure are version-specific, not verification of this main adaptation. Keep the feature experimental until those checks and the remaining lifecycle design review are closed.

## Sources and scope

The admission-accounting concept references [matrix72c/sandbox_rl](https://github.com/matrix72c/sandbox_rl), `src/sandbox_rl/agents/request_gate.py` before commit `fd76fba`. The native implementation here uses condition-protected leases and explicit settlement/version publication rather than copying that HTTP gate's forwarding implementation. The inspected reference snapshot contains third-party Apache notices for listed XTuner-derived files, but no root license declaration for unrelated files; this is not presented as a blanket license grant for the reference repository. Maintainers should retain this provenance and resolve any further source-permission requirements before merging.

External frozen MTP configuration and transport are a separate change, [PR #2289](https://github.com/InternLM/xtuner/pull/2289); this lifecycle patch contains no frozen-source hook or LMDeploy modification. Generation-cancellation/resume, online MTP training, new sampling policy and performance claims are outside this change.
