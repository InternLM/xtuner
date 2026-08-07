# Selective checkpointing under `torch.compile`

Why unit-level SAC needs the compiler's cooperation, which routes exist, and what each one
measured. All numbers are Qwen3-MoE-30BA3, `ep_size=4` on 8 GPUs, `dispatcher="deepep"`,
torch 2.10, reported as the mean of steps 5-8 of an 8-step run. Two shapes recur:

- **8k** — `pack_max_length=8192`, no domino. Baseline 8904.4 tgs, 84.32 GB peak.
- **4k** — `pack_max_length=4096` with domino (`intra_layer_micro_batch=2`). Baseline 7812.4 tgs.

The 4k shape is CPU-bound (GPU busy 55-60%), so every CPU-side cost lands fully in wall time there
and is largely hidden at 8k. The same change reads as −0.9% at 8k and −13.9% at 4k; neither number
is wrong, they measure different regimes.

## The problem

The checkpoint policy is asked about every op that reaches the dispatcher, and it has to answer one
question: does this op belong to a unit the user asked to keep. There are exactly two ways to know,
and which one applies is a property of the unit, not a style choice.

**Naming the op.** Works when the op is specific enough to identify the unit on its own —
`flash_attn::_flash_attn_varlen_forward` appears nowhere else in the model. Costs nothing: the
compile set is untouched, no graph breaks, and it works identically inside and outside compiled
code, because the ops worth keeping are exactly the ones inductor cannot fuse and those are the ones
still reaching the dispatcher.

**Naming the callable.** Works for anything, at a price. The marker is ordinary python state — a
`ContextVar` set around the callable — and compiled code neither writes nor reads it: reading one
inside a `fullgraph=True` region is a hard compile error, and a value written during tracing is
traced away. So the callable has to leave the compiled set, via `torch._dynamo.disable`.

This is not a visibility problem. With the whole layer compiled the policy is still consulted
**117597 times per run**, because ops that inductor cannot fuse still go through the dispatcher.
Measured on this model, 26 distinct ops reach it, among them `flash_attn::_flash_attn_varlen_forward_v2`,
`moe::m_grouped_gemm`, `moe::permute`/`unpermute` and `aten::mm.out`; 61% of the calls are
`aten::record_stream`. What is missing for a callable-scoped unit is the region identity, not the op.

## The route taken: op identity where possible, withdrawal where not

`RecomputeTargetMap` binds each unit a model supports to one of the two resolutions. MoE declares
`SAVE_ATTN` as `KeptOps` and the gate and dispatch stages as `KeptCallables`.

Two mechanics of the withdrawal are easy to get wrong:

- Removing an entry from `compile_cfg` does **not** keep it out of the compiled set. Dynamo inlines
  it into whichever compiled caller reaches it, and the marker is traced away exactly as before.
  `torch._dynamo.disable` is what makes it run in python.
- A disabled callee inside a `fullgraph=True` region is a hard error
  (`Skip inlining torch.compiler.disable()d function`), not a split. The surviving entries are
  therefore relaxed to `fullgraph=False`.

Measured, 8k, `save_attn`, when it was still resolved by withdrawing the callable that encloses it:

| | tgs | peak | compiled graphs / captured calls | kept |
|---|---|---|---|---|
| no unit | 8904.4 | 84.32 GB | 6 / 154 | — |
| withdrawal, marker never fired | 8883.1 | 84.32 GB | 6 / 154 | nothing |
| withdrawal, marker firing | 8021.7 | 112.14 GB | 4 / 26 | 42112 tensors |

−9.9% for the same unit that op identity delivers for free. The cost is concentrated, not diffuse:
`_pre_moe_forward` accounts for 128 of the 154 captured calls, because it exists to give
`torch.compile` the ops on either side of attention. Withdrawing it is most of what an MoE layer
compiles. That is why attention resolves by op identity and why `KeptCallables` should name the
*smallest* callable that covers the unit: the compilation given up is the whole callable's.

## What the shipped units cost

Measured on the shipped resolutions, 16k domino (`pack_max_length=8192`,
`intra_layer_micro_batch=2`), `ep_size=4`, deepep. **Three independent runs per setting**, because
one run per setting is not enough to say anything about throughput here:

| unit | tgs (3 runs) | mean | peak allocated | peak reserved |
|---|---|---|---|---|
| none | 10206.0 / 9990.6 / 10205.9 | 10134.2 | 84.4-85.3 GB | 108.5-109.3 GB |
| `save_attn` | 10165.6 / 10204.6 / 10071.2 | 10147.2 | 90.4-91.2 GB | 114.4-115.3 GB |
| `save_moe_gate` | 10095.2 / 10179.5 / 10144.4 | 10139.7 | 93.7-94.5 GB | 117.7-118.7 GB |
| `save_moe_dispatch` | OOM | — | — | — |

**Throughput is unchanged.** The three means are within 0.1% of each other, while the baseline's own
three runs span 2.1%. Any single-run comparison of these units reads as ±2% in whichever direction
the run-to-run variance happened to fall, and means nothing.

**Memory is the reproducible effect**: `save_attn` costs +6.0 GB allocated, `save_moe_gate` +9.2 GB,
each within ±0.4 GB across runs.

So a unit here buys nothing on this model and costs memory. That is not a statement about the
mechanism -- it is a statement about which activations an MoE layer's recompute is actually spent
on, and the census below says why.

`save_moe_dispatch` is not usable at this shape at all. It keeps the permutation and padding buffers
on both sides of the all-to-all, the widest tensors in the layer, and domino already runs at 109 GB
reserved of 140 GB. It OOMs during the first step -- and also OOMs without domino, and at
`pack_max_length=4096`, where it peaks at 111.9 GB against the baseline's 85.9 GB. It is verified
numerically for one step (see below) and declared for smaller models, not as a default here.

## Numerical correctness

Checkpointing changes only the backward pass, so **step-1 loss must be bit-identical** whatever is
kept. It is, across every setting and both shapes -- `2.46262765` at 8k, `2.38410378` at 4k,
including the `save_moe_dispatch` run that OOMs immediately afterwards.

Gradients need a noise floor to interpret, which is why the baseline was run twice under identical
settings:

| | step-1 loss | step-1 grad_norm | vs baseline |
|---|---|---|---|
| baseline | 2.46262765 | 24.39401245 | — |
| baseline, second run | 2.46262765 | 24.39329147 | 3.0e-5 |
| `save_attn` | 2.46262765 | 24.39325905 | 3.1e-5 |
| `save_moe_gate` | 2.46262765 | 24.39400864 | 1.6e-8 |

Every unit sits at or below the floor two identical runs produce. The floor itself comes from
reduction ordering in the grouped GEMM and the all-to-all, and exists with no unit selected.

Measured eager (`torch_compile=False`, all2all, no domino) so that compilation cannot be the source
of a difference; the compiled path is covered by the throughput runs above.

## Whether a unit repays its cost

On this model: no. Every unit raises peak reserved -- keeping activations is the trade -- and none
of them buys back measurable throughput. The census below says why.

A resident-activation census under the shipped configuration (domino, 8k, whole layer checkpointed)
found **245 tensors totalling 15.27 GiB**:

| producer | count | bytes |
|---|---|---|
| `LogSoftmaxBackward0 (16384, 151936) float32` | 1 | 9.27 GiB |
| layer-boundary `(1, 8192, 2048)` bf16 | 94 | 2.94 GiB |
| `TBackward0 (2048, 151936)` bf16 | 1 | 0.58 GiB |
| everything else (60 producers) | 149 | ~2.5 GiB |

The loss logits alone are 61% of the resident set, and no recompute unit addresses them. Nothing
inside the MoE layers survives — the whole-layer checkpoint is doing its job. That bounds how much
any unit-level policy can win here.

## Routes measured

### Keeping the expert GEMM by op identity

The same resolution `SAVE_ATTN` uses, pointed at the ops that dominate an MoE layer's recompute:

| kept | tgs (8k) | peak | captured |
|---|---|---|---|
| baseline | 8904.4 | 84.32 GB | 154 |
| `flash_attn::_flash_attn_varlen_forward_v2` | 8828.2 | 87.31 GB | 154 |
| `moe::m_grouped_gemm` | **9477.3 (+6.4%)** | 105.25 GB | 154 |
| `moe::m_grouped_gemm` + `permute` + `unpermute` | **9616.7 (+8.0%)** | 122.61 GB | 154 |

Not declared as a unit here because +20 to +38 GB is not a trade a user can currently tune — the
selection is per-model, not per-layer. It is the obvious next unit once the selection can say
"in the first N layers". The limit of op identity is expressiveness: it cannot reach anything fused
into a generated kernel, and it cannot scope a unit to part of the model. It is what torchtitan does
(`torchtitan/distributed/activation_checkpoint.py`), with a preset op list and no region concept.

### Withdrawing a smaller callable

Making the excluded callable smaller does not make the exclusion cheaper in proportion, because
`torch._dynamo.disable` breaks the graph once **per level of the inline stack it has to unwind**,
not once per call:

| cut point | inline depth | breaks | captured | tgs (8k) |
|---|---|---|---|---|
| `MultiHeadAttention.forward` | 1 | 2 | 60 | 8101.7 |
| `attn_imp.flash_attention` | 2 | — | 96 | 8664.5 |
| `flash_attn_varlen_func` | 3 | 4 | 104 | 8660.3 |

Cutting deeper leaves more code compiled but unwinds more levels; the two effects cancel. A
withdrawal is cheapest when its boundary is *shallow*, near the compile entry point — the opposite of intuition.

The two resolutions also cannot be made equivalent. Even at the tightest cut, the withdrawal keeps
`aten::alias` alongside the attention kernel, because a `dynamo.disable`d segment runs eagerly and
eager execution dispatches bookkeeping ops that do not exist in the compiled path.

### Letting the marker run while compiling, and taking the graph break

Instead of withdrawing the callable, let the marker execute during tracing and accept the graph
break Dynamo takes at it. The marker becomes live and the policy sees the unit — but only for ops
that reach the dispatcher, so it keeps 752 tensors where withdrawing the callable keeps 42112.

Measured, 8k, `save_attn`: 8578.4 tgs, 86.03 GB, 11 graphs / 90 captured, 5 breaks. Cheaper than
withdrawing the callable and more expensive than op identity, with a third semantics again. Not
adopted because "keep this unit" should not silently mean "keep the handful of ops in it that
inductor happened not to fuse".

### `fx_traceback.annotate` plus a joint-graph pass

The only marker that survives compilation intact. Dynamo executes it during tracing
(`_dynamo/variables/ctx_manager.py`), stamps `node.meta["custom"]`, and AOT carries it through
decomposition and into the backward nodes. Verified on GPU under `fullgraph=True`: 6 graphs, 154
captured calls, **0 graph breaks** — identical to the baseline — with the joint pass seeing 410
annotated nodes and pinning 386.

And it changes nothing. Peak memory stayed at 84.32 GB, byte for byte.

The reason is a level mismatch. `torch.utils.checkpoint(use_reentrant=False)` is implemented with
saved-tensor hooks: its `pack_hook` replaces every tensor autograd saves inside the region with a
holder and frees it. A compiled region is an ordinary `autograd.Function`, so the tensors the
partitioner chose to save are handed to `ctx.save_for_backward` and discarded like everything else.
The partitioner's decision is made and honoured — its *product* is thrown away.

So the tag route requires the partitioner to be the only decider, which means removing the outer
checkpoint.

### Removing the outer checkpoint

Then the partitioner governs, and the tags work. It also costs 61.8 GiB.

| | resident tensors | resident bytes |
|---|---|---|
| whole layer checkpointed | 245 | 15.27 GiB |
| no checkpoint, everything forced to `MUST_RECOMPUTE` | 1706 | 77.04 GiB |

`MUST_RECOMPUTE` applies *within* one joint graph. It cannot recompute across a graph boundary,
because the backward of each graph needs that graph's inputs. Domino + EP splits a layer into many
graphs, and the boundaries land on the widest tensors in the model: one `(49152, 2048)` per layer
(9.00 GiB total), one `(65536, 768)` per layer (4.50 GiB), 96 layer-boundary `(1, 8192, 2048)`
(3.00 GiB). Of the 77.04 GiB, 18.33 GiB has no `grad_fn` at all — pure graph-boundary input.

Forcing every node to recompute changes nothing versus a `budget=0.0`: 131.14 GB against 131.34 GB.
The memory that cannot be reclaimed is not the partitioner's to reclaim.

`torch._functorch.config.activation_memory_budget` on the same shape, with no checkpoint:

| budget | tgs | peak |
|---|---|---|
| baseline (whole layer checkpointed) | 8904.4 | 84.32 GB |
| 0.0 | 10032.5 (+12.7%) | 107.71 GB |
| 0.25 | **10282.0 (+15.5%)** | 111.68 GB |
| 0.5 | 9750.1 (+9.5%) | 113.25 GB |

The fastest result measured anywhere, for +23 GB and no way back down: `budget=0.0` is the floor and
it is 23.4 GB above the baseline. The curve is not monotone (0.25 beats both neighbours) and that is
unexplained; it is a single point per budget, not a characterised curve.

**Under domino this route is unusable.** Domino already needs ~19 GB more reserved, and removing the
checkpoint adds ~24 GB more, which crosses the 140 GB device limit: `budget=0.0` runs at 1925 tgs
(5.4× slower than the checkpointed baseline) with 134.60 GB reserved, and 0.25 and 0.5 both OOM.

## The route that would be complete, and what blocks it

If the checkpoint became a `tag_activation_checkpoint` HOP — that is, if `torch.compile` wrapped the
checkpoint call rather than sitting inside it — the two mechanisms collapse into one. The policy
result is written to `node.meta["recompute"]` during tracing (`torch/utils/checkpoint.py`) and the
partitioner consumes it; there is no second decider to overrule it, no dispatch mode at runtime, and
no callable to withdraw from the compile set.

The HOP only forms if the checkpointed region speculates into a single subgraph. On the domino path
it never does: `torch_all2all.py` reads `permuted_hidden_states.grad_fn` to register a backward
pre-hook for the CUDA-event choreography, and Dynamo cannot trace a tensor's `grad_fn`. Speculation
fails, everything traced is discarded, and the graph before the checkpoint call is empty — measured
at 30B as `unique_graphs=0, calls_captured=0`, a silent degradation to eager rather than an error.

torchtitan reaches this topology because its MoE dispatch is a custom op (`deepep.dispatch.default`),
which enters the graph whole. Making xtuner's dispatcher expressible the same way — replacing the
`grad_fn.register_prehook` choreography — is the one change that unlocks it.

## Memory wall, for context

Independent of checkpointing: domino at 16384 is 4.44× *slower* than not using it (14.20 s vs
3.199 s per step) purely because reserved memory reaches 134.43 GB of 140.06 GB and the caching
allocator thrashes. At 8192 the same code is **16.1% faster** with domino than without (1.583 s vs
1.838 s). `expandable_segments` does not help, so this is total demand rather than fragmentation, and
switching between the all2all and deepep backends does not help either (deepep is 7.1% faster only
when domino is off).

Any activation-memory work on the domino path should check headroom before it profiles kernels.
