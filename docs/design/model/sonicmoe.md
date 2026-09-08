# SonicMoE integration design

## Layering

The integration follows XTuner's model/module/ops split:

- `xtuner/v1/model/moe`: declares the model-level backend choice, validates combinations with EP/FP8/router options,
  and selects compile targets. It must not import the optional `sonicmoe` package directly.
- `xtuner/v1/module/moe_backend`: implements the routed-expert execution strategy. It owns configuration, input
  validation, weight-layout views, routing-mode selection, and composition with `MoEBlock`.
- `xtuner/v1/ops/moe`: owns tensor algorithms and third-party kernel boundaries. It contains Token Rounding metadata
  construction and the optional official SonicMoE functional call.

SonicMoE does not own parameters. XTuner continues to own `gate.weight`, `fused_w1w3`, and `fused_w2`, preserving
optimizer, FSDP, state-dict, and HuggingFace checkpoint behavior.

Install the optional dependency with `pip install -e '.[sonicmoe]'` and select the backend on any `MoEConfig`:

```python
config = Qwen3MoE30BA3Config(expert_backend="sonicmoe")
```

## Token Rounding

Token Rounding is explicitly enabled with:

```python
config = Qwen3MoE30BA3Config(
    expert_backend="sonicmoe",
    sonicmoe_cfg=SonicMoEBackendConfig(
        routing_mode="token_rounding",
        rounding_mode="nearest",  # nearest | up | down
        rounding_quantum=128,
    ),
)
```

The implementation follows the official SonicMoE benchmark algorithm:

1. Obtain full softmax router probabilities and original token-choice top-k expert ids.
2. Count original assignments per expert and round each count to a multiple of the configured quantum.
3. Rank original assignments ahead of non-top-k candidates for every expert.
4. For round-up, add the highest-scoring non-top-k candidates; for round-down, remove the lowest-priority original
   assignments; nearest chooses the closest tile count.
5. Sort the selected variable-length assignments by token id and pass them to `moe_general_routing_inputs`.
6. Gather final scores from the full router probabilities so gradients still propagate to router logits. The discrete
   selection priority is detached.

The initial implementation permits Token Rounding only with a non-grouped softmax `GreedyRouterConfig`,
`norm_topk_prob=True`, and `router_scaling_factor=1.0`, matching the official routing semantics.

## torch.compile

The default SonicMoE compile plan compiles the decoder's attention, router, shared-expert, and post-MoE computation
with `MoEDecoderLayer.forward(fullgraph=False)`. The official SonicMoE functional call is a deliberate Dynamo graph
boundary because it already owns optimized CUDA/CuTe kernels and a custom backward.

This design provides a stable compiled training path without tracing through third-party Python internals or
reimplementing the official backward. A future fullgraph integration should require an official stable
`torch.library`/fake-tensor contract.

## Current constraints

- CUDA BF16 and SwiGLU only.
- `ep_size=1` and `expert_tp_size=1`; no dispatcher, EP, or ExpertTP path yet.
- No SonicMoE FP8 path.
- General routing supports existing XTuner routers; Token Rounding has the stricter router constraints above.
- Token Rounding changes routing semantics and remains opt-in.
