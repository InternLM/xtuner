---
name: xtuner-custom-judger
description: Use when creating, modifying, or reviewing XTuner RL judgers under xtuner/v1/rl/judger or configs that build custom judgers. Enforces the preferred Judger payload contract and the BaseJudger/ComposedJudger constraints.
---

# XTuner Custom Judger

Use this skill whenever you implement or review an XTuner RL judger.

## Preferred Pattern

Prefer subclassing `Judger`, not `BaseJudger`.

For custom scoring logic, override these methods as needed:

```python
class MyJudger(Judger):
    def preprocess(self, rollout_state: RolloutState) -> JudgerPayload:
        ...

    async def judge_payload(self, payload: JudgerPayloadBatch) -> JudgerOutputBatch:
        ...

    def postprocess(self, rollout_state: RolloutState, output: JudgerOutput) -> RolloutState:
        ...
```

Do not override `Judger.judge()` just to read extra fields from `RolloutState`.
Put the field extraction in `preprocess()` instead.

## Why This Contract Exists

`preprocess -> judge_payload -> postprocess` keeps the scoring payload small and
composable:

- `preprocess()` extracts only the fields required for scoring.
- `judge_payload()` can run locally, in a Ray actor, or behind a judger pool.
- `postprocess()` writes the output back to the original `RolloutState`.

This avoids `deepcopy(RolloutState)` in composed and remote judging. Deep-copying
rollout states can add serialization overhead and can be risky for large fields,
Ray object refs, tensors, numpy arrays, or externally owned objects.

## BaseJudger Rule

Subclass `BaseJudger` only when the implementation must own the full
`judge()` / `batch_judge()` flow and cannot reasonably be expressed as a
payload contract.

BaseJudger-only judgers cannot be used as `ComposedJudger` branches.
`ComposedJudger` intentionally calls branch `preprocess()` and
`judge_payload()` rather than arbitrary `BaseJudger.judge()` implementations.
Because it does not deep-copy `RolloutState`, supporting arbitrary
BaseJudger-only branches would allow multiple branches to mutate the same
rollout state concurrently and overwrite `reward` or other fields.

## ComposedJudger Checklist

When adding a branch intended for `ComposedJudger`:

1. Ensure the class inherits `Judger`.
2. Ensure branch-specific fields are copied into the payload in `preprocess()`.
3. Ensure `judge_payload()` returns a `JudgerOutput` for one payload, or a
   `list[JudgerOutput]` for a batch payload.
4. Ensure `postprocess()` writes `rollout_state.reward`.
5. For multi-branch composed judging, ensure `merge_fn` creates the final
   reward shape, including `reward["score"]` when training expects it.

## Tests

For custom judger changes, add focused tests for:

- Single-sample `judge()`.
- `batch_judge()` if batch support is implemented.
- `ComposedJudger` routing if the judger is intended as a composed branch.
- Rejection or explicit non-support if the implementation directly subclasses
  `BaseJudger`.
