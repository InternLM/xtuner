# Runner and environment ownership

`Runner` composes inference and validation stages. The caller owns an episode's
resources. Training policy, replay and model versions remain with the existing
XTuner loop and manager; neither Runner nor an environment provider decides them.

```python
provider = PoolEnvironmentProvider(pool={
    "type": SandboxPool,
    "provider": provider_config,
    "specs": sandbox_specs,
})
runner = Runner(infer=infer_stage, validate=judger)
try:
    async with provider.open(item) as environment:
        result = await runner.run(item, environment)
        await environment.check()
finally:
    await provider.close()
```

An environment exposes three operations:

- `get(name, record=...)`: return a ready stage client, allocating lazily if needed.
- `info(name)`: validate a name and return immutable image, working-directory,
  environment ID and optional endpoint metadata. This does not allocate resources.
- `check()`: raise on infrastructure failure, including renewal failure for providers
  that renew leases. Failure must not become a completed sample with a zero reward.

`EnvironmentProvider.open(item)` yields a fresh episode environment and owns
cleanup. Providers interpret `item.environment`; Runner treats it as opaque.
Create one provider per worker, drain its episode contexts, then call `close()`.
A future backend needs an environment provider and stage-client adapter, with no
backend dispatch in Runner or Judger.

The built-in `PoolEnvironmentProvider` keeps the existing lazy SandboxPool
allocation/retry behavior. It copies the template and merges the per-item request
into that copy. Live SandboxPool instances are not accepted as templates.
Cleanup failures propagate and retain failed resource IDs for a later close attempt.
The framework closes loops during normal and exceptional trainer shutdown.

## Migration

This is an API migration, with no second Runner implementation:

1. Move `Runner(pool=...)` to `PoolEnvironmentProvider(pool=...)`.
2. Move `item.pipeline_overrides["pool"]` to `item.environment`.
3. Set `AgentInSandboxLoopConfig.environment_provider` to the provider config or
   a module attribute containing that config.
4. For direct execution, use the context shown above and call
   `runner.run(item, environment)`.

Environment extraction alone does not implement live partial rollout. An owner
that retains an episode across pauses must keep its context open until that
same episode completes or is discarded. Do not create resources per trace segment,
reuse one environment across episodes, or reuse provider connections after close.
