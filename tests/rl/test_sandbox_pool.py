from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import SandboxPool
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import SandboxSpec, StageRecord


class FakeClient:
    def __init__(self, name: str, events: list[str], *, health_gate: asyncio.Event | None = None):
        self.name = name
        self.url = f"http://{name}"
        self.events = events
        self.health_gate = health_gate

    async def health_check(self) -> dict[str, bool]:
        self.events.append(f"health:{self.name}")
        if self.health_gate is not None:
            await self.health_gate.wait()
        return {"ok": True}

    async def aclose(self) -> None:
        self.events.append(f"close:{self.name}")


class FakeProvider:
    def __init__(
        self,
        events: list[str],
        *,
        fail_image: str | None = None,
        health_gate_image: str | None = None,
    ) -> None:
        self.events = events
        self.fail_image = fail_image
        self.health_gate_image = health_gate_image
        self.health_gate = asyncio.Event()
        self.created = 0

    async def create(self, image_tag: str, ttl_seconds: int, **kwargs: Any) -> tuple[FakeClient, str]:
        del ttl_seconds, kwargs
        self.events.append(f"create:{image_tag}")
        if image_tag == self.fail_image:
            raise RuntimeError(f"create failed: {image_tag}")
        self.created += 1
        env_id = f"env-{self.created}-{image_tag}"
        gate = self.health_gate if image_tag == self.health_gate_image else None
        return FakeClient(env_id, self.events, health_gate=gate), env_id

    async def delete(self, env_id: str) -> None:
        self.events.append(f"delete:{env_id}")


class RecordingProvisioner:
    def __init__(self, events: list[str], *, fail_once: bool = False) -> None:
        self.events = events
        self.fail_once = fail_once
        self.calls = 0

    async def __call__(self, primary: FakeClient, dependencies: Mapping[str, FakeClient]) -> None:
        self.calls += 1
        self.events.append(f"provision:{primary.name}:{','.join(dependencies)}")
        if self.fail_once and self.calls == 1:
            raise RuntimeError("provision failed")


def spec(**overrides: Any) -> SandboxSpec:
    return SandboxSpec(image="agent", **overrides)


def pool(provider: FakeProvider, sandbox_spec: SandboxSpec, **overrides: Any) -> SandboxPool:
    return SandboxPool(
        provider=provider,
        specs={"main": sandbox_spec},
        creates_per_sec=None,
        health_poll_interval_sec=0,
        **overrides,
    )


def test_sandbox_spec_rejects_nested_dependencies_and_dependency_provisioner() -> None:
    with pytest.raises(ValidationError, match="must not have dependencies"):
        spec(dependencies={"target": spec(dependencies={"db": spec()})})

    with pytest.raises(ValidationError, match="must not have a provisioner"):
        spec(dependencies={"target": spec(provisioner=object())})


@pytest.mark.asyncio
async def test_group_creation_provisioning_primary_api_and_release_order() -> None:
    events: list[str] = []
    provider = FakeProvider(events)
    provisioner_cfg = {"type": RecordingProvisioner, "events": events}
    sandbox_pool = pool(
        provider,
        spec(
            dependencies={
                "db": SandboxSpec(image="database"),
                "target": SandboxSpec(image="target"),
            },
            provisioner=provisioner_cfg,
        ),
    )

    client = await sandbox_pool.get("main")

    assert [event for event in events if event.startswith("create:")] == [
        "create:database",
        "create:target",
        "create:agent",
    ]
    assert events.index("health:env-3-agent") < events.index("provision:env-3-agent:db,target")
    assert client.name == "env-3-agent"
    assert sandbox_pool.env_id("main") == "env-3-agent"
    assert sandbox_pool.url("main") == "http://env-3-agent"
    with pytest.raises(KeyError, match="unknown sandbox"):
        sandbox_pool.validate_name("target")

    await sandbox_pool.release_all()

    assert [event for event in events if event.startswith("delete:")] == [
        "delete:env-3-agent",
        "delete:env-2-target",
        "delete:env-1-database",
    ]
    assert [event for event in events if event.startswith("close:")] == [
        "close:env-3-agent",
        "close:env-2-target",
        "close:env-1-database",
    ]


@pytest.mark.asyncio
async def test_provision_failure_rolls_back_whole_attempt_before_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    provider = FakeProvider(events)
    provisioner = RecordingProvisioner(events, fail_once=True)

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    sandbox_pool = pool(
        provider,
        spec(
            dependencies={"target": SandboxSpec(image="target")},
            provisioner=provisioner,
        ),
        max_attempts=2,
    )
    record = StageRecord()

    client = await sandbox_pool.get("main", record=record)

    assert client.name == "env-4-agent"
    assert record.metadata["sandbox_create_attempts"] == 2
    assert [event for event in events if event.startswith("delete:")][:2] == [
        "delete:env-2-agent",
        "delete:env-1-target",
    ]
    assert provisioner.calls == 2
    await sandbox_pool.release_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_image", ["target", "agent"])
async def test_create_failure_rolls_back_only_returned_members(fail_image: str) -> None:
    events: list[str] = []
    provider = FakeProvider(events, fail_image=fail_image)
    sandbox_pool = pool(
        provider,
        spec(dependencies={"db": SandboxSpec(image="database"), "target": SandboxSpec(image="target")}),
        max_attempts=1,
    )

    with pytest.raises(RuntimeError, match="could not acquire sandbox group"):
        await sandbox_pool.get("main")

    created_ids = [event.removeprefix("health:") for event in events if event.startswith("health:")]
    assert [event.removeprefix("delete:") for event in events if event.startswith("delete:")] == list(
        reversed(created_ids)
    )


@pytest.mark.asyncio
async def test_unhealthy_member_rolls_back_the_group(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    provider = FakeProvider(events)
    sandbox_pool = pool(
        provider,
        spec(dependencies={"target": SandboxSpec(image="target")}),
        max_attempts=1,
    )

    async def health(client: FakeClient) -> bool:
        return not client.name.endswith("target")

    monkeypatch.setattr(sandbox_pool, "_wait_healthy", health)
    with pytest.raises(RuntimeError, match="member 'target'.*unhealthy"):
        await sandbox_pool.get("main")

    assert [event for event in events if event.startswith("delete:")] == ["delete:env-1-target"]
    assert [event for event in events if event.startswith("close:")] == ["close:env-1-target"]


@pytest.mark.asyncio
async def test_cancellation_cleans_up_all_returned_members() -> None:
    events: list[str] = []
    provider = FakeProvider(events, health_gate_image="agent")
    sandbox_pool = pool(
        provider,
        spec(dependencies={"target": SandboxSpec(image="target")}),
        max_attempts=1,
    )

    acquire = asyncio.create_task(sandbox_pool.get("main"))
    while "health:env-2-agent" not in events:
        await asyncio.sleep(0)
    acquire.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquire

    assert [event for event in events if event.startswith("delete:")] == [
        "delete:env-2-agent",
        "delete:env-1-target",
    ]
    assert [event for event in events if event.startswith("close:")] == [
        "close:env-2-agent",
        "close:env-1-target",
    ]


@pytest.mark.asyncio
async def test_rate_limiter_is_acquired_for_every_physical_create(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    provider = FakeProvider(events)

    class Limiter:
        calls = 0

        async def acquire(self) -> None:
            self.calls += 1

    limiter = Limiter()
    monkeypatch.setattr(
        "xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox.get_shared_async_token_bucket",
        lambda *_args, **_kwargs: limiter,
    )
    sandbox_pool = SandboxPool(
        provider=provider,
        specs={
            "main": spec(dependencies={"db": SandboxSpec(image="database"), "target": SandboxSpec(image="target")})
        },
        creates_per_sec=1.0,
    )

    await sandbox_pool.get("main")

    assert limiter.calls == 3
    await sandbox_pool.release_all()
