"""CPU coverage for the port's real gate, terminal ownership and proxy wiring."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web

from xtuner.v1.rl.agent_loop.sandbox_agent_loop.agent_in_sandbox_loop import AgentInSandboxLoop
from xtuner.v1.rl.rollout.affinity_router import SessionAffinityRouter
from xtuner.v1.rl.rollout.request_gate import RequestGate, SettlementReceipt
from xtuner.v1.rl.rollout.session_server import SessionServer


def test_router_preserves_native_request_gate_versions():
    async def run():
        router = SessionAffinityRouter()
        servers = []
        gates = []
        calls = []
        try:
            urls = []
            for rank in range(2):
                server = SessionServer.__new__(SessionServer)
                gate = RequestGate(wait_timeout=3)
                server.request_gate = gate
                gates.append(gate)

                async def forward(request, lease, rank=rank):
                    # Only the model/trace forwarder is mocked. Production router,
                    # SessionServer dispatch and RequestGate all run over real HTTP.
                    body = await request.json()
                    calls.append((body["session_id"], rank, lease.model_step, lease.request_id))
                    lease.settlement = SettlementReceipt(True, True, "trace_committed")
                    return web.json_response(
                        {"rank": rank, "version": lease.model_step, "request_id": lease.request_id}
                    )

                server._forward_request = forward
                app = web.Application()
                app.router.add_route("*", "/{path:.*}", server._handle_request)
                runner = web.AppRunner(app, handler_cancellation=True)
                await runner.setup()
                servers.append(runner)
                await web.TCPSite(runner, "127.0.0.1", 0).start()
                urls.append(f"http://127.0.0.1:{runner.addresses[0][1]}")
            router.replace_backends(urls)
            endpoint = await router.start("127.0.0.1") + "/v1/chat/completions"
            async with aiohttp.ClientSession() as client:

                async def post(sid):
                    async with client.post(
                        endpoint, json={"session_id": sid, "messages": [], "return_token_ids": True}
                    ) as response:
                        assert response.status == 200, await response.text()
                        return await response.json()

                first = await asyncio.gather(post("a"), post("b"))
                bindings = dict(router.bindings)
                for gate in gates:
                    await gate.seal(1)
                    await gate.drain(1)
                waiting = asyncio.create_task(post("a"))
                await asyncio.sleep(0.05)
                assert not waiting.done() and len(calls) == 2
                assert router.bindings == bindings
                for gate in gates:
                    await gate.install_version(1, 1)
                    await gate.reopen(1)
                result = await asyncio.wait_for(waiting, 2)
                assert result["rank"] == first[0]["rank"]
                assert result["version"] == 1
                assert result["request_id"] != first[0]["request_id"]
                router.retire_sessions(["a"])
                async with client.post(endpoint, json={"session_id": "a"}) as response:
                    assert response.status == 410
                assert len(calls) == 3
        finally:
            await router.stop()
            for server in servers:
                await server.cleanup()

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_terminal_agent_retires_route_without_releasing_trace(outcome):
    async def run():
        loop = AgentInSandboxLoop.__new__(AgentInSandboxLoop)
        loop.mode = "train"
        loop.logger = MagicMock()
        loop.retire_route_on_completion = True
        retired = AsyncMock()
        loop.rollout_ctl = SimpleNamespace(retire_route_sessions=SimpleNamespace(remote=retired))
        item = SimpleNamespace(model_copy=lambda **kwargs: SimpleNamespace())
        state = SimpleNamespace(extra_fields={"rollout_item": item}, session_id="trial", group_id=1)
        loop._run_item = AsyncMock(return_value=object())
        loop._build_rollout_states = AsyncMock(return_value=[state])
        if outcome == "failure":
            loop._run_item.side_effect = ValueError("runner failed")
        if outcome == "cancel":
            loop._run_item.side_effect = asyncio.CancelledError()
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await loop.generate_sample(state)
        else:
            assert await loop.generate_sample(state) == [state]
        retired.assert_awaited_once_with(["trial"])

    asyncio.run(run())


def test_affinity_disabled_adds_no_terminal_rpc():
    async def run():
        loop = AgentInSandboxLoop.__new__(AgentInSandboxLoop)
        loop.mode = "train"
        loop.retire_route_on_completion = False
        retired = AsyncMock()
        loop.rollout_ctl = SimpleNamespace(retire_route_sessions=SimpleNamespace(remote=retired))
        state = SimpleNamespace(
            extra_fields={"rollout_item": SimpleNamespace(model_copy=lambda **kwargs: SimpleNamespace())},
            session_id="trial",
            group_id=1,
        )
        loop._run_item = AsyncMock()
        loop._build_rollout_states = AsyncMock(return_value=[state])
        await loop.generate_sample(state)
        retired.assert_not_called()

    asyncio.run(run())


def test_proxy_registers_one_entry_and_health_updates_only_internal_backends(monkeypatch):
    from xtuner.v1.rl.rollout import affinity_router, proxy_manager

    operations = []

    class Service:
        url = "http://job-router:1234"

        def __init__(self, host):
            operations.append(("start", host))

        def invoke(self, method, *args):
            operations.append((method, *args))

        def close(self):
            operations.append(("close",))

    monkeypatch.setattr(affinity_router, "SessionAffinityService", Service)
    monkeypatch.setattr(proxy_manager.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(
        proxy_manager, "register_to_routedapiproxy", lambda *args: operations.append(("register", *args))
    )
    monkeypatch.setattr(
        proxy_manager, "delete_from_routedapiproxy", lambda *args: operations.append(("delete", *args))
    )
    config = SimpleNamespace(
        turn_resume_wait_timeout=10,
        model_name="owned-model",
        routed_proxy_url="proxy",
        routed_proxy_admin_url="admin",
        worker_log_dir=None,
        session_affinity=True,
    )
    manager = proxy_manager.RolloutProxyManager(config)
    manager.replace_registered_session_urls(["http://dp0", "http://dp1"])
    registrations = [o for o in operations if o[0] == "register"]
    assert registrations == [("register", "admin", "owned-model", "http://job-router:1234")]
    assert ("replace_backends", ["http://dp0", "http://dp1"]) in operations
    group = SimpleNamespace(workers=[SimpleNamespace(is_request_entrypoint=True, session_url="http://dp0")])
    count = len(operations)
    manager.on_worker_group_inactive(group)
    manager.on_worker_group_active(group)
    assert operations[count:] == [("set_backend", "http://dp0", False), ("set_backend", "http://dp0", True)]
    manager.retire_sessions(["terminal"])
    assert operations[-1] == ("retire_sessions", ["terminal"])
    manager.close()
    manager.close()
    assert operations.count(("close",)) == 1
