"""Real HTTP routing contracts without model weights or Ray workers."""

import asyncio
import importlib.util
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web


spec = importlib.util.spec_from_file_location(
    "affinity_router", Path(__file__).resolve().parents[2] / "xtuner/v1/rl/rollout/affinity_router.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_binding_load_retirement_and_failure():
    r = module.SessionAffinityRouter(max_sessions=3)
    r.replace_backends(["http://dp0", "http://dp1"])
    assert r.acquire("a") == "http://dp0"
    assert r.acquire("b") == "http://dp1"
    assert r.acquire("a") == "http://dp0"
    r.replace_backends(["http://dp0", "http://dp1"])  # publication does not move sessions
    assert r.acquire("a") == "http://dp0"
    r.set_backend("http://dp0", False)
    with pytest.raises(web.HTTPServiceUnavailable):
        r.acquire("a")
    assert r.acquire("c") == "http://dp1"
    with pytest.raises(web.HTTPServiceUnavailable):
        r.acquire("d")
    r.retire_sessions(["a"])
    with pytest.raises(web.HTTPGone):
        r.acquire("a")


def test_http_stream_error_pause_disconnect_and_affinity():
    async def run():
        calls = []
        released = asyncio.Event()
        cancelled = asyncio.Event()
        entered = asyncio.Event()
        version = [0]

        async def handler(req):
            body = await req.json()
            calls.append((req.app["rank"], body["session_id"]))
            if body.get("wait"):
                entered.set()
                try:
                    await released.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            if body.get("error"):
                return web.json_response({"error": {"code": "turn_cancelled"}}, status=409)
            if body.get("stream"):
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(req)
                await response.write(b"data: first\n\n")
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response
            await asyncio.sleep(0.02)
            return web.json_response({"rank": req.app["rank"], "version": version[0], "body": body})

        runners = []
        router = module.SessionAffinityRouter()
        try:
            urls = []
            for rank in range(2):
                app = web.Application()
                app["rank"] = rank
                app.router.add_post("/v1/chat/completions", handler)
                runner = web.AppRunner(app, handler_cancellation=True)
                await runner.setup()
                runners.append(runner)
                await web.TCPSite(runner, "127.0.0.1", 0).start()
                urls.append(f"http://127.0.0.1:{runner.addresses[0][1]}")
            router.replace_backends(urls)
            endpoint = await router.start("127.0.0.1") + "/v1/chat/completions"
            async with aiohttp.ClientSession() as client:

                async def post(sid):
                    async with client.post(endpoint, json={"session_id": sid, "xtuner_route_ack": "ack"}) as res:
                        assert res.status == 200
                        return await res.json()

                first = await asyncio.gather(*(post(str(i)) for i in range(4)))
                assert {x["rank"] for x in first} == {0, 1}
                version[0] = 1  # backend version changes, router does not cache replies or change affinity
                router.replace_backends(urls)
                second = await asyncio.gather(*(post(str(i)) for i in range(4)))
                assert [x["rank"] for x in first] == [x["rank"] for x in second]
                assert all(x["version"] == 1 and x["body"]["xtuner_route_ack"] == "ack" for x in second)
                async with client.post(endpoint, json={"session_id": "0", "stream": True}) as res:
                    assert await res.read() == b"data: first\n\ndata: [DONE]\n\n"
                count = len(calls)
                async with client.post(endpoint, json={"session_id": "0", "error": True}) as res:
                    assert res.status == 409
                    assert (await res.json())["error"]["code"] == "turn_cancelled"
                assert len(calls) == count + 1  # no retry to another backend
                async with client.post(endpoint, json={}) as res:
                    assert res.status == 400
                task = asyncio.create_task(client.post(endpoint, json={"session_id": "pause", "wait": True}))
                await asyncio.wait_for(entered.wait(), 3)
                before = router.bindings["pause"]
                router.replace_backends(urls)
                assert router.bindings["pause"] == before
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                await asyncio.wait_for(cancelled.wait(), 3)
                await asyncio.sleep(0.02)
                assert sum(router.inflight.values()) == 0
                router.retire_sessions(["0"])
                async with client.post(endpoint, json={"session_id": "0"}) as res:
                    assert res.status == 410
        finally:
            await router.stop()
            for runner in runners:
                await runner.cleanup()

    asyncio.run(run())


def test_service_thread_shutdown():
    s = module.SessionAffinityService("127.0.0.1")
    s.invoke("replace_backends", ["http://127.0.0.1:1"])
    s.close()
    assert not s.thread.is_alive()
