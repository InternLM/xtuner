"""Probe v2: reproduce rc6's 3h-hang scenario more faithfully.

Probe v1 showed sandbox ``/exec`` honors ``timeout_sec`` on a fresh,
healthy sandbox.  But rc6 hung to ``timeout_sec + 10 = 10810s``.
Something else is going on.  This script tests additional hypotheses:

  D)  /exec while the sandbox has been busy for a while.  Does timeout
      still fire if the container has accumulated state (stdin pipes
      from long-running ops etc.)?

  E)  Sandbox dies mid-exec (simulate OOM / container crash). We kill
      the container via gateway DELETE while /exec is in flight and see
      what the host sees.  This is the scenario that would explain
      rc6's "no response for 3h" — sandbox quietly dead, httpx holds
      a half-broken TCP connection until its own timeout fires.

  F)  Agent-shaped command that actually prints lots of stdout before
      hanging.  subprocess.run(capture_output=True) buffers in memory;
      if stdout drain blocks after SIGKILL, timeout wouldn't fire.

Run with: conda activate xtuner_dev; python probe_sandbox_timeout_v2.py
"""

from __future__ import annotations

import asyncio
import sys
import time

import httpx


GATEWAY = "http://env-gateway.ailab.ailab.ai"


async def wait_healthy(client: httpx.AsyncClient, max_wait: float = 120) -> bool:
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        try:
            r = await client.get("/health", timeout=5)
            if r.json().get("ok"):
                return True
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def create_sandbox(image: str, ttl: int = 900) -> tuple[str, str]:
    async with httpx.AsyncClient() as g:
        r = await g.post(
            f"{GATEWAY}/envs",
            json={"image_tag": image, "ttl_seconds": ttl},
            timeout=120,
        )
        r.raise_for_status()
        env = r.json()["env"]
        return env["url"], env["env_id"]


async def delete_sandbox(env_id: str) -> None:
    async with httpx.AsyncClient() as g:
        try:
            await g.delete(f"{GATEWAY}/envs/{env_id}", timeout=30)
        except Exception as exc:
            print(f"    gateway delete failed: {exc}")


async def probe_exec(sb: httpx.AsyncClient, *, command: str, timeout_sec: int, httpx_timeout: float) -> None:
    t0 = time.monotonic()
    try:
        r = await sb.post(
            "/exec",
            json={"command": command, "cwd": "/root", "timeout_sec": timeout_sec},
            timeout=httpx_timeout,
        )
        dt = time.monotonic() - t0
        try:
            body = r.json()
        except Exception:
            body = {"_non_json_body": r.text[:500]}
        rc = body.get("return_code")
        stderr_tail = (body.get("stderr") or "")[-300:]
        stdout_len = len(body.get("stdout") or "")
        print(f"    rc={rc} wall={dt:.2f}s stdout_len={stdout_len} stderr={stderr_tail!r}")
    except httpx.ReadTimeout:
        dt = time.monotonic() - t0
        print(f"    HTTPX_READ_TIMEOUT wall={dt:.2f}s (httpx_timeout was {httpx_timeout}s)")
    except httpx.ConnectError as exc:
        dt = time.monotonic() - t0
        print(f"    HTTPX_CONNECT_ERROR wall={dt:.2f}s: {exc}")
    except Exception as exc:
        dt = time.monotonic() - t0
        print(f"    EXC {type(exc).__name__}: {exc} wall={dt:.2f}s")


async def case_D():
    """/exec after sandbox has been busy: does timeout still fire?"""
    print("[D] busy-sandbox /exec: first 10s of busy work, then sleep 60 with timeout=5")
    print("    expected if server is correct: rc=124 wall≈5s")
    url, env_id = await create_sandbox("t-data-processing-v1")
    try:
        async with httpx.AsyncClient(base_url=url) as sb:
            if not await wait_healthy(sb):
                print("    sandbox never became healthy")
                return
            # Prime with some real work first.
            await probe_exec(sb, command="seq 1 1000 > /tmp/out.txt", timeout_sec=10, httpx_timeout=20)
            # Now the hanging one.
            await probe_exec(sb, command="sleep 60", timeout_sec=5, httpx_timeout=65)
    finally:
        await delete_sandbox(env_id)


async def case_E():
    """Sandbox killed mid-exec: does httpx give up, or wait?"""
    print("[E] kill sandbox mid-exec: start sleep 60 then DELETE env while /exec in-flight")
    print("    this is the suspected rc6 scenario (container dies silently)")
    url, env_id = await create_sandbox("t-data-processing-v1")

    async def _kill_after(delay: float):
        await asyncio.sleep(delay)
        print(f"    [killer] DELETE env at wall={delay}s")
        await delete_sandbox(env_id)

    try:
        async with httpx.AsyncClient(base_url=url) as sb:
            if not await wait_healthy(sb):
                print("    sandbox never became healthy")
                return
            # Spawn killer task that deletes the sandbox ~5s into the exec.
            killer = asyncio.create_task(_kill_after(5))
            # Send /exec with a long sleep and timeout_sec big enough that
            # it wouldn't otherwise fire in our httpx window. httpx timeout
            # is 60s — we want to see what httpx sees when sandbox dies.
            t0 = time.monotonic()
            try:
                r = await sb.post(
                    "/exec",
                    json={"command": "sleep 120", "cwd": "/root", "timeout_sec": 120},
                    timeout=60,
                )
                print(f"    sandbox still alive?? rc={r.json().get('return_code')} wall={time.monotonic()-t0:.2f}s")
            except httpx.ReadTimeout:
                print(f"    HTTPX_READ_TIMEOUT wall={time.monotonic()-t0:.2f}s  ← this is rc6 symptom")
            except httpx.RemoteProtocolError as exc:
                print(f"    HTTPX_REMOTE_PROTOCOL_ERROR wall={time.monotonic()-t0:.2f}s: {exc}")
            except httpx.ConnectError as exc:
                print(f"    HTTPX_CONNECT_ERROR wall={time.monotonic()-t0:.2f}s: {exc}")
            except Exception as exc:
                print(f"    EXC {type(exc).__name__}: {exc} wall={time.monotonic()-t0:.2f}s")
            killer.cancel()
    finally:
        # extra cleanup attempt — may already be deleted
        await delete_sandbox(env_id)


async def case_F():
    """Noisy stdout + background daemon: simulates a long agent run."""
    print("[F] stdout-noisy command + nohup daemon + sleep 20, timeout_sec=5")
    print("    tests subprocess.run pipe-drain under pressure")
    url, env_id = await create_sandbox("t-data-processing-v1")
    try:
        async with httpx.AsyncClient(base_url=url) as sb:
            if not await wait_healthy(sb):
                print("    sandbox never became healthy")
                return
            cmd = (
                'nohup bash -c '
                '\'for i in $(seq 1 1000000); do echo "line $i"; done > /tmp/spam.log 2>&1\' '
                '& '
                'echo started-daemon; '
                'sleep 20'
            )
            await probe_exec(sb, command=cmd, timeout_sec=5, httpx_timeout=30)
    finally:
        await delete_sandbox(env_id)


async def main():
    for case in (case_D, case_E, case_F):
        try:
            await case()
        except Exception as exc:
            print(f"    CASE_FAILED: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
