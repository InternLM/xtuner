"""Probe: does sandbox /exec honor ``timeout_sec``?

The 3h hang in rc6 bottomed out on host-side ``httpx.ReadTimeout`` at
exactly ``timeout_sec + 10`` seconds.  This test isolates which layer is
misbehaving: host / sandbox server / kernel-level subprocess.

Scenarios (each uses a fresh sandbox):

  A)  command = ``sleep 60``, timeout_sec = 5
      Expected if server is correct: returns in ~5s with rc=124.
      If it hangs to httpx_timeout (65s): server doesn't enforce timeout_sec.

  B)  command = ``sleep 3``, timeout_sec = 60
      Expected: returns in ~3s with rc=0.  Sanity check that /exec works
      at all and that timeout_sec doesn't double as command duration.

  C)  command = a bash snippet that nohups a python daemon that sleeps
      forever — mirrors lagent_entry.sh's shape.  timeout_sec = 5.
      Expected: returns in ~5s with rc=124 — but this is exactly the
      scenario where ``subprocess.run(shell=True, capture_output=True,
      timeout=T)`` can hang past T if the daemon holds the shell's
      stdout/stderr pipe open.  If C hangs while A returns cleanly, the
      bug is in subprocess.run pipe-draining with backgrounded children.
"""

from __future__ import annotations

import asyncio
import sys
import time

import httpx


GATEWAY = "http://env-gateway.ailab.ailab.ai"
IMAGE = "t-data-processing-v1"  # same image agent_dev tasks use


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


async def with_fresh_sandbox(fn):
    """Create a sandbox → run fn(sb) → delete the sandbox."""
    async with httpx.AsyncClient() as g:
        r = await g.post(
            f"{GATEWAY}/envs",
            json={"image_tag": IMAGE, "ttl_seconds": 900},
            timeout=120,
        )
        r.raise_for_status()
        env = r.json()["env"]
        url, env_id = env["url"], env["env_id"]
        print(f"    sandbox={env_id} url={url}")

    try:
        async with httpx.AsyncClient(base_url=url) as sb:
            if not await wait_healthy(sb):
                print("    sandbox never became healthy within 120s — abort")
                return
            await fn(sb)
    finally:
        async with httpx.AsyncClient() as g:
            try:
                await g.delete(f"{GATEWAY}/envs/{env_id}", timeout=30)
            except Exception as exc:
                print(f"    gateway delete failed: {exc}")


async def probe_exec(sb: httpx.AsyncClient, *, command: str, timeout_sec: int, httpx_timeout: float):
    """Call /exec, report wall clock + result or whatever exception bubbled."""
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
        stderr = (body.get("stderr") or "")[:200]
        print(f"    rc={rc} wall={dt:.2f}s stderr={stderr!r}")
    except httpx.ReadTimeout:
        dt = time.monotonic() - t0
        print(f"    HTTPX_READ_TIMEOUT wall={dt:.2f}s (httpx_timeout was {httpx_timeout}s)")
    except Exception as exc:
        dt = time.monotonic() - t0
        print(f"    EXC {type(exc).__name__}: {exc} wall={dt:.2f}s")


async def case_A():
    print("[A] sleep 60 with timeout_sec=5, httpx_timeout=65")
    print("    expected if server honors timeout_sec: rc=124 wall≈5s")

    async def _run(sb):
        await probe_exec(sb, command="sleep 60", timeout_sec=5, httpx_timeout=65)

    await with_fresh_sandbox(_run)


async def case_B():
    print("[B] sleep 3 with timeout_sec=60, httpx_timeout=30")
    print("    expected: rc=0 wall≈3s (sanity check /exec works)")

    async def _run(sb):
        await probe_exec(sb, command="sleep 3", timeout_sec=60, httpx_timeout=30)

    await with_fresh_sandbox(_run)


async def case_C():
    print("[C] nohup-backgrounded python daemon + sleep 60, timeout_sec=5, httpx_timeout=65")
    print("    mirrors lagent_entry.sh shape; tests subprocess.run pipe-drain")
    print("    expected if server is correct: rc=124 wall≈5s")

    daemon_cmd = (
        "nohup python3 -c "
        "'import time; "
        "print(\\\"daemon started\\\", flush=True); "
        "open(\\\"/tmp/probe_daemon.log\\\", \\\"w\\\").write(\\\"alive\\\"); "
        "time.sleep(3600)' "
        ">> /tmp/probe_daemon.log 2>&1 & "
        "echo started; "
        "sleep 60"
    )

    async def _run(sb):
        await probe_exec(sb, command=daemon_cmd, timeout_sec=5, httpx_timeout=65)

    await with_fresh_sandbox(_run)


async def main():
    for case in (case_A, case_B, case_C):
        try:
            await case()
        except Exception as exc:
            print(f"    CASE_FAILED: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
