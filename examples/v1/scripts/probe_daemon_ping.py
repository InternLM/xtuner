"""Integration probe: does our inline-socket ping work against a real sandbox?

Starts a sandbox, runs lagent_entry.sh daemon setup, sends the exact
python-heredoc ping command that ``_ping_daemon`` uses, verifies:

  - ping returns ``{"status": "ok", "type": "agent"}`` within a few seconds
  - rc=0, stdout parses to JSON
  - the whole path works without needing ``lagent`` on the sandbox's
    system python (i.e. the reason the previous ping was broken)

Run: conda activate xtuner_dev; python probe_daemon_ping.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx


GATEWAY = "http://env-gateway.ailab.ailab.ai"
IMAGE = "t-data-processing-v1"
SOCK_PATH = "/tmp/lagent_agent.sock"

# Mirror the inline script in runner._ping_daemon verbatim.
TIMEOUT = 5
INLINE = (
    "import socket,struct,json,sys\n"
    "s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)\n"
    f"s.settimeout({TIMEOUT})\n"
    "try:\n"
    f"  s.connect({SOCK_PATH!r})\n"
    '  m=json.dumps({"cmd":"ping"}).encode()\n'
    '  s.sendall(struct.pack("!I",len(m))+m)\n'
    '  h=s.recv(4)\n'
    '  assert len(h)==4,"short header"\n'
    '  (n,)=struct.unpack("!I",h)\n'
    '  buf=b""\n'
    '  while len(buf)<n:\n'
    '    chunk=s.recv(n-len(buf))\n'
    '    assert chunk,"short body"\n'
    '    buf+=chunk\n'
    '  sys.stdout.write(buf.decode())\n'
    "except Exception as e:\n"
    '  sys.stderr.write(f"PING_FAIL: {type(e).__name__}: {e}")\n'
    "  sys.exit(1)\n"
)
PING_CMD = f"python3 - <<'PING_EOF'\n{INLINE}\nPING_EOF"


async def wait_healthy(client: httpx.AsyncClient, max_wait: float = 120) -> bool:
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        try:
            if (await client.get("/health", timeout=5)).json().get("ok"):
                return True
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def install_lagent(sb: httpx.AsyncClient) -> None:
    """Mirror what the InstallLagent pre-hook does, without the hook's
    machinery: set up /tmp/lagent-py as a wrapper.  In the real pipeline
    this is done by uploading lagent source and linking /tmp/lagent-py
    — here we do the minimum to get an ``agent daemon start`` to boot.

    Since our ping doesn't need lagent, we only need the daemon itself
    running.  For that we need lagent on the sandbox's python path.
    """
    # Easiest path: install lagent via pip inside the sandbox.  This
    # assumes the sandbox has network.  Real pipeline uses a mount; we
    # fake it for probe purposes.
    probe_msg = (
        "pip install -q lagent 2>&1 | tail -5 || true; "
        "python3 -c 'import lagent; print(\"lagent ok\", lagent.__version__ if hasattr(lagent,\"__version__\") else \"\")'"
    )
    r = await sb.post(
        "/exec",
        json={"command": probe_msg, "cwd": "/root", "timeout_sec": 120},
        timeout=150,
    )
    print(f"    install lagent: rc={r.json().get('return_code')} stdout={r.json().get('stdout','')[:200]!r}")


async def start_daemon(sb: httpx.AsyncClient) -> None:
    """Start the agent daemon in the background.  Needs a minimal
    agent_config to boot — we write one inline.
    """
    config_py = '''
agent_config = {
    "type": "lagent.agents.AsyncAgent",
    "llm": {
        "type": "lagent.llms.AsyncGPTAPI",
        "model_type": "gpt-4o-mini",
        "key": "sk-fake",
        "api_base": "http://127.0.0.1:9999/v1/chat/completions",
    },
}
'''
    # Write config, start daemon with nohup so /exec returns immediately.
    setup = f"""
mkdir -p /tmp
cat > /tmp/agent_config.py <<'CONF_EOF'
{config_py}
CONF_EOF
rm -f {SOCK_PATH}
nohup python3 -m lagent.serving.sandbox.daemon start --mode agent \
  --config /tmp/agent_config.py --sock {SOCK_PATH} \
  >> /tmp/agent_daemon.log 2>&1 &
sleep 2
test -S {SOCK_PATH} && echo "daemon up" || (tail -n 30 /tmp/agent_daemon.log; echo "daemon failed to bind socket"; exit 1)
"""
    r = await sb.post(
        "/exec",
        json={"command": setup, "cwd": "/root", "timeout_sec": 30},
        timeout=40,
    )
    body = r.json()
    print(f"    start daemon: rc={body.get('return_code')} stdout={body.get('stdout','')!r}")
    if body.get("return_code") != 0:
        print(f"    stderr={body.get('stderr','')[:400]!r}")
        raise RuntimeError("daemon failed to start")


async def probe_ping(sb: httpx.AsyncClient) -> None:
    """Run our exact ping command, report timing + result."""
    t0 = time.monotonic()
    try:
        r = await sb.post(
            "/exec",
            json={"command": PING_CMD, "cwd": "/root", "timeout_sec": TIMEOUT},
            timeout=TIMEOUT + 5,
        )
    except Exception as exc:
        print(f"    PING /exec failed after {time.monotonic()-t0:.2f}s: {type(exc).__name__}: {exc}")
        return
    dt = time.monotonic() - t0
    body = r.json()
    rc = body.get("return_code")
    stdout = (body.get("stdout") or "").strip()
    stderr = (body.get("stderr") or "").strip()
    print(f"    ping: rc={rc} wall={dt:.2f}s stdout={stdout[:200]!r}")
    if stderr:
        print(f"    stderr={stderr[:200]!r}")
    if rc != 0:
        print("    ❌ ping failed")
        return
    try:
        parsed = json.loads(stdout)
    except Exception as exc:
        print(f"    ❌ stdout not JSON: {exc}")
        return
    if parsed.get("status") == "ok":
        print(f"    ✓ ping ok, type={parsed.get('type')!r}")
    else:
        print(f"    ❌ ping status != ok: {parsed}")


async def main():
    async with httpx.AsyncClient() as g:
        r = await g.post(
            f"{GATEWAY}/envs",
            json={"image_tag": IMAGE, "ttl_seconds": 600},
            timeout=120,
        )
        r.raise_for_status()
        env = r.json()["env"]
        url, env_id = env["url"], env["env_id"]

    print(f"sandbox created: env_id={env_id} url={url}")
    try:
        async with httpx.AsyncClient(base_url=url) as sb:
            if not await wait_healthy(sb):
                print("sandbox never became healthy")
                return
            print("sandbox healthy")

            try:
                await install_lagent(sb)
                await start_daemon(sb)
            except Exception as exc:
                print(f"daemon setup failed: {exc}")
                # Run ping anyway — it'll fail, but we verify the failure mode.
                print("(running ping against presumably dead daemon)")

            # Happy path: daemon alive, ping should succeed fast.
            print("\n[happy] daemon alive, ping should return ok in <1s:")
            await probe_ping(sb)

            # Kill daemon, ping should fail with a clean error.
            print("\n[dead] kill daemon, ping should fail with connect refused:")
            await sb.post(
                "/exec",
                json={"command": f"pkill -f 'lagent.serving.sandbox.daemon.*{SOCK_PATH}'; rm -f {SOCK_PATH}; sleep 1", "timeout_sec": 10},
                timeout=20,
            )
            await probe_ping(sb)
    finally:
        async with httpx.AsyncClient() as g:
            try:
                await g.delete(f"{GATEWAY}/envs/{env_id}", timeout=30)
            except Exception as exc:
                print(f"gateway delete failed: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
