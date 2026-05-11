"""Verify the inline ping command by running it against a **self-contained
stdlib unix-socket server** that speaks the same protocol as the lagent
AgentDaemon (4-byte big-endian length + JSON body, handles cmd="ping").

This sidesteps the "sandbox has no lagent" problem — we don't need lagent
to validate that our ping command works.  The server is written in pure
stdlib python so it runs anywhere.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
import time

# ── Server: minimal daemon that accepts {"cmd":"ping"} ────────
SOCK_PATH = "/tmp/mini_daemon.sock"
SERVER_SCRIPT = r'''
import asyncio, json, os, socket, struct, sys

SOCK = os.environ["MINI_SOCK"]
HEARTBEAT_RATE = float(os.environ.get("MINI_RATE", "0"))  # 0 = pong only

async def handle(reader, writer):
    try:
        h = await reader.readexactly(4)
        (n,) = struct.unpack("!I", h)
        body = await reader.readexactly(n)
        req = json.loads(body)
        if req.get("cmd") == "ping":
            resp = json.dumps({"status": "ok", "type": "mini"}).encode()
            writer.write(struct.pack("!I", len(resp)) + resp)
            await writer.drain()
        else:
            resp = json.dumps({"error": "unknown"}).encode()
            writer.write(struct.pack("!I", len(resp)) + resp)
            await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()

async def main():
    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass
    srv = await asyncio.start_unix_server(handle, path=SOCK)
    os.chmod(SOCK, 0o777)
    print(f"mini daemon up on {SOCK}", flush=True)
    async with srv:
        await srv.serve_forever()

asyncio.run(main())
'''

# ── Client: exact ping from runner._ping_daemon ──────────────
TIMEOUT = 5
CLIENT_SCRIPT = (
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


async def run_client() -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", CLIENT_SCRIPT,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(), stderr.decode()


async def main() -> int:
    # Start the mini daemon.
    env = os.environ | {"MINI_SOCK": SOCK_PATH}
    srv_proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", SERVER_SCRIPT,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        # Wait for socket to appear
        for _ in range(30):
            if os.path.exists(SOCK_PATH):
                break
            await asyncio.sleep(0.1)
        assert os.path.exists(SOCK_PATH), "mini daemon never bound socket"
        print(f"mini daemon ready at {SOCK_PATH}")

        print("\n[alive] ping a live daemon:")
        t0 = time.monotonic()
        rc, stdout, stderr = await run_client()
        dt = time.monotonic() - t0
        print(f"  rc={rc} wall={dt:.3f}s stdout={stdout!r} stderr={stderr!r}")
        assert rc == 0, f"expected rc=0, got {rc}"
        parsed = json.loads(stdout)
        assert parsed.get("status") == "ok", parsed
        print("  ✓ status=ok, ping loop works")

        # Kill daemon, try ping again — should fail with FileNotFoundError
        # (after we unlink socket) or ConnectionRefusedError.
        srv_proc.terminate()
        await srv_proc.wait()
        try:
            os.unlink(SOCK_PATH)
        except FileNotFoundError:
            pass
        print("\n[dead] ping a dead daemon (socket removed):")
        t0 = time.monotonic()
        rc, stdout, stderr = await run_client()
        dt = time.monotonic() - t0
        print(f"  rc={rc} wall={dt:.3f}s stdout={stdout!r} stderr={stderr!r}")
        assert rc == 1, f"expected rc=1, got {rc}"
        assert "PING_FAIL" in stderr, stderr
        print("  ✓ clean failure, no hang")
        return 0
    finally:
        if srv_proc.returncode is None:
            srv_proc.kill()
            await srv_proc.wait()
        try:
            os.unlink(SOCK_PATH)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
