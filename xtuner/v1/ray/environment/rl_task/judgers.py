"""Judger factories — thin wrappers that build :class:`SandboxStage` instances
for each kind of verifier.

A judger is structurally a :class:`SandboxStage` with:
  - ``pre``: an :class:`UploadHook` shipping the verifier file tree
  - ``entry``: a shell command (bench-provided wrapper or plain script)
  - ``post``: a :class:`ParseJudgerStdout` turning stdout into a JudgerResult

A judger also carries a ``name`` and ``weight`` used by
:class:`validator.JudgerValidator` for aggregation — those live on the
:class:`Judger` wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from hooks import ParseJudgerStdout
from sandbox import Hook, SandboxStage, UploadHook
from schemas import SandboxSpec


@dataclass
class Judger:
    """A named, weighted verifier stage.

    Attributes:
        name (str): Identifier used by aggregation + as ``JUDGER_NAME``.
        stage (SandboxStage): The actual work.
        weight (float): Weight for weighted aggregation.
        sandbox (Literal["shared"] | SandboxSpec): ``"shared"`` reuses the
            infer sandbox; a :class:`SandboxSpec` spins up a fresh one.
        on_isolated_pre (list[Hook]): Extra hooks run once an isolated
            sandbox is acquired (e.g. seed workspace).
    """

    name: str
    stage: SandboxStage
    weight: float = 1.0
    sandbox: Literal["shared"] | SandboxSpec = "shared"
    on_isolated_pre: list[Hook] = field(default_factory=list)


def pytest_ctrf_judger(
    name: str,
    *,
    target: str,
    wrapper: str,
    verifier_root: str = "/tmp/verifier",
    weight: float = 1.0,
    timeout: int = 300,
    sandbox: Literal["shared"] | SandboxSpec = "shared",
    extra_with: str = "",
) -> Judger:
    """Pytest runner + CTRF JSON emitter.

    Args:
        name (str): Judger name.
        target (str): Pytest file path, relative to task root (e.g.
            ``verifier/test_output.py``).  Its parent dir is uploaded as
            test fixtures.
        wrapper (str): Sandbox-abs path of the bench's pytest-runner script.
        verifier_root (str): Where this judger's files land in the sandbox.
        extra_with (str): Passed as ``$PYTEST_EXTRA_WITH``.

    Returns:
        Judger: Ready to register with :class:`validator.JudgerValidator`.
    """
    target_dir = str(Path(target).parent)  # relative to task_root
    target_name = Path(target).name
    stage_target = f"{verifier_root}/{name}/"

    stage = SandboxStage(
        sandbox=sandbox if sandbox != "shared" else None,
        pre=[
            # base=<verifier_dir> so matched files land at target_root/<rel>
            # (not target_root/verifier/<rel>).
            UploadHook([{
                "base": target_dir,
                "source": "**/*",
                "target": stage_target,
            }]),
        ],
        entry=f"bash {wrapper}",
        env=_EnvBuilder(
            name=name,
            judger_dir=f"{verifier_root}/{name}",
            extras={
                "PYTEST_TARGET": f"{verifier_root}/{name}/{target_name}",
                **({"PYTEST_EXTRA_WITH": extra_with} if extra_with else {}),
            },
        ),
        timeout=timeout,
        post=[ParseJudgerStdout(name)],
    )
    return Judger(name=name, stage=stage, weight=weight, sandbox=sandbox)


def bash_script_judger(
    name: str,
    *,
    script: str,
    verifier_root: str = "/tmp/verifier",
    weight: float = 1.0,
    timeout: int = 300,
    sandbox: Literal["shared"] | SandboxSpec = "shared",
    extra_env: dict[str, str] | None = None,
) -> Judger:
    """Arbitrary bash-script judger.

    The script must emit a JudgerResult-shaped JSON line (or a
    ``{"total_score": ...}`` dict) as its last JSON-looking stdout line.
    """
    script_dir = str(Path(script).parent)
    script_name = Path(script).name

    stage = SandboxStage(
        sandbox=sandbox if sandbox != "shared" else None,
        pre=[
            UploadHook([{
                "base": script_dir,
                "source": "**/*",
                "target": f"{verifier_root}/{name}/",
            }]),
        ],
        entry=f"bash {verifier_root}/{name}/{script_name}",
        env=_EnvBuilder(
            name=name,
            judger_dir=f"{verifier_root}/{name}",
            extras=dict(extra_env or {}),
        ),
        timeout=timeout,
        post=[ParseJudgerStdout(name)],
    )
    return Judger(name=name, stage=stage, weight=weight, sandbox=sandbox)


class _EnvBuilder:
    """Callable that builds a judger stage's env vars from ctx.

    A small named class (not a lambda) so reading a judger stage's
    ``env=`` is self-documenting.
    """

    def __init__(self, *, name: str, judger_dir: str, extras: dict[str, str]):
        self.name = name
        self.judger_dir = judger_dir
        self.extras = extras

    def __call__(self, ctx: dict[str, Any]) -> dict[str, str]:
        env = {
            "JUDGER_NAME": self.name,
            "TASK_WORKSPACE": ctx["workspace"],
            "TASK_JUDGER_DIR": self.judger_dir,
        }
        if ctx.get("reference_path"):
            env["TASK_REFERENCE"] = ctx["reference_path"]
        env.update(self.extras)
        return env
