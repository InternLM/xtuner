"""Debug: run claw-bench calendar tasks via solution/solve.sh (skip LLM).

If tasks fail here, the bug is in the validate side (upload, env, verifier,
judger).  If they pass here but fail with ``calendar.py`` config, the bug
is in agent inference.

``KNOWN_BROKEN_SOLUTIONS`` lists upstream tasks whose ``solve.sh`` has bugs
that aren't our concern (literal ``$WORKSPACE`` in python ``.format()``,
missing imports, ``set -u`` + unused positional args, non-serializable
numpy types, missing system deps like ``sqlite3``, missing python pkgs
like ``textblob``).  Upstream's own runner never invokes ``solve.sh``, so
these have never actually executed anywhere — running them through our
pipeline just exposes pre-existing upstream bugs.  Skipping them lets us
validate the infra against the tasks whose solve.sh *does* work.
"""

from claw_bench.dataset import ClawBench
from claw_bench.pipeline import claw_solution_pipeline


KNOWN_BROKEN_SOLUTIONS: set[str] = {
    # ── Upstream script bugs (need solve.sh / setup.sh patches) ────────
    "law-002-lease-analysis",         # setup.sh ``$5,000`` → bash expands $5
    "law-003-compliance-check",       # solve.sh: missing ``import os``
    "law-005-merger-due-diligence",   # solve.sh: unterminated triple-quote
    "bio-004-phylogenetic",           # solve.sh: literal ``$WORKSPACE`` in .format()
    "bio-002-protein-alignment",      # solve.sh: invalid escapes in quoted heredoc
    "bio-005-variant-annotation",     # solve.sh: invalid escapes in quoted heredoc
    "acad-003-statistical-analysis",  # solve.sh: numpy.bool_ in json.dump
    "cs-002-db-migration",            # solve.sh: unexpanded ``$SPEC_FILE``
    "sci-005-optimization",           # solve.sh: literal ``$WORKSPACE`` in python
    "sys-008",                        # solve.sh: bash syntax error line 113
    "ds-002-ab-testing",              # solve.sh: bool / numpy in json.dump
    "reg-002-sox-controls",           # solve.sh expects ``./controls.csv`` in cwd
    "fin-006",                        # solve.sh: ``import:`` command (ImageMagick typo)

    # ── Upstream never committed environment/data/<file> ───────────────
    "db-001", "db-002", "db-003", "db-004", "db-005",
    "debug-001", "debug-002", "debug-003", "debug-004", "debug-005",
    "math-001", "math-002", "math-003", "math-004", "math-005",
    "plan-001", "plan-002", "plan-003", "plan-004", "plan-005",
    "tool-001", "tool-002", "tool-003", "tool-004", "tool-005",
    "fin-008", "sys-008", "fin-006", "ds-004-time-series", "cs-004-ci-pipeline", "sci-001-ode-solver" # solve.sh awk on missing balance_sheet.csv
}


dataset = ClawBench(
    tasks_root="/mnt/shared-storage-user/llmit/user/liukuikun/workspace/bench/claw-bench/tasks",
    pipeline=claw_solution_pipeline(),
    skip_ids=KNOWN_BROKEN_SOLUTIONS,
)
