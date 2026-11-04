# XTuner Unit Test

Run XTuner tests with the user-specified Python and lmdeploy environment. Never report success
before pytest exits and its results have been inspected.

## 1. Select tests

Apply the first matching rule:

1. **Explicit scope**: run exactly what the user requested. Typical targets are `tests`,
   `tests/rl`, `tests/model`, or explicit files and pytest nodeids. A request to rerun everything
   means `tests`, even if previous failures exist.
2. **Generic rerun with saved failures** (for example, “再跑一下单测”): rerun only the saved
   nodeids and show their count before launching. Do not switch to the full suite.
3. **No scope or saved failures**: recommend the smallest credible regression set and wait for
   confirmation before running.

For change-based recommendations, verify `origin/main` with `git rev-parse --verify origin/main`;
if unavailable, ask for the base ref. Inspect all committed, staged, unstaged, and untracked
changes:

```bash
git diff --name-only origin/main --
git ls-files --others --exclude-standard
```

Inspect actual diffs, imports, fixtures, and coverage rather than choosing by filename. Include
modified tests and directly related focused tests; include integration tests only for changed
cross-module contracts. Avoid unrelated adjacent tests and do not default to the full suite.
Present the recommended targets, the changes they cover, and important resource requirements,
then ask the user to run them, run all tests, or provide another target.

## 2. Prepare the environment

Require these values from the request or existing environment; ask for missing values together:

- `XTUNER_DIR`: XTuner repository root.
- `LMDEPLOY_DIR`: lmdeploy repository root.
- `XTUNER_TEST_PYTHON`: Python interpreter for tests.

Do not infer personal or cluster paths. If selected tests need shared assets, also require the
appropriate `CI_SHARE_MODEL` and `CI_SHARE_DATA` values. The user-provided `LMDEPLOY_DIR` is
authoritative; never replace it with a path from another script.

Do not source `ci/scripts/CI_ENV.sh` directly because it runs package installation and other
commands. Create a temporary script containing only its environment-variable exports, excluding
its `PYTHONPATH` assignment:

```bash
cd "$XTUNER_DIR"
export PATH="$(dirname "$XTUNER_TEST_PYTHON"):$PATH"
export PYTHONNOUSERSITE=1
export XTUNER_MAX_CONCURRENCY="${XTUNER_MAX_CONCURRENCY:-1024}"
export UVICORN_LOG_LEVEL="${UVICORN_LOG_LEVEL:-INFO}"

REPO_ID="$(realpath "$XTUNER_DIR" | sha256sum | cut -d' ' -f1)"
STATE_DIR="/tmp/xtuner-unit-test/$REPO_ID"
CACHE_DIR="$STATE_DIR/pytest-cache"
FAILURES_FILE="$STATE_DIR/last_failed.json"
ENV_SCRIPT="$STATE_DIR/ci-env.sh"
mkdir -p "$CACHE_DIR"

"$XTUNER_TEST_PYTHON" - "$XTUNER_DIR/ci/scripts/CI_ENV.sh" "$ENV_SCRIPT" <<'PY'
from pathlib import Path
import sys

source, destination = sys.argv[1:]
lines = []
for line in Path(source).read_text().splitlines(keepends=True):
    if line.startswith("export ") and not line.startswith("export PYTHONPATH="):
        if "$(" in line or "`" in line:
            raise ValueError("Refusing to copy an executable environment assignment")
        lines.append(line)
Path(destination).write_text("".join(lines))
PY

source "$ENV_SCRIPT"
export PYTHONPATH="$XTUNER_DIR:$LMDEPLOY_DIR:${PYTHONPATH:-}"
```

The temporary script must not run `CI_ENV.sh`, `pip`, `proxy_off`, version checks, or any other
non-export command. Verify both repositories, the interpreter, and required assets; report any
missing prerequisite instead of claiming tests ran.

A nonempty valid `FAILURES_FILE` is a JSON object keyed by failed pytest nodeids. If it is absent,
empty, or malformed, there is no reusable failure set.

## 3. Run and save results

Always use `"$XTUNER_TEST_PYTHON" -m pytest` and keep targets in a Bash array. For a new selection:

```bash
TARGETS=(tests/rl)  # Replace with the confirmed targets.
```

For a failed-only rerun:

```bash
mapfile -t TARGETS < <("$XTUNER_TEST_PYTHON" -c \
  'import json, sys; print("\n".join(json.load(open(sys.argv[1]))))' "$FAILURES_FILE")
```

Run with an isolated cache, JUnit report, and log. Use a background task for long runs and wait
for actual completion.

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)"
REPORT="$STATE_DIR/junit-$RUN_ID.xml"
LOG="$STATE_DIR/run-$RUN_ID.log"

"$XTUNER_TEST_PYTHON" -m pytest "${TARGETS[@]}" -ra \
  -o cache_dir="$CACHE_DIR" --junitxml="$REPORT" 2>&1 | tee "$LOG"
TEST_STATUS=${PIPESTATUS[0]}
```

Do not use `set -e` around pytest, append unrelated targets to a failed-only rerun, or replace
the saved nodeids with `--lf`. After every completed run, replace the saved failure set:

```bash
"$XTUNER_TEST_PYTHON" - "$CACHE_DIR/v/cache/lastfailed" "$FAILURES_FILE" <<'PY'
import json
import os
import sys

source, destination = sys.argv[1:]
try:
    with open(source) as f:
        failures = json.load(f)
except FileNotFoundError:
    failures = {}
if not isinstance(failures, dict):
    raise TypeError(f"Unexpected pytest lastfailed format: {type(failures).__name__}")
temporary = destination + ".tmp"
with open(temporary, "w") as f:
    json.dump(failures, f, indent=2, sort_keys=True)
    f.write("\n")
os.replace(temporary, destination)
PY
```

Never merge old failures into a new result. A passing run clears the saved failures. If pytest did
not start or was interrupted before producing a trustworthy result, preserve the previous file.

## 4. Report

Report the selected scope and selection reason, exit status and result counts, every failed/error
nodeid with its key exception, and the log, JUnit report, and failure-state paths. State whether
the next generic request will rerun saved failures or recommend tests from current changes.

A passing subset is not a full-suite pass. Do not modify code or tests merely to make failures
disappear unless the user separately asks for diagnosis or fixes.
