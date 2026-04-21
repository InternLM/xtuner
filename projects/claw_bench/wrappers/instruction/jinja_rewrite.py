#!/usr/bin/env python3
"""Rewrite `{{TASK_*}}` placeholders in a markdown file.

Run during converter time:

    python jinja_rewrite.py \\
        --input  instruction.md \\
        --output _instruction_rewritten.md \\
        --replace "/app/=/{{TASK_WORKSPACE}}/" \\
        --replace "workspace/=/{{TASK_WORKSPACE}}/"

Keeps upstream ``instruction.md`` untouched; writes the rewritten version as a
sibling file that ``task.py`` points at.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def apply_replacements(text: str, replacements: list[tuple[str, str]]) -> str:
    """Apply path-prefix replacements.

    Each replacement is ``(needle, replacement)``; matched only at word boundaries
    (i.e. preceded by a non-word character or start-of-line) so prose mentions of
    the word "workspace" aren't clobbered.
    """
    for needle, replacement in replacements:
        # If needle starts with '/', it's an absolute-path prefix; match anywhere.
        # Otherwise, word-boundary at start.
        if needle.startswith("/"):
            text = text.replace(needle, replacement)
        else:
            pattern = r"(?<![A-Za-z0-9_])" + re.escape(needle)
            text = re.sub(pattern, replacement, text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--replace", action="append", default=[],
                    help='Replacement in "needle=replacement" form. '
                         'Repeat for multiple. Leading / means absolute-path prefix.')
    args = ap.parse_args()

    replacements: list[tuple[str, str]] = []
    for entry in args.replace:
        if "=" not in entry:
            print(f"bad --replace (missing '='): {entry}", file=sys.stderr)
            return 2
        needle, replacement = entry.split("=", 1)
        replacements.append((needle, replacement))

    text = Path(args.input).read_text(encoding="utf-8")
    rewritten = apply_replacements(text, replacements)
    Path(args.output).write_text(rewritten, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
