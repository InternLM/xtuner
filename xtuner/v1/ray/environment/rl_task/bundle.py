"""Bundle helpers — small pure functions for file-map building + text rewriting.

Most upload logic now lives in :class:`sandbox.UploadHook` (glob/regex
source matching).  This module keeps a couple of callers that still need
a low-level mirror builder (reference staging in validator) and a string
rewriter (instruction rendering in hooks).
"""

from __future__ import annotations

from pathlib import Path

from sandbox import walk_files


def mirror_tree(
    src_root: Path,
    dst_root: str,
    *,
    exclude: frozenset[str] | set[str] | tuple[str, ...] = (),
) -> dict[str, Path]:
    """Mirror ``src_root/<rel>`` → ``dst_root/<rel>`` for every non-excluded file.

    Args:
        src_root (Path): Host directory to walk.
        dst_root (str): Absolute sandbox path the tree should land under.
        exclude (frozenset | set | tuple): Top-level names (relative to
            ``src_root``) to skip entirely.

    Returns:
        dict[str, Path]: ``{sandbox_abs_path: host_path}``.
    """
    exclude = frozenset(exclude)
    files: dict[str, Path] = {}
    for f in walk_files(src_root):
        rel = f.relative_to(src_root)
        if rel.parts and rel.parts[0] in exclude:
            continue
        files[f"{dst_root.rstrip('/')}/{rel.as_posix()}"] = f
    return files


def rewrite_text(text: str, substitutions: dict[str, str]) -> str:
    """Apply ordered string substitutions.  Later keys see already-rewritten text."""
    out = text
    for needle, replacement in substitutions.items():
        out = out.replace(needle, replacement)
    return out
