---
name: sphinx-debug
description: >
  Use when `make html` or `sphinx-build` fails with
  `Extension error (sphinx.ext.autosummary)`,
  `ImportExceptionGroup`, or other Sphinx autosummary / autodoc
  import failures. Provides automated pdb diagnosis for
  `autodoc_mock_imports` issues.
---

# Sphinx Debug

## Quick Diagnostic Command

Run the build with maximum verbosity and pdb on exception:

```bash
make html SPHINXOPTS="-vv -T -P"
```

Flags:
- `-vv`: Verbose autodoc mock logging
- `-T`: Full traceback
- `-P`: Drop into pdb on exception

## For agents: automate pdb via monkey-patch

If you are an agent and cannot interact with an interactive pdb session,
**do not redesign a subprocess/pty capture system**. Instead, monkey-patch
`pdb.post_mortem` so it runs scripted commands and prints the output
directly to stdout.

Run this one-shot script from the docs directory (`docs/en` or `docs/zh_cn`):

```bash
cd docs/en   # or docs/zh_cn
python -c "
import pdb, sys, io, traceback

# Hijack pdb.post_mortem to auto-run diagnostic commands
_original_post_mortem = pdb.post_mortem

def scripted_post_mortem(tb=None):
    out = io.StringIO()
    p = pdb.Pdb(stdout=out, stdin=io.StringIO(''))
    p.use_rawinput = False
    p.reset()
    if tb is None:
        tb = sys.exc_info()[2]
    p.setup(None, tb)

    old_stderr = sys.stderr
    sys.stderr = out

    # List all inner exceptions
    p.onecmd('for i, exc in enumerate(exceptions): print(f\"[{i}] {type(exc).__name__}: {exc}\")')
    # Print full traceback of the first TypeError (the real root cause)
    p.onecmd('type_errs = [exc for exc in exceptions if type(exc).__name__ == \"TypeError\"]')
    p.onecmd('if type_errs: import traceback; traceback.print_exception(type(type_errs[0]), type_errs[0], type_errs[0].__traceback__)')
    p.do_quit('')

    sys.stderr = old_stderr
    print('=== PDB AUTOMATED OUTPUT ===')
    print(out.getvalue())
    print('=== END OUTPUT ===')

pdb.post_mortem = scripted_post_mortem

from sphinx.cmd.build import build_main
build_main(['-b', 'html', '-vv', '-T', '-P', '.', '_build/html'])
"
```

This runs the real Sphinx build with `-P`, but when pdb drops in it
automatically executes your commands and dumps the results to stdout.
You read the output exactly as if you had typed the pdb commands yourself.

## What to look for in the automated output

Sphinx autosummary raises `ImportExceptionGroup` when it cannot resolve a module.
The **top-level message is often misleading** (e.g. `no module named xtuner.v1.ray.dataflow`).
The real root cause is hidden inside the grouped exceptions.

### Common root-cause patterns

1. **TypeError about `__version__`**
   ```
   TypeError: expected string or bytes-like object, got '__version__'
   ```
   This happens when a mocked module (e.g. `torch`) is imported for real by an
   intermediate library (e.g. `fla`) that calls `packaging.version.parse()` on
   the mocked `__version__` object.

   **Fix**: Add the intermediate library to `autodoc_mock_imports` in `conf.py`.
   To identify the exact library, read the traceback: the frame just above
   `packaging/version.py` is the culprit (e.g. `fla/utils.py`).

   ```python
   autodoc_mock_imports = [
       ...
       "fla",   # or whatever library triggers the real import
   ]
   ```

2. **Missing-comma string-concatenation bug**

   If `autodoc_mock_imports` contains an entry like `"scipytorchvision"`,
   a comma is missing between two string literals in `conf.py`:

   ```python
   # WRONG
   autodoc_mock_imports = [
       ...
       "scipy"
       "torchvision",
       ...
   ]

   # CORRECT
   autodoc_mock_imports = [
       ...
       "scipy",
       "torchvision",
       ...
   ]
   ```

3. **AttributeError after the TypeErrors**

   ```
   AttributeError: module 'xtuner.v1.ray' has no attribute 'dataflow'
   ```
   This is usually a *secondary* failure caused by the earlier `TypeError`s.
   Fix the `TypeError` first, then rebuild.

## Step-by-step workflow

1. Run the automated pdb script (see "For agents" section above).
2. Read the `=== PDB AUTOMATED OUTPUT ===` block.
3. If you see `TypeError: expected string or bytes-like object, got '__version__'`,
   read the traceback to find the intermediate library name.
4. Add the identified library to `autodoc_mock_imports` in **both**
   `docs/en/conf.py` and `docs/zh_cn/conf.py`.
5. Also eyeball `autodoc_mock_imports` for missing commas while you are there.
6. Re-run `make html` (without `-P`) to verify the fix.
