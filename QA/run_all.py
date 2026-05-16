"""QA test runner.

Discovers every `test_*` function in `QA/unit/`, `QA/integration/`, and
`QA/regression/` and runs them in alphabetical order. Pure stdlib — works
without pytest installed (CI can run it on a vanilla Python image).

Usage:
    python QA/run_all.py            # run everything
    python QA/run_all.py unit       # run only the unit suite
    python QA/run_all.py unit regression

Exit code 0 if every test passed, 1 if anything failed or errored. Each
failure is printed with the traceback so you can jump straight to the line.

Tests are written with plain `assert` statements, so this same suite is
also runnable via `python -m pytest QA/` if you prefer the richer output.
"""
from __future__ import annotations
import importlib
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path


QA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = QA_DIR.parent

# Make `core_*`, `scraper`, `local_companies` importable from inside tests.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(QA_DIR))


SUITES = ("unit", "integration", "regression")


def _discover_test_files(suites):
    for suite in suites:
        suite_dir = QA_DIR / suite
        if not suite_dir.is_dir():
            continue
        for path in sorted(suite_dir.glob("test_*.py")):
            yield suite, path


def _load_module(path: Path):
    """Import a test file by path. Returns the loaded module."""
    mod_name = f"qa_{path.parent.name}_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _run_module(suite: str, path: Path, results: dict):
    """Run every `test_*` callable in the module. Append (status, label, err) per test."""
    rel = path.relative_to(PROJECT_ROOT)
    try:
        module = _load_module(path)
    except Exception:
        results["errors"].append((str(rel), "IMPORT", traceback.format_exc()))
        return

    test_names = sorted(n for n in dir(module) if n.startswith("test_") and callable(getattr(module, n)))
    for name in test_names:
        label = f"{rel}::{name}"
        try:
            getattr(module, name)()
        except AssertionError as e:
            results["failures"].append((label, str(e) or "assert failed", traceback.format_exc()))
        except Exception as e:
            results["errors"].append((label, type(e).__name__, traceback.format_exc()))
        else:
            results["passes"].append(label)


def main():
    requested = [s for s in sys.argv[1:] if s in SUITES] or list(SUITES)
    results = {"passes": [], "failures": [], "errors": []}
    started = time.time()

    files_run = 0
    for suite, path in _discover_test_files(requested):
        files_run += 1
        _run_module(suite, path, results)

    elapsed = time.time() - started
    total = len(results["passes"]) + len(results["failures"]) + len(results["errors"])

    if results["failures"]:
        print("\n----- FAILURES -----")
        for label, msg, tb in results["failures"]:
            print(f"\nFAIL  {label}\n{tb}")

    if results["errors"]:
        print("\n----- ERRORS -----")
        for label, msg, tb in results["errors"]:
            print(f"\nERROR {label}  ({msg})\n{tb}")

    print(
        f"\nQA SUITE: {len(results['passes'])} passed, "
        f"{len(results['failures'])} failed, "
        f"{len(results['errors'])} errored "
        f"across {files_run} file(s) in {elapsed:.1f}s "
        f"({', '.join(requested)})."
    )
    sys.exit(0 if not (results["failures"] or results["errors"]) else 1)


if __name__ == "__main__":
    main()
