#!/usr/bin/env python3
"""
Run all example scripts with --no-plot to verify they execute without errors.

Usage:
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/run_all_examples.py

Requirements:
    - Python 3.10+ (for dataclass slots=True)
    - numpy, scipy (core)
    - pandas (for scenarios historical; optional - some examples skip historical)
    - matplotlib (optional - use --no-plot to skip)
    - tensorflow (optional - for examples/machine_learning/*)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Examples that take a long time or need special args
SKIP_LONG_OR_OPTIONAL = {
    "examples/machine_learning/01_neural_pricer.py",  # needs tensorflow, long
    "examples/machine_learning/02_calibration_ml.py",   # needs tensorflow, long
    "examples/machine_learning/03_neural_sde.py",      # needs tensorflow
    "examples/q_learning/01_hedging_agent.py",        # can be slow
    "examples/q_learning/02_trading_agent.py",         # can be slow
    "examples/ml/02_rl_hedging_agent.py",             # can be slow
}

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    py = sys.executable

    scripts = sorted(examples_dir.rglob("*.py"))
    scripts = [s for s in scripts if s.name != "run_all_examples.py" and "workflows" not in s.parts]
    rel_paths = [str(s.relative_to(repo_root)) for s in scripts]

    passed = []
    failed = []
    skipped = []

    for rel in rel_paths:
        if rel in SKIP_LONG_OR_OPTIONAL:
            skipped.append(rel)
            continue
        cmd = [py, rel, "--no-plot"] if "--no-plot" in subprocess.run([py, rel, "--help"], capture_output=True, cwd=repo_root, env={**__import__("os").environ, "PYTHONPATH": str(repo_root)}).stderr.decode() or subprocess.run([py, rel], capture_output=True, cwd=repo_root, env={**__import__("os").environ, "PYTHONPATH": str(repo_root)}).stdout.decode() else [py, rel]
        # Prefer --no-plot to avoid blocking
        args = [py, rel]
        try:
            result = subprocess.run(
                [py, rel, "--no-plot"],
                cwd=repo_root,
                env={**__import__("os").environ, "PYTHONPATH": str(repo_root)},
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            failed.append((rel, "timeout"))
            continue
        except Exception as e:
            failed.append((rel, str(e)))
            continue

        if result.returncode == 0:
            passed.append(rel)
        else:
            err = (result.stderr or result.stdout or "")[-500:]
            failed.append((rel, err))

    print("=" * 70)
    print("EXAMPLE RUN SUMMARY")
    print("=" * 70)
    print(f"\nPassed: {len(passed)}")
    for p in passed:
        print(f"  OK   {p}")
    print(f"\nSkipped (long/optional deps): {len(skipped)}")
    for s in skipped:
        print(f"  skip {s}")
    print(f"\nFailed: {len(failed)}")
    for path, err in failed:
        print(f"  FAIL {path}")
        print(f"       {err[:200].replace(chr(10), ' ')}")
    print("=" * 70)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
