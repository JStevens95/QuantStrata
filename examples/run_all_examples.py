#!/usr/bin/env python3
"""
Run all example scripts with --no-plot to verify they execute without errors.

Usage:
    cd /path/to/QuantStrata
    PYTHONPATH=. python examples/run_all_examples.py

    # Include long-running and ML/RL examples (longer timeout):
    PYTHONPATH=. python examples/run_all_examples.py --long

Requirements:
    - Python 3.10+ (for dataclass slots=True)
    - numpy, scipy (core)
    - pandas (for scenarios historical; optional - some examples skip historical)
    - matplotlib (optional - use --no-plot to skip)
    - tensorflow (optional - for examples/machine_learning/*)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Examples that take a long time or need optional deps (tensorflow); run only with --long
SKIP_LONG_OR_OPTIONAL = {
    "examples/machine_learning/01_neural_pricer.py",  # needs tensorflow, long
    "examples/machine_learning/02_calibration_ml.py",  # needs tensorflow, long
    "examples/machine_learning/03_neural_sde.py",     # needs tensorflow
    "examples/q_learning/01_hedging_agent.py",       # ~30–60s
    "examples/q_learning/02_trading_agent.py",       # ~30–60s
    "examples/ml/02_rl_hedging_agent.py",            # ~60–90s
}

# Default timeout (seconds); long examples get LONG_TIMEOUT when --long
DEFAULT_TIMEOUT = 120
LONG_TIMEOUT = 300


def _script_supports_no_plot(repo_root: Path, py: str, rel: str, env: dict) -> bool:
    """Return True if script accepts --no-plot."""
    try:
        result = subprocess.run(
            [py, rel, "--help"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = (result.stderr or result.stdout or "").lower()
        return "--no-plot" in out or "no-plot" in out
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all example scripts (with --no-plot where supported).",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="Include long-running and ML/RL examples (longer timeout).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    py = sys.executable
    env = {**__import__("os").environ, "PYTHONPATH": str(repo_root)}

    scripts = sorted(examples_dir.rglob("*.py"))
    scripts = [s for s in scripts if s.name != "run_all_examples.py" and "workflows" not in s.parts]
    rel_paths = [str(s.relative_to(repo_root)) for s in scripts]

    passed = []
    failed = []
    skipped = []
    run_long = args.long

    for rel in rel_paths:
        skip_this = rel in SKIP_LONG_OR_OPTIONAL and not run_long
        if skip_this:
            skipped.append(rel)
            continue

        timeout = LONG_TIMEOUT if run_long and rel in SKIP_LONG_OR_OPTIONAL else DEFAULT_TIMEOUT
        use_no_plot = _script_supports_no_plot(repo_root, py, rel, env)
        run_args = [py, rel, "--no-plot"] if use_no_plot else [py, rel]

        try:
            result = subprocess.run(
                run_args,
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
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
    if run_long:
        print("  (--long: included long/ML/RL examples)")
    print(f"\nPassed: {len(passed)}")
    for p in passed:
        print(f"  OK   {p}")
    print(f"\nSkipped (long/optional; use --long to include): {len(skipped)}")
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
