from __future__ import annotations

from pathlib import Path

from src.orchestrator.artifacts.naming import run_directory, sanitize


def test_sanitize_replaces_unsafe_chars() -> None:
    assert sanitize("hello world") == "hello_world"
    assert sanitize("a/b/c") == "a_b_c"
    assert sanitize("  ") == "unnamed"


def test_run_directory_builds_under_workdir() -> None:
    wd = Path("/tmp/workdir")
    p = run_directory(wd, "run 01")
    assert str(p).endswith("/tmp/workdir/run_01")