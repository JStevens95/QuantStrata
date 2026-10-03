"""Shared fixtures for ``rade_analytics`` UI-backend tests.

Re-exports the small synthetic two-cluster registry fixture from
``tests/rade_ml_pt/pipelines/ensemble/conftest.py`` so backend
integration tests can rely on the same model fixtures the pipeline
suite already validates against — single source of truth for
"registry + ensemble" stand-up.

Pytest does not propagate fixtures across sibling directories, so we
import + re-bind the fixture function explicitly here rather than
relying on autodiscovery.
"""
from __future__ import annotations

from tests.rade_ml_pt.pipelines.ensemble.conftest import (   # noqa: F401
    cluster_mapping,
    registry_with_members,
)
