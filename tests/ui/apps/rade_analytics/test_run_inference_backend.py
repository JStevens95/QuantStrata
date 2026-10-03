"""Backend wrapper integration tests for ``RadeBackend.run_inference``.

Verifies the *Inference Console* backend seam end-to-end:

* Loads the synthetic two-cluster ensemble from the same fixture
  ``test_ensemble_pipelines.py`` already proves works
  (``registry_with_members``).
* Calls :meth:`RadeBackend.run_inference` with the *member_inputs*
  shortcut (so the test never needs real scenario CSVs).
* Asserts the returned :class:`InferenceRunResult` carries:
    - the resolved ensemble version,
    - matching ``n_scenarios`` / ``predictions.shape``,
    - a non-empty ``activity_log`` whose first row is the
      ``Pipeline started`` event the UI uses as the
      "we've started" lifecycle marker.

Tests run in <2 s and pull in no live API client (a
:class:`unittest.mock.MagicMock` stands in for the HTTP layer).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from src.rade_ml_pt.ensemble.registry import EnsembleRegistry
from src.ui.apps.rade_analytics.data.backend import (
    InferenceRunResult,
    NoOpCache,
    RadeBackend,
)


@pytest.fixture
def backend() -> RadeBackend:
    """A backend wired to a stub HTTP client + no-op cache.

    The HTTP client is never touched by ``run_inference`` (the call
    runs in-process), so a :class:`MagicMock` is enough to satisfy
    the constructor.
    """
    return RadeBackend(client=MagicMock(), cache=NoOpCache())


@pytest.fixture
def registered_ensemble(registry_with_members):
    """Register the two-cluster ensemble from conftest under a known tag.

    Returns ``(registry_dir, ensemble_version)`` ready to feed
    straight into :meth:`RadeBackend.run_inference`.
    """
    _registry, versions, config = registry_with_members
    ens_reg = EnsembleRegistry(config.registry_dir)
    ens_reg.register(config, versions, tags=["backend_test"])
    return config.registry_dir, "backend_test"


class TestRunInferenceBackend:
    def test_returns_success_with_typed_payload(
        self, backend, registered_ensemble,
    ):
        registry_dir, version = registered_ensemble
        member_inputs = {
            "cluster_0": {"features": torch.randn(4, 4)},
            "cluster_1": {"features": torch.randn(4, 4)},
        }

        res = backend.run_inference(
            registry_dir=registry_dir,
            ensemble_version=version,
            new_scenario_dir="(unused — member_inputs short-circuits)",
            member_inputs=member_inputs,
        )

        assert res.ok, f"expected ok=True, got error={res.error!r}"
        assert isinstance(res.data, InferenceRunResult)
        assert res.data.predictions.shape[0] == 4
        assert res.data.n_scenarios == 4
        assert res.data.latency_seconds > 0

    def test_activity_log_is_populated(self, backend, registered_ensemble):
        registry_dir, version = registered_ensemble
        member_inputs = {
            "cluster_0": {"features": torch.randn(2, 4)},
            "cluster_1": {"features": torch.randn(2, 4)},
        }
        res = backend.run_inference(
            registry_dir=registry_dir,
            ensemble_version=version,
            new_scenario_dir="(unused)",
            member_inputs=member_inputs,
        )
        assert res.ok
        log = res.data.activity_log
        assert log, "activity log must not be empty"

        first = log[0]
        assert first["stage"]  == "inference"
        assert first["phase"]  == "Pipeline started"
        assert first["status"] == "running"

        last = log[-1]
        assert last["stage"]  == "inference"
        assert last["phase"]  == "Pipeline complete"
        assert last["status"] == "ok"

    def test_unknown_version_returns_failure_not_exception(
        self, backend, registered_ensemble,
    ):
        registry_dir, _version = registered_ensemble
        res = backend.run_inference(
            registry_dir=registry_dir,
            ensemble_version="this_tag_does_not_exist",
            new_scenario_dir="(unused)",
            member_inputs={
                "cluster_0": {"features": torch.randn(1, 4)},
                "cluster_1": {"features": torch.randn(1, 4)},
            },
        )
        assert not res.ok
        assert "this_tag_does_not_exist" in (res.error or "")

    def test_resolved_version_returned(self, backend, registered_ensemble):
        """Tag inputs (``"backend_test"``) must resolve to a concrete
        version string in the result so callbacks can pin subsequent
        artifact reads to the same snapshot."""
        registry_dir, version = registered_ensemble
        res = backend.run_inference(
            registry_dir=registry_dir,
            ensemble_version=version,
            new_scenario_dir="(unused)",
            member_inputs={
                "cluster_0": {"features": torch.randn(1, 4)},
                "cluster_1": {"features": torch.randn(1, 4)},
            },
        )
        assert res.ok
        # Tag should resolve to a concrete ens_* version (not the
        # tag itself).
        assert res.data.ensemble_version != version
        assert res.data.ensemble_version.startswith("ens_")
