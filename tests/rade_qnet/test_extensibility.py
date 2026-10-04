"""
A model defined outside ``rade_qnet`` trains through the whole lifecycle.

This is the framework's central promise, stated as a test. Everything else
is in service of it: a user brings a model, the framework brings the rest.

Why it needs its own file
-------------------------
Every model that ships lives under ``rade_qnet.models``, and every synthetic
model in the rest of the suite lives under ``tests``. Both are inside the
repository, and both are imported by code that already imports the
framework. Neither answers the question a user actually has, which is
whether a model in *their* distribution, importing ``rade_qnet`` as a
third-party dependency, works the same way.

It does, and the mechanism is that there is no mechanism: importing the
module registers it. But "it should work, there is nothing special about
it" is exactly the kind of claim that stops being true without anyone
noticing — the first import-time registry scan or packaging assumption
added for convenience would break it, and nothing else in the suite would
fail.

What is deliberately not done here
----------------------------------
The model below does not import anything private. It uses only the five
public names a user would reach for, and if any of them moves, this test
fails -- which is the point. A test that reached into internals to make
itself work would be testing that the internals exist rather than that the
public surface is sufficient.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from pydantic import Field
from sklearn.ensemble import RandomForestRegressor

# The five public names a third-party model needs, and nothing else.
from src.rade_qnet.api import evaluate, infer, train
from src.rade_qnet.core.capability.simple import TabularModel
from src.rade_qnet.core.runtime.components import model
from src.rade_qnet.core.spec.base import Spec
from src.rade_qnet.engines import sklearn as _sklearn_engine  # noqa: F401
from src.rade_qnet.sources.dataset.module import TabularDataModule
from src.rade_qnet.testkit.fixtures import isolated_registries

#: The name the out-of-tree model claims. Registered inside a fixture rather
#: than at import, so that collecting this module does not leave a component
#: behind for every other test in the session to trip over.
MODEL_NAME = "third_party_forest"


class ThirdPartySpec(Spec):
    """
    Settings for a model the framework has never heard of.

    Parameters
    ----------
    n_estimators
        Trees in the forest.
    max_depth
        Depth of each tree.
    """

    n_estimators: int = Field(default=20, ge=1)
    max_depth: int = Field(default=4, ge=1)


class ThirdPartyForest(TabularModel):
    """
    A model as a user would write it, in a package of their own.

    Not decorated at class scope: the decorator runs inside the fixture, so
    the registration is undone when the test finishes.

    Attributes
    ----------
    spec
        Validates the ``model.params`` block.
    """

    spec = ThirdPartySpec

    def data_module(self, spec: object) -> TabularDataModule:
        """
        Use the framework's tabular reader.

        Parameters
        ----------
        spec
            The validated run specification, unused.

        Returns
        -------
        TabularDataModule
            The framework's own.
        """
        del spec
        return TabularDataModule()

    def build_model(self, spec: object, signature: object) -> RandomForestRegressor:
        """
        Return an unfitted forest.

        Parameters
        ----------
        spec
            The run specification, for the model's parameters.
        signature
            The declared interface, unused by a forest.

        Returns
        -------
        RandomForestRegressor
            Unfitted, seeded so the test is deterministic.
        """
        del signature
        settings = ThirdPartySpec.model_validate(dict(spec.model.params))  # type: ignore[attr-defined]
        return RandomForestRegressor(
            n_estimators=settings.n_estimators,
            max_depth=settings.max_depth,
            random_state=0,
        )


@pytest.fixture
def registered():
    """
    Register the out-of-tree model for one test.

    Yields
    ------
    None
        For the duration of the test.
    """
    with isolated_registries():
        model(MODEL_NAME, engine="sklearn")(ThirdPartyForest)
        yield


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """
    Write a small regression problem.

    Parameters
    ----------
    tmp_path
        Pytest's per-test directory.

    Returns
    -------
    Path
        The CSV file.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 4))
    targets = features @ np.array([1.5, -2.0, 0.5, 3.0])
    path = tmp_path / "book.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"f{index}" for index in range(4)] + ["target"])
        writer.writerows(
            [*row, target] for row, target in zip(features, targets, strict=True)
        )
    return path


def specification(dataset: Path) -> dict:
    """
    Build a run specification naming the out-of-tree model.

    Parameters
    ----------
    dataset
        The CSV to read.

    Returns
    -------
    dict
        An unvalidated specification.
    """
    return {
        "task": "supervised",
        "model": {"name": MODEL_NAME, "params": {"n_estimators": 20}},
        "source": {"kind": "tabular", "path": str(dataset)},
        "training": {"engine": "sklearn"},
        "reports": {"enabled": ["summary", "baselines"]},
        "hardware": {"device": "cpu"},
    }


class TestAModelFromOutsideTheFramework:
    """The framework's central promise, exercised end to end."""

    def test_naming_it_in_a_specification_resolves_it(
        self, registered, dataset: Path, tmp_path: Path
    ) -> None:
        """
        Importing is the whole registration mechanism.

        No entry point, no plugin manifest, no scan. If this ever needs
        one, a user's model has stopped being an ordinary Python class.
        """
        del registered
        result = train(specification(dataset), output_root=tmp_path / "runs")

        assert result.bundle_directory is not None

    def test_it_gets_the_full_lifecycle(
        self, registered, dataset: Path, tmp_path: Path
    ) -> None:
        """
        Train, re-score and predict, with no model-specific support.

        Re-scoring is checked for equality rather than closeness: a bundle
        that scores differently from the run that produced it is a
        different model, whoever wrote it.
        """
        del registered
        result = train(specification(dataset), output_root=tmp_path / "runs")
        bundle = Path(str(result.bundle_directory))

        assert evaluate(bundle).metric("test", "mae") == result.metric("test", "mae")
        assert infer(bundle).values.shape[0] > 0

    def test_it_gets_the_reports_too(
        self, registered, dataset: Path, tmp_path: Path
    ) -> None:
        """
        Reports are not reserved for models that ship with the framework.

        Worth asserting separately because reporting is the part most
        likely to acquire a quiet assumption about where a model lives --
        a lookup keyed on the model name, say.
        """
        del registered
        result = train(specification(dataset), output_root=tmp_path / "runs")

        reports = Path(str(result.bundle_directory)).parents[2] / "reports"
        assert {"summary.md", "baselines.md"} <= {p.name for p in reports.iterdir()}

    def test_its_settings_are_validated_like_any_other(
        self, registered, dataset: Path, tmp_path: Path
    ) -> None:
        """
        A user's own spec class gets the framework's validation.

        The constraint is declared on `ThirdPartySpec`, not anywhere in
        the framework, and it is still enforced -- which is what makes a
        bad configuration a parse error rather than an obscure failure
        inside scikit-learn some minutes later.
        """
        del registered
        invalid = specification(dataset)
        invalid["model"]["params"] = {"n_estimators": 0}

        with pytest.raises(Exception, match=r"n_estimators|greater than or equal"):
            train(invalid, output_root=tmp_path / "runs")
