"""
Shared fixtures for the pipelines that re-load a saved model.

Evaluation and inference both start by opening a bundle, so they both need a
model that is *registered* rather than merely passed in. Training does not --
a train pipeline takes its definition as a constructor argument, which is
what lets its tests use a definition that was never registered anywhere.

That difference is the point of this module. A bundle records a model name,
and resolving a name means going through the registry, so these tests have to
exercise the registration path rather than side-stepping it. Anything
reusable for that lives here instead of being written twice.

The dataset is exactly linear with no noise, and the engine solves least
squares in closed form. Both are deliberate: the fit is *exact*, so a
reproduction test can assert equality rather than a tolerance. A test that
could only check "the metrics are close" would pass against a pipeline that
re-fitted its scalers on a nearly identical source, which is precisely the
failure these tests exist to catch.
"""

from __future__ import annotations

import csv

import numpy as np

from src.rade_qnet.core.capability.supervised import SupervisedModel
from src.rade_qnet.core.runtime.components import engine as register_engine
from src.rade_qnet.core.runtime.components import model as register_model
from src.rade_qnet.core.spec.data import TabularSourceSpec
from src.rade_qnet.core.spec.run import SupervisedRunSpec
from src.rade_qnet.sources.dataset.module import TabularDataModule
from src.rade_qnet.testkit.fixtures import LinearModel, SyntheticEngine

__all__ = [
    "ENGINE_TAG",
    "MODEL_NAME",
    "N_FEATURES",
    "TRUE_COEFFICIENTS",
    "TRUE_INTERCEPT",
    "SyntheticSupervisedModel",
    "make_spec",
    "register_components",
    "write_linear_dataset",
]

#: Engine tag the synthetic engine is registered under. It borrows an existing
#: name because ``TrainingSpec`` is a discriminated union over the engines the
#: framework ships, so a test cannot invent a fourth tag. Closed-form least
#: squares is also an honest description of what ``sklearn`` does.
ENGINE_TAG = "sklearn"

#: The name the model registers under, and therefore the name that ends up in
#: every bundle manifest these tests write.
MODEL_NAME = "synthetic_tabular"

#: Width of the synthetic problem. Small, and solvable exactly.
N_FEATURES = 4

#: True coefficients of the synthetic target, so a correct run recovers them
#: and an incorrect one cannot.
TRUE_COEFFICIENTS = np.array([1.5, -0.7, 0.3, 2.0])
TRUE_INTERCEPT = 0.4


class SyntheticSupervisedModel(SupervisedModel):
    """A model definition needing one line of data code, as advertised."""

    component_name = MODEL_NAME
    component_engine = ENGINE_TAG

    def data_module(self, spec):
        """Return the standard tabular data module."""
        del spec
        return TabularDataModule()

    def build_model(self, spec, signature):
        """Return an unmaterialised linear model."""
        del spec
        return LinearModel(n_features=int(signature.dynamic["features"].shape[-1]))


def register_components() -> None:
    """
    Register the synthetic engine and model under their names.

    Called inside an ``isolated_registries`` block, so the registrations are
    undone afterwards. Both are needed: re-loading a bundle resolves its
    model name *and* its engine name through the registry, and a test that
    registered only one would fail at whichever lookup came second with an
    error about the wrong thing.
    """
    register_engine(ENGINE_TAG)(SyntheticEngine)
    register_model(MODEL_NAME, engine=ENGINE_TAG)(SyntheticSupervisedModel)


def write_linear_dataset(path, *, n_rows: int = 400, seed: int = 11, shift: float = 0.0):
    """
    Write an exactly linear dataset to a CSV file.

    Parameters
    ----------
    path
        Where to write it.
    n_rows
        How many scenarios.
    seed
        Seed for the feature draw. A different seed means different *data*
        under the same relationship, which is how a re-score against new
        observations is simulated.
    shift
        Added to every feature before the target is computed. A way to move
        the feature distribution without changing the relationship, so that
        a re-fitted scaler produces visibly different numbers while a
        correctly reapplied one does not.

    Returns
    -------
    pathlib.Path
        The file that was written.
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_rows, N_FEATURES)) + shift
    targets = features @ TRUE_COEFFICIENTS + TRUE_INTERCEPT

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"f{index}" for index in range(N_FEATURES)] + ["target"])
        writer.writerows(np.column_stack([features, targets]).tolist())
    return path


def make_spec(dataset, **overrides) -> SupervisedRunSpec:
    """
    Build a run spec pointing at a dataset and the synthetic engine.

    Parameters
    ----------
    dataset
        Path to the CSV file.
    **overrides
        Fields to replace.

    Returns
    -------
    SupervisedRunSpec
        The validated spec.
    """
    fields = {
        "model": MODEL_NAME,
        "source": TabularSourceSpec(path=dataset),
        "training": {"engine": ENGINE_TAG},
        "reports": {"enabled": ()},
    }
    fields.update(overrides)
    return SupervisedRunSpec.model_validate(fields)
