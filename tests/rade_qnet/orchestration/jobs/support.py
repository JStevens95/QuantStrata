"""
Shared scaffolding for the job-set tests.

A job set's unit of work is a complete training run, so a naive test of the
set machinery would train a real model forty times to assert something about
a manifest. Everything here exists to make that cheap: the model is a linear
one, the engine solves least squares in closed form, and the dataset is
exactly linear with no noise.

The closed form matters beyond speed. Because the fit is exact, these tests
can assert on *the numbers themselves* rather than on a tolerance, which is
what lets a test distinguish "the set ran every job" from "the set ran one
job three times" -- a failure that a tolerance-based assertion on a noisy
model cannot see.

Module-level rather than in a conftest, and defined with module-level names,
because a job payload must survive pickling: a class defined inside a test
function cannot be sent to a worker process, and the parity test sends one.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.rade_qnet.core.capability.supervised import SupervisedModel
from src.rade_qnet.sources.dataset.tables import read_table
from src.rade_qnet.sources.dataset.tabular import TabularDataModule
from src.rade_qnet.testkit.fixtures import LinearModel

__all__ = [
    "DIRECTORY_MODEL_NAME",
    "ENGINE_TAG",
    "GROUP_FILENAME",
    "MODEL_NAME",
    "N_FEATURES",
    "SyntheticDirectoryModel",
    "SyntheticSupervisedModel",
    "job_set_payload",
    "write_linear_dataset",
]

#: The engine these tests register the synthetic engine under. `TrainingSpec`
#: is a discriminated union over the engine names the framework ships, so a
#: test cannot invent a fourth tag; `sklearn` is also an honest label for an
#: engine whose fit is closed-form least squares.
ENGINE_TAG = "sklearn"

#: The name the synthetic model registers under.
MODEL_NAME = "synthetic_tabular"

#: Deliberately tiny. A job set test is about the set, not the model.
N_FEATURES = 4

#: The relationship the dataset encodes exactly, so a correct run recovers it.
TRUE_COEFFICIENTS = np.array([1.5, -0.7, 0.3, 2.0])
TRUE_INTERCEPT = 0.25


class SyntheticSupervisedModel(SupervisedModel):
    """A model definition needing one line of data code, as advertised."""

    component_name = MODEL_NAME
    component_engine = ENGINE_TAG

    def data_module(self, spec):
        """Return the standard tabular data module."""
        return TabularDataModule()

    def build_model(self, spec, signature):
        """Return an unmaterialised linear model."""
        del spec
        return LinearModel(n_features=int(signature.dynamic["features"].shape[-1]))


def write_linear_dataset(path: Path, *, n_rows: int = 200, seed: int = 11) -> Path:
    """
    Write an exactly-linear dataset to a CSV file.

    No noise, because the engine solves least squares exactly -- so a
    correct run must recover the coefficients, and a job that silently
    trained on the wrong data cannot produce the same score by chance.

    Parameters
    ----------
    path
        Where to write.
    n_rows
        How many rows.
    seed
        Seed for the feature draw.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_rows, N_FEATURES))
    targets = features @ TRUE_COEFFICIENTS + TRUE_INTERCEPT

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"f{index}" for index in range(N_FEATURES)] + ["target"])
        writer.writerows(np.column_stack([features, targets]).tolist())
    return path


def job_set_payload(dataset: Path, output_root: Path, **overrides: object) -> dict[str, object]:
    """
    Build a two-job job-set payload against the synthetic model.

    Parameters
    ----------
    dataset
        Path to the CSV file every job reads.
    output_root
        Where the set writes.
    **overrides
        Top-level job-set keys to replace, most often ``jobs`` or
        ``placement``.

    Returns
    -------
    dict
        A payload for :func:`~rade_qnet.core.spec.jobs.parse_job_set_spec`.
    """
    payload: dict[str, object] = {
        "name": "set",
        "output_root": str(output_root),
        "defaults": {
            "model": {"name": MODEL_NAME},
            "source": {"kind": "tabular", "path": str(dataset)},
            "training": {"engine": ENGINE_TAG},
            "reports": {"enabled": []},
        },
        "jobs": [{"id": "a"}, {"id": "b"}],
    }
    payload.update(overrides)
    return payload


#: The name the directory-reading model registers under. A separate model
#: rather than an option on the first, because the two differ in the one
#: thing that matters here: where their data comes from.
DIRECTORY_MODEL_NAME = "synthetic_directory"

#: The file each group directory is expected to contain.
GROUP_FILENAME = "data.csv"


class DirectoryDataModule(TabularDataModule):
    """
    Reads its table from a directory named in ``source.params``.

    What a data group looks like on disk: a directory per group, which is
    the shape :func:`~rade_qnet.orchestration.jobs.fanout.group_overrides`
    points each job at. The tabular module reads a path from the spec
    directly, so the only thing changed here is where the path comes from.
    """

    def load(self, spec):
        """
        Read the group's table.

        Parameters
        ----------
        spec
            A model source specification whose ``params`` name a directory.

        Returns
        -------
        TableData
            The parsed table.
        """
        return read_table(
            Path(spec.params["directory"]) / GROUP_FILENAME,
            target_column="target",
            feature_columns=None,
            attribute_columns=(),
        )


class SyntheticDirectoryModel(SyntheticSupervisedModel):
    """The synthetic model, reading one directory per cluster."""

    component_name = DIRECTORY_MODEL_NAME

    def data_module(self, spec):
        """Return the directory-reading data module."""
        del spec
        return DirectoryDataModule()
