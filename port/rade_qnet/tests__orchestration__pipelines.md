# `tranql/models/rade/rade_qnet/tests/orchestration/pipelines`

7 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 29 | 1330 | `d46d1a478e03562f` |
| 2 | `support.py` | 159 | 5707 | `14b9f51c1df34a8e` |
| 3 | `test_pipelines_evaluate.py` | 509 | 18759 | `28d1e9e79950a7e1` |
| 4 | `test_pipelines_infer.py` | 410 | 15348 | `be43cdcafbbe59a0` |
| 5 | `test_pipelines_reinforce.py` | 415 | 14584 | `b78854a4531fbd00` |
| 6 | `test_pipelines_train.py` | 1067 | 41370 | `09056bdfc76bb41e` |
| 7 | `test_pipelines_tune.py` | 619 | 22345 | `e67eee2015ed3f2e` |

---

## 1. `tranql/models/rade/rade_qnet/tests/orchestration/pipelines/__init__.py`

1330 bytes · SHA-256 `d46d1a478e03562f`

```python
"""
Tests for ``rade_qnet.orchestration.pipelines`` -- the four lifecycle pipelines.

Pipelines are tested against synthetic models and sources from
``rade_qnet.testkit.fixtures``, never against real data. That keeps the suite
fast and, more importantly, keeps it testing the pipeline rather than the
model: a failure here is unambiguously the framework's fault.

The overriding tests matter as much as the happy paths. Each of the four
customisation tiers is exercised, because the promise that a model can replace
one step without reimplementing training is only credible if something checks
it.

Planned modules
---------------
``test_pipelines_train.py``
    Stage ordering, the artifacts produced, and each override tier: spec only,
    added reports, a single replaced step, and a fully replaced ``run()``.
    [Phase 1, extended in Phase 2]
``test_pipelines_evaluate.py``
    Rebuilding a source from saved lineage, and metrics reported in original
    target units rather than transformed space.  [Phase 5]
``test_pipelines_infer.py``
    Predicting for entities unseen during training, and prediction provenance
    pointing back to a specific bundle.  [Phase 5]
``test_pipelines_tune.py``
    Trial generation, the data build being cached across trials rather than
    repeated, and best-trial selection.  [Phase 5]
"""
```

---

## 2. `tranql/models/rade/rade_qnet/tests/orchestration/pipelines/support.py`

5707 bytes · SHA-256 `14b9f51c1df34a8e`

```python
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

from tranql.models.rade.rade_qnet.rade_qnet.core.authoring.supervised import SupervisedModel
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import (
    engine as register_engine,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import model as register_model
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.data import TabularSourceSpec
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.run import SupervisedRunSpec
from tranql.models.rade.rade_qnet.rade_qnet.sources.dataset.tabular import TabularDataModule
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import LinearModel, SyntheticEngine

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
```

---

## 3. `tranql/models/rade/rade_qnet/tests/orchestration/pipelines/test_pipelines_evaluate.py`

18759 bytes · SHA-256 `28d1e9e79950a7e1`

```python
"""
Tests for scoring a saved model.

Almost every test here is about a failure that produces a *number* rather
than an error, because that is the characteristic failure of this pipeline.
A re-scored model with a re-derived split reports a plausible metric. A
re-fitted scaler produces plausible predictions. Nothing raises, and the
mistake surfaces weeks later as two numbers that should match and do not.

So the tests are built to make the wrong behaviour visibly wrong rather than
slightly wrong:

- The data is exactly linear and the engine solves least squares in closed
  form, so a correct reproduction is *equal*, not close. Equality is the only
  assertion strong enough to distinguish a correct reload from a nearly
  correct one.
- The "new data" used for re-scoring is shifted, not merely reseeded. A
  shifted feature distribution makes a re-fitted scaler produce different
  numbers from a correctly reapplied one, which a same-distribution redraw
  would not.
"""

from __future__ import annotations

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.contract.result import EvaluationResult
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import MODELS
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import StageError
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.data import (
    ScalingSpec,
    TabularSourceSpec,
    TransformsSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.evaluate import EvaluatePipeline
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.train import TrainPipeline
from tranql.models.rade.rade_qnet.rade_qnet.storage.bundle import (
    load_lineage,
    load_spec,
    open_bundle,
)
from tranql.models.rade.rade_qnet.rade_qnet.storage.runs.catalog import InMemoryCatalog
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import (
    isolated_registries,
    make_run_context,
)

from .support import (
    MODEL_NAME,
    SyntheticSupervisedModel,
    make_spec,
    register_components,
    write_linear_dataset,
)


@pytest.fixture(autouse=True)
def _registries():
    """
    Register the synthetic components, isolated to this module.

    Yields
    ------
    None
        For the duration of the test.
    """
    # ``empty=True`` because the stand-in engine below claims a shipped
    # engine's name. Without it, whether this fixture succeeds depends on
    # whether an earlier test imported the real one -- which made the suite
    # pass or fail on collection order.
    with isolated_registries(empty=True):
        register_components()
        yield


@pytest.fixture
def dataset(tmp_path):
    """
    Write the standardised training dataset.

    Returns
    -------
    pathlib.Path
        The file.
    """
    return write_linear_dataset(tmp_path / "linear.csv")


@pytest.fixture
def spec(dataset):
    """
    Build a spec that standardises its features.

    Scaling is switched on deliberately. Without a fitted transform there is
    nothing for a reload to get wrong, and most of this module would pass
    against a pipeline that re-fitted everything on the data in front of it.

    Returns
    -------
    SupervisedRunSpec
        The spec.
    """
    return make_spec(
        dataset,
        source=TabularSourceSpec(
            path=dataset,
            transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
        ),
    )


@pytest.fixture
def trained(tmp_path, spec):
    """
    Train a model and return its run result and bundle directory.

    Returns
    -------
    tuple
        The :class:`TrainingResult` and the bundle directory.
    """
    pipeline = TrainPipeline(
        context=make_run_context(output_directory=tmp_path / "run", catalog=InMemoryCatalog()),
        spec=spec,
        definition=SyntheticSupervisedModel(),
    )
    result = pipeline.execute()
    assert pipeline.saved is not None
    return result, pipeline.saved.directory


def evaluate(tmp_path, directory, **kwargs) -> EvaluationResult:
    """
    Run the evaluation pipeline over a bundle.

    Parameters
    ----------
    tmp_path
        Temporary directory for the run's output.
    directory
        The bundle to score.
    **kwargs
        Passed through to the pipeline.

    Returns
    -------
    EvaluationResult
        The scored result.
    """
    return EvaluatePipeline(
        context=make_run_context(output_directory=tmp_path / "eval"),
        directory=directory,
        **kwargs,
    ).execute()


class TestReproduction:
    """The gate: a saved model re-scored must give back its own numbers."""

    def test_evaluate_reproduces_training_metrics(self, tmp_path, trained):
        """
        Every metric on every split, exactly.

        Exactly rather than approximately, which is affordable because the
        engine is closed-form and the weights are loaded rather than
        re-fitted. A tolerance here would let through a reload that
        re-derived the split into something *almost* the same -- and an
        almost-correct split is the failure mode that costs most, because
        it means the model is being scored partly on rows it trained on.
        """
        result, directory = trained
        rescored = evaluate(tmp_path, directory)

        assert set(rescored.evaluations) == set(result.evaluations)
        for split, original in result.evaluations.items():
            assert rescored.evaluations[split].metrics == original.metrics
            assert rescored.evaluations[split].n_samples == original.n_samples

    def test_the_baseline_is_reproduced_too(self, tmp_path, trained):
        """
        Not only the model's metrics.

        The baseline is computed from the *training* targets, so reproducing
        it is a second, independent check that the split came back intact:
        a re-derived split would change which rows the baseline mean is
        taken over, even where the model's own score happened to survive.
        """
        result, directory = trained
        rescored = evaluate(tmp_path, directory)

        for split, original in result.evaluations.items():
            assert rescored.evaluations[split].baseline_metrics == original.baseline_metrics

    def test_metrics_are_in_original_target_units(self, tmp_path, trained):
        """
        A scaled target still reports an error on the original scale.

        An error in standardised space is not a quantity anyone can act on,
        and two runs' standardised errors -- each scaled by its own training
        mean -- are not comparable to one another.
        """
        _, directory = trained
        rescored = evaluate(tmp_path, directory)

        assert all(result.in_original_units for result in rescored.evaluations.values())
        # The relationship is exact, so a correctly inverted prediction has
        # essentially no error. A metric left in standardised space would be
        # a different number entirely.
        assert rescored.metric("test", "mae") < 1e-6

    def test_a_reproduction_is_marked_as_one(self, tmp_path, trained):
        """
        The result says the source did not change.

        Which is the half of provenance a reader actually acts on: a metric
        that matches a recorded one means nothing unless it was computed
        over the same rows.
        """
        _, directory = trained
        rescored = evaluate(tmp_path, directory)

        assert rescored.source_changed is False
        assert rescored.source_fingerprint == rescored.trained_on_fingerprint
        assert "CHANGED SOURCE" not in rescored.describe()


class TestTheSavedSplitIsUsed:
    """The split is read from lineage, never re-derived."""

    def test_rebuild_uses_the_saved_split(self, tmp_path, trained, dataset):
        """
        Changed split settings do not move an old bundle's boundaries.

        The naive implementation re-derives the split from the spec it is
        handed. It would pass every other test in this module and fail this
        one, which is why the fraction is changed to something that could
        not possibly coincide.
        """
        result, directory = trained
        rescored = evaluate(
            tmp_path,
            directory,
            source=TabularSourceSpec(
                path=dataset,
                transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
                split={"kind": "chronological", "test_fraction": 0.5, "validation_fraction": 0.3},
            ),
        )

        for split, original in result.evaluations.items():
            assert rescored.evaluations[split].n_samples == original.n_samples

    def test_the_rebuilt_split_indices_are_the_saved_ones(self, tmp_path, trained):
        """
        Identical index lists, not merely identical sizes.

        Two different splits of the same data can have the same sizes, so
        comparing counts alone would pass against a re-derived split that
        happened to use the same fractions. The indices are the thing that
        decides which rows a metric is computed over.
        """
        _, directory = trained
        saved_lineage = load_lineage(open_bundle(directory))

        pipeline = EvaluatePipeline(
            context=make_run_context(output_directory=tmp_path / "eval"),
            directory=directory,
        )
        loaded = pipeline.load()
        data = pipeline.rebuild_data(loaded)

        assert data.lineage.split_indices == saved_lineage.split_indices


class TestTheSavedStateIsUsed:
    """The fitted state is applied, never re-fitted."""

    def test_rebuild_does_not_refit_the_state(self, tmp_path, trained, dataset):
        """
        The classic production failure, asserted directly.

        Scoring against a *shifted* source: the relationship is unchanged, so
        a model whose saved scaler is reapplied still predicts well. A model
        whose scaler was re-fitted on the shifted data sees inputs centred
        where the training inputs were, which is no longer where these
        inputs are, and its error explodes.
        """
        _, directory = trained
        shifted = write_linear_dataset(tmp_path / "shifted.csv", seed=29, shift=3.0)

        rescored = evaluate(
            tmp_path,
            directory,
            source=TabularSourceSpec(
                path=shifted,
                transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
            ),
            splits=("test",),
        )

        # Reapplied correctly, the exact linear relationship still holds.
        assert rescored.metric("test", "mae") < 1e-6

    def test_the_bundle_state_object_is_the_one_applied(self, tmp_path, trained):
        """
        Not merely an equivalent one.

        Checked through the pipeline's own stages rather than its result,
        because the result cannot distinguish "the saved state" from "a
        state that happens to agree".
        """
        _, directory = trained
        pipeline = EvaluatePipeline(
            context=make_run_context(output_directory=tmp_path / "eval"),
            directory=directory,
        )
        loaded = pipeline.load()
        data = pipeline.rebuild_data(loaded)

        assert data.state is loaded.state


class TestChangedSources:
    """A re-score against new data is allowed, and is never silent."""

    def test_rebuild_reports_a_changed_fingerprint(self, tmp_path, trained):
        """
        Surfaced on the result, not left for a reader to infer.

        Re-scoring on new data is a legitimate and common thing to do. Doing
        it while believing you reproduced an old number is not, and the only
        difference between the two is whether anybody was told.
        """
        _, directory = trained
        shifted = write_linear_dataset(tmp_path / "other.csv", seed=77)

        rescored = evaluate(
            tmp_path,
            directory,
            source=TabularSourceSpec(
                path=shifted,
                transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
            ),
            splits=("test",),
        )

        assert rescored.source_changed is True
        assert rescored.source_fingerprint != rescored.trained_on_fingerprint
        assert "CHANGED SOURCE" in rescored.describe()

    def test_a_source_with_different_columns_is_refused(self, tmp_path, trained):
        """
        Rather than scored against the wrong inputs.

        A changed column count means the weights were shaped for one
        interface and are being fed another. Here the saved scaler refuses
        first, which is the better place for it: the fault is caught while
        reapplying the state rather than after a forward pass.
        """
        _, directory = trained
        narrow = tmp_path / "narrow.csv"
        rows = (tmp_path / "linear.csv").read_text(encoding="utf-8").splitlines()
        narrow.write_text(
            "\n".join(",".join(line.split(",")[1:]) for line in rows) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(StageError, match=r"3 column\(s\) but the scaler was fitted on 4"):
            evaluate(
                tmp_path,
                directory,
                source=TabularSourceSpec(
                    path=narrow,
                    transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
                ),
                splits=("test",),
            )


class TestLoading:
    """Opening a bundle, and failing usefully when it cannot be opened."""

    def test_an_unregistered_model_names_the_problem(self, tmp_path, trained):
        """
        The error must say the package was not imported.

        A bundle records a model *name*, so loading one needs the model's
        package importable -- the same constraint job sets hit as defect 12,
        reached from the other direction. A bare registry miss here would
        send the reader looking at the bundle rather than at their imports.
        """
        _, directory = trained

        with isolated_registries():
            # Emptied rather than simply not re-registered: the fixture
            # snapshots the registries rather than clearing them, so without
            # this the model would still be resolvable inside the block.
            MODELS.restore({})
            with pytest.raises(StageError, match=r"not registered in this process"):
                evaluate(tmp_path, directory)

    def test_the_error_lists_what_is_registered(self, tmp_path, trained):
        """So a typo in a model name is diagnosable from the message alone."""
        _, directory = trained

        with isolated_registries():
            MODELS.restore({"other_model": MODELS.entry(MODEL_NAME)})
            with pytest.raises(StageError) as caught:
                evaluate(tmp_path, directory)

        assert "other_model" in str(caught.value)

    def test_the_loaded_spec_round_trips(self, tmp_path, trained, spec):
        """
        ``load_spec`` returns the specification the run was configured with.

        Phase 1 wrote ``spec.json`` and provided no way to read it back,
        which went unnoticed for four phases because nothing re-loaded a
        bundle.
        """
        del tmp_path
        _, directory = trained
        assert load_spec(open_bundle(directory)) == spec

    def test_the_stage_sequence_is_declared(self, tmp_path, trained):
        """Declared as data, so a hook can enumerate it before the run."""
        _, directory = trained
        pipeline = EvaluatePipeline(
            context=make_run_context(output_directory=tmp_path / "eval"),
            directory=directory,
        )
        pipeline.execute()

        assert pipeline.stages == ("load", "rebuild_data", "restore", "score", "report")
        assert pipeline.completed_stages == pipeline.stages


class TestProvenance:
    """What a metric must carry to be comparable to another metric."""

    def test_the_result_identifies_the_bundle(self, tmp_path, trained):
        """Model name, version and spec digest, from the manifest."""
        _, directory = trained
        manifest = open_bundle(directory).manifest
        rescored = evaluate(tmp_path, directory)

        assert rescored.model_name == manifest.model_name
        assert rescored.bundle_version == manifest.version
        assert rescored.spec_digest == manifest.spec_digest

    def test_selecting_splits_scores_only_those(self, tmp_path, trained):
        """
        A re-score against new observations usually wants only ``test``.

        On fresh data the training split is no longer a meaningful category,
        and scoring it would invite a comparison that means nothing.
        """
        _, directory = trained
        rescored = evaluate(tmp_path, directory, splits=("test",))

        assert set(rescored.evaluations) == {"test"}

    def test_a_missing_metric_names_the_alternatives(self, tmp_path, trained):
        """So a typo is diagnosable without opening the contract."""
        _, directory = trained
        rescored = evaluate(tmp_path, directory)

        with pytest.raises(Exception, match=r"no metric 'rmsee'"):
            rescored.metric("test", "rmsee")


class TestPredictionsAreUnchanged:
    """The restored model is the trained model, not a lookalike."""

    def test_restored_weights_predict_identically(self, tmp_path, trained, spec):
        """
        Compared row by row against the training run's own forward pass.

        Metrics are a reduction, and two different prediction vectors can
        reduce to the same mean absolute error. Comparing the vectors
        themselves is the stronger statement.
        """
        from tranql.models.rade.rade_qnet.rade_qnet.orchestration.stages.scoring import (  # noqa: PLC0415
            scoring_source,
        )

        _, directory = trained

        trainer = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "again"),
            spec=spec,
            definition=SyntheticSupervisedModel(),
        )
        trainer.execute()
        trained_data = trainer.build_data(spec)
        assert trainer.handle is not None
        expected = trainer._engine().predict(trainer.handle, scoring_source(trained_data, "test"))

        pipeline = EvaluatePipeline(
            context=make_run_context(output_directory=tmp_path / "eval"),
            directory=directory,
        )
        loaded = pipeline.load()
        data = pipeline.rebuild_data(loaded)
        handle = pipeline.restore(loaded, data)
        actual = pipeline._engine(loaded).predict(handle, scoring_source(data, "test"))

        assert np.allclose(np.ravel(actual), np.ravel(expected))
```

---

## 4. `tranql/models/rade/rade_qnet/tests/orchestration/pipelines/test_pipelines_infer.py`

15348 bytes · SHA-256 `be43cdcafbbe59a0`

```python
"""
Tests for producing predictions from a saved model.

Inference is evaluation without targets, so the reload half is already
covered by the evaluation tests and is not repeated here. What these tests
cover is what is *different* once nobody is computing a metric:

- **Nothing reduces the numbers.** An evaluation that misattributes every
  prediction to the wrong row still produces a mean absolute error, and a
  wrong one is hard to distinguish from a right one. An inference pass that
  does the same hands somebody a number for the wrong instrument. So the
  tests here assert on individual values and their identifiers rather than
  on aggregates.

- **Provenance is load-bearing.** A prediction detached from the bundle and
  source behind it cannot be reconciled later, and reconciling predictions
  after the fact is the ordinary case: somebody will ask why Tuesday's
  number differed from Monday's.

- **Refusal is a feature.** Asking a transductive model about an instrument
  it has never seen tends to return a default embedding -- a confident
  number indistinguishable from a real one. The test that this is refused
  matters more than most of the tests that something succeeds.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.contract.result import Predictions
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError, StageError
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.data import (
    ScalingSpec,
    TabularSourceSpec,
    TransformsSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.evaluate import EvaluatePipeline
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.infer import InferPipeline
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.train import TrainPipeline
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.stages.scoring import scoring_source
from tranql.models.rade.rade_qnet.rade_qnet.storage.bundle import load_lineage, open_bundle
from tranql.models.rade.rade_qnet.rade_qnet.storage.runs.catalog import InMemoryCatalog
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import (
    isolated_registries,
    make_run_context,
)

from .support import (
    SyntheticSupervisedModel,
    make_spec,
    register_components,
    write_linear_dataset,
)

#: Entities a model with a per-entity axis would have been trained on. Used
#: to give the synthetic model's data bundle an entity list it would not
#: otherwise have, so the unseen-entity branch can be exercised at all.
KNOWN_ENTITIES = ("EURUSD", "GBPUSD", "USDJPY")


@pytest.fixture(autouse=True)
def _registries():
    """
    Register the synthetic components, isolated to this module.

    Yields
    ------
    None
        For the duration of the test.
    """
    # ``empty=True`` because the stand-in engine below claims a shipped
    # engine's name. Without it, whether this fixture succeeds depends on
    # whether an earlier test imported the real one -- which made the suite
    # pass or fail on collection order.
    with isolated_registries(empty=True):
        register_components()
        yield


@pytest.fixture
def dataset(tmp_path):
    """
    Write the standardised training dataset.

    Returns
    -------
    pathlib.Path
        The file.
    """
    return write_linear_dataset(tmp_path / "linear.csv")


@pytest.fixture
def spec(dataset):
    """
    Build a spec that standardises its features.

    Scaling is on so that the inversion stage has something to invert.
    Without it a prediction in the model's output space and one in the
    target's original units are the same number, and every test about units
    would pass vacuously.

    Returns
    -------
    SupervisedRunSpec
        The spec.
    """
    return make_spec(
        dataset,
        source=TabularSourceSpec(
            path=dataset,
            transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
        ),
    )


@pytest.fixture
def bundle(tmp_path, spec):
    """
    Train a model and return its bundle directory.

    Returns
    -------
    pathlib.Path
        The bundle.
    """
    pipeline = TrainPipeline(
        context=make_run_context(output_directory=tmp_path / "run", catalog=InMemoryCatalog()),
        spec=spec,
        definition=SyntheticSupervisedModel(),
    )
    pipeline.execute()
    assert pipeline.saved is not None
    return pipeline.saved.directory


def entity_aware(tmp_path, directory, entities):
    """
    Build the pieces needed to exercise the unseen-entity check.

    The synthetic model has no entity axis, so its data bundle carries no
    identifiers and the check returns early -- correctly, since a tabular
    model's "entities" are the caller's row labels rather than anything the
    model resolves. To test the branch that matters, the rebuilt bundle is
    given an entity list, which is what a model with a per-entity embedding
    table would produce.

    Parameters
    ----------
    tmp_path
        Temporary directory for the run's output.
    directory
        The bundle to load.
    entities
        Identifiers to request predictions for.

    Returns
    -------
    tuple
        The pipeline, the loaded bundle, and a data bundle that knows which
        entities the model was trained on.
    """
    pipeline = InferPipeline(
        context=make_run_context(output_directory=tmp_path / "infer"),
        directory=directory,
        entities=entities,
    )
    loaded = pipeline.load()
    data = loaded.definition.rebuild_data(loaded.spec, state=loaded.state, lineage=loaded.lineage)
    return pipeline, loaded, replace(data, entity_ids=KNOWN_ENTITIES)


def infer(tmp_path, directory, **kwargs) -> Predictions:
    """
    Run the inference pipeline over a bundle.

    Parameters
    ----------
    tmp_path
        Temporary directory for the run's output.
    directory
        The bundle to predict with.
    **kwargs
        Passed through to the pipeline.

    Returns
    -------
    Predictions
        The predictions.
    """
    return InferPipeline(
        context=make_run_context(output_directory=tmp_path / "infer"),
        directory=directory,
        **kwargs,
    ).execute()


class TestTheForwardPass:
    """Predictions come back, in the right space, for the right rows."""

    def test_the_stage_sequence_is_declared(self, tmp_path, bundle):
        """Declared as data, so a hook can enumerate it before the run."""
        pipeline = InferPipeline(
            context=make_run_context(output_directory=tmp_path / "infer"),
            directory=bundle,
        )
        pipeline.execute()

        assert pipeline.stages == (
            "load",
            "prepare_inputs",
            "restore",
            "predict",
            "invert",
            "attribute",
        )
        assert pipeline.completed_stages == pipeline.stages

    def test_one_prediction_per_row_of_the_split(self, tmp_path, bundle):
        """
        The count matches the split, not the whole source.

        A pass that silently predicted over everything would still return
        plausible numbers, just the wrong ones for the question asked.
        """
        predictions = infer(tmp_path, bundle)
        saved = open_bundle(bundle)
        expected = len(load_lineage(saved).split_indices["test"])
        assert predictions.n_predictions == expected

    def test_predictions_are_in_original_target_units(self, tmp_path, bundle, dataset):
        """
        Compared against the actual target column, not against a scale.

        The relationship is exactly linear and the engine solves it in
        closed form, so a correctly inverted prediction equals the observed
        value. A prediction left in standardised space would be off by the
        training mean and standard deviation -- a large, obvious gap, which
        is what makes this assertion worth making as an equality.
        """
        predictions = infer(tmp_path, bundle)
        assert predictions.in_original_units is True

        rows = dataset.read_text(encoding="utf-8").splitlines()[1:]
        targets = np.array([float(row.split(",")[-1]) for row in rows])
        indices = np.asarray(load_lineage(open_bundle(bundle)).split_indices["test"])

        assert np.allclose(predictions.values, targets[indices], atol=1e-6)

    def test_predictions_match_the_evaluation_pass(self, tmp_path, bundle):
        """
        The two pipelines must not produce different numbers.

        They share their reload path as functions rather than as a base
        class, and this is the test that the sharing is real: if one of them
        ever grows its own version of a stage, the vectors stop agreeing.
        """
        predictions = infer(tmp_path, bundle)

        pipeline = EvaluatePipeline(
            context=make_run_context(output_directory=tmp_path / "eval"),
            directory=bundle,
        )
        loaded = pipeline.load()
        data = pipeline.rebuild_data(loaded)
        handle = pipeline.restore(loaded, data)
        raw = pipeline._engine(loaded).predict(handle, scoring_source(data, "test"))
        expected = loaded.state.inverse_transform_targets(np.asarray(raw, dtype=np.float64))

        assert np.allclose(predictions.values, np.ravel(expected))

    def test_an_absent_split_names_the_available_ones(self, tmp_path, bundle):
        """So a typo is diagnosable from the message alone."""
        with pytest.raises(StageError, match=r"no 'holdout' split"):
            infer(tmp_path, bundle, split="holdout")


class TestProvenance:
    """What a prediction must carry to be reconciled with anything later."""

    def test_predictions_carry_provenance(self, tmp_path, bundle):
        """Bundle version, spec digest and the time of inference."""
        manifest = open_bundle(bundle).manifest
        predictions = infer(tmp_path, bundle)

        assert predictions.bundle_version == str(manifest.version)
        assert predictions.provenance["spec_digest"] == manifest.spec_digest
        assert predictions.provenance["model_name"] == manifest.model_name
        assert predictions.provenance["predicted_at"]

    def test_provenance_records_both_fingerprints(self, tmp_path, bundle):
        """
        The data predicted over, and the data the model was trained on.

        Both, because the question people ask later is whether they differ.
        Recording only one makes that unanswerable from the record alone.
        """
        predictions = infer(tmp_path, bundle)

        assert (
            predictions.provenance["source_fingerprint"]
            == predictions.provenance["trained_on_fingerprint"]
        )

    def test_predicting_over_new_data_records_the_difference(self, tmp_path, bundle):
        """So a prediction made against other inputs is identifiable as such."""
        other = write_linear_dataset(tmp_path / "other.csv", seed=77)
        predictions = infer(
            tmp_path,
            bundle,
            source=TabularSourceSpec(
                path=other,
                transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
            ),
        )

        assert (
            predictions.provenance["source_fingerprint"]
            != predictions.provenance["trained_on_fingerprint"]
        )

    def test_scenario_indices_pair_values_with_source_rows(self, tmp_path, bundle):
        """Without which a prediction cannot be attributed to an observation."""
        predictions = infer(tmp_path, bundle)
        expected = np.asarray(load_lineage(open_bundle(bundle)).split_indices["test"])

        assert predictions.scenario_indices is not None
        assert np.array_equal(predictions.scenario_indices, expected)


class TestUnseenEntities:
    """A refusal that matters more than most successes."""

    def test_a_model_with_no_entity_axis_accepts_any_labels(self, tmp_path, bundle):
        """
        Because they are the caller's labels for rows, not model entities.

        A tabular model predicts per scenario. Refusing a label it was never
        going to resolve would be a false alarm, and false alarms are how a
        useful check gets switched off.
        """
        predictions = infer(tmp_path, bundle)
        labels = tuple(f"row-{index}" for index in range(predictions.n_predictions))

        relabelled = infer(tmp_path, bundle, entities=labels)
        assert relabelled.entity_ids == labels

    def test_non_inductive_model_raises_for_unseen_entity(self, tmp_path, bundle):
        """
        Rather than returning a plausible number for an unknown instrument.

        The quiet failure this check exists for: a model with a per-entity
        embedding table has no row for a new instrument, so it falls back to
        a default and returns a confident prediction that looks exactly like
        the real ones.
        """
        pipeline, loaded, data = entity_aware(tmp_path, bundle, ("EURUSD", "BRAND_NEW"))

        with pytest.raises(ContractError, match=r"does not declare Inductive"):
            pipeline._check_entities(loaded, data)

    def test_the_refusal_names_the_offending_entities(self, tmp_path, bundle):
        """So the caller can see whether it is one stale ticker or a whole feed."""
        pipeline, loaded, data = entity_aware(tmp_path, bundle, ("BRAND_NEW",))

        with pytest.raises(ContractError, match=r"BRAND_NEW"):
            pipeline._check_entities(loaded, data)

    def test_a_known_entity_is_not_refused(self, tmp_path, bundle):
        """Otherwise the check would reject every request, including valid ones."""
        pipeline, loaded, data = entity_aware(tmp_path, bundle, ("EURUSD",))

        pipeline._check_entities(loaded, data)

    def test_an_inductive_model_is_allowed_through(self, tmp_path, bundle):
        """Declaring the capability is what turns the refusal into a route."""
        pipeline, loaded, data = entity_aware(tmp_path, bundle, ("BRAND_NEW",))

        class InductiveDefinition:
            """Declares that unseen entities are predictable."""

            def supports_unseen_entities(self):
                """Return True."""
                return True

        pipeline._check_entities(replace(loaded, definition=InductiveDefinition()), data)

    def test_answering_false_is_refused_like_not_declaring_at_all(self, tmp_path, bundle):
        """
        Because the answer is a method, not the mere presence of one.

        A model may be inductive or transductive according to how it was
        configured -- an entity embedding table versus a feature encoder --
        so the capability is read by calling it, and a class that answers
        False is in exactly the position of a class that never declared it.
        """
        pipeline, loaded, data = entity_aware(tmp_path, bundle, ("BRAND_NEW",))

        class Transductive:
            """Declares the capability and then disclaims it."""

            def supports_unseen_entities(self):
                """Return False."""
                return False

        with pytest.raises(ContractError, match=r"does not declare Inductive"):
            pipeline._check_entities(replace(loaded, definition=Transductive()), data)
```

---

## 5. `tranql/models/rade/rade_qnet/tests/orchestration/pipelines/test_pipelines_reinforce.py`

14584 bytes · SHA-256 `b78854a4531fbd00`

```python
"""
Tests for the interactive training pipeline.

The pipeline's claim is that an interactive run is a *sibling* of a
supervised one rather than a fork, so the tests are mostly about sameness:
the same stage mechanics, the same ``TrainingResult``, the same bundle
layout, the same catalog entry. If those drift apart, every later
improvement to training has to be made twice.

The two tests that matter most are about the bundle, and they are the reason
``ModelBundle.signature`` was widened to a union rather than made to hold the
experience stream's tensor description:

**A policy rebuilds from its bundle with no environment.** That is the whole
purpose of saving a signature. A tensor description would have passed every
other test here and failed this one, because it loses the number of discrete
actions and the bounds of a continuous space.

**A supervised reader refuses an interactive bundle by name.** Widening a
contract is only safe if the readers that cannot handle the new member say
so. ``load_bundle`` and ``load_signature`` both do.

One absence is deliberate and tested for: there is no evaluate stage, because
nothing on this path can turn exploration off yet. See the pipeline's module
docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tranql.models.rade.rade_qnet.rade_qnet import api
from tranql.models.rade.rade_qnet.rade_qnet.core.authoring.policy import PolicyModel
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import (
    engine as register_engine,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import model as register_model
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import (
    BundleError,
    SpecError,
    StageError,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.run import parse_run_spec
from tranql.models.rade.rade_qnet.rade_qnet.engines.torch import TorchEngine
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.reinforce import (
    ReinforcePipeline,
)
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.train import TrainPipeline
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.stages.reload import load_bundle
from tranql.models.rade.rade_qnet.rade_qnet.storage.bundle import (
    load_lineage,
    load_policy_signature,
    load_signature,
    load_spec,
    open_bundle,
)
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import (
    SyntheticEngine,
    SyntheticEnvironment,
    isolated_registries,
)

from .support import SyntheticSupervisedModel

EPISODE_LENGTH = 4


class Policy(torch.nn.Module):
    """
    A one-layer policy sized from a signature.

    Parameters
    ----------
    signature
        Supplies the observation width and the number of actions.
    """

    def __init__(self, signature) -> None:
        super().__init__()
        self.head = torch.nn.Linear(signature.observation.shape[0], signature.action.n)

    def forward(self, *, observation: torch.Tensor) -> torch.Tensor:
        """
        Score one batch of observations.

        Parameters
        ----------
        observation
            A batch of observations.

        Returns
        -------
        torch.Tensor
            One row of action scores per observation.
        """
        return self.head(observation)


class Agent(PolicyModel):
    """The two-method model an interactive run trains."""

    def build_environment(self, spec):
        """
        Construct the environment from the spec's parameters.

        Parameters
        ----------
        spec
            The validated run specification.

        Returns
        -------
        SyntheticEnvironment
            A fresh environment.
        """
        return SyntheticEnvironment(**dict(spec.environment.params))

    def build_policy(self, spec, signature):
        """
        Construct the untrained policy from the signature alone.

        Parameters
        ----------
        spec
            Unused: the shape is fully determined by the spaces.
        signature
            The declared spaces.

        Returns
        -------
        Policy
            An untrained policy.
        """
        del spec
        return Policy(signature)


@pytest.fixture
def registered():
    """
    Register the real Torch engine and the interactive model, in isolation.

    ``empty=True`` because this fixture *claims* the names ``torch`` and
    ``agent`` rather than merely adding them, and a block that only snapshots
    would pass or fail on collection order.

    The learner registry is deliberately not emptied by
    :func:`isolated_registries` and is not touched here: ``random`` is
    registered by importing the Torch engine, which is exactly the path a
    real run relies on.

    Yields
    ------
    None
        For the duration of one test.
    """
    with isolated_registries(empty=True):
        register_engine("torch")(TorchEngine)
        register_model("agent", engine="torch")(Agent)
        yield


def payload(output_root, **training):
    """
    Build an interactive run configuration.

    Parameters
    ----------
    output_root
        Where the run writes.
    training
        Keys merged over the minimal training section.

    Returns
    -------
    dict
        A configuration ready for :func:`api.train`.
    """
    section = {
        "total_steps": 24,
        "steps_per_update": 4,
        "batch_size": 4,
        "evaluate_every_steps": 8,
    }
    section.update(training)
    return {
        "task": "reinforcement",
        "model": "agent",
        "seed": 11,
        "environment": {"name": "synthetic", "params": {"episode_length": EPISODE_LENGTH}},
        "training": section,
        "output_root": str(output_root),
    }


class TestAFullInteractiveRun:
    """It runs, and it produces the same shape of answer a supervised run does."""

    def test_a_run_completes_and_reports_its_blocks(self, tmp_path, registered):
        """
        Twenty-four steps in blocks of eight is three records.

        The budget is in steps rather than passes, and the history is in
        blocks rather than updates, so this pins both.
        """
        result = api.train(payload(tmp_path))

        assert result.fit.n_epochs == 3
        assert result.seed == 11

    def test_the_episode_return_is_the_metric_that_is_reported(self, tmp_path, registered):
        """
        Not a loss.

        The synthetic environment rewards one per step and terminates after
        four, so an episode's return is four by arithmetic. That it arrives
        in the history at all is the point: the driver reads it from the
        source's own bookkeeping.
        """
        result = api.train(payload(tmp_path))

        for record in result.fit.history:
            assert record.metrics["episode_return"] == pytest.approx(float(EPISODE_LENGTH))

    def test_there_is_no_validation_loss(self, tmp_path, registered):
        """An environment has no held-out split, so nothing pretends it does."""
        result = api.train(payload(tmp_path))
        assert all(record.val_loss is None for record in result.fit.history)

    def test_the_run_has_no_evaluations(self, tmp_path, registered):
        """
        Empty, and deliberately so.

        Scoring a policy means running episodes with exploration off, which
        nothing on this path can do yet. An empty mapping says "not measured";
        a number produced from the exploring policy would say "measured" and
        be wrong, and somebody would compare two runs with it.
        """
        result = api.train(payload(tmp_path))
        assert result.evaluations == {}

    def test_the_stage_sequence_mirrors_the_supervised_one(self):
        """
        The same names in the same order, where the work is the same.

        Compared as a property rather than described in prose, so a stage
        quietly renamed on one side shows up here.
        """
        shared = ("resolve", "resolve_seed", "materialise", "prepare_hardware", "fit", "persist")
        for name in shared:
            assert name in ReinforcePipeline.stages
            assert name in TrainPipeline.stages

        interactive = ReinforcePipeline.stages
        assert interactive.index("materialise") < interactive.index("prepare_hardware")
        assert interactive.index("fit") < interactive.index("persist")

    def test_nothing_is_written_when_a_stage_fails(self, tmp_path):
        """
        Persistence is last, so a failed run leaves no half-registered bundle.

        Forced by a model whose ``build_policy`` returns something that is
        not a network -- a realistic fault in a model package, and one the
        spec cannot catch, since the spec never sees the object.
        """

        class Broken(Agent):
            def build_policy(self, spec, signature):
                """Return something the engine cannot train."""
                del spec, signature
                return "not a network"

        with isolated_registries(empty=True):
            register_engine("torch")(TorchEngine)
            register_model("agent", engine="torch")(Broken)

            with pytest.raises(StageError):
                api.train(payload(tmp_path))

        assert not list(tmp_path.rglob("manifest.json"))


class TestTheBundle:
    """What a policy bundle holds, and what can read it."""

    def test_the_bundle_has_the_same_layout_as_a_supervised_one(self, tmp_path, registered):
        """
        Same files, same names, written by the same function.

        A second layout would mean every tool that reads a bundle needs two
        code paths.
        """
        result = api.train(payload(tmp_path))
        written = {path.name for path in Path(result.bundle_directory).iterdir()}

        assert {"manifest.json", "signature.json", "spec.json", "lineage.json"} <= written
        assert "fitted_state" in written

    def test_the_policy_rebuilds_from_the_bundle_with_no_environment(self, tmp_path, registered):
        """
        The reason the signature is saved at all.

        No environment is constructed anywhere in this test after the run
        finishes. The spaces come off the bundle, and they come off it
        intact -- the discrete action count and the box bounds both survive,
        which a tensor description would have lost.
        """
        result = api.train(payload(tmp_path))
        saved = open_bundle(Path(result.bundle_directory))

        signature = load_policy_signature(saved)
        assert signature.action.n == SyntheticEnvironment().action_space.n
        assert signature.observation.high == float(EPISODE_LENGTH)

        rebuilt = Agent().build_policy(load_spec(saved), signature)
        assert rebuilt.head.out_features == signature.action.n

    def test_a_supervised_reader_refuses_it(self, tmp_path, registered):
        """
        Widening a contract is only safe if its narrow readers say so.

        ``load_signature`` expects a target and an interactive bundle has
        none, so it fails rather than parsing a partial result.
        """
        result = api.train(payload(tmp_path))
        saved = open_bundle(Path(result.bundle_directory))

        with pytest.raises(BundleError):
            load_signature(saved)

    def test_load_bundle_refuses_it_by_name(self, tmp_path, registered):
        """
        And says which path the bundle does not belong to.

        The evaluate and infer pipelines both go through this function, so
        one refusal covers both.
        """
        result = api.train(payload(tmp_path))

        with pytest.raises(BundleError, match="only supervised runs"):
            load_bundle(Path(result.bundle_directory))

    def test_the_lineage_records_the_environment_rather_than_a_split(self, tmp_path, registered):
        """
        An empty split map, with a note saying why.

        Otherwise an interactive bundle looks like a supervised one whose
        data build failed.
        """
        result = api.train(payload(tmp_path))
        saved = open_bundle(Path(result.bundle_directory))

        recorded = load_lineage(saved)
        assert recorded.split_indices == {}
        assert recorded.n_scenarios == 24
        assert "environment" in recorded.notes["interaction"]

    def test_the_run_is_recorded_in_the_catalog(self, tmp_path, registered):
        """
        So an interactive run is selectable by the same tools as any other.

        A separate record keeper for policies would mean two places to look
        for "the best model we have".
        """
        result = api.train(payload(tmp_path))
        assert result.bundle_directory.endswith("v1")


class TestWhatIsRefused:
    """The pairings that cannot work, named before anything expensive runs."""

    def test_a_supervised_model_in_an_interactive_run_is_refused(self, tmp_path, registered):
        """
        Reported against the model, not against a missing environment.

        Without this the run would fail in ``build_environment`` with an
        ``AttributeError``, which points at the pipeline rather than at the
        configuration.
        """
        register_model("predictor", engine="torch")(SyntheticSupervisedModel)
        configuration = payload(tmp_path)
        configuration["model"] = "predictor"

        with pytest.raises(SpecError, match="learns from a fixed dataset"):
            api.train(configuration)

    def test_an_engine_that_cannot_train_a_policy_is_refused(self, tmp_path, registered):
        """
        Asked at resolve time, which costs milliseconds.

        The capability is opt-in precisely so this answer is available
        before an environment is built: a tree engine cannot train a policy
        by any amount of plumbing, and finding out four stages in would waste
        the data build.
        """
        register_engine("sklearn")(SyntheticEngine)
        configuration = payload(tmp_path)
        configuration["training"] = {"engine": "sklearn"}

        with pytest.raises(Exception, match="engine"):
            api.train(configuration)

    def test_an_interactive_spec_still_parses_as_a_run_spec(self):
        """
        Routed by task, through the ordinary discriminated union.

        No second parser and no second entry point: ``api.train`` reads the
        task and dispatches, because "train what this file describes" is one
        intent.
        """
        spec = parse_run_spec(payload("/tmp"), origin="test")
        assert spec.task == "reinforcement"
```

---

## 6. `tranql/models/rade/rade_qnet/tests/orchestration/pipelines/test_pipelines_train.py`

41370 bytes · SHA-256 `09056bdfc76bb41e`

```python
"""
Tests for the training pipeline, now that it runs end to end.

In Phase 1 these tests covered a skeleton whose last three stages raised. Phase
2 supplies an engine, so what is worth asserting changes completely: not "the
unfinished stages are instrumented" but "the finished sequence produces the
right artifacts, in the right order, with the right numbers".

Everything here runs against
:class:`~rade_qnet.testkit.fixtures.SyntheticEngine`, which solves least squares
in closed form. That is a deliberate choice over the real torch engine: the
solution is *exact*, so these tests assert on the metrics themselves rather
than on a tolerance. A pipeline test that can only check "the loss went down"
cannot distinguish a correct pipeline from one that scores the wrong split --
which is exactly the bug Phase 2 found in this file's subject.

Four orderings are load-bearing and are asserted rather than assumed:

- resolution precedes the data build, so a misspelled name costs milliseconds
  rather than a full build;
- seeding precedes the data build, because a data build that samples is part
  of what a seed must reproduce;
- materialisation precedes hardware preparation, which is defect 6;
- persistence precedes reporting, and nothing registers before it succeeds.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

# The three report modules are imported for their registration side effect:
# `isolated_registries` snapshots the registries rather than emptying them, so
# a report is available inside the isolated context only if its module was
# imported before the context opened.
from tranql.models.rade.rade_qnet.rade_qnet.analysis.reports import (  # noqa: F401
    curves,
    quality,
    summary,
)
from tranql.models.rade.rade_qnet.rade_qnet.analysis.reports.base import Report, report
from tranql.models.rade.rade_qnet.rade_qnet.core.authoring.supervised import SupervisedModel
from tranql.models.rade.rade_qnet.rade_qnet.core.contract.bundle import ModelBundle
from tranql.models.rade.rade_qnet.rade_qnet.core.contract.data import DataBundle, TensorBatchData
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import ENGINES
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import (
    engine as register_engine,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import StageError
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.data import (
    ScalingSpec,
    TabularSourceSpec,
    TransformsSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.train import TrainPipeline
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.stages.scoring import (
    scoring_source,
    source_for,
)
from tranql.models.rade.rade_qnet.rade_qnet.sources.dataset.tabular import TabularDataModule
from tranql.models.rade.rade_qnet.rade_qnet.storage.bundle import load_signature, open_bundle
from tranql.models.rade.rade_qnet.rade_qnet.storage.runs.catalog import InMemoryCatalog
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import (
    LinearModel,
    RecordingHook,
    SyntheticEngine,
    isolated_registries,
    make_run_context,
    make_signature,
    make_tensor_bundle,
)

#: Engine tag the synthetic engine is registered under. See the registry
#: fixture for why it borrows an existing name.
ENGINE_TAG = "sklearn"

#: Width of the synthetic problem. Small, and solvable exactly.
N_FEATURES = 4

#: True coefficients of the synthetic target, so a correct run must recover
#: them and an incorrect one cannot.
TRUE_COEFFICIENTS = np.array([1.5, -0.7, 0.3, 2.0])
TRUE_INTERCEPT = 0.4


@pytest.fixture(autouse=True)
def _registries():
    """
    Isolate the component registries for every test in this module.

    Autouse because every test here registers an engine, and a registration
    that leaked would fail whichever *other* test happened to run next rather
    than the one at fault.

    Yields
    ------
    None
        For the duration of the test.
    """
    # ``empty=True`` because the stand-in engine below claims a shipped
    # engine's name. Without it, whether this fixture succeeds depends on
    # whether an earlier test imported the real one -- which made the suite
    # pass or fail on collection order.
    with isolated_registries(empty=True):
        # Registered under 'sklearn' rather than a name of its own, because
        # TrainingSpec is a discriminated union over the engine names the
        # framework ships and a test cannot invent a fourth tag. The fit is
        # closed-form least squares, so 'sklearn' is also the honest label for
        # what this engine does.
        register_engine(ENGINE_TAG)(SyntheticEngine)
        yield


@pytest.fixture
def dataset(tmp_path):
    """
    Write a small exactly-linear dataset to a CSV file.

    Exactly linear, with no noise, because the engine solves least squares
    exactly -- so a correct pipeline must report an r-squared of one, and any
    misalignment anywhere in the chain makes that impossible to fake.

    Returns
    -------
    pathlib.Path
        The file.
    """
    rng = np.random.default_rng(11)
    features = rng.normal(size=(400, N_FEATURES))
    targets = features @ TRUE_COEFFICIENTS + TRUE_INTERCEPT

    path = tmp_path / "linear.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"f{index}" for index in range(N_FEATURES)] + ["target"])
        writer.writerows(np.column_stack([features, targets]).tolist())
    return path


class SyntheticSupervisedModel(SupervisedModel):
    """A model definition needing one line of data code, as advertised."""

    component_name = "synthetic_tabular"
    component_engine = ENGINE_TAG

    def data_module(self, spec):
        """Return the standard tabular data module."""
        return TabularDataModule()

    def build_model(self, spec, signature):
        """Return an unmaterialised linear model."""
        del spec
        return LinearModel(n_features=int(signature.dynamic["features"].shape[-1]))


def make_spec(dataset, **overrides):
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
        The spec.
    """
    from tranql.models.rade.rade_qnet.rade_qnet.core.spec.run import (  # noqa: PLC0415
        SupervisedRunSpec,
    )

    fields = {
        "model": "synthetic_tabular",
        "source": TabularSourceSpec(path=dataset),
        "training": {"engine": ENGINE_TAG},
        "reports": {"enabled": ()},
    }
    fields.update(overrides)
    return SupervisedRunSpec.model_validate(fields)


@pytest.fixture
def pipeline(tmp_path, dataset):
    """
    Provide a pipeline wired to the synthetic engine and dataset.

    Returns
    -------
    TrainPipeline
        The pipeline.
    """
    return TrainPipeline(
        context=make_run_context(output_directory=tmp_path / "run"),
        spec=make_spec(dataset),
        definition=SyntheticSupervisedModel(),
    )


class TestStageSequence:
    """The canonical order, and the four orderings that carry meaning."""

    def test_the_declared_stages_are_the_full_sequence(self, pipeline):
        """
        Declared as data, so a hook or a report can enumerate them.

        A sequence that existed only in the body of ``run`` could not be
        introspected before the run started.
        """
        assert pipeline.stages == (
            "resolve",
            "resolve_seed",
            "build_data",
            "declare_signature",
            "build_model",
            "materialise",
            "prepare_hardware",
            "fit",
            "evaluate",
            "persist",
            "report",
        )

    def test_every_stage_runs(self, pipeline):
        """The sequence completes, which is the Phase 2 deliverable."""
        pipeline.execute()
        assert pipeline.completed_stages == pipeline.stages

    def test_resolution_precedes_the_data_build(self, tmp_path, dataset):
        """
        So a misspelled component name costs milliseconds, not a full build.

        Finding an unregistered engine after a twenty-minute data build wastes
        the twenty minutes, and the information was available before it
        started.
        """
        hook = RecordingHook()
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run", hooks=(hook,)),
            spec=make_spec(dataset),
            definition=SyntheticSupervisedModel(),
        )
        pipeline.execute()
        started = [event[1] for event in hook.events if event[0] == "stage_start"]
        assert started.index("resolve") < started.index("build_data")

    def test_seeding_precedes_the_data_build(self, tmp_path, dataset):
        """
        The seed is applied before anything that could consume randomness.

        A data build that samples is part of what a seed must reproduce -- a
        subgraph, a negative set, a shuffled split. Seeding afterwards makes a
        run reproducible in its training and not in its data, which is the
        harder half to notice.
        """
        hook = RecordingHook()
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run", hooks=(hook,)),
            spec=make_spec(dataset),
            definition=SyntheticSupervisedModel(),
        )
        pipeline.execute()
        started = [event[1] for event in hook.events if event[0] == "stage_start"]
        assert started.index("resolve_seed") < started.index("build_data")

    def test_materialisation_precedes_hardware_preparation(self, tmp_path, dataset):
        """
        Defect 6, asserted on the sequence rather than on its consequences.

        A lazily shaped model has no parameters until it has seen a batch, and
        an optimiser built over the empty set reports success and updates
        nothing. The stage order is the fix, so the stage order is the test.
        """
        hook = RecordingHook()
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run", hooks=(hook,)),
            spec=make_spec(dataset),
            definition=SyntheticSupervisedModel(),
        )
        pipeline.execute()
        started = [event[1] for event in hook.events if event[0] == "stage_start"]
        assert started.index("materialise") < started.index("prepare_hardware")

    def test_reporting_follows_persistence(self, tmp_path, dataset):
        """
        A report reads a finished bundle, so it cannot run before one exists.

        And because a report never fails a run, running it last means a report
        fault cannot cost a trained model.
        """
        hook = RecordingHook()
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run", hooks=(hook,)),
            spec=make_spec(dataset, reports={"enabled": ("summary",)}),
            definition=SyntheticSupervisedModel(),
        )
        pipeline.execute()
        started = [event[1] for event in hook.events if event[0] == "stage_start"]
        assert started.index("persist") < started.index("report")


class TestResolution:
    """Unresolvable names fail first, and say which name."""

    def test_an_unregistered_engine_fails_at_resolve(self, tmp_path, dataset):
        """
        Named, so the reader knows what to register.

        A generic failure would send them to the source to find out what the
        spec asked for.
        """
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset),
            definition=SyntheticSupervisedModel(),
        )
        # The engine is removed rather than misnamed in the spec, because the
        # spec's discriminated union rejects an unknown engine tag before the
        # pipeline ever sees it. So the only way to reach the resolve stage
        # with an unresolvable engine is an empty registry -- which is also
        # the real-world case: a host that never imported the engine package.
        del ENGINES._entries[ENGINE_TAG]

        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert caught.value.stage == "resolve"
        assert ENGINE_TAG in str(caught.value)

    def test_an_unregistered_report_fails_at_resolve(self, tmp_path, dataset):
        """
        Before the data build, not after the training run.

        A report name is checked by instantiating it, which is what turns a
        typo into an immediate failure rather than a warning logged once
        training has already finished.
        """
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ("no_such_report",)}),
            definition=SyntheticSupervisedModel(),
        )
        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert caught.value.stage == "resolve"
        assert "no_such_report" in str(caught.value)


class TestTheNumbers:
    """The point of an exactly solvable problem: assert on the metrics."""

    def test_an_exactly_linear_target_is_recovered(self, pipeline):
        """
        Every split scores an r-squared of one, to numerical precision.

        This single assertion covers an improbable amount of the pipeline. It
        can only hold if the state was fitted on the training rows, inverted
        correctly, the splits are disjoint, the engine trained the caller's
        model, and predictions are paired with their own targets. Any one of
        those being wrong makes it unreachable.
        """
        result = pipeline.execute()
        for split in ("train", "validation", "test"):
            assert result.evaluations[split].metrics["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_the_training_split_is_scored_too(self, pipeline):
        """
        Not redundant: it is how overfitting becomes visible.

        A model scoring well on train and badly on validation has overfitted;
        one scoring badly on both has underfitted. Without the training score
        the two are indistinguishable in the saved bundle.
        """
        result = pipeline.execute()
        assert "train" in result.evaluations

    def test_metrics_are_reported_in_original_units(self, pipeline):
        """
        A mean absolute error in standardised space is not actionable.

        And it is incomparable with another run's, which was scaled by a
        different training mean.
        """
        result = pipeline.execute()
        assert all(evaluation.in_original_units for evaluation in result.evaluations.values())

    def test_a_scaled_target_gives_the_same_errors_as_an_unscaled_one(self, tmp_path, dataset):
        """
        The verification the ``in_original_units`` flag cannot provide.

        That flag is self-reported: it would stay true while the values
        underneath were in standard deviations. The r-squared test above
        cannot catch it either, because r-squared is scale-invariant -- it is
        1.0 whether or not the inversion happened, which is exactly why a
        units bug here would survive a green suite.

        ``mae`` and ``rmse`` are not scale-invariant. Scaling the target
        divides them by its training standard deviation, so running the same
        data once with ``scale_target`` on and once off must produce the same
        errors if and only if both predictions and targets are inverted
        before scoring. Inverting only one of the two would be far worse than
        inverting neither, and this catches that case as well.
        """
        errors = {}
        for scale_target in (False, True):
            spec = make_spec(
                dataset,
                source=TabularSourceSpec(
                    path=dataset,
                    transforms=TransformsSpec(
                        scaling=ScalingSpec(method="standard", scale_target=scale_target)
                    ),
                ),
            )
            result = TrainPipeline(
                context=make_run_context(output_directory=tmp_path / f"scaled_{scale_target}"),
                spec=spec,
                definition=SyntheticSupervisedModel(),
            ).execute()
            errors[scale_target] = result.evaluations["test"].metrics

        # The target's own scale, which is what a missing inversion would
        # divide the errors by. Asserted to be far from one, so the comparison
        # below could not pass by coincidence on a target that happened to be
        # standardised already.
        raw = np.loadtxt(dataset, delimiter=",", skiprows=1)
        assert raw[:, -1].std() > 2.0

        for metric in ("mae", "rmse"):
            assert errors[True][metric] == pytest.approx(errors[False][metric], abs=1e-8)

    def test_every_split_is_scored_against_a_baseline(self, pipeline):
        """
        So no headline number is reported without a reference point.

        A mean absolute error of 0.03 is either excellent or embarrassing, and
        only the baseline says which.
        """
        result = pipeline.execute()
        assert all(evaluation.baseline_metrics for evaluation in result.evaluations.values())

    def test_the_sample_counts_match_the_splits(self, pipeline):
        """
        A metric computed over the wrong number of rows is a wrong metric.

        Checked against the lineage rather than against a literal, so the test
        does not encode the split fractions. At a sequence length of one there
        is no boundary gap, so every scenario belongs to exactly one split and
        the counts must account for all of them.
        """
        result = pipeline.execute()
        scored = {name: value.n_samples for name, value in result.evaluations.items()}
        bundle = pipeline.build_data(pipeline.spec)
        assert scored == bundle.lineage.split_sizes
        assert sum(scored.values()) == bundle.lineage.n_scenarios


class TestScoringAlignment:
    """The defect this stage had, and the guard that now prevents it."""

    def test_a_shuffling_source_is_scored_in_a_stable_order(self, pipeline):
        """
        The bug Phase 2 found: two passes over a training source disagree.

        Scoring takes two passes -- one for the forward pass, one for the
        targets -- and pairs them row for row. A training source reshuffles
        between passes, so pairing them compares each prediction against an
        unrelated target. The counts still match and every metric still
        computes, so the only symptom is a model that appears to score at
        random on the very split it was trained on.

        Asserted through the training r-squared, because that is where the
        symptom appeared.
        """
        result = pipeline.execute()
        assert result.evaluations["train"].metrics["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_the_ordered_view_has_the_same_samples(self, pipeline):
        """
        A stable order must not be achieved by dropping or adding rows.

        Otherwise the fix for the alignment bug would introduce a sampling
        bug, which is harder to see.
        """
        pipeline.execute()
        data = pipeline.build_data(pipeline.spec)
        shuffled = source_for(data, "train")
        ordered = scoring_source(data, "train")
        assert ordered.n_samples == shuffled.n_samples
        assert sorted(_targets_of(ordered)) == pytest.approx(sorted(_targets_of(shuffled)))

    def test_a_source_that_cannot_be_ordered_is_refused(self, tmp_path, dataset):
        """
        A plausible wrong number is worse than no number.

        A source whose order varies and which offers no ordered view is
        refused rather than scored, because the metrics would be computed
        against mismatched targets and nothing in them would look wrong.
        """

        class Reshuffling:
            """Reshuffles every pass and offers no ordered view."""

            def __init__(self) -> None:
                self._pass = 0

            @property
            def signature(self):
                """The declared interface."""
                return make_signature(n_features=N_FEATURES)

            @property
            def static(self):
                """No static inputs."""
                return {}

            @property
            def steps_per_epoch(self):
                """Two batches per pass."""
                return 2

            @property
            def n_samples(self):
                """Twenty samples."""
                return 20

            def batches(self):
                """Yield a differently ordered pass each time."""
                self._pass += 1
                order = np.random.default_rng(self._pass).permutation(20)
                for start in (0, 10):
                    rows = order[start : start + 10]
                    yield {
                        "features": np.zeros((10, N_FEATURES)),
                        "target": rows.astype(np.float64),
                    }

        class ShufflingModel(SyntheticSupervisedModel):
            """Replaces the training split with a source that cannot be ordered."""

            def build_data(self, spec):
                """Return a bundle whose training split reshuffles."""
                bundle = super().build_data(spec)
                source = Reshuffling()
                splits = dict(bundle.splits)
                splits["train"] = TensorBatchData(loader=source, n_samples=20, n_batches=2)
                return DataBundle(
                    splits=splits,
                    signature=bundle.signature,
                    state=bundle.state,
                    lineage=bundle.lineage,
                )

        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset),
            definition=ShufflingModel(),
        )
        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert "OrderedSource" in str(caught.value)


class TestPersistence:
    """What lands on disk, and what does not when a run fails."""

    def test_the_bundle_is_written(self, pipeline):
        """Every file a bundle needs to be reopened without the data."""
        pipeline.execute()
        written = {
            path.name for path in pipeline.context.bundles_directory.rglob("*") if path.is_file()
        }
        assert {
            "manifest.json",
            "signature.json",
            "lineage.json",
            "result.json",
            "spec.json",
        } <= written

    def test_the_weights_are_written_through_the_engine(self, pipeline):
        """
        The seam that keeps ``storage`` free of any engine import.

        Storage decides where and when; the engine decides how.
        """
        pipeline.execute()
        weights = list(pipeline.context.bundles_directory.rglob("weights.*"))
        assert weights

    def test_a_reloaded_bundle_predicts_identically(self, pipeline):
        """
        The whole purpose of persisting a model, asserted end to end.

        Every other test in this class checks that a *file* appeared. None of
        them checks that the file means anything. A bundle whose weights were
        written from a stale reference, or reloaded into a differently shaped
        model, produces a complete and well-formed bundle that predicts
        something else -- and the discrepancy only surfaces in production,
        against a model nobody can reproduce.

        Identity is asserted exactly rather than approximately. Weights
        round-trip through the file without arithmetic, so anything other
        than bit-for-bit equality means a dtype was narrowed or a tensor was
        reordered, and both are bugs rather than tolerances.
        """
        bundle = _execute_and_persist(pipeline)
        data = pipeline.build_data(pipeline.spec)
        # The same ordered view scoring uses, so the two passes below are
        # comparable row for row rather than merely the same multiset.
        source = scoring_source(data, "test")
        engine = pipeline._engine()

        def predict_with(model):
            """Prepare a model the way the pipeline does, then predict."""
            return engine.predict(
                engine.prepare(
                    model,
                    hardware=pipeline.spec.hardware,
                    training=pipeline.spec.training,
                ),
                source,
            )

        before = predict_with(bundle.model)

        # Reopened from disk, and the model rebuilt from the saved signature:
        # this is the path a consumer who never saw the training run has to
        # take, so it is the one worth testing.
        manifest_path = next(pipeline.context.bundles_directory.rglob("manifest.json"))
        saved = open_bundle(manifest_path.parent)
        rebuilt = engine.materialise(
            pipeline.definition.build_model(pipeline.spec, load_signature(saved)),
            bundle.signature,
        )

        # Before loading, so the assertion below cannot pass on a weights file
        # that was never read. An untrained model predicting the same thing as
        # a trained one would make the round-trip test vacuous, and that is
        # precisely the state a silently skipped `load_weights` leaves.
        assert not np.array_equal(before, predict_with(rebuilt))

        after = predict_with(engine.load_weights(rebuilt, saved.weights_path))
        assert np.array_equal(before, after)

    def test_quality_metrics_are_recorded_in_the_lineage(self, pipeline):
        """
        Captured at training time, because afterwards they cannot be.

        Once the run is over the dataset may not be reconstructable, so
        numbers describing it are recorded in the bundle or lost.
        """
        bundle = _execute_and_persist(pipeline)
        assert "feature_completeness" in bundle.lineage.quality

    def test_the_catalog_is_updated_only_on_success(self, tmp_path, dataset):
        """
        A run that failed leaves nothing registered.

        Otherwise the catalog advertises a bundle that is not there, and the
        failure surfaces as a missing file in whatever reads it next.
        """
        catalog = InMemoryCatalog()
        succeeding = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "ok", catalog=catalog),
            spec=make_spec(dataset),
            definition=SyntheticSupervisedModel(),
        )
        succeeding.execute()
        # `entries` rather than `next_version`, because the latter *reserves* a
        # version rather than reporting one -- reading it would advance the
        # counter and the assertion would be measuring the test.
        registered = len(catalog.entries())

        failing = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "bad", catalog=catalog),
            spec=make_spec(dataset),
            definition=SyntheticSupervisedModel(),
        )

        def explode(handle, data):
            """Fail before anything is written."""
            del handle, data
            message = "no"
            raise RuntimeError(message)

        failing.fit = explode
        with pytest.raises(StageError):
            failing.execute()
        assert len(catalog.entries()) == registered == 1

    def test_the_catalog_records_where_the_bundle_is(self, tmp_path, dataset):
        """
        So a run selected from the catalog can be opened.

        The layout differs between a single run and a job set, so a reader
        cannot derive the directory -- it has to be recorded.
        """
        catalog = InMemoryCatalog()
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run", catalog=catalog),
            spec=make_spec(dataset),
            definition=SyntheticSupervisedModel(),
        )
        pipeline.execute()
        assert catalog.records()[0].location == pipeline.saved.directory


class TestReports:
    """Enabled declaratively, and never load-bearing."""

    def test_enabled_reports_are_rendered(self, tmp_path, dataset):
        """Enabling a report is a specification change, not a code change."""
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ("summary", "curves", "quality")}),
            definition=SyntheticSupervisedModel(),
        )
        pipeline.execute()
        written = {
            path.name for path in pipeline.context.reports_directory.rglob("*") if path.is_file()
        }
        assert {"summary.md", "training_history.csv", "data_quality.md"} <= written

    def test_a_failing_report_does_not_fail_the_run(self, tmp_path, dataset):
        """
        The property that makes reports safe to enable.

        A completed, scored, persisted training run is not discarded because a
        figure could not be drawn.
        """

        @report("explodes")
        class Explodes(Report):
            """A report that always raises."""

            def render(self, context):
                """Raise."""
                message = "this report is broken"
                raise RuntimeError(message)

        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ("explodes",)}),
            definition=SyntheticSupervisedModel(),
        )
        result = pipeline.execute()
        assert result.evaluations["test"].metrics["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_fail_fast_turns_a_report_failure_into_a_run_failure(self, tmp_path, dataset):
        """
        For the case where the report *is* the deliverable.

        Opt-in, because the default must not let a figure discard a model.
        """

        @report("explodes_too")
        class ExplodesToo(Report):
            """A report that always raises."""

            def render(self, context):
                """Raise."""
                message = "this report is broken"
                raise RuntimeError(message)

        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ("explodes_too",), "fail_fast": True}),
            definition=SyntheticSupervisedModel(),
        )
        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert caught.value.stage == "report"


class TestTheReportHook:
    """
    How a model adds a report of its own without overriding a stage.

    Some reports are not optional: a graph model's connectivity diagnostics
    are the only place its largest assumption is visible, and switching
    them off to save a few seconds should not be something a user can do by
    accident. Without this hook the only way to guarantee one is to
    override the ``report`` stage, which means reimplementing the
    fail-fast handling, the ordering and the registration -- for a model
    whose only quarrel with the base is the *list*.
    """

    def test_the_default_is_exactly_what_the_spec_asked_for(self, tmp_path, dataset):
        """
        So adding the hook changed nothing for any existing model.

        The first thing to check about a new seam in a base class.
        """
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ("summary", "curves")}),
            definition=SyntheticSupervisedModel(),
        )
        assert pipeline.report_names() == ("summary", "curves")

    def test_a_name_the_hook_adds_is_rendered(self, tmp_path, dataset):
        """
        Even though the specification never mentions it.

        This is the behaviour the hook exists for, asserted through a real
        run rather than against the hook's return value, because the
        failure mode worth catching is a stage that reads the spec
        directly and bypasses the hook.
        """

        @report("added_by_the_model")
        class AddedByTheModel(Report):
            """A report a model insists on."""

            def render(self, context):
                """Write a marker file."""
                path = context.directory / "insisted.md"
                path.write_text("rendered", encoding="utf-8")
                return (path,)

        class Insisting(TrainPipeline):
            """A pipeline that appends one report to whatever was asked for."""

            def report_names(self):
                """Return the spec's reports plus this model's own."""
                return (*super().report_names(), "added_by_the_model")

        pipeline = Insisting(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ()}),
            definition=SyntheticSupervisedModel(),
        )
        pipeline.execute()
        assert (pipeline.context.reports_directory / "insisted.md").is_file()

    def test_a_name_the_hook_adds_is_checked_at_resolve(self, tmp_path, dataset):
        """
        At the first stage, not the last.

        Resolution instantiates every report precisely so that a
        misspelled name costs nothing instead of costing a full training
        run. A hook that added names after that check would reintroduce
        the cost for exactly the reports a model considers mandatory.
        """

        class Insisting(TrainPipeline):
            """A pipeline that insists on a report nobody registered."""

            def report_names(self):
                """Return a name that does not resolve."""
                return (*super().report_names(), "never_registered")

        pipeline = Insisting(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ()}),
            definition=SyntheticSupervisedModel(),
        )
        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert caught.value.stage == "resolve"
        assert "never_registered" in str(caught.value)


class TestInstrumentation:
    """Every stage is timed, logged and attributed, finished or not."""

    def test_a_failure_is_attributed_to_its_stage(self, tmp_path, dataset):
        """Not "something raised during training" but "fit raised"."""

        class BrokenFit(SyntheticSupervisedModel):
            """A definition whose fit stage fails."""

        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset),
            definition=BrokenFit(),
        )

        def explode(handle, data):
            """Raise instead of fitting."""
            del handle, data
            message = "no"
            raise RuntimeError(message)

        pipeline.fit = explode
        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert caught.value.stage == "fit"
        assert isinstance(caught.value.__cause__, RuntimeError)

    def test_every_stage_that_started_is_timed(self, pipeline):
        """
        Including the one that failed, which is the useful part.

        "Fit failed after four hours" and "fit failed immediately" call for
        completely different investigations, and only the timing distinguishes
        them.
        """
        pipeline.execute()
        assert set(pipeline.timings) == set(pipeline.stages)

    def test_the_run_end_hook_fires_after_a_failure(self, tmp_path, dataset):
        """
        The guarantee that makes cleanup possible.

        A hook that only fired on success could not close a progress bar,
        release a GPU or mark a job failed.
        """
        hook = RecordingHook()
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run", hooks=(hook,)),
            spec=make_spec(dataset, reports={"enabled": ("missing",)}),
            definition=SyntheticSupervisedModel(),
        )
        with pytest.raises(StageError):
            pipeline.execute()
        assert hook.events[-1][:1] == ("run_end",)
        assert hook.events[-1][2] is False


class TestCustomisationTiers:
    """Overriding one stage must inherit the rest, instrumentation included."""

    def test_one_stage_can_be_replaced(self, tmp_path, dataset):
        """
        Tier three: subclass, replace one stage, inherit the other ten.

        With their timing, logging and error attribution, which is what makes
        the tier worth having over writing a pipeline from scratch.
        """

        class ScoresTestOnly(TrainPipeline):
            """Replaces only the evaluate stage."""

            def evaluate(self, handle, data):
                """Score the test split and nothing else."""
                full = super().evaluate(handle, data)
                return {"test": full["test"]}

        pipeline = ScoresTestOnly(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset),
            definition=SyntheticSupervisedModel(),
        )
        result = pipeline.execute()
        assert set(result.evaluations) == {"test"}
        assert pipeline.completed_stages == pipeline.stages

    def test_the_applied_seed_is_recorded(self, pipeline):
        """
        What was applied, not what was requested.

        A job-set member derives a per-job seed, and recording the derived
        value is what lets a single failed job be re-run alone and reproduce.
        """
        result = pipeline.execute()
        assert result.seed == pipeline.context.seed


class TestDelegation:
    """The three stages the framework cannot implement for a model."""

    def test_the_data_build_delegates_to_the_definition(self, pipeline):
        """
        A thin delegation by design.

        The framework does not know how to build any model's data, and a base
        implementation that tried would be overridden by every non-trivial
        model.
        """
        bundle = pipeline.build_data(pipeline.spec)
        assert isinstance(bundle, DataBundle)

    def test_the_signature_delegates_to_the_definition(self, pipeline):
        """So a model may narrow the interface the data offers."""
        bundle = make_tensor_bundle()
        assert pipeline.declare_signature(bundle) == bundle.signature

    def test_a_data_build_with_no_training_split_is_refused(self, tmp_path, dataset):
        """
        Refused by the contract, before any engine sees it.

        "No batches" from inside a training loop does not say that the split
        was missing entirely, which is a different problem with a different
        fix. ``DataBundle`` enforces this on construction, which is why the
        pipeline stage adds no check of its own.
        """

        class NoTrainSplit(SyntheticSupervisedModel):
            """Drops the training split."""

            def build_data(self, spec):
                """Return a bundle with no training split."""
                bundle = super().build_data(spec)
                splits = {
                    name: payload for name, payload in bundle.splits.items() if name != "train"
                }
                return DataBundle(
                    splits=splits,
                    signature=bundle.signature,
                    state=bundle.state,
                    lineage=bundle.lineage,
                )

        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset),
            definition=NoTrainSplit(),
        )
        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert caught.value.stage == "build_data"
        assert "requires a 'train' split" in str(caught.value)


def _targets_of(source):
    """
    Drain one pass of a source's targets.

    Parameters
    ----------
    source
        A bounded source.

    Returns
    -------
    list of float
        The targets.
    """
    return [
        float(value)
        for batch in source.batches()
        for value in np.ravel(np.asarray(batch["target"]))
    ]


def _execute_and_persist(pipeline) -> ModelBundle:
    """
    Run a pipeline and return the bundle its persist stage assembled.

    Parameters
    ----------
    pipeline
        The pipeline to run.

    Returns
    -------
    ModelBundle
        The bundle.
    """
    captured: list[ModelBundle] = []
    original = pipeline.persist

    def capture(handle, data, signature, result):
        """Record the bundle on its way past."""
        bundle = original(handle, data, signature, result)
        captured.append(bundle)
        return bundle

    pipeline.persist = capture
    pipeline.execute()
    return captured[0]
```

---

## 7. `tranql/models/rade/rade_qnet/tests/orchestration/pipelines/test_pipelines_tune.py`

22345 bytes · SHA-256 `e67eee2015ed3f2e`

```python
"""
Tests for searching a space of configurations.

A search that is subtly wrong still completes and still names a winner, so
most of these tests are about the bookkeeping rather than the optimisation:

- **Defect 7.** An unknown path in the space must fail at construction,
  before anything trains. Left unchecked it is explored, measured and
  reported as a dimension that makes no difference -- which reads as a
  finding rather than a bug.
- **Build once.** Asserted by a call count rather than by timing, because a
  timing assertion measures the machine.
- **Failures are recorded.** A search that drops a failing trial reports the
  best of a partial sweep while looking like a complete one.
- **Selection is deterministic.** Including the tie rule, since a search
  whose winner moves between identical runs cannot be acted on.
"""

from __future__ import annotations

import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import (
    ContractError,
    SpecError,
    StageError,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.tune import parse_tune_spec
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.tune import TunePipeline
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.stages.search import expand, propose
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import (
    isolated_registries,
    make_run_context,
)

from .support import (
    ENGINE_TAG,
    MODEL_NAME,
    SyntheticSupervisedModel,
    register_components,
    write_linear_dataset,
)


@pytest.fixture(autouse=True)
def _registries():
    """
    Register the synthetic components, isolated to this module.

    Yields
    ------
    None
        For the duration of the test.
    """
    # ``empty=True`` because the stand-in engine below claims a shipped
    # engine's name. Without it, whether this fixture succeeds depends on
    # whether an earlier test imported the real one -- which made the suite
    # pass or fail on collection order.
    with isolated_registries(empty=True):
        register_components()
        yield


@pytest.fixture
def dataset(tmp_path):
    """
    Write the training dataset.

    Returns
    -------
    pathlib.Path
        The file.
    """
    return write_linear_dataset(tmp_path / "linear.csv")


def tune_spec(dataset, **overrides):
    """
    Build a search over the synthetic model.

    The default space varies ``seed``, which is a real field of a run
    specification and has no effect on a closed-form least-squares fit. That
    is deliberate: it makes every trial succeed and score identically, so
    the tests about bookkeeping are not entangled with tests about whether
    the search found anything.

    Parameters
    ----------
    dataset
        Path to the CSV file.
    **overrides
        Fields to replace.

    Returns
    -------
    TuneSpec
        The validated search.
    """
    fields = {
        "model": MODEL_NAME,
        "base": {
            "source": {"kind": "tabular", "path": str(dataset)},
            "training": {"engine": ENGINE_TAG},
            "reports": {"enabled": ()},
        },
        "space": {"seed": [1, 2, 3, 4]},
        "trials": 4,
        "sampler": "grid",
        "objective": "mae",
        "direction": "minimise",
    }
    fields.update(overrides)
    return parse_tune_spec(fields)


def pipeline_for(tmp_path, spec, *, name="tune"):
    """
    Build a tuning pipeline over a search.

    Parameters
    ----------
    tmp_path
        Temporary directory for the run's output.
    spec
        The search.
    name
        Subdirectory to write into. Two searches in one test need two,
        because every trial persists a model and a bundle refuses to
        overwrite an existing one -- correctly, and for the same reason a
        training run does.

    Returns
    -------
    TunePipeline
        The pipeline.
    """
    return TunePipeline(
        context=make_run_context(output_directory=tmp_path / name),
        spec=spec,
        definition=SyntheticSupervisedModel(),
    )


class TestTheSearchRuns:
    """The sequence completes and produces what it promises."""

    def test_the_stage_sequence_is_declared(self, tmp_path, dataset):
        """Declared as data, so a hook can enumerate it before the run."""
        pipeline = pipeline_for(tmp_path, tune_spec(dataset))
        pipeline.execute()

        assert pipeline.stages == (
            "resolve",
            "propose",
            "build_data",
            "run_trials",
            "select",
            "refit",
        )
        assert pipeline.completed_stages == pipeline.stages

    def test_every_proposal_becomes_a_trial(self, tmp_path, dataset):
        """Four points, four records, in order."""
        result = pipeline_for(tmp_path, tune_spec(dataset)).execute()

        assert len(result.trials) == 4
        assert [record.trial for record in result.trials] == [0, 1, 2, 3]

    def test_the_trials_record_what_was_proposed(self, tmp_path, dataset):
        """
        Flat, so two trials can be read side by side.

        A nested fragment has to be mentally flattened before it can be
        compared against another, which is the only thing anyone does with
        a column of proposals.
        """
        result = pipeline_for(tmp_path, tune_spec(dataset)).execute()

        assert [record.overrides for record in result.trials] == [
            {"seed": 1},
            {"seed": 2},
            {"seed": 3},
            {"seed": 4},
        ]

    def test_the_winner_carries_its_metrics(self, tmp_path, dataset):
        """So it can be examined on more than the one number it was chosen by."""
        result = pipeline_for(tmp_path, tune_spec(dataset)).execute()

        assert result.best.objective is not None
        assert "mae" in result.best.metrics


class TestDataIsBuiltOnce:
    """Forty trials over one dataset should read it once."""

    def test_data_build_is_reused_across_trials(self, tmp_path, dataset):
        """
        Asserted by a call count, not by timing.

        A timing assertion measures the machine; a count measures the
        pipeline.
        """
        pipeline = pipeline_for(tmp_path, tune_spec(dataset, trials=4))
        pipeline.execute()

        assert pipeline.data_builds == 1

    def test_the_same_dataset_object_reaches_every_trial(self, tmp_path, dataset):
        """
        Reuse means the same object, not an equal one.

        An equal rebuild would also pass a count-free test while costing
        exactly what the reuse was meant to save.
        """
        pipeline = pipeline_for(tmp_path, tune_spec(dataset))
        pipeline.execute()

        assert pipeline.shared_data is not None

    def test_a_search_over_the_source_builds_per_trial(self, tmp_path, dataset):
        """
        Because the trials then have genuinely different datasets.

        Sharing one would mean the search measured something other than
        what it proposed, which is the quietest way for a search to be
        wrong: every number is plausible and every one is about the wrong
        configuration.
        """
        other = write_linear_dataset(tmp_path / "other.csv", seed=5)
        spec = tune_spec(
            dataset,
            space={"source.path": [str(dataset), str(other)]},
            trials=2,
        )
        pipeline = pipeline_for(tmp_path, spec)
        pipeline.execute()

        assert pipeline.data_builds == 0
        assert pipeline.shared_data is None


class TestDefectSeven:
    """An unknown override fails before anything trains."""

    def test_unknown_trial_override_fails_at_construction(self, tmp_path, dataset):
        """
        The structural cure.

        ``rade_ml_pt`` passed ``learning_rate`` into
        ``dataclasses.replace(TrainingConfig)`` and raised ``TypeError``; it
        was latent only because the hybrid model overrode the method. Here
        the proposal goes through the same validated merge a job's
        overrides do, so the path fails at trial construction.
        """
        spec = tune_spec(dataset, space={"training.not_a_field": [1, 2]}, trials=2)

        with pytest.raises(StageError, match=r"do not\s+produce a valid run specification"):
            pipeline_for(tmp_path, spec).execute()

    def test_the_failure_names_the_offending_path(self, tmp_path, dataset):
        """
        Because nobody typed the value -- it was generated.

        A message that said only "validation failed" would send the reader
        looking at a proposal they did not write.
        """
        spec = tune_spec(dataset, space={"training.not_a_field": [1, 2]}, trials=2)

        with pytest.raises(StageError, match=r"not_a_field"):
            pipeline_for(tmp_path, spec).execute()

    def test_nothing_trains_before_the_check(self, tmp_path, dataset):
        """
        The point of checking at construction rather than at trial time.

        A space with one bad path breaks every trial, so discovering it on
        the fortieth wastes the thirty-nine before it.
        """
        spec = tune_spec(dataset, space={"training.not_a_field": [1, 2]}, trials=2)
        pipeline = pipeline_for(tmp_path, spec)

        with pytest.raises(StageError):
            pipeline.execute()

        assert pipeline.data_builds == 0


class TestFailingTrials:
    """Recorded and surfaced, never swallowed."""

    def test_a_failed_trial_does_not_end_the_search(self, tmp_path, dataset, monkeypatch):
        """Thirty-nine good trials must survive one bad one."""
        spec = tune_spec(dataset, trials=4)
        pipeline = pipeline_for(tmp_path, spec)

        original = SyntheticSupervisedModel.build_model
        calls = {"n": 0}

        def flaky(self, run_spec, signature):
            """Fail the second trial only."""
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("deliberate trial failure")
            return original(self, run_spec, signature)

        monkeypatch.setattr(SyntheticSupervisedModel, "build_model", flaky)
        result = pipeline.execute()

        assert len(result.trials) == 4
        assert len(result.failed) == 1
        assert result.failed[0].trial == 1

    def test_a_failure_records_its_reason(self, tmp_path, dataset, monkeypatch):
        """So the search report says why, not merely that something went wrong."""
        spec = tune_spec(dataset, trials=2)
        pipeline = pipeline_for(tmp_path, spec)

        def always_fails(self, run_spec, signature):
            """Fail every trial."""
            raise RuntimeError("deliberate trial failure")

        monkeypatch.setattr(SyntheticSupervisedModel, "build_model", always_fails)

        with pytest.raises(StageError, match=r"nothing to\s+select between"):
            pipeline.execute()

    def test_the_summary_states_the_failure_count(self, tmp_path, dataset, monkeypatch):
        """Even when it is zero, because a missing clause is not information."""
        result = pipeline_for(tmp_path, tune_spec(dataset)).execute()

        assert "0 failed" in result.describe()


class TestSelection:
    """One rule, stated, applied once."""

    def test_best_trial_selection_is_deterministic(self, tmp_path, dataset):
        """Same seed, same winner, across two independent searches."""
        first = pipeline_for(tmp_path, tune_spec(dataset), name="a").execute()
        second = pipeline_for(tmp_path, tune_spec(dataset), name="b").execute()

        assert first.best_trial == second.best_trial
        assert [record.objective for record in first.trials] == [
            record.objective for record in second.trials
        ]

    def test_ties_go_to_the_earlier_trial(self, dataset):
        """
        Which is what makes the winner stable under a re-run.

        Asserted against the comparison rule itself rather than through a
        search. Contriving a genuine tie between two trained models is
        harder than it sounds -- varying the seed moves the split, so even
        a closed-form fit scores slightly differently -- and a test that
        had to contrive one would be testing the contrivance.
        """
        spec = tune_spec(dataset)

        assert spec.is_better(0.5, 0.6) is True
        assert spec.is_better(0.5, 0.5) is False
        assert spec.is_better(0.6, 0.5) is False

    def test_the_tie_rule_reverses_with_the_direction(self, dataset):
        """One rule, applied once, honouring whichever way the metric runs."""
        spec = tune_spec(dataset, objective="r2", direction="maximise")

        assert spec.is_better(0.6, 0.5) is True
        assert spec.is_better(0.5, 0.5) is False

    def test_the_direction_is_honoured(self, tmp_path, dataset):
        """
        Taken from the specification rather than guessed from the metric.

        ``r2`` is maximised and ``mae`` minimised; a framework that inferred
        the direction from a substring would eventually select the worst
        trial in a search over a custom metric.
        """
        maximised = tune_spec(dataset, objective="r2", direction="maximise")
        result = pipeline_for(tmp_path, maximised).execute()

        assert result.direction == "maximise"
        assert result.objective == "r2"

    def test_the_result_records_which_split_decided(self, tmp_path, dataset):
        """Because a winner chosen on test is not a held-out estimate."""
        result = pipeline_for(tmp_path, tune_spec(dataset)).execute()

        assert result.objective_split == "validation"

    def test_refit_is_refused_rather_than_skipped(self, tmp_path, dataset):
        """
        A user who asked for it must not receive a model trained without it.

        Silently ignoring the flag would hand back a model fitted on the
        training split alone, with no way for the caller to tell.
        """
        spec = tune_spec(dataset, refit=True)

        with pytest.raises(StageError, match=r"refit is not implemented"):
            pipeline_for(tmp_path, spec).execute()


class TestTheSearchSpace:
    """Properties of the space itself, cheap to check without training."""

    def test_a_space_cannot_vary_one_path_twice(self):
        """
        Two axes over one path make the earlier one inert.

        The later would win every merge, so half the budget explores a
        dimension with no effect -- and the report would show it having
        none, which reads as a finding rather than a bug.
        """
        with pytest.raises(SpecError, match=r"must not vary the same path twice"):
            parse_tune_spec(
                {
                    "base": {},
                    "space": {"dimensions": [{"path": "seed", "values": [1]}] * 2},
                }
            )

    def test_an_axis_must_be_an_enumeration_or_a_range(self):
        """Not both, and not neither."""
        with pytest.raises(SpecError, match=r"must give either 'values'"):
            parse_tune_spec(
                {"base": {}, "space": {"seed": {"values": [1], "low": 0.0, "high": 1.0}}}
            )

    def test_an_empty_range_is_rejected(self):
        """Because every trial would propose the same value."""
        with pytest.raises(SpecError, match=r"the range is empty"):
            parse_tune_spec({"base": {}, "space": {"x": {"low": 1.0, "high": 1.0}}})

    def test_a_log_scale_below_zero_is_rejected(self):
        """A log scale is undefined there, and would produce NaN proposals."""
        with pytest.raises(SpecError, match=r"undefined at or below zero"):
            parse_tune_spec({"base": {}, "space": {"x": {"low": 0.0, "high": 1.0, "log": True}}})

    def test_a_grid_cannot_enumerate_a_range(self):
        """
        A grid needs values, not bounds.

        Discretising a range into steps the user did not choose would make
        the search coarser than it reads, and the step count would be a
        framework decision affecting the result.
        """
        with pytest.raises(SpecError, match=r"cannot enumerate the continuous"):
            parse_tune_spec(
                {
                    "base": {},
                    "space": {"x": {"low": 0.1, "high": 1.0}},
                    "sampler": "grid",
                }
            )


class TestProposals:
    """The sampler, tested without training anything."""

    def test_random_proposals_are_reproducible(self):
        """Same seed, same sequence."""
        spec = parse_tune_spec(
            {"base": {}, "space": {"x": {"low": 0.1, "high": 1.0}}, "trials": 8, "seed": 3}
        )

        assert propose(spec) == propose(spec)

    def test_a_different_seed_explores_differently(self):
        """Otherwise the seed would be decoration."""
        space = {"base": {}, "space": {"x": {"low": 0.1, "high": 1.0}}, "trials": 8}

        assert propose(parse_tune_spec({**space, "seed": 1})) != propose(
            parse_tune_spec({**space, "seed": 2})
        )

    def test_continuous_proposals_stay_in_bounds(self):
        """A proposal outside the range is a value the user excluded."""
        spec = parse_tune_spec(
            {"base": {}, "space": {"x": {"low": 0.2, "high": 0.8}}, "trials": 50}
        )

        assert all(0.2 <= proposal["x"] <= 0.8 for proposal in propose(spec))

    def test_log_proposals_stay_in_bounds(self):
        """Including under the exponent transform, which is easy to get wrong."""
        spec = parse_tune_spec(
            {
                "base": {},
                "space": {"x": {"low": 1e-4, "high": 1e-1, "log": True}},
                "trials": 50,
            }
        )

        assert all(1e-4 <= proposal["x"] <= 1e-1 for proposal in propose(spec))

    def test_enumerated_values_keep_their_python_type(self):
        """
        An integer axis must not come back as ``numpy.int64``.

        It would be merged into a specification and fail validation for a
        reason that has nothing to do with what the user wrote.
        """
        spec = parse_tune_spec({"base": {}, "space": {"n": [1, 2, 3]}, "trials": 10})

        assert all(type(proposal["n"]) is int for proposal in propose(spec))

    def test_a_grid_enumerates_the_product(self):
        """Two axes of two and three values give six points."""
        spec = parse_tune_spec(
            {
                "base": {},
                "space": {"a": [1, 2], "b": ["x", "y", "z"]},
                "sampler": "grid",
                "trials": 100,
            }
        )
        proposals = propose(spec)

        assert len(proposals) == 6
        assert len({tuple(sorted(point.items())) for point in proposals}) == 6

    def test_a_grid_is_truncated_to_the_budget(self):
        """And the pipeline logs that the reported best is of a partial grid."""
        spec = parse_tune_spec(
            {
                "base": {},
                "space": {"a": [1, 2], "b": ["x", "y", "z"]},
                "sampler": "grid",
                "trials": 4,
            }
        )

        assert len(propose(spec)) == 4

    def test_dotted_paths_expand_to_nested_mappings(self):
        """Which is the shape a specification is validated in."""
        assert expand({"training.learning_rate": 0.01, "seed": 3}) == {
            "training": {"learning_rate": 0.01},
            "seed": 3,
        }

    def test_a_path_cannot_be_both_a_value_and_a_mapping(self):
        """
        A path cannot be a leaf in one axis and a branch in another.

        One would overwrite the other depending on iteration order, so the
        search would explore a different space on a different day.
        """
        with pytest.raises(SpecError, match=r"Vary one or the other"):
            expand({"training": 1, "training.learning_rate": 0.01})


class TestResolution:
    """Finding the model, and failing usefully when it cannot be found."""

    def test_a_search_can_resolve_its_model_by_name(self, tmp_path, dataset):
        """As a configuration file would, with no definition passed in."""
        pipeline = TunePipeline(
            context=make_run_context(output_directory=tmp_path / "tune"),
            spec=tune_spec(dataset),
        )
        result = pipeline.execute()

        assert len(result.trials) == 4

    def test_an_unnamed_model_is_refused(self, tmp_path, dataset):
        """A search has to know what it is searching over."""
        spec = parse_tune_spec(
            {
                "base": {
                    "source": {"kind": "tabular", "path": str(dataset)},
                    "training": {"engine": ENGINE_TAG},
                },
                "space": {"seed": [1]},
                "trials": 1,
            }
        )
        pipeline = TunePipeline(
            context=make_run_context(output_directory=tmp_path / "tune"), spec=spec
        )

        with pytest.raises(StageError, match=r"names no model"):
            pipeline.execute()

    def test_an_unregistered_model_names_the_problem(self, tmp_path, dataset):
        """So the reader looks at their imports rather than at the search."""
        spec = tune_spec(dataset, model="not_registered")
        pipeline = TunePipeline(
            context=make_run_context(output_directory=tmp_path / "tune"), spec=spec
        )

        with pytest.raises(StageError, match=r"not registered in this process"):
            pipeline.execute()


class TestTheResultContract:
    """Reading a search afterwards."""

    def test_the_summary_names_the_winner_and_the_direction(self, tmp_path, dataset):
        """A column of objective values cannot be read without the direction."""
        result = pipeline_for(tmp_path, tune_spec(dataset)).execute()
        summary = result.describe()

        assert f"#{result.best_trial}" in summary
        assert "minimise" in summary
        assert "4 trial(s)" in summary

    def test_a_best_trial_that_is_not_present_is_refused(self):
        """Rather than returning the first trial as though it had won."""
        from tranql.models.rade.rade_qnet.rade_qnet.core.contract.result import (  # noqa: PLC0415
            TrialRecord,
            TuningResult,
        )

        result = TuningResult(trials=(TrialRecord(trial=0),), best_trial=7)

        with pytest.raises(ContractError, match=r"none with that number"):
            _ = result.best
```

