"""
Tests for the front door.

``api`` adds no behaviour. Every function here assembles the same pieces a
caller could assemble by hand, which is the property that keeps it from
becoming a second, divergent way to run a model -- so these tests are mostly
about *equivalence*: that the one-line path produces what the long path
produces.

The one thing ``api`` genuinely contributes is reaching across layers that
are not allowed to see each other. ``orchestration`` may not import
``domains``; :func:`~rade_xl.api.train_portfolio` reads a book, expands it
into jobs, and hands them to a runner that has no idea a cluster exists.
"""

from __future__ import annotations

import json

import pytest
import yaml

from src.rade_xl import api
from src.rade_xl.core.runtime.components import engine as register_engine
from src.rade_xl.core.runtime.components import model as register_model
from src.rade_xl.core.runtime.errors import SpecError
from src.rade_xl.domains.pnl.portfolio import MANIFEST_FILENAME
from src.rade_xl.orchestration.compute.local import LocalExecutor
from src.rade_xl.testkit.fixtures import SyntheticEngine, isolated_registries

from .orchestration.jobs.support import (
    CLUSTER_FILENAME,
    DIRECTORY_MODEL_NAME,
    ENGINE_TAG,
    MODEL_NAME,
    SyntheticDirectoryModel,
    SyntheticTabularModel,
    job_set_payload,
    write_linear_dataset,
)


@pytest.fixture(autouse=True)
def _registries():
    """
    Isolate the component registries for every test here.

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
        register_engine(ENGINE_TAG)(SyntheticEngine)
        register_model(MODEL_NAME, engine=ENGINE_TAG)(SyntheticTabularModel)
        register_model(DIRECTORY_MODEL_NAME, engine=ENGINE_TAG)(SyntheticDirectoryModel)
        yield


@pytest.fixture
def dataset(tmp_path):
    """
    Write the exactly-linear dataset every run reads.

    Returns
    -------
    pathlib.Path
        The CSV file.
    """
    return write_linear_dataset(tmp_path / "linear.csv")


def run_payload(dataset, output_root):
    """
    Build a single-run configuration mapping.

    Parameters
    ----------
    dataset
        The CSV file.
    output_root
        Where the run writes.

    Returns
    -------
    dict
        The configuration.
    """
    return {
        "model": MODEL_NAME,
        "source": {"kind": "tabular", "path": str(dataset)},
        "training": {"engine": ENGINE_TAG},
        "reports": {"enabled": []},
        "output_root": str(output_root),
    }


class TestTrainingOneModel:
    """`api.train`, in the three forms a configuration arrives in."""

    def test_a_mapping_trains(self, dataset, tmp_path):
        """
        The notebook path.

        The fit is closed-form over exactly linear data, so an r-squared of
        one is the only correct answer and a tolerance would not be needed
        even if one were offered.
        """
        result = api.train(run_payload(dataset, tmp_path / "out"))

        assert result.evaluations["test"].metrics["r2"] == pytest.approx(1.0)

    def test_the_output_root_may_be_a_string(self, dataset, tmp_path):
        """
        A string directory is accepted, like every other path argument.

        `evaluate` and `infer` already took a bundle as either, and a
        specification can be a string, so an output root that took only
        `Path` was an inconsistency rather than a rule. It also failed
        badly: the string travelled several frames inwards before
        `root / run_id` raised `TypeError: unsupported operand type(s)
        for /`, naming neither the argument nor the mistake.
        """
        result = api.train(
            run_payload(dataset, tmp_path / "out"), output_root=str(tmp_path / "strings")
        )

        assert result.bundle_directory is not None
        assert str(tmp_path / "strings") in result.bundle_directory

    def test_a_yaml_file_trains(self, dataset, tmp_path):
        """The production path."""
        path = tmp_path / "run.yaml"
        path.write_text(yaml.safe_dump(run_payload(dataset, tmp_path / "out")), encoding="utf-8")

        assert api.train(path).evaluations["test"].metrics["r2"] == pytest.approx(1.0)

    def test_an_already_validated_specification_trains(self, dataset, tmp_path):
        """
        The test path, and the one that nearly broke silently.

        `RunSpec` is an annotated discriminated union, so checking against
        it in an instance test raises rather than returning `False` -- a
        mistake no passing test would reveal by accident.
        """
        from src.rade_xl.core.spec.run import parse_run_spec  # noqa: PLC0415

        spec = parse_run_spec(run_payload(dataset, tmp_path / "out"))

        assert api.train(spec).evaluations["test"].metrics["r2"] == pytest.approx(1.0)

    def test_the_run_directory_is_derived_from_the_specification(self, dataset, tmp_path):
        """
        Two runs of one configuration share a directory.

        The same reasoning as a job set's run identifier: a timestamped
        name leaves a trail of near-identical directories nobody can tell
        apart, and a re-run after a crash should complete the first rather
        than start a second.
        """
        output_root = tmp_path / "out"
        api.train(run_payload(dataset, output_root))

        before = {path.name for path in output_root.iterdir()}
        api.train(run_payload(dataset, output_root))

        assert {path.name for path in output_root.iterdir()} == before

    def test_a_changed_configuration_lands_somewhere_else(self, dataset, tmp_path):
        """
        Derived from the digest, so a different run is a different place.

        Otherwise a changed configuration would overwrite the results of
        the one it replaced.
        """
        output_root = tmp_path / "out"
        api.train(run_payload(dataset, output_root))
        api.train({**run_payload(dataset, output_root), "seed": 99})

        # Directories only: the catalog writes its index into the same root.
        assert len([path for path in output_root.iterdir() if path.is_dir()]) == 2

    def test_output_root_can_be_overridden(self, dataset, tmp_path):
        """
        A caller writing somewhere other than the file says.

        Common when one configuration is run against several destinations
        and editing the file for each would be the wrong way round.
        """
        elsewhere = tmp_path / "elsewhere"

        api.train(run_payload(dataset, tmp_path / "out"), output_root=elsewhere)

        assert elsewhere.exists()

    def test_a_reinforcement_configuration_is_refused_clearly(self, tmp_path):
        """
        Said here, not several stages later.

        Interactive training arrives in Phase 7, and the alternative to
        this message is a failure about a missing data source.
        """
        with pytest.raises(SpecError, match="supervised"):
            api.train(
                {
                    "task": "reinforcement",
                    "model": MODEL_NAME,
                    "environment": {"name": "hedging"},
                    "training": {"engine": "torch"},
                    "output_root": str(tmp_path),
                }
            )


class TestTrainingAJobSet:
    """`api.train_jobs`."""

    def test_a_mapping_runs_every_job(self, dataset, tmp_path):
        """Two jobs in, two records out."""
        manifest = api.train_jobs(
            job_set_payload(dataset, tmp_path / "out"), executor=LocalExecutor()
        )

        assert len(manifest.succeeded) == 2

    def test_a_yaml_file_runs_every_job(self, dataset, tmp_path):
        """The documented path from `ARCHITECTURE.md` §12."""
        path = tmp_path / "set.yaml"
        path.write_text(
            yaml.safe_dump(job_set_payload(dataset, tmp_path / "out")), encoding="utf-8"
        )

        assert len(api.train_jobs(path, executor=LocalExecutor()).succeeded) == 2

    def test_a_failing_job_is_recorded_rather_than_raised(self, dataset, tmp_path):
        """
        The partial-failure policy, surfaced through the front door.

        Thirty-nine models that trained are thirty-nine results, and
        discarding them over one typo in the fortieth is the wrong trade.
        """
        manifest = api.train_jobs(
            job_set_payload(
                dataset,
                tmp_path / "out",
                jobs=[
                    {"id": "good"},
                    {"id": "bad", "overrides": {"source": {"path": str(tmp_path / "gone.csv")}}},
                ],
            ),
            executor=LocalExecutor(),
        )

        assert [record.job_id for record in manifest.succeeded] == ["good"]
        assert [record.job_id for record in manifest.failed] == ["bad"]


class TestTrainingAPortfolio:
    """`api.train_portfolio`, which is why this module sits above the layers."""

    @pytest.fixture
    def book(self, tmp_path):
        """
        Lay out a two-cluster portfolio, one data file per cluster.

        Each cluster gets its own file, which is the layout
        `cluster_overrides` assumes: a directory per cluster, named in the
        source parameters.

        Returns
        -------
        pathlib.Path
            The portfolio directory.
        """
        root = tmp_path / "book"
        clusters = [{"name": "alpha"}, {"name": "beta"}]
        root.mkdir()
        for entry in clusters:
            directory = root / entry["name"]
            directory.mkdir()
            write_linear_dataset(directory / CLUSTER_FILENAME)
        (root / MANIFEST_FILENAME).write_text(json.dumps({"clusters": clusters}), encoding="utf-8")
        return root

    @pytest.fixture
    def defaults(self):
        """
        Provide the shared run-specification fragment.

        A model source, because that is the shape a cluster override
        produces: `domains.pnl` names a directory, and reading a directory
        is the model's business rather than the framework's.

        Returns
        -------
        dict
            The defaults.
        """
        return {
            "model": {"name": DIRECTORY_MODEL_NAME},
            "source": {"kind": "model"},
            "training": {"engine": ENGINE_TAG},
            "reports": {"enabled": []},
        }

    def test_one_job_runs_per_cluster(self, book, defaults, tmp_path):
        """
        Read, expand, dispatch -- the three steps, as one call.

        The middle one is the step that is easy to get subtly wrong, which
        is why it is worth a function rather than a docstring.
        """
        manifest = api.train_portfolio(
            book,
            defaults=defaults,
            output_root=tmp_path / "out",
            executor=LocalExecutor(),
        )

        assert [record.job_id for record in manifest.jobs] == ["alpha", "beta"]

    def test_a_cluster_filter_restricts_the_set(self, book, defaults, tmp_path):
        """Running part of a book runs part of a book."""
        manifest = api.train_portfolio(
            book,
            defaults=defaults,
            output_root=tmp_path / "out",
            clusters=["beta"],
            executor=LocalExecutor(),
        )

        assert [record.job_id for record in manifest.jobs] == ["beta"]

    def test_per_cluster_overrides_reach_the_jobs(self, book, defaults, tmp_path):
        """
        The hook that makes a portfolio a job set rather than a loop.

        Asserted through a setting that survives into the manifest, since
        the exact fit makes both clusters score identically.
        """
        manifest = api.train_portfolio(
            book,
            defaults=defaults,
            output_root=tmp_path / "out",
            overrides_for=lambda cluster: {"seed": 7 if cluster.name == "alpha" else 9},
            executor=LocalExecutor(),
        )

        assert manifest.record("alpha").seed != manifest.record("beta").seed

    def test_a_missing_portfolio_fails_before_anything_runs(self, defaults, tmp_path):
        """
        Read first, dispatch second.

        A book that is not there is knowable without training anything.
        """
        with pytest.raises(SpecError):
            api.train_portfolio(
                tmp_path / "absent", defaults=defaults, output_root=tmp_path / "out"
            )


def tune_payload(dataset, output_root):
    """
    Build a small search configuration mapping.

    Parameters
    ----------
    dataset
        The CSV file.
    output_root
        Where the search writes.

    Returns
    -------
    dict
        The configuration.
    """
    del output_root
    return {
        "model": MODEL_NAME,
        "base": {
            "source": {"kind": "tabular", "path": str(dataset)},
            "training": {"engine": ENGINE_TAG},
            "reports": {"enabled": []},
        },
        "space": {"seed": [1, 2, 3]},
        "trials": 3,
        "sampler": "grid",
        "name": "smoke",
    }


def trained_bundle(tmp_path, dataset):
    """
    Train one model through the front door and return its bundle.

    Returns
    -------
    pathlib.Path
        The bundle directory.
    """
    root = tmp_path / "runs"
    api.train(run_payload(dataset, root), output_root=root)
    bundles = sorted(root.rglob("manifest.json"))
    assert bundles, "training wrote no bundle"
    return bundles[0].parent


class TestEvaluatingASavedModel:
    """`api.evaluate`, which is the whole phase in one call."""

    def test_a_bundle_can_be_rescored(self, tmp_path, dataset):
        """The headline case, in one line as advertised."""
        result = api.evaluate(trained_bundle(tmp_path, dataset))

        assert "test" in result.evaluations
        assert result.evaluations["test"].in_original_units is True

    def test_rescoring_reproduces_the_training_metrics(self, tmp_path, dataset):
        """
        Exactly, which is the gate the whole phase is built around.

        Compared against the metrics recorded in the bundle rather than
        against the training call's return value, because the bundle is
        what anybody actually has weeks later.
        """
        bundle = trained_bundle(tmp_path, dataset)
        recorded = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
        rescored = api.evaluate(bundle)

        for split, evaluation in recorded["evaluations"].items():
            assert rescored.evaluations[split].metrics == evaluation["metrics"]

    def test_a_reproduction_says_the_source_did_not_change(self, tmp_path, dataset):
        """Which is the half of provenance a reader acts on."""
        result = api.evaluate(trained_bundle(tmp_path, dataset))

        assert result.source_changed is False

    def test_selecting_splits_scores_only_those(self, tmp_path, dataset):
        """A re-score against new observations usually wants only test."""
        result = api.evaluate(trained_bundle(tmp_path, dataset), splits=("test",))

        assert set(result.evaluations) == {"test"}

    def test_the_run_is_rooted_beside_the_bundle(self, tmp_path, dataset):
        """
        Rather than in whatever directory the caller happened to be in.

        Asserted on the context rather than on the filesystem, because
        evaluation deliberately writes nothing: it returns a result, and a
        caller who wants it kept keeps it. Checking for a directory would
        be asserting a side effect this phase chose not to have.
        """
        bundle = trained_bundle(tmp_path, dataset)
        context = api._bundle_context(bundle, action="evaluate", output_root=None)

        assert context.output_directory.parent == bundle.parent
        assert context.metadata["bundle"] == str(bundle)

    def test_the_evaluation_run_has_no_catalog(self, tmp_path, dataset):
        """
        Because it writes no bundle, so there is nothing to record.

        Handing it one would invite a future version to register an
        evaluation as though it were a model.
        """
        bundle = trained_bundle(tmp_path, dataset)
        context = api._bundle_context(bundle, action="evaluate", output_root=None)

        assert context.catalog is None


class TestPredictingWithASavedModel:
    """`api.infer`, and the provenance it insists on."""

    def test_a_bundle_produces_predictions(self, tmp_path, dataset):
        """In the target's original units, as the contract says."""
        predictions = api.infer(trained_bundle(tmp_path, dataset))

        assert predictions.n_predictions > 0
        assert predictions.in_original_units is True

    def test_predictions_carry_their_provenance(self, tmp_path, dataset):
        """Reconciling predictions later is the ordinary case, not the exception."""
        bundle = trained_bundle(tmp_path, dataset)
        predictions = api.infer(bundle)

        assert predictions.bundle_version is not None
        assert predictions.provenance["model_name"] == MODEL_NAME
        assert predictions.provenance["predicted_at"]

    def test_predictions_match_the_evaluation_pass(self, tmp_path, dataset):
        """
        The two share their reload path.

        This is the test that the sharing is real rather than coincidental:
        if either grows its own version of a stage, the counts diverge.
        """
        import numpy as np  # noqa: PLC0415

        bundle = trained_bundle(tmp_path, dataset)
        predictions = api.infer(bundle)
        scored = api.evaluate(bundle, splits=("test",))

        assert predictions.n_predictions == scored.evaluations["test"].n_samples
        assert np.all(np.isfinite(predictions.values))


class TestSearchingASpace:
    """`api.tune`, in the three forms a configuration arrives in."""

    def test_a_search_runs_from_a_mapping(self, tmp_path, dataset):
        """The notebook case."""
        result = api.tune(
            tune_payload(dataset, tmp_path), output_root=tmp_path / "searches"
        )

        assert len(result.trials) == 3

    def test_a_search_runs_from_a_file(self, tmp_path, dataset):
        """The production case."""
        path = tmp_path / "search.yaml"
        path.write_text(yaml.safe_dump(tune_payload(dataset, tmp_path)), encoding="utf-8")

        result = api.tune(path, output_root=tmp_path / "searches")
        assert len(result.trials) == 3

    def test_the_winner_is_identified(self, tmp_path, dataset):
        """With the direction and split recorded alongside it."""
        result = api.tune(
            tune_payload(dataset, tmp_path), output_root=tmp_path / "searches"
        )

        assert result.best.succeeded
        assert result.direction == "minimise"
        assert result.objective_split == "validation"

    def test_each_trial_keeps_its_bundle(self, tmp_path, dataset):
        """Which is the difference between acting on a search and repeating it."""
        result = api.tune(
            tune_payload(dataset, tmp_path), output_root=tmp_path / "searches"
        )

        assert result.best.bundle_directory is not None
        assert (tmp_path / "searches").exists()

    def test_the_winning_bundle_can_be_evaluated(self, tmp_path, dataset):
        """
        The two halves of the phase meeting.

        A search produces a bundle, and that bundle can be re-scored
        without retraining anything.
        """
        from pathlib import Path  # noqa: PLC0415

        result = api.tune(
            tune_payload(dataset, tmp_path), output_root=tmp_path / "searches"
        )
        rescored = api.evaluate(Path(result.best.bundle_directory))

        assert rescored.metric("validation", "mae") == pytest.approx(
            result.best.objective
        )

    def test_a_search_is_named_in_its_directory(self, tmp_path, dataset):
        """So two searches in one root are told apart."""
        api.tune(tune_payload(dataset, tmp_path), output_root=tmp_path / "searches")

        assert any(path.name.startswith("smoke-") for path in (tmp_path / "searches").iterdir())
