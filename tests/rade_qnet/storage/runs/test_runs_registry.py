"""
Tests for choosing a trained run by tag, metric or alias.

The registry answers the questions that follow any batch of training --
*which runs belong to this experiment, which was best, which is in
production* -- and records the answer to the last. Three properties matter
more than the convenience, and most of the tests are about them:

- **Nothing in a bundle is rewritten.** A tag added afterwards and an alias
  move are events in a log beside the catalog; the manifest a run was trained
  with is never touched.
- **Concurrent decisions are not lost.** The registry being replaced read an
  index file, changed it and wrote it back, so two simultaneous promotions
  kept only one. A genuine multi-process probe checks that this cannot happen.
- **Ambiguity is refused rather than guessed.** A metric's direction is
  required, an alias may not imitate a version, and a training-time tag
  cannot be withdrawn.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.rade_qnet.core.contract.bundle import Manifest
from src.rade_qnet.core.lifecycle.errors import BundleError, SpecError
from src.rade_qnet.storage.runs.catalog import JsonlCatalog
from src.rade_qnet.storage.runs.registry import (
    REGISTRY_FILENAME,
    RegisteredRun,
    RunRegistry,
)

#: A fixed origin, so "most recent" in a test is decided by the test.
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _record(
    root: Path,
    version: int,
    *,
    model: str = "ridge",
    job: str | None = None,
    tags: tuple[str, ...] = (),
    metrics: dict[str, float] | None = None,
    with_directory: bool = True,
) -> Path:
    """
    Record one run in the catalog at ``root``, as a training run would.

    Parameters
    ----------
    root
        The catalog directory.
    version
        Version number; also decides ``created_at``, so later versions are
        newer.
    model
        Registered model name.
    job
        Job identifier, or ``None``.
    tags
        Training-time tags.
    metrics
        Headline metrics.
    with_directory
        Whether to create the bundle directory and record its location.

    Returns
    -------
    Path
        The bundle directory, whether or not it was created.
    """
    directory = root / "bundles" / model / (job or "") / f"v{version}"
    if with_directory:
        directory.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(
        model_name=model,
        engine="sklearn",
        version=version,
        created_at=_EPOCH + timedelta(hours=version),
        framework_version="0.1.0",
        spec_digest="deadbeef",
        job_id=job,
        tags=tags,
        metrics=metrics or {},
    )
    JsonlCatalog(root).record(manifest, location=directory if with_directory else None)
    return directory


def _tag_in_subprocess(root: str, prefix: str) -> None:
    """
    Add ten tags to one run from a separate process.

    Defined at module level because a process pool must be able to import it.

    Parameters
    ----------
    root
        The registry root.
    prefix
        Distinguishes this process's tags from the other's.
    """
    registry = RunRegistry(root)
    run = registry.get("ridge", version=1)
    for index in range(10):
        registry.tag(run, f"{prefix}-{index}")


@pytest.fixture
def registry(tmp_path):
    """
    Provide a registry over a sweep of three runs and one other model.

    ``ridge`` v1 to v3 are a learning-rate sweep with ``mae`` of 0.30, 0.10
    and 0.20, so v2 is best by ``mae``. v3 also records ``r2``. ``xgb`` v1 is
    outside the sweep.

    Returns
    -------
    RunRegistry
        The registry.
    """
    _record(tmp_path, 1, tags=("lr-sweep",), metrics={"mae": 0.30})
    _record(tmp_path, 2, tags=("lr-sweep",), metrics={"mae": 0.10})
    _record(tmp_path, 3, tags=("lr-sweep", "wide"), metrics={"mae": 0.20, "r2": 0.9})
    _record(tmp_path, 1, model="xgb", metrics={"mae": 0.05})
    return RunRegistry(tmp_path)


class TestListing:
    """Runs come straight from the catalog, filtered."""

    def test_every_recorded_run_is_listed(self, registry):
        """No separate registration step: training is registration."""
        assert [run.identifier for run in registry.runs()] == [
            "ridge/v1",
            "ridge/v2",
            "ridge/v3",
            "xgb/v1",
        ]

    def test_runs_filter_by_model(self, registry):
        """The narrowest common question."""
        assert {run.model_name for run in registry.runs(model="xgb")} == {"xgb"}

    def test_runs_filter_by_tag(self, registry):
        """Selecting an experiment by the tag its runs were trained with."""
        assert [run.version for run in registry.runs(tags=["lr-sweep"])] == [1, 2, 3]

    def test_several_tags_must_all_match(self, registry):
        """All rather than any: narrowing is the common question."""
        assert [run.version for run in registry.runs(tags=["lr-sweep", "wide"])] == [3]

    def test_runs_filter_by_job(self, tmp_path):
        """Each member of a group set is its own job."""
        _record(tmp_path, 1, job="north")
        _record(tmp_path, 1, job="south")
        assert [run.job_id for run in RunRegistry(tmp_path).runs(job="south")] == ["south"]

    def test_an_empty_store_lists_nothing(self, tmp_path):
        """Not an error: a new store is a normal state."""
        assert RunRegistry(tmp_path).runs() == ()

    def test_a_run_carries_its_metrics_and_directory(self, registry, tmp_path):
        """Everything needed to judge and open it, without opening it."""
        run = registry.get("ridge", version=2)
        assert run.metric("mae") == pytest.approx(0.10)
        assert run.directory == (tmp_path / "bundles" / "ridge" / "v2").resolve()

    def test_a_metric_the_run_did_not_record_is_none(self, registry):
        """Absent, not zero: a zero error would look like a perfect model."""
        assert registry.get("ridge", version=1).metric("r2") is None


class TestGettingOneRun:
    """By version, by alias, or the latest."""

    def test_the_default_is_the_latest(self, registry):
        """The usual meaning of "the ridge model"."""
        assert registry.get("ridge").version == 3

    def test_latest_is_accepted_as_an_alias(self, registry):
        """So a caller passing an alias from configuration can ask for it."""
        assert registry.get("ridge", alias="latest").version == 3

    def test_a_version_is_returned_exactly(self, registry):
        """Not the nearest, not the latest."""
        assert registry.get("ridge", version=1).version == 1

    def test_a_missing_version_names_those_that_exist(self, registry):
        """The useful half of the error."""
        with pytest.raises(BundleError, match=r"versions: \[1, 2, 3\]"):
            registry.get("ridge", version=9)

    def test_an_unknown_model_is_refused(self, registry):
        """Saying where it looked, since a wrong root is the usual cause."""
        with pytest.raises(BundleError, match="nothing is recorded for lasso"):
            registry.get("lasso")

    def test_a_version_and_an_alias_together_are_refused(self, registry):
        """Two answers to one question; neither is silently preferred."""
        with pytest.raises(SpecError, match="not both"):
            registry.get("ridge", version=1, alias="production")

    def test_a_single_run_is_not_confused_with_a_job(self, tmp_path):
        """
        ``job=None`` means the single run, not any job.

        A version number only means something within one job, so letting
        ``None`` match every job would return a different model's v1.
        """
        _record(tmp_path, 1, job="north")
        with pytest.raises(BundleError, match="nothing is recorded for ridge "):
            RunRegistry(tmp_path).get("ridge")


class TestBest:
    """The best run by one metric, in a stated direction."""

    def test_minimise_picks_the_lowest(self, registry):
        """The sweep's v2 has the lowest error."""
        assert registry.best("mae", direction="minimise", model="ridge").version == 2

    def test_maximise_picks_the_highest(self, registry):
        """Runs without the metric are not candidates."""
        assert registry.best("r2", direction="maximise").version == 3

    def test_filters_narrow_the_candidates(self, registry):
        """Without the tag filter, xgb's lower error would win."""
        assert registry.best("mae", direction="minimise").model_name == "xgb"
        assert registry.best("mae", direction="minimise", tags=["lr-sweep"]).model_name == "ridge"

    def test_a_tie_goes_to_the_most_recent(self, tmp_path):
        """Usually the one somebody meant."""
        _record(tmp_path, 1, metrics={"mae": 0.1})
        _record(tmp_path, 2, metrics={"mae": 0.1})
        assert RunRegistry(tmp_path).best("mae", direction="minimise").version == 2

    def test_a_misspelt_metric_names_the_recorded_ones(self, registry):
        """The usual cause, so the message answers it."""
        with pytest.raises(BundleError, match=r"recorded metrics: \['mae', 'r2'\]"):
            registry.best("MAE", direction="minimise")

    def test_no_matching_run_is_refused(self, registry):
        """An empty selection has no best."""
        with pytest.raises(BundleError, match="no run matches"):
            registry.best("mae", direction="minimise", tags=["never-used"])

    def test_tags_may_be_a_one_shot_iterable(self, registry):
        """A generator is consumed once, not once per use."""
        best = registry.best("mae", direction="minimise", tags=(tag for tag in ["lr-sweep"]))
        assert best.version == 2


class TestAliases:
    """A name pointing at exactly one version, per model and job."""

    def test_a_promoted_run_is_found_by_its_alias(self, registry):
        """The point of an alias."""
        registry.promote(registry.get("ridge", version=2), "production")
        assert registry.get("ridge", alias="production").version == 2

    def test_promoting_again_moves_the_alias(self, registry):
        """One version per alias: the old holder loses it."""
        registry.promote(registry.get("ridge", version=1), "production")
        registry.promote(registry.get("ridge", version=3), "production")
        assert registry.get("ridge", alias="production").version == 3
        assert registry.get("ridge", version=1).aliases == ()

    def test_a_run_lists_its_aliases(self, registry):
        """Sorted, so a listing is stable."""
        run = registry.get("ridge", version=2)
        registry.promote(run, "production")
        registry.promote(run, "champion")
        assert registry.get("ridge", version=2).aliases == ("champion", "production")

    def test_aliases_are_scoped_per_job(self, tmp_path):
        """In a group set each group has its own production model."""
        _record(tmp_path, 1, job="north")
        _record(tmp_path, 1, job="south")
        _record(tmp_path, 2, job="south")
        registry = RunRegistry(tmp_path)
        registry.promote(registry.get("ridge", job="north", version=1), "production")
        registry.promote(registry.get("ridge", job="south", version=2), "production")
        assert registry.get("ridge", job="north", alias="production").version == 1
        assert registry.get("ridge", job="south", alias="production").version == 2

    def test_an_unknown_alias_names_those_that_are_set(self, registry):
        """So a typo is obvious."""
        registry.promote(registry.get("ridge", version=2), "production")
        with pytest.raises(BundleError, match=r"aliases set: \['production'\]"):
            registry.get("ridge", alias="prod")

    def test_demoting_clears_the_alias(self, registry):
        """The alias then points at nothing."""
        registry.promote(registry.get("ridge", version=2), "production")
        registry.demote("ridge", "production")
        with pytest.raises(BundleError, match="no alias 'production'"):
            registry.get("ridge", alias="production")

    def test_demoting_an_alias_that_is_not_set_is_refused(self, registry):
        """Almost always a typo; succeeding silently would hide it."""
        with pytest.raises(BundleError, match="no alias 'prodution' to clear"):
            registry.demote("ridge", "prodution")

    @pytest.mark.parametrize("alias", ["latest", "v2", "two words", "a/b", ""])
    def test_an_ambiguous_alias_is_refused(self, registry, alias):
        """Each would shadow a version reference or be awkward to type."""
        with pytest.raises(SpecError):
            registry.promote(registry.get("ridge", version=1), alias)

    def test_a_run_from_another_store_cannot_be_promoted(self, registry, tmp_path):
        """Its alias would point at nothing here."""
        _record(tmp_path / "other", 7)
        stranger = RunRegistry(tmp_path / "other").get("ridge", version=7)
        with pytest.raises(BundleError, match="no version 7"):
            registry.promote(stranger, "production")


class TestTagsAddedLater:
    """What a run turned out to be, recorded after the fact."""

    def test_an_added_tag_is_selectable(self, registry):
        """Indistinguishable from a training-time tag when filtering."""
        registry.tag(registry.get("ridge", version=1), "reviewed")
        assert [run.version for run in registry.runs(tags=["reviewed"])] == [1]

    def test_training_time_tags_come_first(self, registry):
        """The run's own record, then what was added."""
        registry.tag(registry.get("ridge", version=3), "reviewed")
        assert registry.get("ridge", version=3).tags == ("lr-sweep", "wide", "reviewed")

    def test_adding_a_tag_twice_is_a_no_op(self, registry):
        """Idempotent, so a scheduled job can re-run safely."""
        run = registry.get("ridge", version=1)
        registry.tag(run, "reviewed")
        registry.tag(run, "reviewed")
        assert registry.get("ridge", version=1).added_tags == ("reviewed",)

    def test_an_added_tag_can_be_removed(self, registry):
        """A review can be reversed."""
        run = registry.get("ridge", version=1)
        registry.tag(run, "reviewed")
        registry.untag(run, "reviewed")
        assert registry.get("ridge", version=1).added_tags == ()

    def test_a_training_time_tag_cannot_be_removed(self, registry):
        """It is part of the bundle's record of how it was made."""
        with pytest.raises(BundleError, match="trained with tag 'lr-sweep'"):
            registry.untag(registry.get("ridge", version=1), "lr-sweep")

    def test_removing_a_tag_the_run_lacks_is_refused(self, registry):
        """A typo, again."""
        with pytest.raises(BundleError, match="no tag 'reviewd'"):
            registry.untag(registry.get("ridge", version=1), "reviewd")

    def test_the_manifest_is_never_rewritten(self, registry, tmp_path):
        """Decisions live beside the catalog, not in it."""
        before = (tmp_path / "catalog.jsonl").read_bytes()
        registry.tag(registry.get("ridge", version=1), "reviewed")
        registry.promote(registry.get("ridge", version=1), "production")
        assert (tmp_path / "catalog.jsonl").read_bytes() == before
        assert registry.get("ridge", version=1).manifest.tags == ("lr-sweep",)


class TestHistory:
    """The log is the audit trail."""

    def test_every_decision_is_recorded_in_order(self, registry):
        """Who promoted what, and when, is the question a review asks."""
        registry.promote(registry.get("ridge", version=1), "production")
        registry.promote(registry.get("ridge", version=2), "production")
        registry.tag(registry.get("ridge", version=2), "reviewed")
        kinds = [(event.kind, event.label, event.version) for event in registry.history()]
        assert kinds == [
            ("promote", "production", 1),
            ("promote", "production", 2),
            ("tag", "reviewed", 2),
        ]

    def test_history_filters_by_model(self, registry):
        """One model's story, without the others'."""
        registry.promote(registry.get("xgb", version=1), "production")
        registry.promote(registry.get("ridge", version=1), "production")
        assert [event.model_name for event in registry.history(model="xgb")] == ["xgb"]

    def test_an_event_describes_itself(self, registry):
        """One readable line per decision."""
        registry.promote(registry.get("ridge", version=2), "production")
        assert registry.history()[0].describe().endswith("promote production -> ridge/v2")

    def test_a_torn_line_costs_one_decision(self, registry, tmp_path):
        """As in the catalog: damage is contained, not fatal."""
        registry.promote(registry.get("ridge", version=1), "production")
        with (tmp_path / REGISTRY_FILENAME).open("a", encoding="utf-8") as handle:
            handle.write('{"kind": "prom\n')
        registry.tag(registry.get("ridge", version=1), "reviewed")
        assert len(registry.history()) == 2


class TestDirectories:
    """Opening a selected run."""

    def test_a_run_recorded_without_a_location_says_so(self, tmp_path):
        """Rather than handing back ``None`` to fail somewhere later."""
        _record(tmp_path, 1, with_directory=False)
        with pytest.raises(BundleError, match="without a location"):
            _ = RunRegistry(tmp_path).get("ridge").directory

    def test_a_deleted_bundle_says_so(self, tmp_path):
        """The catalog outlived the directory; the message says which."""
        directory = _record(tmp_path, 1)
        directory.rmdir()
        with pytest.raises(BundleError, match="no longer exists"):
            _ = RunRegistry(tmp_path).get("ridge").directory


class TestDescribing:
    """One line per run, for a listing."""

    def test_a_run_describes_its_aliases_and_tags(self, registry):
        """Everything a person scanning a list needs."""
        run = registry.get("ridge", version=3)
        registry.promote(run, "production")
        assert registry.get("ridge", version=3).describe() == (
            "ridge/v3 [production] tags: lr-sweep, wide"
        )

    def test_a_plain_run_is_just_its_identifier(self):
        """No empty brackets or labels."""
        manifest = Manifest(
            model_name="ridge",
            engine="sklearn",
            version=1,
            created_at=_EPOCH,
            framework_version="0.1.0",
            spec_digest="deadbeef",
        )
        assert RegisteredRun(manifest=manifest).describe() == "ridge/v1"


class TestConcurrency:
    """Simultaneous decisions from separate processes are all kept."""

    def test_two_processes_tagging_lose_nothing(self, tmp_path):
        """
        The defect the event log exists to prevent.

        A read-modify-write index keeps whichever process wrote last. With
        the log, both processes' ten tags must all survive.
        """
        _record(tmp_path, 1)
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_tag_in_subprocess, str(tmp_path), name) for name in "ab"]
            for future in futures:
                future.result()
        added = RunRegistry(tmp_path).get("ridge").added_tags
        assert sorted(added) == sorted(f"{name}-{index}" for name in "ab" for index in range(10))
