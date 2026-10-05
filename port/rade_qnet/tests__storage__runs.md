# `tests/rade_qnet/storage/runs`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 1 | 40 | `3c6b979e4a3d6f08` |
| 2 | `test_runs_catalog.py` | 482 | 17709 | `973463f5d4c635b2` |
| 3 | `test_runs_registry.py` | 463 | 19355 | `25025909f204c9e9` |
| 4 | `test_runs_tracker.py` | 227 | 7875 | `259a9fe4ee5c9963` |

---

## 1. `tests/rade_qnet/storage/runs/__init__.py`

40 bytes · SHA-256 `3c6b979e4a3d6f08`

```python
"""Mirror of rade_qnet.storage.runs."""
```

---

## 2. `tests/rade_qnet/storage/runs/test_runs_catalog.py`

17709 bytes · SHA-256 `973463f5d4c635b2`

```python
"""
Tests for the catalog of what has been trained.

This module exists to fix a specific, reproducible defect: assigning a version
by reading the catalog, computing ``max(versions) + 1`` and writing the whole
file back. With one run at a time it works. With a job set of two hundred
members running eight at a time -- the normal case, not an edge case -- two
workers read the same state, compute the same version, and the second write
erases the first worker's entry. The bundle is still on disk; it has simply
become invisible, and nothing reports an error.

So the tests here are mostly about concurrency and about damage containment,
including a genuine multi-process probe. A single-process test cannot
demonstrate that the fix works, and the whole point of the fix is multi-process
behaviour.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.rade_qnet.core.contract.bundle import Manifest
from src.rade_qnet.core.lifecycle.errors import BundleError
from src.rade_qnet.storage.runs.catalog import (
    CATALOG_FILENAME,
    LOCK_FILENAME,
    InMemoryCatalog,
    JsonlCatalog,
)


def _manifest(version=1, *, model_name="demo", job_id=None, tags=()):
    """
    Build a manifest with plausible metadata.

    Parameters
    ----------
    version
        Version number.
    model_name
        Registered model name.
    job_id
        Job identifier, or ``None``.
    tags
        Free-form labels.

    Returns
    -------
    Manifest
        The manifest.
    """
    return Manifest(
        model_name=model_name,
        engine="sklearn",
        version=version,
        created_at=datetime.now(UTC),
        framework_version="0.1.0",
        spec_digest="deadbeef",
        job_id=job_id,
        tags=tags,
    )


def _reserve_in_subprocess(root):
    """
    Reserve four versions from a separate process.

    Defined at module level because a process pool must be able to import it.

    Parameters
    ----------
    root
        Catalog root directory.

    Returns
    -------
    list of int
        The reserved versions.
    """
    catalog = JsonlCatalog(Path(root))
    return [catalog.next_version("demo") for _ in range(4)]


@pytest.fixture
def catalog(tmp_path):
    """
    Provide a file-backed catalog in a temporary directory.

    Returns
    -------
    JsonlCatalog
        The catalog.
    """
    return JsonlCatalog(tmp_path)


class TestVersionAssignment:
    """Versions are unique, contiguous and scoped to a job."""

    def test_the_first_version_is_one(self, catalog):
        """
        Not zero, which would collide with "no version assigned".

        And not an arbitrary start, since a user reads these.
        """
        assert catalog.next_version("demo") == 1

    def test_versions_increase_without_recording_anything(self, catalog):
        """
        The defect, in its single-process form.

        A reservation followed by a long data build must not hand the same
        number to the next caller just because nothing has been recorded yet.
        """
        assert [catalog.next_version("demo") for _ in range(3)] == [1, 2, 3]

    def test_a_recorded_bundle_advances_the_sequence(self, catalog):
        """A catalog restored from disk continues where it left off."""
        catalog.record(_manifest(version=5))
        assert catalog.next_version("demo") == 6

    def test_versions_are_scoped_per_model(self, catalog):
        """Two unrelated models should not share a numbering sequence."""
        catalog.next_version("demo")
        assert catalog.next_version("other") == 1

    def test_versions_are_scoped_per_job(self, catalog):
        """
        Each member of a job set has its own sequence starting at one.

        Sharing one sequence across two hundred members would make a
        member's version number meaningless on its own.
        """
        catalog.next_version("demo", job_id="EURUSD")
        assert catalog.next_version("demo", job_id="USDJPY") == 1

    def test_a_job_and_a_single_run_are_different_sequences(self, catalog):
        """
        ``None`` is a distinct scope, not a wildcard.

        Treating it as a wildcard would make a single run collide with every
        job of the same model.
        """
        catalog.next_version("demo", job_id="EURUSD")
        assert catalog.next_version("demo", job_id=None) == 1

    def test_a_reservation_survives_a_new_catalog_object(self, tmp_path):
        """
        The reservation is on disk, not in the object.

        Which is the only way a second *process* can see it.
        """
        JsonlCatalog(tmp_path).next_version("demo")
        assert JsonlCatalog(tmp_path).next_version("demo") == 2


class TestConcurrency:
    """The failure this module exists to prevent, demonstrated."""

    def test_concurrent_processes_never_share_a_version(self, tmp_path):
        """
        A real multi-process probe, because nothing less proves it.

        Four workers reserving four versions each must produce sixteen
        distinct, contiguous numbers. Under the read-modify-write scheme
        this replaces, the numbers collide and entries are lost.
        """
        with ProcessPoolExecutor(max_workers=4) as pool:
            batches = list(pool.map(_reserve_in_subprocess, [str(tmp_path)] * 4))

        assigned = sorted(version for batch in batches for version in batch)
        assert assigned == list(range(1, 17))

    def test_concurrent_records_are_all_retained(self, tmp_path):
        """
        Append-only is the structural fix, beyond the lock itself.

        No writer rewrites existing content, so no writer can erase another's
        entry even if the lock were somehow lost.
        """
        catalog = JsonlCatalog(tmp_path)
        for version in range(1, 11):
            catalog.record(_manifest(version=version))
        assert len(catalog.entries()) == 10

    def test_the_lock_file_is_separate_from_the_catalog(self, catalog):
        """
        So locking never depends on the catalog existing.

        A first writer would otherwise have to create the thing it is trying
        to lock, which is the classic place for a race.
        """
        catalog.next_version("demo")
        assert (catalog.root / LOCK_FILENAME).exists()
        assert catalog.lock_path != catalog.path


class TestRecordingAndReading:
    """What goes in comes back out."""

    def test_a_recorded_manifest_is_returned(self, catalog):
        """The basic round trip, through the file."""
        catalog.record(_manifest(version=1))
        assert catalog.entries()[0].version == 1

    def test_the_latest_is_the_highest_version(self, catalog):
        """
        By version, not by position in the file.

        A job set's members finish out of order, so file order says nothing
        about which version is newest.
        """
        for version in (3, 1, 2):
            catalog.record(_manifest(version=version))
        assert catalog.latest("demo").version == 3

    def test_the_latest_of_an_unknown_model_is_none(self, catalog):
        """
        A question, not an error.

        "Has this ever been trained?" is asked routinely before a first run.
        """
        assert catalog.latest("never-trained") is None

    def test_the_latest_respects_the_job_scope(self, catalog):
        """Otherwise one member's version would be reported for another."""
        catalog.record(_manifest(version=9, job_id="EURUSD"))
        assert catalog.latest("demo", job_id="USDJPY") is None

    def test_entries_can_be_filtered_by_model(self, catalog):
        """A shared store holds many models."""
        catalog.record(_manifest(model_name="demo"))
        catalog.record(_manifest(model_name="other"))
        assert len(catalog.entries(model_name="demo")) == 1

    def test_entries_can_be_filtered_by_tag(self, catalog):
        """
        Which is what tags are recorded for.

        "Show me the nightly runs" should not require reading every bundle.
        """
        catalog.record(_manifest(version=1, tags=("nightly",)))
        catalog.record(_manifest(version=2))
        assert len(catalog.entries(tag="nightly")) == 1

    def test_entries_are_returned_in_a_stable_order(self, catalog):
        """
        Sorted by model, job and version rather than by write order.

        A listing that reordered itself between calls would make diffing two
        stores impossible.
        """
        for version in (3, 1, 2):
            catalog.record(_manifest(version=version))
        assert [m.version for m in catalog.entries()] == [1, 2, 3]

    def test_reservations_are_not_reported_as_entries(self, catalog):
        """
        A reservation occupies a version; it does not describe a bundle.

        Reporting it would make the catalog claim a model exists when the
        run that would produce it is still building its data.
        """
        catalog.next_version("demo")
        assert catalog.entries() == ()

    def test_entries_are_an_immutable_sequence(self, catalog):
        """
        A tuple, so a caller cannot mutate the catalog's view by accident.

        Both implementations return the same type, which is what makes them
        substitutable.
        """
        catalog.record(_manifest())
        assert isinstance(catalog.entries(), tuple)


class TestDamageContainment:
    """One torn write costs one entry, not the index."""

    def test_an_unparseable_line_is_skipped(self, catalog):
        """
        A process killed mid-append leaves a partial line.

        Raising on it would make one interrupted run destroy the record of
        every run before it.
        """
        catalog.record(_manifest(version=1))
        with catalog.path.open("a", encoding="utf-8") as handle:
            handle.write('{"model_name": "demo", "ver\n')
        catalog.record(_manifest(version=2))
        assert [m.version for m in catalog.entries()] == [1, 2]

    def test_a_valid_line_of_the_wrong_shape_is_skipped(self, catalog):
        """
        Parsing is not validating.

        A line written by an older layout would otherwise raise and take the
        whole catalog with it.
        """
        catalog.record(_manifest(version=1))
        with catalog.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"unexpected": True}) + "\n")
        assert len(catalog.entries()) == 1

    def test_blank_lines_are_tolerated(self, catalog):
        """Trailing newlines appear whenever a file is edited by hand."""
        catalog.record(_manifest(version=1))
        with catalog.path.open("a", encoding="utf-8") as handle:
            handle.write("\n\n")
        assert len(catalog.entries()) == 1

    def test_an_absent_catalog_reads_as_empty(self, tmp_path):
        """
        A first run must not have to create the file before reading it.

        Which would be another read-then-write race, reintroduced in the
        place the lock was meant to remove it.
        """
        assert JsonlCatalog(tmp_path).entries() == ()

    def test_the_catalog_file_is_line_delimited_json(self, catalog):
        """
        Readable by ``tail`` and by anything else.

        Which matters when the thing reading it is an operator at 3am with
        no framework installed.
        """
        catalog.record(_manifest())
        lines = catalog.path.read_text(encoding="utf-8").strip().splitlines()
        assert all(json.loads(line) for line in lines)

    def test_an_unopenable_lock_is_reported_clearly(self, tmp_path):
        """
        Not as a bare ``OSError`` from deep inside the catalog.

        A read-only store is a configuration problem, and the message should
        say which path could not be locked.
        """
        catalog = JsonlCatalog(tmp_path)
        catalog.lock_path.mkdir()
        with pytest.raises(BundleError, match="lock"):
            catalog.next_version("demo")


class TestInMemoryCatalog:
    """The substitutable implementation, which must behave the same."""

    @pytest.fixture
    def catalog(self):
        """
        Provide an in-memory catalog.

        Returns
        -------
        InMemoryCatalog
            The catalog.
        """
        return InMemoryCatalog()

    def test_versions_increase_without_recording(self, catalog):
        """
        The same reservation semantics as the file-backed catalog.

        A test that passed against one and failed against the other would
        make the protocol a lie.
        """
        assert [catalog.next_version("demo") for _ in range(3)] == [1, 2, 3]

    def test_versions_are_scoped_per_job(self, catalog):
        """Matching the file-backed behaviour."""
        catalog.next_version("demo", job_id="EURUSD")
        assert catalog.next_version("demo", job_id="USDJPY") == 1

    def test_the_latest_is_the_highest_version(self, catalog):
        """Matching the file-backed behaviour."""
        for version in (3, 1, 2):
            catalog.record(_manifest(version=version))
        assert catalog.latest("demo").version == 3

    def test_the_latest_of_an_unknown_model_is_none(self, catalog):
        """Matching the file-backed behaviour."""
        assert catalog.latest("never-trained") is None

    def test_entries_are_an_immutable_sequence(self, catalog):
        """
        A tuple, as the file-backed catalog returns.

        These two disagreeing on return type was a real defect: a caller
        written against one would mutate the other's internal list.
        """
        catalog.record(_manifest())
        assert isinstance(catalog.entries(), tuple)

    def test_nothing_is_written_to_disk(self, tmp_path, catalog):
        """
        The reason it exists.

        A pipeline test needs no filesystem and no locking, and a notebook
        should not litter the working directory.
        """
        catalog.record(_manifest())
        assert not (tmp_path / CATALOG_FILENAME).exists()

    def test_entries_filter_as_the_file_backed_catalog_does(self, catalog):
        """
        The same keyword filters, and the same ordering.

        This catalog once took no filters at all, so a test written against
        it could not exercise the query a production caller makes.
        """
        catalog.record(_manifest(version=2, tags=("nightly",)))
        catalog.record(_manifest(version=1, tags=("nightly",)))
        catalog.record(_manifest(version=1, model_name="other"))
        assert [m.version for m in catalog.entries(model_name="demo", tag="nightly")] == [1, 2]

    def test_a_location_is_kept_as_given(self, tmp_path, catalog):
        """Nothing to make portable: the catalog never leaves the process."""
        catalog.record(_manifest(), location=tmp_path / "bundle")
        assert catalog.records()[0].location == tmp_path / "bundle"


class TestLocations:
    """Each entry records where its bundle is, portably."""

    def test_a_location_beneath_the_catalog_reads_back_absolute(self, catalog):
        """The reader gets a usable path, wherever it is running from."""
        bundle = catalog.root / "bundles" / "demo" / "v1"
        catalog.record(_manifest(), location=bundle)
        assert catalog.records()[0].location == bundle.resolve()

    def test_a_location_beneath_the_catalog_is_stored_relative(self, catalog):
        """
        Relative, so the store can be copied and still resolve.

        To another machine, or from macOS to Windows.
        """
        catalog.record(_manifest(), location=catalog.root / "bundles" / "demo" / "v1")
        stored = json.loads(catalog.path.read_text().strip())
        assert stored["location"] == "bundles/demo/v1"

    def test_a_moved_store_still_resolves(self, tmp_path):
        """The property the relative form exists for."""
        original = JsonlCatalog(tmp_path / "before")
        original.record(_manifest(), location=original.root / "bundles" / "v1")
        moved = tmp_path / "after"
        original.root.rename(moved)
        assert JsonlCatalog(moved).records()[0].location == moved / "bundles" / "v1"

    def test_a_location_outside_the_catalog_is_stored_absolute(self, tmp_path):
        """There is no relative path to store, so the full one is kept."""
        catalog = JsonlCatalog(tmp_path / "catalog")
        elsewhere = tmp_path / "bundles" / "v1"
        catalog.record(_manifest(), location=elsewhere)
        assert catalog.records()[0].location == elsewhere.resolve()

    def test_an_entry_without_a_location_reads_back_as_none(self, catalog):
        """Including every entry written before locations were recorded."""
        catalog.path.write_text(_manifest().model_dump_json() + "\n", encoding="utf-8")
        assert catalog.records()[0].location is None

    def test_the_location_does_not_leak_into_the_manifest(self, catalog):
        """
        The manifest and the location are read back separately.

        The manifest is a strict contract; the location is a fact about the
        store.
        """
        catalog.record(_manifest(), location=catalog.root / "v1")
        assert catalog.entries()[0] == catalog.records()[0].manifest
        assert "location" not in catalog.entries()[0].model_dump()

    def test_records_take_the_same_filters_as_entries(self, catalog):
        """One query vocabulary, whichever view is asked for."""
        catalog.record(_manifest(version=1, tags=("sweep",)), location=catalog.root / "v1")
        catalog.record(_manifest(version=2), location=catalog.root / "v2")
        assert [entry.manifest.version for entry in catalog.records(tag="sweep")] == [1]
```

---

## 3. `tests/rade_qnet/storage/runs/test_runs_registry.py`

19355 bytes · SHA-256 `25025909f204c9e9`

```python
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
```

---

## 4. `tests/rade_qnet/storage/runs/test_runs_tracker.py`

7875 bytes · SHA-256 `259a9fe4ee5c9963`

```python
"""
Tests for experiment tracking.

One rule governs this module: tracking is never load-bearing. A model that
trained successfully but could not reach a tracking server has still trained
successfully, and the bundle on disk is the system of record. So the tests
here spend most of their effort on failure paths -- a read-only directory, an
unserialisable value -- confirming that none of them can take a run down.

The second thing under test is that the default does nothing. If the baseline
behaviour required a backend, every test, notebook and quick experiment would
need one configured, and tracking would be something you disable rather than
something you add.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rade_qnet.core.lifecycle.context import Tracker
from src.rade_qnet.storage.runs.tracker import JsonlTracker, NullTracker


@pytest.fixture
def tracker(tmp_path):
    """
    Provide a file-backed tracker.

    Returns
    -------
    JsonlTracker
        The tracker.
    """
    return JsonlTracker(tmp_path / "events.jsonl")


def _events(tracker):
    """
    Read back every event a tracker has written.

    Parameters
    ----------
    tracker
        The tracker.

    Returns
    -------
    list of dict
        Decoded events, in order.
    """
    text = tracker.path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class TestTheDefaultDoesNothing:
    """``NullTracker`` is a working implementation of "do not track"."""

    def test_it_satisfies_the_protocol(self):
        """
        So it substitutes for a real tracker with no special-casing.

        A pipeline that had to check for ``None`` before every call would
        put the "is tracking configured?" question in a dozen places.
        """
        assert isinstance(NullTracker(), Tracker)

    def test_no_method_raises(self, tmp_path):
        """
        Empty, not ``NotImplementedError``.

        This is a complete implementation of a real decision, not a stub
        waiting to be filled in.
        """
        tracker = NullTracker()
        tracker.log_params({"seed": 1})
        tracker.log_metrics({"mae": 0.1}, step=0)
        tracker.log_artifact(tmp_path / "figure.png", name="curve")
        tracker.finish(succeeded=True)

    def test_it_writes_nothing(self, tmp_path):
        """A run with default tracking leaves no stray files behind."""
        NullTracker().log_params({"seed": 1})
        assert list(tmp_path.iterdir()) == []


class TestEventRecording:
    """What the file-backed tracker writes."""

    def test_it_satisfies_the_protocol(self, tracker):
        """Both implementations have to be interchangeable."""
        assert isinstance(tracker, Tracker)

    def test_parameters_are_recorded(self, tracker):
        """The configuration a run was launched with."""
        tracker.log_params({"seed": 3, "model": "demo"})
        assert _events(tracker)[0]["params"]["seed"] == 3

    def test_parameters_are_canonicalised(self, tracker):
        """
        Through the same canonical form the spec digest uses.

        So two runs whose configurations differ only in key order produce
        identical parameter records, and a diff between them is meaningful.
        """
        tracker.log_params({"b": 1, "a": 2})
        assert list(_events(tracker)[0]["params"]) == ["a", "b"]

    def test_metrics_carry_their_step(self, tracker):
        """
        Which is what makes a sequence of events a curve.

        Without it, fifty epoch records are fifty unordered points.
        """
        tracker.log_metrics({"loss": 0.5}, step=7)
        assert _events(tracker)[0]["step"] == 7

    def test_a_final_metric_has_no_step(self, tracker):
        """
        ``None`` distinguishes a summary metric from an epoch metric.

        A test-set score plotted as epoch zero would be actively misleading.
        """
        tracker.log_metrics({"mae": 0.25})
        assert _events(tracker)[0]["step"] is None

    def test_an_artifact_defaults_to_its_file_name(self, tracker, tmp_path):
        """A sensible name beats an empty one in a listing."""
        tracker.log_artifact(tmp_path / "training_curve.png")
        assert _events(tracker)[0]["name"] == "training_curve.png"

    def test_an_artifact_can_be_named_explicitly(self, tracker, tmp_path):
        """
        Because the file name is an implementation detail.

        ``metrics_test.png`` means less to a reader than "test metrics".
        """
        tracker.log_artifact(tmp_path / "f.png", name="test metrics")
        assert _events(tracker)[0]["name"] == "test metrics"

    def test_the_terminal_event_records_the_outcome(self, tracker):
        """
        So an interrupted run is distinguishable from a finished one.

        A run whose events simply stop could be either, and treating a
        crashed run's last metric as its final metric is how a bad model
        gets promoted.
        """
        tracker.finish(succeeded=False)
        assert _events(tracker)[0] == {"event": "finish", "succeeded": False}

    def test_events_are_appended_in_order(self, tracker):
        """
        One line per event, so ``tail -f`` follows a run in progress.

        And a crashed run still leaves everything written before the crash.
        """
        tracker.log_params({"seed": 1})
        tracker.log_metrics({"loss": 1.0}, step=0)
        tracker.finish(succeeded=True)
        assert [event["event"] for event in _events(tracker)] == [
            "params",
            "metrics",
            "finish",
        ]

    def test_the_parent_directory_is_created(self, tmp_path):
        """
        A run's output directory may not exist when the tracker is built.

        Failing at construction would make tracking load-bearing at the
        earliest possible moment.
        """
        JsonlTracker(tmp_path / "nested" / "deeper" / "events.jsonl").finish(succeeded=True)
        assert (tmp_path / "nested" / "deeper" / "events.jsonl").is_file()


class TestTrackingIsNeverLoadBearing:
    """Every failure path, confirmed harmless."""

    def test_an_unwritable_path_does_not_raise(self, tmp_path):
        """
        The central guarantee.

        A full disk, a revoked permission or an unmounted share must not
        fail a run that has otherwise completed.
        """
        directory = tmp_path / "events.jsonl"
        directory.mkdir()
        JsonlTracker(directory).log_metrics({"mae": 0.1})

    def test_a_failure_is_warned_about_rather_than_hidden(self, tmp_path, caplog):
        """
        Swallowed is not the same as silent.

        Someone has to be able to find out why their dashboard is empty.
        """
        directory = tmp_path / "events.jsonl"
        directory.mkdir()
        with caplog.at_level("WARNING"):
            JsonlTracker(directory).finish(succeeded=True)
        assert "tracking event" in caplog.text

    def test_a_run_can_finish_after_a_failed_write(self, tmp_path):
        """
        One failure must not poison the tracker.

        Otherwise a transient error partway through a run would silently
        stop recording everything after it.
        """
        directory = tmp_path / "events.jsonl"
        directory.mkdir()
        tracker = JsonlTracker(directory)
        tracker.log_params({"seed": 1})
        tracker.finish(succeeded=True)

    def test_an_unserialisable_value_is_coerced_rather_than_raised(self, tracker):
        """
        A spec may legitimately carry a path or an enum.

        Refusing to record it, or worse raising, would make tracking fail on
        exactly the configurations worth recording.
        """
        tracker.log_artifact(Path("/tmp/figure.png"), name="curve")
        assert _events(tracker)[0]["path"].endswith("figure.png")
```

