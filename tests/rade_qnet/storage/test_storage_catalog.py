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
from src.rade_qnet.core.runtime.errors import BundleError
from src.rade_qnet.storage.catalog import (
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
