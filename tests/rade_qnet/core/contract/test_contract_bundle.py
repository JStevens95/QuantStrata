"""
Tests for the bundle contracts.

A bundle is self-describing: given only a directory, the framework can say
which model produced it, from which spec, against which data, at which code
version. These tests cover the in-memory and metadata half of that claim; the
filesystem half is covered under ``tests/rade_qnet/storage``.

The manifest's content hashes are the part that turns "we think this bundle is
intact" into something checkable, so corruption fails at load time with a
message naming the file rather than surfacing as predictions that are subtly
wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.rade_qnet.core.contract.bundle import (
    BUNDLE_SCHEMA_VERSION,
    FITTED_STATE_DIRNAME,
    Manifest,
    ManifestEntry,
    SavedBundle,
)
from src.rade_qnet.core.runtime.errors import BundleError
from src.rade_qnet.testkit.fixtures import make_model_bundle

DIGEST = "a" * 64


def _entry(path, digest=DIGEST, size=10):
    """
    Build a manifest entry.

    Parameters
    ----------
    path
        Relative path.
    digest
        Hexadecimal sha256.
    size
        File size in bytes.

    Returns
    -------
    ManifestEntry
        The entry.
    """
    return ManifestEntry(relative_path=path, sha256=digest, size_bytes=size)


def _manifest(files=(), **overrides):
    """
    Build a manifest with plausible metadata.

    Parameters
    ----------
    files
        Manifest entries.
    overrides
        Fields to override.

    Returns
    -------
    Manifest
        The manifest.
    """
    fields = {
        "model_name": "demo",
        "engine": "sklearn",
        "version": 1,
        "created_at": datetime.now(UTC),
        "framework_version": "0.1.0",
        "spec_digest": "deadbeef",
        "files": tuple(files),
    }
    return Manifest(**{**fields, **overrides})


class TestManifestEntry:
    """One file, hashed."""

    def test_a_well_formed_entry_is_accepted(self):
        """The normal case."""
        assert _entry("spec.json").size_bytes == 10

    @pytest.mark.parametrize("digest", ["a" * 63, "a" * 65, ""])
    def test_a_digest_of_the_wrong_length_is_rejected(self, digest):
        """
        A sha256 is sixty-four hex characters, always.

        Anything else means the writer used a different algorithm, and a
        verifier comparing across algorithms would report every file as
        corrupt.
        """
        with pytest.raises(ValidationError):
            ManifestEntry(relative_path="spec.json", sha256=digest, size_bytes=10)

    def test_a_negative_size_is_rejected(self):
        """Not a quantity that can be negative."""
        with pytest.raises(ValidationError):
            ManifestEntry(relative_path="spec.json", sha256=DIGEST, size_bytes=-1)

    def test_an_empty_file_is_representable(self):
        """Zero bytes is a legitimate file and a legitimate size."""
        assert _entry("empty", size=0).size_bytes == 0


class TestManifest:
    """The bundle's description of itself."""

    def test_the_schema_version_defaults_to_the_current_one(self):
        """
        So a reader can refuse a bundle it cannot understand.

        Loading a future layout partially is worse than refusing it, because
        the result looks like a working model.
        """
        assert _manifest().schema_version == BUNDLE_SCHEMA_VERSION

    def test_a_duplicated_file_path_is_rejected(self):
        """
        Verification would be ambiguous about which hash is authoritative.

        And an ambiguous verifier is one that can be made to pass by writing
        the file twice.
        """
        with pytest.raises(ValidationError, match="more than once"):
            _manifest([_entry("spec.json"), _entry("spec.json", digest="b" * 64)])

    def test_distinct_paths_are_accepted(self):
        """The normal case, with two files."""
        assert len(_manifest([_entry("spec.json"), _entry("result.json")]).files) == 2

    def test_file_paths_are_returned_sorted(self):
        """
        Not in write order, so two bundles are comparable.

        A manifest whose ordering depended on which file the writer happened
        to flush first would make diffing two bundles useless.
        """
        manifest = _manifest([_entry("result.json"), _entry("spec.json")])
        assert manifest.file_paths == ("result.json", "spec.json")

    def test_an_entry_is_retrievable_by_path(self):
        """How the verifier finds the expected hash."""
        manifest = _manifest([_entry("spec.json", size=42)])
        assert manifest.entry_for("spec.json").size_bytes == 42

    def test_an_unlisted_path_lists_what_is_present(self):
        """
        The usual cause is a renamed file between writer and reader.

        Listing what is there makes that visible immediately.
        """
        with pytest.raises(BundleError, match=r"result\.json"):
            _manifest([_entry("result.json")]).entry_for("spec.json")

    def test_a_version_below_one_is_rejected(self):
        """
        Versions are assigned by the catalog and start at one.

        A zero would collide with "no version assigned yet".
        """
        with pytest.raises(ValidationError):
            _manifest(version=0)

    def test_metrics_are_duplicated_into_the_manifest(self):
        """
        So a catalog listing need not open every result file.

        Listing a hundred bundles should not mean a hundred extra reads.
        """
        assert _manifest(metrics={"mae": 0.25}).metrics["mae"] == 0.25

    def test_it_round_trips_through_json(self):
        """The manifest is itself a file in the bundle."""
        manifest = _manifest([_entry("spec.json")], tags=("nightly",))
        assert Manifest.model_validate_json(manifest.model_dump_json()) == manifest


class TestIdentifier:
    """What a bundle is called on a command line."""

    def test_a_job_bundle_includes_its_job(self):
        """
        A job set writes many bundles for one model.

        Without the job, two clusters' bundles are indistinguishable by name.
        """
        assert _manifest(job_id="EURUSD", version=3).identifier == "demo/EURUSD/v3"

    def test_a_single_run_omits_the_job(self):
        """
        Rather than inserting an empty segment.

        ``demo//v3`` would be both ugly and ambiguous.
        """
        assert _manifest(version=3).identifier == "demo/v3"


class TestModelBundle:
    """The in-memory form, holding live objects."""

    def test_an_unwritten_bundle_knows_it_is_not_persisted(self):
        """
        Answered without touching the filesystem.

        Which matters because the question is asked on paths where the
        bundle may never be written at all -- a tuning trial, say.
        """
        assert not make_model_bundle().is_persisted

    def test_a_written_bundle_carries_its_manifest(self):
        """The manifest is the evidence that it was written."""
        assert make_model_bundle(with_manifest=True).is_persisted

    def test_it_holds_the_live_model_object(self):
        """
        Typed as ``object``, which is stricter than ``Any``.

        ``core`` cannot name an engine's model type, and ``object`` makes a
        type checker enforce that nothing here tries to use it.
        """
        marker = object()
        bundle = make_model_bundle(model=marker)
        assert bundle.model is marker

    def test_it_holds_an_invertible_state(self):
        """
        The thing that makes a six-month-old bundle usable.

        Without it, the saved model produces numbers in a space whose
        meaning was lost with the original run.
        """
        bundle = make_model_bundle()
        assert bundle.state.inverse_transform_targets is not None

    def test_the_bundle_is_frozen(self):
        """A finished run is a record, not a workspace."""
        with pytest.raises(AttributeError):
            make_model_bundle().model = object()


class TestSavedBundle:
    """The on-disk form, with every path derived from one directory."""

    @pytest.fixture
    def saved(self, tmp_path):
        """
        Provide a saved bundle rooted at a temporary directory.

        Returns
        -------
        SavedBundle
            The located bundle.
        """
        return SavedBundle(directory=tmp_path, manifest=_manifest())

    @pytest.mark.parametrize(
        ("attribute", "name"),
        [
            ("spec_path", "spec.json"),
            ("signature_path", "signature.json"),
            ("lineage_path", "lineage.json"),
            ("result_path", "result.json"),
            ("weights_path", "weights.bin"),
        ],
    )
    def test_each_path_is_derived_from_the_directory(self, saved, attribute, name):
        """
        Named constants, not literals repeated in writer and reader.

        The two disagreeing is a failure that only shows up at load time, by
        which point the run that produced the bundle is long gone.
        """
        assert getattr(saved, attribute) == saved.directory / name

    def test_the_fitted_state_is_a_directory(self, saved):
        """
        Because a state is naturally several arrays.

        Forcing a scaler's statistics, a graph's three arrays and a selected
        feature list through one file reintroduces the fragility the
        directory-based interface removes.
        """
        assert saved.fitted_state_directory.name == FITTED_STATE_DIRNAME
