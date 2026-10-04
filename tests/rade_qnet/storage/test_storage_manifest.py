"""
Tests for manifest construction and verification.

A manifest is what turns a directory of files into a bundle that can be
trusted. The failures it catches -- a file truncated by a full disk, a bundle
half-copied off a file share, an edit from someone's debugging session six
weeks ago -- all produce a model that loads and predicts, just not the model
anyone believes it is.

Verification is deliberately not automatic, so these tests also pin the three
discrepancy kinds apart. "Missing", "changed" and "extra" are different
problems with different causes, and collapsing them into one message sends the
reader after the wrong explanation.
"""

from __future__ import annotations

import pytest

from src.rade_qnet.core.contract.bundle import BUNDLE_SCHEMA_VERSION
from src.rade_qnet.core.runtime.errors import BundleError
from src.rade_qnet.storage.manifest import (
    EXCLUDED_DIRECTORIES,
    EXCLUDED_NAMES,
    build_manifest,
    collect_entries,
    verify_manifest,
)


def _populate(directory, files):
    """
    Write files into a directory, creating parents as needed.

    Parameters
    ----------
    directory
        Target directory.
    files
        Mapping of relative POSIX path to text content.

    Returns
    -------
    pathlib.Path
        The directory.
    """
    for name, content in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return directory


@pytest.fixture
def bundle_directory(tmp_path):
    """
    Provide a small written-out bundle directory.

    Returns
    -------
    pathlib.Path
        A directory with two files, one of them nested.
    """
    return _populate(tmp_path, {"spec.json": "{}", "fitted_state/scaler.npy": "data"})


def _manifest_for(directory, **overrides):
    """
    Build a manifest for a directory with plausible metadata.

    Parameters
    ----------
    directory
        The bundle directory.
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
        "framework_version": "0.1.0",
        "spec_digest": "deadbeef",
    }
    return build_manifest(directory, **{**fields, **overrides})


class TestCollectingEntries:
    """What goes into a manifest, and what deliberately does not."""

    def test_every_file_is_listed(self, bundle_directory):
        """Including nested ones, since a fitted state is a directory."""
        paths = [entry.relative_path for entry in collect_entries(bundle_directory)]
        assert paths == ["fitted_state/scaler.npy", "spec.json"]

    def test_paths_are_posix_form(self, bundle_directory):
        """
        So a bundle written on Windows verifies on Linux.

        A backslash in a recorded path would make every nested file appear
        missing on the other platform.
        """
        assert all("\\" not in entry.relative_path for entry in collect_entries(bundle_directory))

    def test_entries_are_sorted_by_path(self, tmp_path):
        """
        Not in filesystem order, which is arbitrary.

        The manifest's own digest is computed over its JSON, so a stable
        order is what lets two bundles with identical contents produce
        identical manifests.
        """
        _populate(tmp_path, {"z.json": "1", "a.json": "2", "m.json": "3"})
        paths = [entry.relative_path for entry in collect_entries(tmp_path)]
        assert paths == sorted(paths)

    def test_the_manifest_itself_is_excluded(self, bundle_directory):
        """
        It cannot list itself.

        Its own digest would have to be known before it was written.
        """
        _populate(bundle_directory, {"manifest.json": "{}"})
        assert "manifest.json" not in [e.relative_path for e in collect_entries(bundle_directory)]

    @pytest.mark.parametrize("debris", sorted(EXCLUDED_NAMES - {"manifest.json"}))
    def test_filesystem_debris_is_excluded(self, bundle_directory, debris):
        """
        Opening a bundle directory in Finder must not invalidate it.

        A bundle that fails verification because someone looked at it would
        train everyone to ignore verification failures.
        """
        _populate(bundle_directory, {debris: "junk"})
        assert debris not in [e.relative_path for e in collect_entries(bundle_directory)]

    @pytest.mark.parametrize("ignored", sorted(EXCLUDED_DIRECTORIES))
    def test_excluded_directories_are_skipped_entirely(self, bundle_directory, ignored):
        """Their contents are debris too, however they are named."""
        _populate(bundle_directory, {f"{ignored}/cached": "junk"})
        paths = [entry.relative_path for entry in collect_entries(bundle_directory)]
        assert all(ignored not in path for path in paths)

    def test_the_recorded_size_matches_the_file(self, tmp_path):
        """
        Recorded so a truncated file is detectable without hashing it.

        Which makes the common corruption case cheap to catch.
        """
        _populate(tmp_path, {"a.txt": "12345"})
        assert collect_entries(tmp_path)[0].size_bytes == 5

    def test_an_empty_directory_yields_no_entries(self, tmp_path):
        """Empty is a valid state, not an error."""
        assert collect_entries(tmp_path) == ()

    def test_a_missing_directory_is_rejected(self, tmp_path):
        """
        Distinguished from an empty one.

        An empty manifest for a path that does not exist would verify
        happily and describe nothing.
        """
        with pytest.raises(BundleError, match="not a directory"):
            collect_entries(tmp_path / "absent")

    def test_identical_contents_produce_identical_entries(self, tmp_path):
        """
        Two bundles written from the same data are comparable.

        Which is what makes "did anything actually change?" answerable.
        """
        first = _populate(tmp_path / "one", {"a.txt": "x", "b.txt": "y"})
        second = _populate(tmp_path / "two", {"b.txt": "y", "a.txt": "x"})
        assert collect_entries(first) == collect_entries(second)


class TestBuildingAManifest:
    """Metadata, plus whatever is on disk at the moment it is called."""

    def test_the_current_schema_version_is_recorded(self, bundle_directory):
        """So a future reader can refuse a layout it does not understand."""
        assert _manifest_for(bundle_directory).schema_version == BUNDLE_SCHEMA_VERSION

    def test_every_file_present_is_listed(self, bundle_directory):
        """
        It hashes what it finds, which is why it is called last.

        A file written afterwards produces a manifest that no longer
        describes the bundle.
        """
        assert len(_manifest_for(bundle_directory).files) == 2

    def test_headline_metrics_are_carried(self, bundle_directory):
        """
        So a catalog listing need not open every result file.

        Listing a hundred bundles should not cost a hundred extra reads.
        """
        manifest = _manifest_for(bundle_directory, metrics={"mae": 0.25})
        assert manifest.metrics["mae"] == 0.25

    def test_a_freshly_built_manifest_verifies(self, bundle_directory):
        """The round trip, which everything else here is a variation on."""
        verify_manifest(bundle_directory, _manifest_for(bundle_directory))


class TestVerification:
    """Three discrepancy kinds, kept distinct."""

    def test_a_missing_file_is_reported(self, bundle_directory):
        """An incomplete or partially copied bundle."""
        manifest = _manifest_for(bundle_directory)
        (bundle_directory / "spec.json").unlink()
        with pytest.raises(BundleError, match="not present"):
            verify_manifest(bundle_directory, manifest)

    def test_changed_contents_are_reported(self, bundle_directory):
        """
        Corruption, or an edit.

        The case the digests exist for: the file is the right size and the
        right name, and the wrong model.
        """
        manifest = _manifest_for(bundle_directory)
        (bundle_directory / "spec.json").write_text("{}" + " ", encoding="utf-8")
        with pytest.raises(BundleError):
            verify_manifest(bundle_directory, manifest)

    def test_contents_changed_without_a_size_change_are_still_caught(self, bundle_directory):
        """
        The size check is a shortcut, not the guarantee.

        A single flipped byte keeps the size identical, so the digest has to
        be what finally decides.
        """
        manifest = _manifest_for(bundle_directory)
        (bundle_directory / "spec.json").write_text("[]", encoding="utf-8")
        with pytest.raises(BundleError, match="contents have changed"):
            verify_manifest(bundle_directory, manifest)

    def test_an_extra_file_is_reported(self, bundle_directory):
        """
        Something wrote into the bundle after it was sealed.

        Every listed file is intact, and the bundle is still no longer the
        thing the manifest describes.
        """
        manifest = _manifest_for(bundle_directory)
        _populate(bundle_directory, {"stray.txt": "written later"})
        with pytest.raises(BundleError, match="not listed"):
            verify_manifest(bundle_directory, manifest)

    def test_every_discrepancy_is_collected_before_raising(self, bundle_directory):
        """
        All three discrepancy kinds appear in one message.

        Being told one file of two hundred is wrong, when eighty are, sends
        the reader after the wrong explanation entirely.
        """
        manifest = _manifest_for(bundle_directory)
        (bundle_directory / "spec.json").unlink()
        (bundle_directory / "fitted_state" / "scaler.npy").write_text("x", encoding="utf-8")
        _populate(bundle_directory, {"stray.txt": "later"})
        with pytest.raises(BundleError) as caught:
            verify_manifest(bundle_directory, manifest)
        message = str(caught.value)
        assert "spec.json" in message
        assert "scaler.npy" in message
        assert "stray.txt" in message

    def test_an_unknown_schema_version_is_refused_outright(self, bundle_directory):
        """
        Checked before any file is read.

        Loading a future layout partially is worse than refusing it, because
        a partially loaded bundle looks like a working model.
        """
        manifest = _manifest_for(bundle_directory).model_copy(
            update={"schema_version": BUNDLE_SCHEMA_VERSION + 1}
        )
        with pytest.raises(BundleError, match="schema version"):
            verify_manifest(bundle_directory, manifest)

    def test_debris_added_after_sealing_does_not_fail_verification(self, bundle_directory):
        """
        The exclusion list has to apply on both sides.

        Otherwise a bundle verified on a Mac fails because Finder wrote a
        ``.DS_Store`` into it, and the exclusion would be pointless.
        """
        manifest = _manifest_for(bundle_directory)
        _populate(bundle_directory, {".DS_Store": "junk"})
        verify_manifest(bundle_directory, manifest)
