"""
Tests for writing and reading a bundle on disk.

Two disciplines are under test, and both exist because of specific ways model
stores go wrong.

Writes are atomic: a crash, a cancellation or a full disk leaves either no
bundle or a complete one, never a directory holding weights but no manifest --
which looks loadable and is not. The implementation being replaced wrote files
in place, so an interrupted run left behind a bundle that failed at inference
time rather than at write time.

And weights are serialised by the engine, not here. ``storage`` never imports
a training library, which is what lets one bundle format hold a network's
tensors and a boosted-tree dump.
"""

from __future__ import annotations

import json
import re
import shutil

import pytest

from src.rade_xl.core.contract.bundle import (
    FITTED_STATE_DIRNAME,
    MANIFEST_FILENAME,
    SPEC_FILENAME,
    WEIGHTS_FILENAME,
    Manifest,
)
from src.rade_xl.core.runtime.errors import BundleError
from src.rade_xl.storage.bundle import (
    TEMPORARY_PREFIX,
    bundle_directory,
    load_fitted_state,
    load_lineage,
    load_manifest,
    load_result,
    load_signature,
    open_bundle,
    write_bundle,
)
from src.rade_xl.testkit.fixtures import StandardisingState, make_model_bundle


def _write_weights(path):
    """
    Stand in for an engine's parameter serialiser.

    Parameters
    ----------
    path
        Where to write.
    """
    path.write_bytes(b"weights")


def _write(root, bundle=None, **overrides):
    """
    Write a bundle with plausible metadata.

    Parameters
    ----------
    root
        The model store root.
    bundle
        The in-memory bundle, defaulting to the standard fixture.
    overrides
        Arguments to override.

    Returns
    -------
    SavedBundle
        The written bundle.
    """
    arguments = {
        "root": root,
        "version": 1,
        "framework_version": "0.1.0",
        "engine": "sklearn",
        "model_name": "demo",
        "write_weights": _write_weights,
    }
    return write_bundle(bundle or make_model_bundle(), **{**arguments, **overrides})


class TestLayout:
    """Where a bundle lives, and why it lives there."""

    def test_a_single_run_omits_the_job_level(self, tmp_path):
        """No empty path segment for a run that has no job."""
        path = bundle_directory(tmp_path, model_name="demo", version=3, job_id=None)
        assert path == tmp_path / "demo" / "v3"

    def test_a_job_run_nests_under_its_job(self, tmp_path):
        """
        Nested rather than flattened to ``model-job-vN``.

        A job set with three hundred members stays navigable with ``ls``,
        which matters more than it sounds when something has gone wrong.
        """
        path = bundle_directory(tmp_path, model_name="demo", version=3, job_id="EURUSD")
        assert path == tmp_path / "demo" / "EURUSD" / "v3"


class TestWriting:
    """What a write produces, and what it refuses to do."""

    def test_every_expected_file_is_present(self, tmp_path):
        """
        The complete set, stated in one place in the writer.

        A bundle missing one of these loads partially, which is worse than
        not loading at all.
        """
        saved = _write(tmp_path)
        present = {entry.relative_path for entry in saved.manifest.files}
        assert present == {
            "spec.json",
            "signature.json",
            "lineage.json",
            "result.json",
            f"{FITTED_STATE_DIRNAME}/standardiser.npy",
            WEIGHTS_FILENAME,
        }

    def test_the_manifest_is_written_but_not_self_listed(self, tmp_path):
        """
        It cannot list itself: its digest would precede its own contents.

        But it must exist on disk, or the bundle cannot be verified at all.
        """
        saved = _write(tmp_path)
        assert (saved.directory / MANIFEST_FILENAME).is_file()
        assert MANIFEST_FILENAME not in saved.manifest.file_paths

    def test_the_engine_supplies_the_weights(self, tmp_path):
        """
        ``storage`` writes the bytes it is handed and nothing more.

        Which is what keeps a training library out of this layer, and what
        the layering test enforces independently.
        """
        saved = _write(tmp_path)
        assert (saved.directory / WEIGHTS_FILENAME).read_bytes() == b"weights"

    def test_headline_metrics_reach_the_manifest(self, tmp_path):
        """So a catalog listing need not open every result file."""
        assert "mae" in _write(tmp_path).manifest.metrics

    def test_an_existing_version_is_never_overwritten(self, tmp_path):
        """
        A version is final once it exists.

        A bundle that has been catalogued, evaluated and perhaps promoted
        must not change underneath those decisions.
        """
        _write(tmp_path)
        with pytest.raises(BundleError, match="refusing to overwrite"):
            _write(tmp_path)

    def test_distinct_versions_coexist(self, tmp_path):
        """The normal case for a model that is retrained."""
        first = _write(tmp_path, version=1)
        second = _write(tmp_path, version=2)
        assert first.directory != second.directory
        assert first.directory.is_dir()

    def test_tags_are_recorded(self, tmp_path):
        """Free-form labels, used for querying a store later."""
        assert _write(tmp_path, tags=("nightly",)).manifest.tags == ("nightly",)


class TestAtomicity:
    """A failed write leaves nothing behind."""

    def test_a_failure_leaves_no_bundle(self, tmp_path):
        """
        Either no bundle or a complete one, never a half-written one.

        A directory holding weights but no manifest looks loadable and is
        not -- the failure then surfaces at inference time, long after the
        run that caused it.
        """

        def exploding(path):
            """Fail partway through writing."""
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            _write(tmp_path, write_weights=exploding)
        assert not bundle_directory(tmp_path, model_name="demo", version=1, job_id=None).exists()

    def test_a_failure_leaves_no_staging_directory(self, tmp_path):
        """
        Otherwise abandoned partial writes accumulate in the store.

        Which eventually fills the disk that caused the first failure.
        """

        def exploding(path):
            """Fail partway through writing."""
            raise OSError("disk full")

        with pytest.raises(OSError):
            _write(tmp_path, write_weights=exploding)
        assert not list((tmp_path / "demo").glob(f"{TEMPORARY_PREFIX}*"))

    def test_an_interruption_is_cleaned_up_too(self, tmp_path):
        """
        ``KeyboardInterrupt`` does not derive from ``Exception``.

        Catching only ``Exception`` would mean a cancelled run is exactly the
        case that litters the store, and cancelling runs is common.
        """

        def interrupting(path):
            """Simulate the user pressing control-C mid-write."""
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            _write(tmp_path, write_weights=interrupting)
        assert not list((tmp_path / "demo").glob(f"{TEMPORARY_PREFIX}*"))

    def test_a_retry_after_a_failure_succeeds(self, tmp_path):
        """
        The practical consequence of cleaning up.

        If the failed attempt left anything behind, the retry would hit the
        overwrite guard and the version would be unusable forever.
        """

        def exploding(path):
            """Fail the first attempt."""
            raise OSError("disk full")

        with pytest.raises(OSError):
            _write(tmp_path, write_weights=exploding)
        assert _write(tmp_path).directory.is_dir()


class TestReading:
    """Everything written comes back unchanged."""

    @pytest.fixture
    def saved(self, tmp_path):
        """
        Provide a written bundle.

        Returns
        -------
        SavedBundle
            The located bundle.
        """
        return _write(tmp_path)

    def test_a_written_bundle_verifies_on_open(self, saved):
        """
        Verification is on by default, and must pass on a fresh write.

        A writer and a verifier that disagreed would make every bundle
        unloadable, so this is the single most important test here.
        """
        assert open_bundle(saved.directory).manifest == saved.manifest

    def test_the_signature_round_trips(self, saved):
        """
        Signature plus spec is what rebuilds the model.

        Without the data build, which is the property that makes a
        six-month-old bundle usable rather than archaeological.
        """
        assert load_signature(saved) == make_model_bundle().signature

    def test_the_lineage_round_trips(self, saved):
        """
        Including the exact split indices.

        Which is what lets a saved model be re-evaluated against the data it
        was actually trained against, even after the split logic changed.
        """
        assert load_lineage(saved) == make_model_bundle().lineage

    def test_the_result_round_trips(self, saved):
        """Metrics and the full training history."""
        assert load_result(saved) == make_model_bundle().result

    def test_the_fitted_state_round_trips(self, saved):
        """
        The state that makes the model's output interpretable.

        Loaded into a type the caller names, because recording a class path
        would mean renaming the class breaks every bundle written before.
        """
        assert load_fitted_state(saved, StandardisingState) == make_model_bundle().state

    def test_verification_can_be_skipped(self, saved):
        """
        Verification is a choice, because it is not free.

        Hashing a multi-gigabyte checkpoint to list bundles in a UI would
        make the UI unusable, so the caller decides.
        """
        (saved.directory / SPEC_FILENAME).write_text("tampered", encoding="utf-8")
        assert open_bundle(saved.directory, verify=False) is not None

    def test_verification_catches_tampering(self, saved):
        """And the default is the safe one."""
        (saved.directory / SPEC_FILENAME).write_text("tampered", encoding="utf-8")
        with pytest.raises(BundleError):
            open_bundle(saved.directory)


class TestReadFailures:
    """Every failure names the file, because a bundle holds five of them."""

    def test_a_missing_manifest_is_reported(self, tmp_path):
        """The most common cause is pointing at the wrong directory."""
        with pytest.raises(BundleError, match=re.escape(MANIFEST_FILENAME)):
            load_manifest(tmp_path)

    def test_a_missing_contract_file_names_it(self, tmp_path):
        """
        "Validation failed" without a path is nearly useless here.

        There are five JSON files, and the reader has to know which.
        """
        saved = _write(tmp_path)
        saved.signature_path.unlink()
        with pytest.raises(BundleError, match=r"signature\.json"):
            load_signature(saved)

    def test_malformed_json_names_the_file_and_the_type(self, tmp_path):
        """
        Both, because the type says what the file was supposed to be.

        A truncated signature and a truncated lineage look identical
        otherwise.
        """
        saved = _write(tmp_path)
        saved.result_path.write_text("{not json", encoding="utf-8")
        with pytest.raises(BundleError, match="TrainingResult"):
            load_result(saved)

    def test_valid_json_of_the_wrong_shape_is_rejected(self, tmp_path):
        """
        Parsing is not the same as validating.

        A file that happens to be JSON but is not a result would otherwise
        load as an empty result and report no metrics at all.
        """
        saved = _write(tmp_path)
        saved.result_path.write_text('{"unexpected": 1}', encoding="utf-8")
        with pytest.raises(BundleError, match="not a valid"):
            load_result(saved)

    def test_a_missing_fitted_state_directory_is_reported(self, tmp_path):
        """
        Distinguished from an empty one, and reported as a version problem.

        A bundle written by an incompatible version is the likely cause, and
        saying so saves the reader a filesystem investigation.
        """
        saved = _write(tmp_path)
        shutil.rmtree(saved.fitted_state_directory)
        with pytest.raises(BundleError, match="fitted state"):
            load_fitted_state(saved, StandardisingState)

    def test_a_manifest_from_a_future_layout_is_refused(self, tmp_path):
        """
        Refused outright rather than loaded partially.

        A partially loaded bundle looks like a working model, which is the
        failure this schema version exists to prevent.
        """
        saved = _write(tmp_path)
        future = saved.manifest.model_copy(
            update={"schema_version": saved.manifest.schema_version + 1}
        )
        (saved.directory / MANIFEST_FILENAME).write_text(future.model_dump_json(), encoding="utf-8")
        with pytest.raises(BundleError, match="schema version"):
            open_bundle(saved.directory)


class TestNoPickling:
    """The format stores parameters, never objects."""

    def test_the_stored_spec_is_readable_json(self, tmp_path):
        """
        Not a pickle, so it survives the model's code being reorganised.

        Pickling embeds the class's import path and source semantics into the
        file, so a refactor breaks every bundle ever written -- and loading
        one executes whatever the file says.
        """
        saved = _write(tmp_path)
        assert saved.spec_path.read_text(encoding="utf-8").lstrip().startswith("{")

    def test_the_manifest_is_readable_without_the_framework(self, tmp_path):
        """
        Plain JSON, so a bundle can be inspected by anything.

        Which matters when the thing inspecting it is a monitoring script
        that has no reason to install a training library.
        """
        saved = _write(tmp_path)
        payload = json.loads((saved.directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert payload["model_name"] == "demo"
        assert Manifest.model_validate(payload) == saved.manifest
