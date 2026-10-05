# `tests/rade_qnet/core/contract`

8 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 32 | 1381 | `455461a7d762debd` |
| 2 | `test_contract_bundle.py` | 291 | 9501 | `e7f0fc8dc09c7503` |
| 3 | `test_contract_data.py` | 275 | 9775 | `73b19d1e89316134` |
| 4 | `test_contract_requirement.py` | 226 | 9536 | `6017b8f7ead178ed` |
| 5 | `test_contract_result.py` | 336 | 12445 | `7a2525781c1cdef6` |
| 6 | `test_contract_signature.py` | 320 | 12534 | `2ac93fb372973aef` |
| 7 | `test_contract_source.py` | 244 | 8360 | `d6c4f0f4a3b93cb1` |
| 8 | `test_contract_state.py` | 209 | 7845 | `85aeb1274d4d54a1` |

---

## 1. `tests/rade_qnet/core/contract/__init__.py`

1381 bytes · SHA-256 `455461a7d762debd`

```python
"""
Tests for ``rade_qnet.core.contract`` -- the typed stage hand-offs.

A contract is only useful if it is enforced, so these tests check that a
malformed payload is rejected at construction rather than surfacing as a shape
error inside a training loop. They also pin the behaviour that makes contracts
portable: a contract must survive being written to disk and read back, because
that is how a job crosses a process boundary and how a bundle is reloaded
months later.

Planned modules
---------------
``test_contract_signature.py``
    ``TensorSpec`` and ``InputSignature`` construction and equality -- the
    basis for rebuilding a model from a bundle without the original data.
    [Phase 1]
``test_contract_data.py``
    ``DataBundle``, ``TensorBatchData`` and ``DataLineage``.
    [Phase 1]
``test_contract_source.py``
    The ``BatchSource`` protocol, checked against a trivial implementation to
    prove the protocol is satisfiable without inheritance.  [Phase 1]
``test_contract_result.py``
    ``FitOutcome``, ``TrainingResult``, ``EvalResult`` and ``Predictions``.
    [Phase 1]
``test_contract_bundle.py``
    ``ModelBundle`` and ``Manifest``.  [Phase 1]
``test_contract_state.py``
    ``FittedState`` save and load round-trips, and target inversion.  [Phase 1]
``test_contract_experience.py``
    ``Transition``, ``Trajectory`` and ``EpisodeStats``.  [Phase 7]
"""
```

---

## 2. `tests/rade_qnet/core/contract/test_contract_bundle.py`

9501 bytes · SHA-256 `e7f0fc8dc09c7503`

```python
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
from src.rade_qnet.core.lifecycle.errors import BundleError
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
```

---

## 3. `tests/rade_qnet/core/contract/test_contract_data.py`

9775 bytes · SHA-256 `73b19d1e89316134`

```python
"""
Tests for the data payloads a build hands to an engine.

Two things here are worth more than the rest. :class:`SplitIndices` rejects
overlapping splits on construction, because an overlap produces an encouraging
validation score and a model that fails in production -- the most expensive
mistake available at this layer. And :class:`DataBundle` is generic over the
payload type, which is what lets one pipeline serve a batched Torch run and a
one-shot XGBoost run without branching on the engine.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from src.rade_qnet.core.contract.data import (
    SPLIT_NAMES,
    DataBundle,
    DataLineage,
    SplitIndices,
    TensorBatchData,
)
from src.rade_qnet.core.lifecycle.errors import ContractError, SpecError
from src.rade_qnet.testkit.fixtures import (
    make_lineage,
    make_signature,
    make_tensor_bundle,
)


def _indices(train, validation=(), test=()):
    """
    Build split indices from plain sequences.

    Parameters
    ----------
    train, validation, test
        Scenario indices.

    Returns
    -------
    SplitIndices
        The constructed splits.
    """
    return SplitIndices(
        train=np.array(train, dtype=np.int64),
        validation=np.array(validation, dtype=np.int64),
        test=np.array(test, dtype=np.int64),
    )


class TestSplitIndices:
    """Disjointness is enforced where it is cheapest to enforce."""

    def test_disjoint_splits_are_accepted(self):
        """The normal case."""
        assert _indices([0, 1, 2], [3, 4], [5]).sizes == {
            "train": 3,
            "validation": 2,
            "test": 1,
        }

    def test_an_empty_training_split_is_rejected(self):
        """There is nothing to fit."""
        with pytest.raises(SpecError, match="training"):
            _indices([])

    def test_empty_validation_and_test_splits_are_permitted(self):
        """
        A final refit on everything is a legitimate configuration.

        Requiring held-out data would make that inexpressible.
        """
        assert _indices([0, 1]).sizes == {"train": 2, "validation": 0, "test": 0}

    @pytest.mark.parametrize(
        ("first", "second", "third"),
        [
            ([0, 1], [1, 2], [3]),
            ([0, 1], [2], [1]),
            ([0], [1, 2], [2]),
        ],
    )
    def test_any_overlapping_pair_is_rejected(self, first, second, third):
        """
        Every pair is checked, not just train against validation.

        A test index that also appears in validation inflates the final
        number just as effectively.
        """
        with pytest.raises(SpecError, match="disjoint"):
            _indices(first, second, third)

    def test_the_overlap_message_names_the_offending_splits(self):
        """
        So the reader knows which boundary moved.

        A message saying only "splits overlap" sends them to read all three.
        """
        with pytest.raises(SpecError, match="train and validation"):
            _indices([0, 1], [1])

    def test_an_unknown_split_name_is_rejected_on_lookup(self):
        """A typo must not return ``None`` and train on nothing."""
        with pytest.raises(ContractError, match="unknown split"):
            _indices([0])["holdout"]

    def test_lineage_form_is_json_encodable(self):
        """
        Indices are stored, not the fractions that produced them.

        Which is what lets a saved bundle be re-evaluated against exactly the
        data it was trained against, even after the split logic changes.
        """
        rendered = _indices([0, 1], [2]).as_lineage()
        assert rendered["train"] == (0, 1)
        assert all(isinstance(value, int) for value in rendered["validation"])

    def test_lineage_form_covers_every_split(self):
        """An absent split records as empty rather than as missing."""
        assert set(_indices([0]).as_lineage()) == set(SPLIT_NAMES)


class TestTensorBatchData:
    """The batched payload, and the static-input field that earns its place."""

    def test_static_inputs_default_to_empty(self):
        """Most models have none."""
        assert TensorBatchData(loader=[]).static == {}

    def test_static_inputs_are_held_once(self):
        """
        The field that retires a concrete defect.

        The previous implementation merged every static tensor into every
        sample and then compared them across the batch during collation to
        confirm something true by construction. Holding them here delivers
        the same tensors to the same place with no comparison at all.
        """
        adjacency = np.eye(3)
        payload = TensorBatchData(loader=[], static={"adjacency": adjacency})
        assert payload.static["adjacency"] is adjacency

    def test_an_unbounded_source_records_no_batch_count(self):
        """
        ``None`` is the honest answer for a stream.

        An interactive source has no epoch length, and inventing one would
        make the loop stop at an arbitrary point.
        """
        assert TensorBatchData(loader=[], n_batches=None).n_batches is None

    def test_the_payload_is_frozen(self):
        """A stage must not mutate another stage's data."""
        with pytest.raises(AttributeError):
            TensorBatchData(loader=[]).n_samples = 5


class TestDataLineage:
    """Lineage is JSON and must stay JSON."""

    def test_it_round_trips_exactly(self):
        """
        Stored in a bundle, so it has to reload identically.

        A lineage that does not round-trip cannot support the claim that a
        bundle records how its data was built.
        """
        lineage = make_lineage()
        assert DataLineage.model_validate_json(lineage.model_dump_json()) == lineage

    def test_notes_default_to_empty(self):
        """Annotations are optional."""
        assert make_lineage().notes == {}

    def test_notes_carry_free_form_annotations(self):
        """
        Where a reward-shaping change or a parity flag becomes visible.

        Without it, two runs that are not comparable look comparable.
        """
        lineage = make_lineage(notes={"reward_shaping": "v2"})
        assert lineage.notes["reward_shaping"] == "v2"

    def test_a_negative_scenario_count_is_rejected(self):
        """Not a quantity that can be negative."""
        with pytest.raises(Exception, match="greater than or equal"):
            DataLineage(
                source_fingerprint="a",
                spec_digest="b",
                split_indices={},
                n_scenarios=-1,
                framework_version="0.1.0",
                created_at=datetime.now(UTC),
            )

    def test_an_absent_entity_axis_is_expressible(self):
        """Not every problem has one, and zero is not the same as none."""
        assert make_lineage(n_entities=None).n_entities is None


class TestDataBundle:
    """The generic container, and the two invariants it enforces."""

    def test_a_bundle_without_training_data_is_rejected(self):
        """Nothing downstream can proceed, so it fails at the boundary."""
        with pytest.raises(ContractError, match="train"):
            DataBundle(
                splits={"test": object()},
                signature=make_signature(),
                state=make_tensor_bundle().state,
                lineage=make_lineage(),
            )

    def test_an_unknown_split_name_is_rejected(self):
        """
        Fixed names, so reports and metric tables agree on what to call them.

        A ``holdout`` key would silently never be evaluated.
        """
        with pytest.raises(ContractError, match="holdout"):
            DataBundle(
                splits={"train": object(), "holdout": object()},
                signature=make_signature(),
                state=make_tensor_bundle().state,
                lineage=make_lineage(),
            )

    def test_split_names_are_returned_in_canonical_order(self):
        """
        Not insertion order, so a report's column order is stable.

        Two runs whose splits were assembled in different orders should
        produce identically ordered tables.
        """
        bundle = make_tensor_bundle(splits=("test", "train", "validation"))
        assert bundle.split_names == ("train", "validation", "test")

    def test_an_absent_split_lists_what_is_available(self):
        """
        The usual cause is a run with no validation fraction.

        Meeting a callback that monitors validation loss, which is exactly
        the case where listing the present splits answers the question.
        """
        bundle = make_tensor_bundle(splits=("train",))
        with pytest.raises(ContractError, match="available splits"):
            bundle.split("validation")

    def test_presence_can_be_tested_without_raising(self):
        """So a caller can branch rather than catch."""
        bundle = make_tensor_bundle(splits=("train",))
        assert bundle.has_split("train")
        assert not bundle.has_split("test")

    def test_the_same_bundle_type_carries_either_payload(self):
        """
        The property that lets one pipeline serve every engine.

        Both of these are ``DataBundle``; only the payload differs, and no
        pipeline stage has to know which.
        """
        assert isinstance(make_tensor_bundle(), DataBundle)

    def test_the_payload_type_is_preserved(self):
        """The generic parameter is real, not decorative."""
        assert isinstance(make_tensor_bundle().split("train"), TensorBatchData)

    def test_the_bundle_is_frozen(self):
        """A build's output must not be edited by a later stage."""
        with pytest.raises(AttributeError):
            make_tensor_bundle().splits = {}
```

---

## 4. `tests/rade_qnet/core/contract/test_contract_requirement.py`

9536 bytes · SHA-256 `6017b8f7ead178ed`

```python
"""
Tests for the input contract: what a model consumes, declared and checked.

The cases below are organised around the two defects this type was
introduced to kill, rather than around its methods. Both were silent --
they produced a trained model and a plausible loss curve -- so the tests
that matter most are the ones asserting that something now *fails*.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rade_qnet.core.contract.requirement import InputRequirement, RequiredInput
from src.rade_qnet.core.contract.signature import InputSignature, TensorSpec
from src.rade_qnet.core.lifecycle.errors import ContractError

WINDOW = TensorSpec(shape=(None, 5, 4), dtype="float32")
FLAT = TensorSpec(shape=(None, 4), dtype="float32")
TARGET = TensorSpec(shape=(None, 1), dtype="float32")


def signature(**dynamic: TensorSpec) -> InputSignature:
    """
    Build a signature with the given dynamic inputs.

    Parameters
    ----------
    **dynamic
        Name to spec.

    Returns
    -------
    InputSignature
        With a scalar target.
    """
    return InputSignature(dynamic=dynamic, target=TARGET)


class TestTheAmbiguousInputDefect:
    """
    A model that reads "one input" given two of them.

    Before this contract existed, the recurrent baseline scanned the batch
    and took the first tensor it found. Two feature blocks meant it trained
    on whichever the dictionary yielded first, so reordering the data build
    changed the model and nothing said so.
    """

    def test_one_unnamed_input_is_satisfied_by_any_single_input(self) -> None:
        """
        The name is the source's choice, so the model must not assume it.

        A model that hard-coded ``features`` would work against every
        fixture in this repository and fail against the first user whose
        block is called something else.
        """
        requirement = InputRequirement(dynamic=(RequiredInput(rank=3),))
        requirement.check(signature(whatever_they_called_it=WINDOW), model="m")

    def test_two_inputs_are_refused_when_one_is_required(self) -> None:
        """
        The headline. This is the run that used to succeed and be wrong.

        The message names both inputs, because the user's next question is
        always "which one did it want?" and the answer is "it cannot tell,
        and neither can you".
        """
        requirement = InputRequirement(dynamic=(RequiredInput(rank=3),))
        with pytest.raises(ContractError, match="exactly one unnamed"):
            requirement.check(signature(prices=WINDOW, volumes=WINDOW), model="m")

    def test_the_failure_names_both_candidates(self) -> None:
        """An error a user cannot act on is barely better than silence."""
        requirement = InputRequirement(dynamic=(RequiredInput(rank=3),))
        with pytest.raises(ContractError) as caught:
            requirement.check(signature(prices=WINDOW, volumes=WINDOW), model="m")
        assert "prices" in str(caught.value)
        assert "volumes" in str(caught.value)


class TestTheWrongRankDefect:
    """
    A tensor of the wrong shape that the library accepts anyway.

    ``torch.atleast_3d`` appends its axis, turning ``(samples, features)``
    into ``(samples, features, 1)``. The recurrence accepts that happily
    and learns nonsense from it.
    """

    def test_an_accepted_rank_passes(self) -> None:
        """Rank 2 is legal for this model, as a window of length one."""
        requirement = InputRequirement(dynamic=(RequiredInput(rank=(3, 2)),))
        requirement.check(signature(features=FLAT), model="m")

    def test_an_unaccepted_rank_is_refused(self) -> None:
        """
        Rank 4 is not quietly reshaped into something plausible.

        Declaring *which* ranks are acceptable is the point: a model that
        handles two of them should say so, rather than leaving a reader to
        infer it from a reshape buried in a forward pass.
        """
        requirement = InputRequirement(dynamic=(RequiredInput(rank=3),))
        with pytest.raises(ContractError, match="rank 2, but rank 3"):
            requirement.check(signature(features=FLAT), model="m")


class TestNamedRequirements:
    """For a model whose inputs are not interchangeable."""

    def test_a_missing_named_input_is_refused(self) -> None:
        """
        The flagship's case: swapping two statics trains a wrong model.

        ``adjacency_values`` and ``target_indices`` are not
        interchangeable, and a build that supplied one for the other would
        produce a decreasing loss and attribute every prediction to the
        wrong instrument -- invisible in every metric a run reports.
        """
        requirement = InputRequirement(dynamic=(RequiredInput(name="pnl_history", rank=3),))
        with pytest.raises(ContractError, match="'pnl_history' is required"):
            requirement.check(signature(something_else=WINDOW), model="m")

    def test_a_named_requirement_does_not_consume_an_unnamed_ones_input(
        self,
    ) -> None:
        """
        Named inputs are matched first, so order cannot change the outcome.

        If the unnamed requirement were matched first it could claim the
        very input a named one asked for, and whether it did would depend
        on dictionary ordering -- which is the defect, reintroduced.
        """
        requirement = InputRequirement(
            dynamic=(
                RequiredInput(name="pnl_history", rank=3),
                RequiredInput(rank=2),
            )
        )
        requirement.check(signature(pnl_history=WINDOW, extras=FLAT), model="m")

    def test_an_unconsumed_input_is_refused_by_default(self) -> None:
        """
        A build producing something the model never reads is a warning sign.

        Either the work is wasted, or -- the dangerous case -- the user
        believes the model is using information it cannot see. The second
        is invisible in every metric, so it is refused rather than logged.
        """
        requirement = InputRequirement(dynamic=(RequiredInput(name="pnl_history", rank=3),))
        with pytest.raises(ContractError, match="does not consume"):
            requirement.check(signature(pnl_history=WINDOW, spare=FLAT), model="m")


class TestOptionalRefinements:
    """Dtype and shape, for the cases where they are genuine requirements."""

    def test_a_wrong_dtype_is_refused_when_declared(self) -> None:
        """An index tensor of floats is a bug, not a conversion."""
        requirement = InputRequirement(dynamic=(RequiredInput(rank=2, dtype="int64"),))
        with pytest.raises(ContractError, match="dtype float32"):
            requirement.check(signature(indices=FLAT), model="m")

    def test_a_declared_shape_constrains_the_axes_it_names(self) -> None:
        """
        Edge endpoints come in pairs; that is a requirement, not a reading.

        The wildcard axis is left free because the edge count is a property
        of the book. Pinning it would duplicate a number the signature
        already carries, and two copies of a number can disagree.
        """
        requirement = InputRequirement(dynamic=(RequiredInput(rank=2, shape=(None, 2)),))
        requirement.check(signature(edges=TensorSpec(shape=(99, 2), dtype="int64")), model="m")
        with pytest.raises(ContractError, match="is required"):
            requirement.check(signature(edges=TensorSpec(shape=(99, 3), dtype="int64")), model="m")

    def test_unconstrained_accepts_anything(self) -> None:
        """
        The honest declaration for a model that flattens what it is given.

        Spelled rather than left unset, so a reader knows the question was
        asked and answered.
        """
        InputRequirement.unconstrained().check(signature(a=WINDOW, b=FLAT), model="ridge")


class TestTheDeclarationItselfIsChecked:
    """A requirement that cannot be satisfied should fail its author."""

    def test_two_unnamed_entries_are_refused(self) -> None:
        """
        They are indistinguishable, so matching would depend on ordering.

        That is precisely the defect this type removes, so permitting it
        here would let a model reintroduce it while appearing to declare a
        contract.
        """
        with pytest.raises(ValidationError, match="more than one unnamed"):
            InputRequirement(dynamic=(RequiredInput(rank=3), RequiredInput(rank=3)))

    def test_a_duplicated_name_is_refused(self) -> None:
        """Two requirements on one input cannot both be the contract."""
        with pytest.raises(ValidationError, match="name an input twice"):
            InputRequirement(
                dynamic=(
                    RequiredInput(name="x", rank=3),
                    RequiredInput(name="x", rank=2),
                )
            )

    def test_a_shape_contradicting_the_rank_is_refused(self) -> None:
        """
        Caught at authoring time, so the model's author hears about it.

        Left to check time, this would surface as a requirement no data
        build can ever satisfy, reported to a user who did nothing wrong.
        """
        with pytest.raises(ValidationError, match="not among the accepted ranks"):
            RequiredInput(rank=3, shape=(None, 2))

    def test_a_rank_below_one_is_refused(self) -> None:
        """Every batched input has at least a batch axis."""
        with pytest.raises(ValidationError, match="minimum rank is 1"):
            RequiredInput(rank=0)
```

---

## 5. `tests/rade_qnet/core/contract/test_contract_result.py`

12445 bytes · SHA-256 `7a2525781c1cdef6`

```python
"""
Tests for the result contracts.

The field doing the most work here is ``in_original_units``. A mean absolute
error of 0.03 is either excellent or meaningless depending on that one
boolean, and the number alone does not say which -- so it travels with the
number rather than being remembered.

The rest of this module is about rejecting results that are wrong in ways that
still render: a ``best_epoch`` that indexes nothing, a ``NaN`` metric, a
prediction array longer than its entity identifiers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from src.rade_qnet.core.contract.result import (
    EpochRecord,
    EvalResult,
    FitOutcome,
    Predictions,
    TrainingResult,
)
from src.rade_qnet.core.lifecycle.errors import ContractError
from src.rade_qnet.testkit.fixtures import make_training_result


def _history(*losses, val=True):
    """
    Build an epoch history from a sequence of training losses.

    Parameters
    ----------
    losses
        One training loss per epoch.
    val
        Whether to record a validation loss alongside each.

    Returns
    -------
    tuple of EpochRecord
        The history.
    """
    return tuple(
        EpochRecord(epoch=index, train_loss=loss, val_loss=loss + 0.1 if val else None)
        for index, loss in enumerate(losses)
    )


class TestEpochRecord:
    """One epoch, recorded in enough detail to diagnose a run."""

    def test_a_minimal_record_is_accepted(self):
        """Training loss is the only thing every engine has."""
        assert EpochRecord(epoch=0, train_loss=1.0).val_loss is None

    def test_a_negative_epoch_index_is_rejected(self):
        """Epochs are zero-based and count upward."""
        with pytest.raises(ValidationError):
            EpochRecord(epoch=-1, train_loss=1.0)

    def test_the_learning_rate_is_recorded(self):
        """
        So a schedule's behaviour is visible in the saved history.

        Otherwise it has to be inferred from the shape of the loss curve,
        which is guesswork.
        """
        assert EpochRecord(epoch=0, train_loss=1.0, learning_rate=1e-3).learning_rate == 1e-3

    def test_a_zero_learning_rate_is_rejected(self):
        """
        A rate of zero means no epoch after this one changes anything.

        Recording it without comment would make a frozen run look like a
        converged one.
        """
        with pytest.raises(ValidationError):
            EpochRecord(epoch=0, train_loss=1.0, learning_rate=0.0)

    def test_negative_wall_time_is_rejected(self):
        """Not a quantity that can be negative."""
        with pytest.raises(ValidationError):
            EpochRecord(epoch=0, train_loss=1.0, seconds=-1.0)


class TestFitOutcome:
    """One type for a gradient loop and a boosting run alike."""

    def test_an_empty_history_is_accepted(self):
        """
        A fit that has not run yet is representable.

        Which is what lets a pipeline construct the outcome before the loop
        rather than assembling it from fragments afterwards.
        """
        assert FitOutcome().n_epochs == 0

    def test_the_best_record_is_resolved_by_epoch_number(self):
        """
        Looked up by ``epoch``, not by position in the tuple.

        An engine that records every other epoch would otherwise return the
        wrong record.
        """
        outcome = FitOutcome(
            history=(
                EpochRecord(epoch=0, train_loss=1.0),
                EpochRecord(epoch=2, train_loss=0.5),
            ),
            best_epoch=2,
        )
        assert outcome.best_record.train_loss == 0.5

    def test_a_best_epoch_absent_from_the_history_is_rejected(self):
        """
        Catches an off-by-one between a callback and the history.

        Which would otherwise surface as the wrong checkpoint being restored
        -- a run that reports epoch 40's metrics with epoch 41's weights.
        """
        with pytest.raises(ValidationError, match="best_epoch"):
            FitOutcome(history=_history(1.0, 0.5), best_epoch=7)

    def test_no_best_epoch_is_permitted(self):
        """A run that monitored nothing has no best epoch."""
        assert FitOutcome(history=_history(1.0)).best_record is None

    def test_restoring_the_best_epoch_is_recorded_not_assumed(self):
        """
        When false, the reported metrics and the saved weights disagree.

        That is a legitimate configuration, but it has to be visible in the
        bundle or the two will be compared as though they described the same
        model.
        """
        assert FitOutcome().restored_best is False

    def test_a_curve_preserves_missing_validation_losses(self):
        """
        ``None`` rather than zero, because they mean different things.

        A gap in a curve and a loss of zero look nothing alike to a reader
        and nothing alike to a plotting function.
        """
        outcome = FitOutcome(history=_history(1.0, 0.5, val=False))
        assert outcome.curve("val_loss") == (None, None)

    def test_a_curve_returns_one_value_per_record(self):
        """The series a report plots."""
        assert FitOutcome(history=_history(1.0, 0.8, 0.5)).curve("train_loss") == (1.0, 0.8, 0.5)

    def test_it_round_trips_through_json(self):
        """Written into a bundle, so it must reload exactly."""
        outcome = FitOutcome(history=_history(1.0, 0.5), best_epoch=1, stopped_early=True)
        assert FitOutcome.model_validate_json(outcome.model_dump_json()) == outcome


class TestEvalResult:
    """Metrics, with the two things that make them interpretable."""

    def test_metrics_are_assumed_to_be_in_original_units(self):
        """
        The safe default is the one that is true for most models.

        A model that transforms its target has to say so, which is the right
        way round: silence should not mean "these numbers are in a space
        nobody can interpret".
        """
        assert EvalResult(split="test", metrics={"mae": 0.1}, n_samples=10).in_original_units

    def test_model_internal_units_are_expressible(self):
        """
        Because sometimes they genuinely are internal.

        Hiding that would be worse than reporting it.
        """
        result = EvalResult(
            split="test", metrics={"mae": 0.1}, n_samples=10, in_original_units=False
        )
        assert not result.in_original_units

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_metric_is_rejected(self, value):
        """
        Almost always a degenerate split or a divide-by-zero in a metric.

        Far cheaper to diagnose here than after it has been written to a
        bundle and compared against other runs.
        """
        with pytest.raises(ValidationError, match="finite"):
            EvalResult(split="test", metrics={"mae": value}, n_samples=10)

    def test_the_offending_metric_is_named(self):
        """So the reader does not have to inspect every metric."""
        with pytest.raises(ValidationError, match="r2"):
            EvalResult(
                split="test",
                metrics={"mae": 0.1, "r2": float("nan")},
                n_samples=10,
            )

    def test_baseline_metrics_travel_with_the_result(self):
        """
        A headline metric without a reference point is not interpretable.

        Carrying the baseline alongside means a report cannot show one
        without the other.
        """
        result = EvalResult(
            split="test",
            metrics={"mae": 0.1},
            n_samples=10,
            baseline_metrics={"mae": 0.5},
        )
        assert result.baseline_metrics["mae"] == 0.5

    def test_an_empty_split_is_representable(self):
        """Zero samples is a fact to report, not an error to raise."""
        assert EvalResult(split="test", metrics={}, n_samples=0).n_samples == 0


class TestTrainingResult:
    """The complete outcome, and its lookups."""

    def test_a_metric_is_retrieved_by_split_and_name(self):
        """The normal access path."""
        result = make_training_result()
        assert isinstance(result.metric("test", "mae"), float)

    def test_an_unevaluated_split_lists_what_was_evaluated(self):
        """
        The usual cause is a run configured without that split.

        Listing the evaluated splits answers the question immediately.
        """
        with pytest.raises(ContractError, match="evaluated splits"):
            make_training_result().metric("holdout", "mae")

    def test_an_unknown_metric_lists_what_is_available(self):
        """
        Metric names vary by task.

        A caller asking for ``accuracy`` on a regression run should see what
        it could have asked for instead.
        """
        with pytest.raises(ContractError, match="available"):
            make_training_result().metric("test", "accuracy")

    def test_the_headline_prefers_the_test_split(self):
        """What a run is normally judged on."""
        result = make_training_result()
        assert result.headline() == result.evaluations["test"].metrics

    def test_the_headline_falls_back_when_there_is_no_test_split(self):
        """
        A run configured without held-out test data still has a headline.

        Returning nothing would make the summary page blank for a legitimate
        configuration.
        """
        result = make_training_result()
        without_test = result.model_copy(
            update={"evaluations": {"validation": result.evaluations["validation"]}}
        )
        assert without_test.headline() == result.evaluations["validation"].metrics

    def test_the_headline_is_empty_when_nothing_was_evaluated(self):
        """An empty mapping rather than an exception, since this is a report path."""
        assert TrainingResult(fit=FitOutcome()).headline() == {}

    def test_it_round_trips_through_json(self):
        """The whole result is written into a bundle."""
        result = make_training_result()
        assert TrainingResult.model_validate_json(result.model_dump_json()) == result


class TestPredictions:
    """Output that can be acted on, which means output that can be attributed."""

    def test_values_alone_are_sufficient(self):
        """Not every problem has entity identifiers."""
        assert Predictions(values=np.zeros(5)).n_predictions == 5

    def test_misaligned_entity_ids_are_rejected(self):
        """
        A prediction attributed to the wrong instrument is worse than none.

        It is actionable and wrong, which is the most expensive combination.
        """
        with pytest.raises(ContractError, match="entity_ids"):
            Predictions(values=np.zeros(5), entity_ids=("EURUSD",))

    def test_misaligned_scenario_indices_are_rejected(self):
        """The same failure along the time axis."""
        with pytest.raises(ContractError, match="scenario_indices"):
            Predictions(values=np.zeros(5), scenario_indices=np.array([0, 1], dtype=np.int64))

    def test_aligned_metadata_is_accepted(self):
        """Both axes, correctly supplied."""
        predictions = Predictions(
            values=np.zeros(2),
            entity_ids=("EURUSD", "USDJPY"),
            scenario_indices=np.array([10, 11], dtype=np.int64),
        )
        assert predictions.n_predictions == 2

    def test_the_originating_bundle_is_recorded(self):
        """
        So a prediction can be traced to the model that made it.

        Without it, a prediction file found six months later is unattributable.
        """
        assert Predictions(values=np.zeros(1), bundle_version="demo/job/v3").bundle_version == (
            "demo/job/v3"
        )

    def test_predictions_are_frozen(self):
        """Output must not be edited in place by a downstream consumer."""
        with pytest.raises(AttributeError):
            Predictions(values=np.zeros(1)).values = np.ones(1)

    def test_a_non_finite_prediction_is_permitted(self):
        """
        Unlike a metric, which is rejected.

        A model that diverged genuinely produced ``NaN``, and hiding that
        would be a lie. A metric computed *from* such output, on the other
        hand, is meaningless and is rejected where it is constructed.
        """
        assert math.isnan(float(Predictions(values=np.array([np.nan])).values[0]))
```

---

## 6. `tests/rade_qnet/core/contract/test_contract_signature.py`

12534 bytes · SHA-256 `2ac93fb372973aef`

```python
"""
Tests for the input signature.

:class:`InputSignature` is the most load-bearing contract in the framework
because three unrelated mechanisms depend on its exact shape: static-input
device placement, lazy parameter materialisation, and rebuilding a model from a
saved bundle. The static/dynamic split in particular is what lets an engine
treat the two differently without inspecting any data -- which is what retires
the per-sample static tensor comparison in the implementation being replaced.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rade_qnet.core.contract.signature import (
    InputSignature,
    PolicySignature,
    SpaceSpec,
    TensorSpec,
)
from src.rade_qnet.core.lifecycle.errors import ContractError, SpecError


@pytest.fixture
def signature():
    """
    Provide a signature with one dynamic input, one static input and a target.

    Returns
    -------
    InputSignature
        A signature exercising both input groups.
    """
    return InputSignature(
        dynamic={"features": TensorSpec(shape=(None, 8), dtype="float32")},
        static={"adjacency": TensorSpec(shape=(12, 12), dtype="float32")},
        target=TensorSpec(shape=(None, 1), dtype="float32"),
    )


class TestTensorSpec:
    """A tensor spec describes a shape without naming a library's dtype."""

    def test_the_dtype_is_a_plain_string(self):
        """
        Library-agnostic by construction.

        ``core`` may not import a training library, and a signature written by
        the Torch engine has to be readable by the XGBoost one.
        """
        assert TensorSpec(shape=(4,), dtype="float32").dtype == "float32"

    def test_a_wildcard_dimension_is_accepted(self):
        """``None`` marks the batch dimension, whose size is not yet known."""
        assert TensorSpec(shape=(None, 8), dtype="float32").rank == 2

    def test_a_zero_dimension_is_rejected(self):
        """
        A zero dimension produces an empty tensor.

        Which trains without error and learns nothing -- far harder to
        diagnose later than a rejection here.
        """
        with pytest.raises(ValidationError, match="positive"):
            TensorSpec(shape=(0, 4), dtype="float32")

    def test_a_negative_dimension_is_rejected(self):
        """Not a valid shape under any interpretation."""
        with pytest.raises(ValidationError, match="positive"):
            TensorSpec(shape=(-1, 4), dtype="float32")

    def test_a_scalar_shape_is_accepted(self):
        """A zero-rank tensor is a legitimate loss or scalar feature."""
        assert TensorSpec(shape=(), dtype="float32").rank == 0


class TestCompatibility:
    """Compatibility is weaker than equality, deliberately."""

    def test_a_wildcard_matches_a_concrete_size(self):
        """
        What lets a declared signature be checked against a real batch.

        The signature says "any batch size"; the batch says 16.
        """
        declared = TensorSpec(shape=(None, 8), dtype="float32")
        actual = TensorSpec(shape=(16, 8), dtype="float32")
        assert declared.is_compatible_with(actual)

    def test_a_mismatched_dtype_is_incompatible(self):
        """float32 and float64 data are not interchangeable."""
        assert not TensorSpec(shape=(16, 8), dtype="float32").is_compatible_with(
            TensorSpec(shape=(16, 8), dtype="float64")
        )

    def test_a_mismatched_concrete_dimension_is_incompatible(self):
        """Eight features are not sixteen features."""
        assert not TensorSpec(shape=(16, 8), dtype="float32").is_compatible_with(
            TensorSpec(shape=(16, 16), dtype="float32")
        )

    def test_a_mismatched_rank_is_incompatible(self):
        """
        Checked before the dimensions are compared.

        Otherwise zipping two shapes of different length would either raise or
        silently compare a prefix.
        """
        assert not TensorSpec(shape=(16, 8), dtype="float32").is_compatible_with(
            TensorSpec(shape=(16, 8, 1), dtype="float32")
        )

    def test_compatibility_is_symmetric_for_wildcards(self):
        """A wildcard on either side matches."""
        wild = TensorSpec(shape=(None, 8), dtype="float32")
        concrete = TensorSpec(shape=(16, 8), dtype="float32")
        assert wild.is_compatible_with(concrete) == concrete.is_compatible_with(wild)


class TestConcreteShape:
    """Resolving wildcards is how a dummy batch is synthesised."""

    def test_wildcards_resolve_to_the_given_batch_size(self):
        """
        The mechanism behind lazy parameter materialisation.

        A model whose shapes are known only after the data build is
        materialised by one forward pass over a synthetic batch, built from
        this.
        """
        assert TensorSpec(shape=(None, 8), dtype="float32").concrete_shape(16) == (16, 8)

    def test_concrete_dimensions_are_unchanged(self):
        """A static input's shape is already fully known."""
        assert TensorSpec(shape=(12, 12), dtype="float32").concrete_shape(16) == (12, 12)

    def test_a_non_positive_batch_size_is_rejected(self):
        """A dummy batch of zero samples materialises nothing."""
        with pytest.raises(SpecError):
            TensorSpec(shape=(None, 8), dtype="float32").concrete_shape(0)


class TestInputSignature:
    """The static/dynamic split is enforced, not merely documented."""

    def test_input_names_cover_both_groups(self, signature):
        """A caller asking what the model consumes gets everything."""
        assert signature.input_names == ("adjacency", "features")

    def test_an_empty_dynamic_set_is_rejected(self):
        """
        A model with only static inputs cannot vary with the data.

        It would produce the same output for every sample, which is not a
        model.
        """
        with pytest.raises(ValidationError, match="dynamic"):
            InputSignature(dynamic={}, target=TensorSpec(shape=(None,), dtype="float32"))

    def test_a_name_in_both_groups_is_rejected(self, signature):
        """
        A duplicated name is unresolvable.

        The engine would have to guess whether to collate it per sample or
        upload it once, and either choice is silently wrong half the time.
        """
        spec = TensorSpec(shape=(None, 8), dtype="float32")
        with pytest.raises(ValidationError, match="both"):
            InputSignature(dynamic={"x": spec}, static={"x": spec}, target=spec)

    def test_static_inputs_default_to_empty(self):
        """Most models have none, and should not have to say so."""
        spec = TensorSpec(shape=(None, 8), dtype="float32")
        assert InputSignature(dynamic={"x": spec}, target=spec).static == {}

    def test_a_spec_can_be_looked_up_from_either_group(self, signature):
        """A caller need not know which group a name is in."""
        assert signature.spec_for("features").dtype == "float32"
        assert signature.spec_for("adjacency").shape == (12, 12)

    def test_an_unknown_name_lists_the_declared_inputs(self, signature):
        """
        The usual cause is a renamed key in a data module.

        Listing what is declared makes that immediately visible.
        """
        with pytest.raises(ContractError, match="adjacency"):
            signature.spec_for("feature")


class TestBatchKeyValidation:
    """Keys are checked; shapes are the engine's business."""

    def test_a_complete_batch_passes(self, signature):
        """The normal case."""
        signature.validate_batch_keys({"features": 1, "target": 2}, where="train")

    def test_a_missing_dynamic_input_is_rejected(self, signature):
        """
        Caught here rather than as a dimension mismatch in a forward pass.

        The message names both sides, which is what makes a renamed key
        obvious instead of mysterious.
        """
        with pytest.raises(ContractError, match="features"):
            signature.validate_batch_keys({"target": 2}, where="train")

    def test_an_undeclared_key_is_rejected(self, signature):
        """
        An extra key means the data module and the model disagree.

        Silently ignoring it would mean a feature the user believes is in use
        is not.
        """
        with pytest.raises(ContractError, match="junk"):
            signature.validate_batch_keys({"features": 1, "junk": 3}, where="train")

    def test_a_static_input_in_the_batch_is_permitted(self, signature):
        """
        Permitted but not required.

        A source may deliver static inputs separately, which is the path that
        avoids collating them per sample.
        """
        signature.validate_batch_keys({"features": 1, "adjacency": 2, "target": 3}, where="train")

    def test_a_batch_without_a_target_is_permitted(self, signature):
        """An inference batch legitimately has no target."""
        signature.validate_batch_keys({"features": 1}, where="infer")

    def test_the_caller_is_named_in_the_error(self, signature):
        """
        So the failure says which stage produced the bad batch.

        A message naming only the key leaves the reader guessing between the
        training loader, the validation loader and the inference path.
        """
        with pytest.raises(ContractError, match="validation loader"):
            signature.validate_batch_keys({}, where="validation loader")

    def test_the_target_key_can_be_renamed(self, signature):
        """A model whose target is named differently is still checkable."""
        signature.validate_batch_keys(
            {"features": 1, "label": 2}, where="train", target_key="label"
        )


class TestSpaceSpec:
    """Observation and action spaces, for the interactive case."""

    def test_a_box_space_is_accepted(self):
        """The continuous case."""
        space = SpaceSpec(kind="box", shape=(4,), dtype="float32", low=-1.0, high=1.0)
        assert space.shape == (4,)

    def test_a_discrete_space_requires_its_size(self):
        """
        A discrete space with no action count is not a space.

        Any default would be arbitrary, and an arbitrary action count produces
        a policy head of the wrong width.
        """
        with pytest.raises(ValidationError, match="'n'"):
            SpaceSpec(kind="discrete")

    def test_an_action_count_on_a_box_space_is_rejected(self):
        """
        ``n`` has no meaning for a continuous space.

        Accepting it would let two contradictory descriptions of one space
        coexist.
        """
        with pytest.raises(ValidationError, match="'n'"):
            SpaceSpec(kind="box", shape=(4,), n=3)

    def test_a_zero_action_count_is_rejected(self):
        """A policy must have something to choose between."""
        with pytest.raises(ValidationError):
            SpaceSpec(kind="discrete", n=0)


class TestDescribeAndRoundTrip:
    """Signatures are rendered for humans and stored in bundles."""

    def test_a_tensor_spec_describes_compactly(self):
        """Used in error messages and reports."""
        assert TensorSpec(shape=(None, 20, 64), dtype="float32").describe() == (
            "float32[?, 20, 64]"
        )

    def test_a_signature_describes_every_input(self, signature):
        """A reader can see the whole interface at a glance."""
        rendered = signature.describe()
        assert "features" in rendered
        assert "adjacency" in rendered
        assert "target" in rendered

    def test_a_signature_round_trips_through_json(self, signature):
        """
        The property that makes a model rebuildable from a bundle.

        The signature is stored as JSON; if it did not reload exactly, a
        six-month-old bundle could not reconstruct its model.
        """
        assert InputSignature.model_validate_json(signature.model_dump_json()) == signature

    def test_a_policy_signature_round_trips(self):
        """The interactive counterpart, stored the same way."""
        policy = PolicySignature(
            observation=SpaceSpec(kind="box", shape=(4,), low=-1.0, high=1.0),
            action=SpaceSpec(kind="discrete", n=3, dtype="int64"),
        )
        assert PolicySignature.model_validate_json(policy.model_dump_json()) == policy

    def test_a_signature_is_frozen(self, signature):
        """A stage must not be able to redeclare the interface mid-run."""
        with pytest.raises(ValidationError):
            signature.target = TensorSpec(shape=(None, 2), dtype="float32")
```

---

## 7. `tests/rade_qnet/core/contract/test_contract_source.py`

8360 bytes · SHA-256 `d6c4f0f4a3b93cb1`

```python
"""
Tests for the batch source protocol.

:class:`BatchSource` is the one abstraction that lets a supervised run and an
interactive run share a training loop. A dataset yields batches until the
epoch ends; an environment rollout yields batches until told to stop. The only
difference the loop sees is ``steps_per_epoch``, where ``None`` means unbounded
-- and that single field is what decides which side drives the loop.

The protocol is structural, so a source need not import the framework to
satisfy it. These tests check that this really is the case, because a protocol
that quietly requires inheritance is just an ABC with extra steps.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.core.contract.source import BatchSource
from src.rade_qnet.testkit.conformance import check_source
from src.rade_qnet.testkit.fixtures import SyntheticTensorSource, make_signature


@pytest.fixture
def source():
    """
    Provide a small compliant source.

    Returns
    -------
    SyntheticTensorSource
        Six samples in batches of two.
    """
    return SyntheticTensorSource(
        features=np.zeros((6, 4)),
        targets=np.zeros(6),
        batch_size=2,
    )


class TestStructuralConformance:
    """Satisfying the protocol must not require importing the framework."""

    def test_a_compliant_source_is_recognised(self, source):
        """The runtime-checkable protocol routes on shape alone."""
        assert isinstance(source, BatchSource)

    def test_an_unrelated_object_is_not_recognised(self):
        """A protocol that matches everything would route nothing."""
        assert not isinstance(object(), BatchSource)

    def test_a_source_defined_without_the_import_is_recognised(self):
        """
        The point of making this structural.

        A user's dataset class, written against no framework type at all,
        plugs in. Requiring a base class would mean every data module in
        every model imports ``core``.
        """

        class Standalone:
            """A source that has never heard of rade_qnet."""

            signature = make_signature()
            static: dict[str, object] = {}
            steps_per_epoch = 1
            n_samples = 2

            def batches(self):
                """Yield one batch."""
                return iter([{"features": np.zeros((2, 4)), "target": np.zeros(2)}])

        assert isinstance(Standalone(), BatchSource)

    def test_a_source_missing_batches_is_not_recognised(self):
        """
        The member that does the work.

        ``isinstance`` is what routes a payload to the batched path, so a
        near-miss has to be rejected rather than routed and then crash.
        """

        class Incomplete:
            """Declares everything but the method that yields data."""

            signature = make_signature()
            static: dict[str, object] = {}
            steps_per_epoch = 1
            n_samples = 2

        assert not isinstance(Incomplete(), BatchSource)


class TestBoundedAndUnbounded:
    """``steps_per_epoch`` is the ML/RL unifier."""

    def test_a_bounded_source_reports_its_length(self, source):
        """
        Six samples in batches of two is three steps.

        The loop uses this for progress reporting and for scheduler steps
        per epoch.
        """
        assert source.steps_per_epoch == 3

    def test_a_partial_final_batch_is_counted(self):
        """
        Seven samples in batches of two is four steps, not three.

        Rounding down would silently drop the tail of every epoch.
        """
        partial = SyntheticTensorSource(
            features=np.zeros((7, 4)), targets=np.zeros(7), batch_size=2
        )
        assert partial.steps_per_epoch == 4

    def test_an_unbounded_source_reports_none(self):
        """
        ``None`` is how an interactive source says "I do not end".

        The loop then takes its stopping condition from the training spec
        rather than from the source, which is exactly the behavioural
        difference between the supervised and interactive cases.
        """
        unbounded = SyntheticTensorSource(
            features=np.zeros((6, 4)),
            targets=np.zeros(6),
            batch_size=2,
            unbounded=True,
        )
        assert unbounded.steps_per_epoch is None

    def test_an_unbounded_source_keeps_yielding(self):
        """
        Demonstrated rather than asserted from the field.

        A source that declares itself unbounded and then stops would hang a
        loop waiting for data that never arrives.
        """
        unbounded = SyntheticTensorSource(
            features=np.zeros((4, 4)),
            targets=np.zeros(4),
            batch_size=2,
            unbounded=True,
        )
        produced = 0
        for _ in unbounded.batches():
            produced += 1
            if produced > 5:
                break
        assert produced > 2


class TestIterationContract:
    """Re-iterability is required, and it is easy to get wrong."""

    def test_a_source_can_be_traversed_more_than_once(self, source):
        """
        A training loop traverses the source once per epoch.

        A bare generator satisfies every other clause of the protocol and
        then yields nothing from epoch two onward -- which looks like a model
        that stopped learning, not like a data bug.
        """
        first = [batch["target"].shape[0] for batch in source.batches()]
        second = [batch["target"].shape[0] for batch in source.batches()]
        assert first == second
        assert first

    def test_each_traversal_returns_a_fresh_iterator(self, source):
        """Two concurrent passes must not consume each other's batches."""
        assert source.batches() is not source.batches()

    def test_every_batch_matches_the_declared_signature(self, source):
        """
        The keys the model will look for are the keys that arrive.

        Checked against the source's own signature, so a renamed feature
        fails here rather than as a shape error inside a forward pass.
        """
        for batch in source.batches():
            source.signature.validate_batch_keys(batch, where="test source")

    def test_the_sample_count_matches_what_is_yielded(self, source):
        """
        A dishonest count silently rescales every per-sample metric.

        Which produces a plausible wrong number rather than an error.
        """
        yielded = sum(batch["target"].shape[0] for batch in source.batches())
        assert yielded == source.n_samples


class TestStaticInputs:
    """Static inputs are declared, not discovered."""

    def test_static_inputs_default_to_empty(self, source):
        """Most sources have none."""
        assert source.static == {}

    def test_declared_static_inputs_are_in_the_signature(self):
        """
        A static input absent from the signature would never be placed.

        The engine uploads what the signature declares; anything else is
        carried and ignored.
        """
        adjacency = np.eye(4)
        with_static = SyntheticTensorSource(
            features=np.zeros((4, 4)),
            targets=np.zeros(4),
            batch_size=2,
            static={"adjacency": adjacency},
        )
        assert "adjacency" in with_static.signature.static

    def test_static_inputs_are_not_repeated_in_every_batch(self):
        """
        The behaviour the static field exists to produce.

        Collating an adjacency matrix per sample is pure waste, and the
        previous implementation then compared the copies to each other.
        """
        with_static = SyntheticTensorSource(
            features=np.zeros((4, 4)),
            targets=np.zeros(4),
            batch_size=2,
            static={"adjacency": np.eye(4)},
        )
        assert all("adjacency" not in batch for batch in with_static.batches())


class TestConformanceAgreement:
    """The shipped checker agrees with these tests."""

    def test_a_compliant_source_passes_the_conformance_suite(self, source):
        """
        What a model author runs instead of writing the tests above.

        If this disagreed with the tests in this file, one of the two would
        be giving authors the wrong answer.
        """
        assert check_source(source).passed
```

---

## 8. `tests/rade_qnet/core/contract/test_contract_state.py`

7845 bytes · SHA-256 `85aeb1274d4d54a1`

```python
"""
Tests for fitted state.

The decisive property is that ``inverse_transform_targets`` is **abstract**. A
model that scales its target and cannot invert the scaling reports a mean
absolute error in standardised space -- a number nobody can act on, and one
that looks perfectly reasonable. Making the method abstract means that failure
has to be chosen rather than reached by omission.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.rade_qnet.core.contract.state import FittedState, IdentityFittedState
from src.rade_qnet.testkit.fixtures import StandardisingState


class TestAbstractInterface:
    """The base demands exactly the three things a state must do."""

    def test_the_base_cannot_be_instantiated(self):
        """An incomplete state is not a usable state."""
        with pytest.raises(TypeError):
            FittedState()

    @pytest.mark.parametrize("method", ["save", "load", "inverse_transform_targets"])
    def test_each_required_method_is_abstract(self, method):
        """
        All three are required, and the inverse especially.

        A default inverse would be an identity, so a model that scales its
        target would silently report metrics in the wrong units.
        """
        assert getattr(FittedState, method).__isabstractmethod__

    def test_a_subclass_missing_the_inverse_cannot_be_instantiated(self):
        """The guarantee, demonstrated."""

        class Incomplete(FittedState):
            """Implements saving but not the inverse."""

            def save(self, directory):
                """Write nothing."""

            @classmethod
            def load(cls, directory):
                """Read nothing."""
                return cls()

        with pytest.raises(TypeError, match="inverse_transform_targets"):
            Incomplete()

    def test_describe_is_not_abstract(self):
        """
        Describing is optional, unlike the other three.

        A state that cannot describe itself produces a less informative
        report, not a wrong one.
        """
        assert not getattr(FittedState.describe, "__isabstractmethod__", False)

    def test_save_receives_a_directory_not_a_file(self):
        """
        Directory-based, because a state is naturally several arrays.

        Forcing a scaler's mean and scale, a graph's three arrays and a
        selected feature list through one pickle reintroduces the fragility
        this class exists to remove.
        """
        assert "directory" in inspect.signature(FittedState.save).parameters


class TestIdentityState:
    """The explicit "this model transforms nothing" state."""

    def test_the_inverse_returns_the_input_unchanged(self):
        """The honest behaviour for an untransformed target."""
        values = np.array([1.0, -2.0, 3.5])
        assert np.array_equal(IdentityFittedState().inverse_transform_targets(values), values)

    def test_it_round_trips_through_save_and_load(self, tmp_path):
        """Every state must survive a bundle write and read."""
        IdentityFittedState().save(tmp_path)
        assert IdentityFittedState.load(tmp_path) == IdentityFittedState()

    def test_saving_writes_a_marker_rather_than_nothing(self, tmp_path):
        """
        An empty directory is indistinguishable from a failed write.

        A marker lets bundle verification confirm the state was saved
        deliberately.
        """
        IdentityFittedState().save(tmp_path)
        assert list(tmp_path.iterdir())

    def test_instances_compare_equal(self):
        """
        All instances are equivalent, so they compare so.

        Needed for a bundle round-trip test to be able to assert equality.
        """
        assert IdentityFittedState() == IdentityFittedState()

    def test_it_is_hashable(self):
        """
        Defining equality without a hash would make it unhashable.

        A state can legitimately end up in a set or as a dictionary key while
        a pipeline is assembling a bundle.
        """
        assert len({IdentityFittedState(), IdentityFittedState()}) == 1

    def test_it_does_not_compare_equal_to_another_state(self):
        """
        An identity state is not a standardiser.

        If it compared equal, a bundle round-trip test could pass while
        having loaded the wrong state entirely.
        """
        assert IdentityFittedState() != StandardisingState(mean=0.0, scale=1.0)

    def test_describe_names_its_type(self):
        """Reported in the run summary, so it must identify itself."""
        assert IdentityFittedState().describe()["type"] == "IdentityFittedState"


class TestStandardisingState:
    """A state with a real inverse, used throughout the test suite."""

    def test_the_inverse_undoes_the_transform(self):
        """
        The property every state must have and this one demonstrates.

        If this round trip did not hold, every metric computed downstream
        would be in the wrong units.
        """
        state = StandardisingState.fit(np.array([1.0, 2.0, 3.0, 4.0]))
        values = np.array([1.0, 2.5, 4.0])
        assert np.allclose(state.inverse_transform_targets(state.transform(values)), values)

    def test_it_fits_only_what_it_is_given(self):
        """
        Fitted on the training split alone.

        The signature takes ``train_targets`` precisely to make fitting on
        everything awkward -- fitting on all scenarios leaks the held-out
        period's distribution into training.
        """
        state = StandardisingState.fit(np.array([0.0, 2.0]))
        assert state.mean == pytest.approx(1.0)

    def test_a_constant_target_does_not_produce_a_zero_scale(self):
        """
        A zero scale would make the transform divide by zero.

        A constant target legitimately occurs in a short window, so the scale
        falls back to one rather than raising.
        """
        assert StandardisingState.fit(np.array([3.0, 3.0, 3.0])).scale == 1.0

    def test_a_non_positive_scale_is_rejected_on_construction(self):
        """Constructed directly, an invalid scale is caught immediately."""
        with pytest.raises(ValueError, match="scale"):
            StandardisingState(mean=0.0, scale=0.0)

    def test_it_round_trips_through_save_and_load(self, tmp_path):
        """
        The statistics survive a bundle write.

        Which is what lets a six-month-old bundle invert its target scaling
        without the original run.
        """
        state = StandardisingState(mean=1.5, scale=2.5)
        state.save(tmp_path)
        assert StandardisingState.load(tmp_path) == state

    def test_describe_reports_the_fitted_statistics(self):
        """
        What was fitted appears in the run summary.

        Often the first place an anomaly -- a scale of 1e-9, say -- becomes
        visible.
        """
        described = StandardisingState(mean=1.5, scale=2.5).describe()
        assert described["mean"] == 1.5
        assert described["scale"] == 2.5

    def test_states_with_different_statistics_differ(self):
        """So a bundle round trip cannot pass by loading the wrong values."""
        assert StandardisingState(mean=1.0, scale=1.0) != StandardisingState(mean=2.0, scale=1.0)


class TestSubclassRecognition:
    """A concrete state is recognised as a FittedState."""

    @pytest.mark.parametrize("state", [IdentityFittedState(), StandardisingState()])
    def test_concrete_states_are_instances_of_the_base(self, state):
        """
        Checked with ``isinstance``, which the conformance suite relies on.

        Unlike the capability protocols, this is nominal: a state must
        inherit, because the framework calls ``load`` as a classmethod on the
        type the caller supplies.
        """
        assert isinstance(state, FittedState)
```

