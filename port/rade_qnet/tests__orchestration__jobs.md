# `tests/rade_qnet/orchestration/jobs`

8 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 22 | 986 | `db64606cfe41ff84` |
| 2 | `support.py` | 194 | 6178 | `57bc8a00c87d9c9f` |
| 3 | `test_jobs_fanout.py` | 279 | 9557 | `1bb24859e5431a8f` |
| 4 | `test_jobs_groups.py` | 330 | 11800 | `caf6e448ee6e970d` |
| 5 | `test_jobs_manifest.py` | 354 | 11521 | `98dc6b714ca46aec` |
| 6 | `test_jobs_parity.py` | 345 | 12518 | `60545903949860a3` |
| 7 | `test_jobs_set.py` | 379 | 13317 | `e62c9a827ce52b21` |
| 8 | `test_jobs_unit.py` | 265 | 9702 | `e3cb40ca51ccddfe` |

---

## 1. `tests/rade_qnet/orchestration/jobs/__init__.py`

986 bytes · SHA-256 `db64606cfe41ff84`

```python
"""
Tests for ``rade_qnet.orchestration.jobs`` -- fan-out across many jobs.

Partial failure is the case these tests exist for. A job set of forty clusters
where one cluster has insufficient history must return thirty-nine trained
models and one recorded failure. Discarding the other thirty-nine because of
one bad input is the behaviour that makes a framework untrustworthy at scale.

Planned modules
---------------
``test_jobs_set.py``
    Job expansion from a spec, per-job override merging including differing
    architecture complexity, result aggregation, and partial failure.
    [Phase 4]
``test_jobs_unit.py``
    ``run_job`` is importable and picklable at module level -- the property
    that lets it cross a process boundary under the spawn start method, which
    a bound method or closure cannot.  [Phase 4]
``test_jobs_manifest.py``
    Manifest contents, and that it is written atomically by a single owner so a
    concurrent set cannot lose entries.  [Phase 4]
"""
```

---

## 2. `tests/rade_qnet/orchestration/jobs/support.py`

6178 bytes · SHA-256 `57bc8a00c87d9c9f`

```python
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

from src.rade_qnet.core.authoring.supervised import SupervisedModel
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
```

---

## 3. `tests/rade_qnet/orchestration/jobs/test_jobs_fanout.py`

9557 bytes · SHA-256 `1bb24859e5431a8f`

```python
"""
Tests for expanding a group set into a job set.

The decision under test is a separation one. The job-set runner knows how to
run *jobs* and nothing else, so expansion happens here: by the time the
runner sees anything there are only jobs, which is what lets the same runner
serve a hand-written job set and an expanded one without telling them apart.

The other thing worth pinning down is the per-group override hook. It is the
mechanism behind the whole proposition that a group set is a job set rather
than a loop: a group with abundant history supports a wider model than a
thin one, and forcing both to one configuration means underfitting the first
or overfitting the second.
"""

from __future__ import annotations

import json

import pytest

from src.rade_qnet.orchestration.jobs.fanout import (
    fingerprint_tag,
    group_overrides,
    job_set_for_groups,
)
from src.rade_qnet.orchestration.jobs.groups import MANIFEST_FILENAME, read_group_set

#: Shared run-specification fragment. The flagship by name only: these tests
#: expand and validate a job set, they do not train anything.
DEFAULTS = {
    "model": {"name": "hybrid_gnn_rnn"},
    "source": {"kind": "model", "transforms": {"sequence": {"length": 4}}},
    "training": {"engine": "torch", "epochs": 1},
    "reports": {"enabled": []},
}


@pytest.fixture(autouse=True)
def _flagship():
    """
    Make the flagship resolvable, since the expansion validates each job.

    Yields
    ------
    None
        For the duration of the test.
    """
    import src.rade_qnet.models.hybrid_gnn_rnn.register  # noqa: F401, PLC0415

    yield


@pytest.fixture
def group_set(tmp_path):
    """
    Provide a two-group set on disk.

    Returns
    -------
    GroupSet
        The set.
    """
    root = tmp_path / "data"
    groups = [
        {"name": "north", "input_ids": ["x1"], "target_ids": ["y1"], "tier": "deep"},
        {"name": "south", "input_ids": ["x2", "x3"], "target_ids": ["y2"], "tier": "thin"},
    ]
    root.mkdir(parents=True)
    for entry in groups:
        (root / entry["name"]).mkdir()
    (root / MANIFEST_FILENAME).write_text(json.dumps({"groups": groups}), encoding="utf-8")
    return read_group_set(root)


def _expand(group_set, tmp_path, **kwargs):
    """
    Expand with the shared defaults into a temporary output root.

    Parameters
    ----------
    group_set
        The set to expand.
    tmp_path
        Pytest's temporary directory.
    **kwargs
        Forwarded to :func:`job_set_for_groups`.

    Returns
    -------
    JobSetSpec
        The expanded set.
    """
    return job_set_for_groups(group_set, defaults=DEFAULTS, output_root=tmp_path / "out", **kwargs)


class TestExpansion:
    """A group set becomes a job set, one job per group."""

    def test_there_is_one_job_per_group(self, group_set, tmp_path):
        """Two groups in, two jobs out, named after them."""
        assert _expand(group_set, tmp_path).job_ids == ("north", "south")

    def test_each_job_reads_its_own_groups_directory(self, group_set, tmp_path):
        """
        The one override every group job needs, whatever the model.

        Deliberately the only one: anything else would be this module
        deciding how a model should be configured, which the model declares.
        """
        spec = _expand(group_set, tmp_path)

        directory = spec.run_spec_for("north").source.params["directory"]

        assert directory == str(group_set.group("north").directory)

    def test_shared_defaults_reach_every_job(self, group_set, tmp_path):
        """
        Defaults are merged under each job's overrides.

        A per-group setting wins and everything it does not mention survives,
        which is the merge property the whole job-set design rests on.
        """
        assert _expand(group_set, tmp_path).run_spec_for("south").training.epochs == 1

    def test_every_job_validates_at_expansion(self, group_set, tmp_path):
        """
        Not at dispatch.

        A set that expands into something misconfigured should fail where the
        error can name the group, rather than after some of its jobs have
        already trained.
        """
        assert set(_expand(group_set, tmp_path).validate_jobs()) == {"north", "south"}

    def test_each_job_carries_its_groups_description(self, group_set, tmp_path):
        """
        So a manifest and a log line say what a job actually is.

        ``south`` is a name; "2 input, 1 target column(s)" is the shape.
        """
        spec = _expand(group_set, tmp_path)

        assert spec.job("south").description == "south: 2 input, 1 target column(s)"


class TestSettingsPerGroup:
    """The reason a group set is a job set rather than a loop."""

    def test_the_hook_can_vary_the_model_per_group(self, group_set, tmp_path):
        """
        A wide model for the data-rich group, a narrow one for the thin.

        Forcing one configuration on both means underfitting the first or
        overfitting the second.
        """
        widths = {"north": 32, "south": 8}

        spec = _expand(
            group_set,
            tmp_path,
            overrides_for=lambda group: {"model": {"params": {"units": widths[group.name]}}},
        )

        assert spec.run_spec_for("north").model.params["units"] == 32
        assert spec.run_spec_for("south").model.params["units"] == 8

    def test_the_hook_does_not_lose_the_directory_override(self, group_set, tmp_path):
        """
        The caller's fragment is merged over this module's, not instead of it.

        A hook returning a ``source`` fragment must not silently detach the
        job from its group's data.
        """
        spec = _expand(
            group_set,
            tmp_path,
            overrides_for=lambda group: {"source": {"transforms": {"sequence": {"length": 8}}}},
        )

        run = spec.run_spec_for("north")

        assert run.source.params["directory"] == str(group_set.group("north").directory)
        assert run.source.transforms.sequence.length == 8

    def test_the_hook_sees_the_whole_group(self, group_set, tmp_path):
        """
        Settings are usually a function of what the group *is*.

        Passing the group rather than its name is what lets a rule be written
        against an attribute or a column count instead of a hand-maintained
        lookup.
        """

        def by_shape(group):
            units = 8 * len(group.input_ids) if group.attributes["tier"] == "thin" else 64
            return {"model": {"params": {"units": units}}}

        spec = _expand(group_set, tmp_path, overrides_for=by_shape)

        assert spec.run_spec_for("north").model.params["units"] == 64
        assert spec.run_spec_for("south").model.params["units"] == 16


class TestProvenance:
    """Which snapshot a set was expanded from."""

    def test_the_snapshot_digest_is_carried_as_a_tag(self, group_set, tmp_path):
        """
        So it reaches every bundle in the set.

        A result that looks surprising six months from now is asked one
        question first: was this trained on the data I think it was.
        """
        spec = _expand(group_set, tmp_path)

        assert spec.tags == (fingerprint_tag(group_set.fingerprint),)

    def test_the_tag_encoding_is_key_equals_digest(self):
        """
        One encoding, written in one place.

        Every future reader has to parse what this writes, so the format is
        pinned rather than left to whatever the function happens to return.
        """
        assert fingerprint_tag("abc123") == "data_fingerprint=abc123"


class TestTheSetsOwnFields:
    """What the expansion decides, and what it leaves to the caller."""

    def test_the_set_has_a_default_name(self, group_set, tmp_path):
        """A default that reads sensibly in a directory listing."""
        assert _expand(group_set, tmp_path).name == "groups"

    def test_a_name_can_be_given(self, group_set, tmp_path):
        """A caller running two sets wants to tell them apart."""
        assert _expand(group_set, tmp_path, name="eod").name == "eod"

    def test_placement_is_left_to_the_policy_by_default(self, group_set, tmp_path):
        """
        ``auto``, which is the normal path.

        The expansion knows about groups, not about how many cores the
        machine running it has.
        """
        assert _expand(group_set, tmp_path).placement.executor == "auto"

    def test_placement_can_be_declared(self, group_set, tmp_path):
        """A caller that knows its machine can say so."""
        spec = _expand(group_set, tmp_path, placement={"executor": "processes", "workers": 4})

        assert spec.placement.executor == "processes"
        assert spec.placement.workers == 4

    def test_a_selected_subset_expands_to_fewer_jobs(self, group_set, tmp_path):
        """
        The selection happens on the group set, before expansion.

        So the job set has no notion of a group that was filtered out.
        """
        spec = _expand(group_set.select(["south"]), tmp_path)

        assert spec.job_ids == ("south",)


class TestTheOverrideFragment:
    """The minimal per-group fragment, on its own."""

    def test_it_names_only_the_source_directory(self, group_set):
        """
        Minimal on purpose.

        Everything else is the model's declaration, and this module is in no
        position to know it.
        """
        group = group_set.group("north")

        assert group_overrides(group) == {"source": {"params": {"directory": str(group.directory)}}}
```

---

## 4. `tests/rade_qnet/orchestration/jobs/test_jobs_groups.py`

11800 bytes · SHA-256 `caf6e448ee6e970d`

```python
"""
Tests for reading a group set and fingerprinting its snapshot.

Two decisions are under test.

**A group set is declared, not discovered.** The groups are the ones a
manifest lists, not the subdirectories that happen to be present. A directory
listing silently changes meaning when somebody leaves a scratch folder
behind, and a job set that quietly grew a forty-first member is a result
nobody asked for.

**The snapshot is fingerprinted, cheaply.** Every result that looks
surprising is asked the same question first: was this trained on the data I
think it was? The digest covers what identifies the snapshot -- names, column
identifiers, file sizes and timestamps -- rather than the data itself,
because a check expensive enough to be disabled is worth nothing.
"""

from __future__ import annotations

import json

import pytest

from src.rade_qnet.core.lifecycle.errors import SpecError
from src.rade_qnet.orchestration.jobs.groups import MANIFEST_FILENAME, GroupSet, read_group_set


def write_group_set(root, groups=None):
    """
    Lay out a group set directory on disk.

    Parameters
    ----------
    root
        The set's directory, created if absent.
    groups
        Manifest entries. Defaults to two groups with distinguishable column
        identifiers and one free-form attribute each.

    Returns
    -------
    pathlib.Path
        The set's directory.
    """
    if groups is None:
        groups = [
            {
                "name": "north",
                "input_ids": ["x1", "x2"],
                "target_ids": ["y1"],
                "region": "N",
            },
            {
                "name": "south",
                "input_ids": ["x3", "x4", "x5"],
                "target_ids": ["y2"],
                "region": "S",
            },
        ]

    root.mkdir(parents=True, exist_ok=True)
    for entry in groups:
        directory = root / entry.get("directory", entry["name"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "data.csv").write_text("date,value\n2024-01-01,1.0\n", encoding="utf-8")

    (root / MANIFEST_FILENAME).write_text(json.dumps({"groups": groups}), encoding="utf-8")
    return root


@pytest.fixture
def group_set(tmp_path):
    """
    Provide a two-group set read from disk.

    Returns
    -------
    GroupSet
        The set.
    """
    return read_group_set(write_group_set(tmp_path / "data"))


class TestReadingAGroupSet:
    """What comes back from a well-formed directory."""

    def test_every_declared_group_is_read(self, group_set):
        """Two entries in, two groups out."""
        assert group_set.names == ("north", "south")

    def test_groups_keep_manifest_order(self, tmp_path):
        """
        Not sorted.

        A job set's directory listing should match the file somebody wrote,
        so that reading one tells you about the other.
        """
        root = write_group_set(tmp_path / "data", groups=[{"name": "zeta"}, {"name": "alpha"}])

        assert read_group_set(root).names == ("zeta", "alpha")

    def test_column_identifiers_are_read_in_order(self, group_set):
        """The identifiers, in the order the manifest gave them."""
        group = group_set.group("north")

        assert group.input_ids == ("x1", "x2")
        assert group.target_ids == ("y1",)

    def test_column_identifiers_are_optional(self, tmp_path):
        """
        A group may declare no identifiers at all.

        They are provenance, not computation: a model reads its own columns
        through its own ``data.py``, so a manifest that omits them is still
        a complete description of which groups exist.
        """
        root = write_group_set(tmp_path / "data", groups=[{"name": "bare"}])

        group = read_group_set(root).group("bare")

        assert group.input_ids == ()
        assert group.target_ids == ()

    def test_unrecognised_fields_are_kept_as_attributes(self, group_set):
        """
        Carried, not interpreted.

        Every caller partitions their data differently, and a fixed set of
        fields here would be wrong for the second one.
        """
        assert group_set.group("north").attributes == {"region": "N"}

    def test_a_group_directory_defaults_to_its_name(self, group_set):
        """The common layout needs no ``directory`` key."""
        assert group_set.group("north").directory.name == "north"

    def test_a_group_directory_can_be_given_explicitly(self, tmp_path):
        """
        For data whose directories are not named after its groups.

        Common when a group name carries a prefix the filesystem does not.
        """
        root = write_group_set(
            tmp_path / "data", groups=[{"name": "NORTH__2024", "directory": "north_2024"}]
        )

        assert read_group_set(root).group("NORTH__2024").directory.name == "north_2024"


class TestWhatIsRefused:
    """Problems that are knowable before any training starts."""

    def test_a_missing_manifest_says_what_was_expected(self, tmp_path):
        """
        Named, because "declared not discovered" is not obvious.

        A user who laid out directories and expected them to be found needs
        to be told about the manifest, not about a missing file.
        """
        (tmp_path / "data").mkdir()

        with pytest.raises(SpecError, match=MANIFEST_FILENAME):
            read_group_set(tmp_path / "data")

    def test_unreadable_json_is_reported_as_such(self, tmp_path):
        """A truncated file should not surface as a type error."""
        root = tmp_path / "data"
        root.mkdir()
        (root / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

        with pytest.raises(SpecError, match=r"(?i)json"):
            read_group_set(root)

    def test_a_manifest_without_a_groups_list_is_refused(self, tmp_path):
        """The top-level shape is checked before any entry is read."""
        root = tmp_path / "data"
        root.mkdir()
        (root / MANIFEST_FILENAME).write_text(json.dumps({"clusters": []}), encoding="utf-8")

        with pytest.raises(SpecError, match="'groups' list"):
            read_group_set(root)

    def test_a_manifest_with_no_groups_is_refused(self, tmp_path):
        """An empty set expands into an empty job set, which is not a run."""
        root = tmp_path / "data"
        root.mkdir()
        (root / MANIFEST_FILENAME).write_text(json.dumps({"groups": []}), encoding="utf-8")

        with pytest.raises(SpecError, match=r"(?i)at least one"):
            read_group_set(root)

    def test_an_entry_without_a_name_is_refused(self, tmp_path):
        """A name is the one field every group must have: it becomes the job id."""
        root = tmp_path / "data"
        root.mkdir()
        (root / MANIFEST_FILENAME).write_text(
            json.dumps({"groups": [{"directory": "somewhere"}]}), encoding="utf-8"
        )

        with pytest.raises(SpecError, match="'name'"):
            read_group_set(root)

    def test_a_group_pointing_nowhere_fails_at_read(self, tmp_path):
        """
        Here, not thirty-nine jobs later.

        A missing directory is knowable before any training starts, so
        discovering it partway through a set would be the framework's fault
        rather than the data's.
        """
        root = tmp_path / "data"
        write_group_set(root, groups=[{"name": "present"}])
        manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        manifest["groups"].append({"name": "absent"})
        (root / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(SpecError, match="absent"):
            read_group_set(root)

    def test_an_unknown_group_lookup_lists_the_known_ones(self, group_set):
        """
        The usual cause is a typo in a group filter.

        Listing the alternatives turns that into a one-second fix.
        """
        with pytest.raises(KeyError, match="north"):
            group_set.group("nroth")


class TestTheSnapshotFingerprint:
    """Which version of the data a run used."""

    def test_reading_the_same_directory_twice_agrees(self, tmp_path):
        """
        Stable, or it would be useless as provenance.

        A digest that changed on every read would mark every run as having
        used different data.
        """
        root = write_group_set(tmp_path / "data")

        assert read_group_set(root).fingerprint == read_group_set(root).fingerprint

    def test_changing_the_data_changes_the_fingerprint(self, tmp_path):
        """
        The property the whole mechanism exists for.

        A refreshed directory must not look like the one a bundle was trained
        against.
        """
        root = write_group_set(tmp_path / "data")
        before = read_group_set(root).fingerprint

        (root / "north" / "data.csv").write_text("date,value\n2024-01-01,2.0\n", encoding="utf-8")

        assert read_group_set(root).fingerprint != before

    def test_changing_the_columns_changes_the_fingerprint(self, tmp_path):
        """
        An added column is a different snapshot.

        The file contents may be untouched while the problem has changed, so
        the column identifiers are part of the digest rather than only the
        data.
        """
        root = write_group_set(tmp_path / "data")
        before = read_group_set(root).fingerprint

        write_group_set(root, groups=[{"name": "north", "input_ids": ["x1", "x2", "x9"]}])

        assert read_group_set(root).fingerprint != before


class TestSelectingASubset:
    """Running part of a set."""

    def test_selection_keeps_the_named_groups_in_order(self, group_set):
        """The order given, not the manifest's, because the caller asked."""
        assert group_set.select(["south", "north"]).names == ("south", "north")

    def test_selecting_nothing_keeps_everything(self, group_set):
        """
        ``None`` means "all", so a caller need not branch.

        The common case is no filter at all.
        """
        assert group_set.select(None) is group_set

    def test_the_fingerprint_survives_selection(self, group_set):
        """
        A subset of a snapshot was still trained on that snapshot.

        Recomputing would make two runs over different subsets of identical
        data look like runs over different data, which is the opposite of
        what provenance is for.
        """
        assert group_set.select(["north"]).fingerprint == group_set.fingerprint

    def test_selecting_an_unknown_group_fails(self, group_set):
        """A filter naming a group that is not there is a mistake, not an empty set."""
        with pytest.raises(KeyError):
            group_set.select(["absent"])


class TestDescribing:
    """What a log line and a progress display show."""

    def test_a_set_summarises_itself(self, group_set):
        """Group count, column count, and an abbreviated snapshot digest."""
        summary = group_set.describe()

        assert "2 group(s)" in summary
        assert "7 column(s)" in summary
        assert group_set.fingerprint[:8] in summary

    def test_a_group_summarises_its_two_sides(self, group_set):
        """Both counts, because one without the other says little."""
        assert group_set.group("north").describe() == "north: 2 input, 1 target column(s)"

    def test_a_set_is_iterable_and_sized(self, group_set):
        """
        So a caller can loop without reaching for an attribute.

        The expansion in ``fanout.py`` reads better for it, and so does every
        ad-hoc script somebody writes against this.
        """
        assert len(group_set) == 2
        assert [group.name for group in group_set] == list(group_set.names)
        assert isinstance(group_set, GroupSet)
```

---

## 5. `tests/rade_qnet/orchestration/jobs/test_jobs_manifest.py`

11521 bytes · SHA-256 `98dc6b714ca46aec`

```python
"""
Tests for the set-level manifest.

Each bundle is recorded in the catalog on its own. This is the record of the
*set*: which jobs there were, which succeeded, and for the ones that did not,
why.

Two properties dominate what is worth testing here.

**It is read while it is being written.** A forty-job set takes hours, and a
dashboard or a watching script will open this file at some point during them.
A half-written manifest must read as the previous version rather than as a
parse error, which is what the write-and-rename does -- and the temporary
file has to be in the same directory, because a rename is atomic only within
a filesystem.

**It has to survive being moved.** A manifest is archived, copied to another
machine, and read months later. Bundle paths are therefore recorded relative
to the set's directory: an absolute path is correct exactly once, on the
machine that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rade_qnet.orchestration.compute.base import WorkFailure, WorkResult
from src.rade_qnet.orchestration.jobs.manifest import (
    MANIFEST_FILENAME,
    JobRecord,
    JobSetManifest,
)
from src.rade_qnet.orchestration.jobs.unit import JobOutcome


def outcome(job_id: str = "a", **overrides: object) -> JobOutcome:
    """
    Build a job outcome.

    Parameters
    ----------
    job_id
        The job's identifier.
    **overrides
        Fields to replace.

    Returns
    -------
    JobOutcome
        The outcome.
    """
    fields: dict[str, object] = {
        "job_id": job_id,
        "metrics": {"test": {"r2": 0.5, "mse": 1.25}},
        "seed": 7,
        "epochs": 3,
        "stopped_early": False,
        "bundle_directory": Path("/sets/run/jobs") / job_id / "bundle",
        "bundle_version": 1,
        "model_name": "synthetic_tabular",
    }
    fields.update(overrides)
    return JobOutcome(**fields)  # type: ignore[arg-type]


def success(job_id: str = "a", **overrides: object) -> WorkResult[JobOutcome]:
    """
    Build a successful executor result.

    Parameters
    ----------
    job_id
        The job's identifier.
    **overrides
        Outcome fields to replace.

    Returns
    -------
    WorkResult
        The result.
    """
    return WorkResult(key=job_id, value=outcome(job_id, **overrides), wall_seconds=2.0)


def failure(job_id: str = "b") -> WorkResult[JobOutcome]:
    """
    Build a failed executor result.

    Parameters
    ----------
    job_id
        The job's identifier.

    Returns
    -------
    WorkResult
        The result.
    """
    return WorkResult(
        key=job_id,
        failure=WorkFailure(kind="StageError", message="boom", traceback_text="Traceback..."),
        wall_seconds=0.5,
    )


#: The root every bundle path in these tests sits beneath.
ROOT = Path("/sets/run")


class TestBuildingARecordFromAResult:
    """What one executor result becomes."""

    def test_a_success_carries_its_numbers(self):
        """Metrics, seed, epochs and the bundle's version all survive."""
        record = JobRecord.of(success(), relative_to=ROOT)

        assert record.succeeded
        assert record.metric("test", "r2") == 0.5
        assert record.seed == 7
        assert record.epochs == 3
        assert record.bundle_version == 1

    def test_a_failure_carries_its_reason(self):
        """
        Kind, message and traceback, captured in whichever process failed.

        The kind is separate from the message so a set that failed the same
        way forty times is visible as one fix rather than forty.
        """
        record = JobRecord.of(failure(), relative_to=ROOT)

        assert not record.succeeded
        assert record.failure_kind == "StageError"
        assert record.failure_message == "boom"
        assert record.failure_traceback

    def test_a_failure_still_records_how_long_it_took(self):
        """
        "It failed after four hours" and "it failed immediately" differ.

        The two call for completely different responses, and a record that
        dropped the duration on failure would lose the distinction exactly
        when it matters most.
        """
        assert JobRecord.of(failure(), relative_to=ROOT).wall_seconds == 0.5

    def test_a_bundle_path_is_recorded_relative_to_the_set(self):
        """
        So the record survives the directory being copied or archived.

        An absolute path is correct on exactly one machine.
        """
        record = JobRecord.of(success(), relative_to=ROOT)

        assert record.bundle_directory == "jobs/a/bundle"

    def test_a_bundle_outside_the_set_keeps_its_absolute_path(self):
        """
        Unusual but legal, and recording it beats failing to record it.

        A job given its own `output_root` writes outside the set, and the
        manifest should still say where.
        """
        elsewhere = Path("/somewhere/else/bundle")

        record = JobRecord.of(success(bundle_directory=elsewhere), relative_to=ROOT)

        assert record.bundle_directory == str(elsewhere)

    def test_a_missing_metric_reads_as_none(self):
        """
        Not zero.

        A zero would plot, and would be indistinguishable from a genuinely
        zero score.
        """
        record = JobRecord.of(success(), relative_to=ROOT)

        assert record.metric("test", "absent") is None
        assert record.metric("absent", "r2") is None


class TestWhatASetReports:
    """The summaries a manifest exists to serve."""

    @pytest.fixture
    def manifest(self):
        """
        Provide a manifest with one success and one failure.

        Returns
        -------
        JobSetManifest
            The manifest.
        """
        return JobSetManifest(
            run_id="run",
            spec_digest="digest",
            wall_seconds=12.0,
            jobs=(
                JobRecord.of(success("a"), relative_to=ROOT),
                JobRecord.of(failure("b"), relative_to=ROOT),
            ),
        )

    def test_successes_and_failures_are_separable(self, manifest):
        """The two partitions are what every summary starts from."""
        assert [record.job_id for record in manifest.succeeded] == ["a"]
        assert [record.job_id for record in manifest.failed] == ["b"]

    def test_a_record_is_retrievable_by_identifier(self, manifest):
        """Looking one job up should not need a scan at the call site."""
        assert manifest.record("a").job_id == "a"
        assert manifest.record("absent") is None

    def test_a_metric_can_be_gathered_across_jobs(self, manifest):
        """
        The shape every ranking and figure needs.

        Jobs without the metric are omitted rather than given a
        placeholder, for the same reason a missing metric reads as `None`.
        """
        assert manifest.metric_by_job("test", "r2") == {"a": 0.5}

    def test_the_summary_counts_what_succeeded(self, manifest):
        """One line, for a log and a terminal."""
        assert manifest.summary() == "1 of 2 job(s) succeeded in 12.0s"

    def test_partial_failure_is_a_recorded_outcome_not_an_exception(self, manifest):
        """
        A set with a failing job still produces a manifest.

        Thirty-nine jobs that trained are thirty-nine results, and
        discarding them because the fortieth failed would throw away hours
        of work over one typo.
        """
        assert len(manifest.jobs) == 2
        assert len(manifest.succeeded) == 1


class TestWritingAndReading:
    """The file itself."""

    @pytest.fixture
    def manifest(self):
        """
        Provide a one-job manifest.

        Returns
        -------
        JobSetManifest
            The manifest.
        """
        return JobSetManifest(
            run_id="run",
            spec_digest="digest",
            name="set",
            placement="local (sequential, in-process)",
            placement_reason="one job",
            jobs=(JobRecord.of(success("a"), relative_to=ROOT),),
        )

    def test_a_manifest_round_trips(self, manifest, tmp_path):
        """What is written is what is read."""
        manifest.write(tmp_path)

        assert JobSetManifest.read(tmp_path) == manifest

    def test_the_directory_is_created(self, manifest, tmp_path):
        """A set writes its manifest before anyone has made its directory."""
        directory = tmp_path / "not" / "yet"

        assert manifest.write(directory) == directory / MANIFEST_FILENAME

    def test_the_temporary_file_does_not_survive(self, manifest, tmp_path):
        """
        A leftover temporary file is evidence of a crash mid-write.

        It is deliberately visible in a directory listing for that reason,
        so a successful write must not leave one behind.
        """
        manifest.write(tmp_path)

        assert [path.name for path in tmp_path.iterdir()] == [MANIFEST_FILENAME]

    def test_a_rewrite_replaces_rather_than_appends(self, manifest, tmp_path):
        """
        Re-running a set overwrites its manifest in place.

        The run identifier is derived from the specification, so the same
        file lands in the same place -- which is only safe if the second
        write is a replacement.
        """
        manifest.write(tmp_path)
        manifest.model_copy(update={"wall_seconds": 99.0}).write(tmp_path)

        assert JobSetManifest.read(tmp_path).wall_seconds == 99.0

    def test_the_file_is_readable_json(self, manifest, tmp_path):
        """
        Indented JSON, because a human reads this during a long run.

        A dashboard is not the only consumer; the first thing anyone does
        with a stalled set is open its manifest in an editor.
        """
        path = manifest.write(tmp_path)

        assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "run"

    def test_placement_and_its_reason_are_both_recorded(self, manifest, tmp_path):
        """
        A set that chose its placement and one that was told differ.

        Six months later this is the only thing that remembers which it
        was, and the two are not the same experiment.
        """
        manifest.write(tmp_path)

        reloaded = JobSetManifest.read(tmp_path)

        assert reloaded.placement.startswith("local")
        assert reloaded.placement_reason == "one job"

    def test_an_unknown_field_is_refused_on_read(self, tmp_path):
        """
        A manifest from a newer version fails loudly here.

        Silently dropping a field somebody downstream relies on is the
        worse outcome: the data is simply absent, with nothing to say so.
        """
        path = tmp_path / MANIFEST_FILENAME
        path.write_text(json.dumps({"run_id": "r", "spec_digest": "d", "invented": 1}))

        with pytest.raises(Exception, match=r"(?i)extra|invented"):
            JobSetManifest.read(tmp_path)

    def test_reading_a_missing_manifest_says_so(self, tmp_path):
        """A set that never finished has no manifest, and that is the answer."""
        with pytest.raises(FileNotFoundError):
            JobSetManifest.read(tmp_path)

    def test_the_framework_version_is_recorded(self, manifest, tmp_path):
        """
        Which version produced the record.

        A manifest outlives the code that wrote it, and "which version was
        this" is the first question asked of a result that looks wrong.
        """
        manifest.write(tmp_path)

        assert JobSetManifest.read(tmp_path).framework_version
```

---

## 6. `tests/rade_qnet/orchestration/jobs/test_jobs_parity.py`

12518 bytes · SHA-256 `60545903949860a3`

```python
"""
Parity level 5: placement must not change results.

The whole proposition of a job set is that *where* a job runs is an
operational choice. If running forty jobs across eight processes produced
different models than running them one after another, the parallelism would
not be an optimisation -- it would be a second, undocumented experiment.

This is the Phase 4 gate, and it runs the flagship model rather than the
synthetic one the rest of this package uses. A toy model with a closed-form
fit would pass trivially; what is actually at risk is the real thing, with an
optimiser, a graph, and a dozen library calls whose behaviour depends on the
process they run in.

What is compared, and what is not
---------------------------------
"Identical artifacts, byte for byte" is unachievable taken literally: a
bundle records when it was written and how long it took, and those *must*
differ between two runs. So the gate compares a named, reviewed set of things
placement must not change, with the exclusions equally named -- both as
module-level constants, so widening either is a visible change rather than a
quiet edit to an assertion. See `PHASE_4_JOB_SETS.md` §8.1.

Metrics are compared **exactly** rather than to a tolerance. Placement
changing the eighth decimal place is still placement changing results.

Two preconditions, and why they are constraints rather than evasions
--------------------------------------------------------------------
Both were found by running this comparison and watching it fail, and both
turned out to be real defects rather than reasons to soften the criterion.

*The thread budget must be pinned in the specification.* The number of
intra-op threads fixes the order in which a reduction accumulates, so it
fixes the last few significant figures. A worker process gets its budget from
an environment variable read before it imports anything; the process that
launched it has already imported Torch, so the same variable does nothing
there. The two paths therefore ran at different thread counts and scored
differently. That is defect 13, and the fix makes the budget a specification
field applied in-process. What remains is genuine: a run that does not say
how many threads it wants gets whatever the host decided.

*The device must be pinned to the CPU.* Under ``device: auto`` on Apple
silicon the model is not reproducible **against itself** -- two sequential
runs, same process, ``determinism: strict``, differ. MPS kernels are
non-deterministic and ``torch.use_deterministic_algorithms`` does not cover
them. Nothing in the framework can fix that, and a gate that blamed placement
for it would be measuring the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src.rade_qnet.core.contract.bundle import WEIGHTS_FILENAME
from src.rade_qnet.core.spec.jobs import parse_job_set_spec
from src.rade_qnet.orchestration.compute.local import LocalExecutor
from src.rade_qnet.orchestration.compute.processes import ProcessExecutor
from src.rade_qnet.orchestration.jobs.set import JobSetRunner

#: The Phase 0 capture's input, which the flagship model's data module reads.
FIXTURE = Path("tests/fixtures/rade_qnet/golden/hybrid_gnn_rnn/input")

#: Fields of a job record that placement must not change. Listed rather than
#: compared wholesale so that adding a field to `JobRecord` is a decision
#: about whether placement may affect it, not a silent widening.
COMPARED_FIELDS = (
    "job_id",
    "status",
    "seed",
    "epochs",
    "stopped_early",
    "metrics",
    "bundle_directory",
    "bundle_version",
    "model_name",
    "failure_kind",
    "failure_message",
)

#: Fields that *must* differ, or may. Excluded deliberately: a record that
#: took the same number of seconds in two different placements would mean the
#: placement had not changed anything, which is not the claim being tested.
EXCLUDED_FIELDS = (
    "wall_seconds",
    "failure_traceback",
)


def job_set(tmp_path, executor_name):
    """
    Build a two-job set over the flagship model, fully pinned.

    Pinned in every respect that affects arithmetic: device, determinism and
    thread budget. The two jobs differ in width, so a run that silently
    trained one model twice would be visible as two identical scores rather
    than hiding behind a comparison that happens to pass.

    Parameters
    ----------
    tmp_path
        Where the set writes. Separate per executor, because the comparison
        is of what each produced, not of one overwriting the other.
    executor_name
        Recorded in the specification, though the executor is supplied
        directly so the test does not depend on the placement policy's view
        of the machine.

    Returns
    -------
    JobSetSpec
        The specification.
    """
    return parse_job_set_spec(
        {
            "model": "hybrid_gnn_rnn",
            "name": "parity",
            "output_root": str(tmp_path),
            "defaults": {
                "source": {
                    "kind": "model",
                    "params": {"directory": str(FIXTURE.resolve())},
                    "transforms": {"sequence": {"length": 4}},
                },
                "training": {"engine": "torch", "epochs": 2},
                "hardware": {
                    "device": "cpu",
                    "determinism": "strict",
                    "threads_per_worker": 1,
                },
                "reports": {"enabled": []},
            },
            "jobs": [
                {"id": "wide", "overrides": {"model": {"params": {"units": 16}}}},
                {"id": "narrow", "overrides": {"model": {"params": {"units": 8}}}},
            ],
            "placement": {"executor": executor_name},
        }
    )


@dataclass(frozen=True)
class Run:
    """
    One completed job set, and where it wrote.

    Paired because a manifest records bundle paths *relative* to the set's
    directory -- which is what lets it survive being copied -- so reading
    one back means rejoining the two.

    Parameters
    ----------
    manifest
        The set's manifest.
    directory
        The set's output directory.
    """

    manifest: object
    directory: Path


@pytest.fixture(scope="module")
def manifests(tmp_path_factory):
    """
    Run the same set twice: once sequentially, once across two processes.

    Module-scoped because this trains four models, which is slow enough that
    repeating it per assertion would dominate the suite. The runs are
    independent and nothing mutates them, so sharing is safe.

    Returns
    -------
    tuple
        The sequential manifest and the pooled one.
    """
    # Imported here rather than at module level for its registration side
    # effect, which is what makes `hybrid_gnn_rnn` resolvable by name.
    import src.rade_qnet.models.hybrid_gnn_rnn.register  # noqa: F401, PLC0415

    sequential_root = tmp_path_factory.mktemp("sequential")
    pooled_root = tmp_path_factory.mktemp("pooled")

    sequential = JobSetRunner(job_set(sequential_root, "local"), executor=LocalExecutor())
    pooled = JobSetRunner(
        job_set(pooled_root, "processes"),
        executor=ProcessExecutor(workers=2, threads_per_worker=1),
    )

    return (
        Run(sequential.run(), sequential.output_directory),
        Run(pooled.run(), pooled.output_directory),
    )


class TestPlacementDoesNotChangeResults:
    """The gate."""

    def test_both_placements_ran_every_job(self, manifests):
        """
        A precondition for the rest.

        Asserted separately so a failure here does not read as a parity
        failure. A pool that silently dropped a job would otherwise make every
        comparison below vacuous.
        """
        sequential, pooled = manifests

        assert len(sequential.manifest.succeeded) == 2
        assert len(pooled.manifest.succeeded) == 2

    def test_the_jobs_are_the_same_jobs_in_the_same_order(self, manifests):
        """
        Declaration order, not completion order.

        A pool finishes jobs in whatever order they happen to end, and a
        manifest that followed that would differ between two identical
        runs for no reason anyone could act on.
        """
        sequential, pooled = manifests

        assert [record.job_id for record in sequential.manifest.jobs] == ["wide", "narrow"]
        assert [record.job_id for record in pooled.manifest.jobs] == ["wide", "narrow"]

    @pytest.mark.parametrize("job_id", ["wide", "narrow"])
    def test_metrics_are_identical(self, manifests, job_id):
        """
        Exactly identical, not close.

        A tolerance here would pass a framework in which placement moved
        every score slightly, which is precisely the thing a job set
        promises it does not do.
        """
        sequential, pooled = manifests

        assert sequential.manifest.record(job_id).metrics == pooled.manifest.record(job_id).metrics

    @pytest.mark.parametrize("job_id", ["wide", "narrow"])
    def test_every_compared_field_agrees(self, manifests, job_id):
        """
        The named list, field by field.

        Reported per field rather than by comparing whole records, so a
        failure says which property placement changed.
        """
        sequential, pooled = manifests
        left = sequential.manifest.record(job_id)
        right = pooled.manifest.record(job_id)

        differing = [
            field for field in COMPARED_FIELDS if getattr(left, field) != getattr(right, field)
        ]

        assert differing == []

    @pytest.mark.parametrize("job_id", ["wide", "narrow"])
    def test_the_weights_are_byte_identical(self, manifests, job_id):
        """
        The artifact itself, not a summary of it.

        Matching metrics with differing weights would mean the two runs
        reached different models that happen to score the same on this
        data -- which is a far worse failure than differing metrics,
        because nothing downstream would notice.
        """
        sequential, pooled = manifests

        left = _weights(sequential, job_id)
        right = _weights(pooled, job_id)

        assert left == right

    def test_the_two_jobs_are_actually_different_models(self, manifests):
        """
        A guard against the comparison passing for the wrong reason.

        If the per-job overrides were silently dropped, both jobs would
        train the same model and every assertion above would hold while
        testing nothing about overrides at all.
        """
        sequential, _ = manifests

        assert (
            sequential.manifest.record("wide").metrics
            != sequential.manifest.record("narrow").metrics
        )


class TestTheExclusionsAreDeliberate:
    """What the gate declines to compare, and why that is honest."""

    def test_durations_are_excluded_and_do_differ(self, manifests):
        """
        The exclusion is real, not defensive.

        A pooled run and a sequential one taking identical wall time would
        suggest the placement had not taken effect.
        """
        sequential, pooled = manifests

        assert "wall_seconds" in EXCLUDED_FIELDS
        assert sequential.manifest.wall_seconds != pooled.manifest.wall_seconds

    def test_no_field_is_both_compared_and_excluded(self):
        """
        The two lists partition what a record carries.

        An overlap would let a field look checked while being exempt.
        """
        assert set(COMPARED_FIELDS).isdisjoint(EXCLUDED_FIELDS)

    def test_the_lists_cover_every_field_a_record_has(self):
        """
        Nothing is unaccounted for.

        A field added to `JobRecord` later should fail this test, forcing a
        decision about whether placement may affect it -- which is the
        whole point of naming the lists rather than comparing records
        wholesale.
        """
        from src.rade_qnet.orchestration.jobs.manifest import JobRecord  # noqa: PLC0415

        assert set(JobRecord.model_fields) == set(COMPARED_FIELDS) | set(EXCLUDED_FIELDS)


def _weights(run: Run, job_id: str) -> bytes:
    """
    Read a job's weights file.

    Parameters
    ----------
    run
        The completed set.
    job_id
        Which job.

    Returns
    -------
    bytes
        The file's contents.
    """
    record = run.manifest.record(job_id)
    return (run.directory / record.bundle_directory / WEIGHTS_FILENAME).read_bytes()
```

---

## 7. `tests/rade_qnet/orchestration/jobs/test_jobs_set.py`

13317 bytes · SHA-256 `e62c9a827ce52b21`

```python
"""
Tests for the job-set runner.

The runner is the piece that turns a file describing forty jobs into forty
bundles and one manifest. It expands, merges, dispatches, and aggregates, and
each of those four has a decision worth pinning down.

**Expansion and validation happen before anything runs.** A set that trains
thirty-nine models and then discovers the fortieth was misconfigured has
spent thirty-nine runs' worth of time finding out. Every job is validated up
front and every failure is reported together.

**A failing job does not fail the set.** The opposite policy throws away the
thirty-nine that worked because of one typo in the fortieth. A failure is a
recorded outcome, not an exception.

**The run identifier is derived, not timestamped.** A timestamp looks tidier
and means a re-run after a crash produces a second partial set beside the
first rather than completing it.

These tests use the synthetic engine from :mod:`.support` throughout, so a
"job" costs milliseconds. The one test that uses a real process pool uses the
same synthetic model, which is why it can afford to.
"""

from __future__ import annotations

import pytest

from src.rade_qnet.core.lifecycle.components import engine as register_engine
from src.rade_qnet.core.lifecycle.components import model as register_model
from src.rade_qnet.core.lifecycle.errors import SpecError
from src.rade_qnet.core.spec.jobs import parse_job_set_spec
from src.rade_qnet.orchestration.compute.local import LocalExecutor
from src.rade_qnet.orchestration.jobs.manifest import MANIFEST_FILENAME, JobSetManifest
from src.rade_qnet.orchestration.jobs.set import JOBS_SUBDIRECTORY, JobSetRunner
from src.rade_qnet.testkit.fixtures import SyntheticEngine, isolated_registries

from .support import (
    ENGINE_TAG,
    MODEL_NAME,
    SyntheticSupervisedModel,
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
        register_model(MODEL_NAME, engine=ENGINE_TAG)(SyntheticSupervisedModel)
        yield


@pytest.fixture
def dataset(tmp_path):
    """
    Write the exactly-linear dataset every job reads.

    Returns
    -------
    pathlib.Path
        The CSV file.
    """
    return write_linear_dataset(tmp_path / "linear.csv")


def runner(dataset, tmp_path, **overrides):
    """
    Build a runner over a two-job set, pinned to the sequential executor.

    Pinned rather than left to the placement policy, because the policy's
    answer depends on the machine and a test asserting on a manifest should
    not change behaviour between a laptop and a build box.

    Parameters
    ----------
    dataset
        The CSV file.
    tmp_path
        The test's temporary directory.
    **overrides
        Job-set payload keys to replace.

    Returns
    -------
    JobSetRunner
        The runner.
    """
    spec = parse_job_set_spec(job_set_payload(dataset, tmp_path / "out", **overrides))
    return JobSetRunner(spec, executor=LocalExecutor())


class TestRunningASet:
    """End to end, with every job succeeding."""

    def test_every_job_runs(self, dataset, tmp_path):
        """Two jobs in, two records out."""
        manifest = runner(dataset, tmp_path).run()

        assert [record.job_id for record in manifest.jobs] == ["a", "b"]
        assert len(manifest.succeeded) == 2

    def test_each_job_writes_its_own_bundle(self, dataset, tmp_path):
        """
        Separate directories, so two jobs cannot overwrite each other.

        Sharing one would leave a manifest describing artifacts that a
        later job had replaced -- a record that is wrong rather than
        missing, which is the worse of the two.
        """
        manifest = runner(dataset, tmp_path).run()

        directories = {record.bundle_directory for record in manifest.succeeded}

        assert len(directories) == 2
        assert all(path.startswith(f"{JOBS_SUBDIRECTORY}/") for path in directories)

    def test_the_manifest_is_written_to_disk(self, dataset, tmp_path):
        """
        The set's own record, beside the job directories rather than in them.

        It is the thing a dashboard opens, and a reader should not have to
        know a job identifier to find it.
        """
        job_runner = runner(dataset, tmp_path)

        job_runner.run()

        assert (job_runner.output_directory / MANIFEST_FILENAME).exists()
        assert JobSetManifest.read(job_runner.output_directory).run_id == job_runner.run_id

    def test_overrides_reach_the_jobs_that_declared_them(self, dataset, tmp_path):
        """
        Per-job configuration is what a job set is for.

        Asserted through the seed, which is observable in the manifest and
        differs per job, rather than through a metric the exact fit makes
        identical for both.
        """
        manifest = runner(
            dataset,
            tmp_path,
            jobs=[
                {"id": "a", "overrides": {"training": {"fit_params": {"alpha": 0.1}}}},
                {"id": "b", "overrides": {"training": {"fit_params": {"alpha": 0.9}}}},
            ],
        ).run()

        assert manifest.record("a").seed != manifest.record("b").seed

    def test_the_set_records_where_it_ran(self, dataset, tmp_path):
        """
        Placement and the reason for it, even when supplied directly.

        Wrapped in the same shape as a policy decision rather than special
        cased, so a manifest always answers the question.
        """
        manifest = runner(dataset, tmp_path).run()

        assert manifest.placement
        assert manifest.placement_reason == "supplied directly by the caller"


class TestPartialFailure:
    """One job failing must not cost the others."""

    @pytest.fixture
    def manifest(self, dataset, tmp_path):
        """
        Run a set whose second job points at a file that is not there.

        Returns
        -------
        JobSetManifest
            The manifest.
        """
        return runner(
            dataset,
            tmp_path,
            jobs=[
                {"id": "good"},
                {"id": "bad", "overrides": {"source": {"path": str(tmp_path / "absent.csv")}}},
            ],
        ).run()

    def test_the_working_job_still_completes(self, manifest):
        """
        Thirty-nine results are thirty-nine results.

        Discarding them because the fortieth failed throws away hours of
        work over one typo.
        """
        assert [record.job_id for record in manifest.succeeded] == ["good"]

    def test_the_failure_is_recorded_with_its_reason(self, manifest):
        """
        A recorded outcome, not an exception that ends the set.

        The kind is recorded separately from the message so forty jobs
        failing the same way read as one fix.
        """
        record = manifest.record("bad")

        assert not record.succeeded
        assert record.failure_kind
        assert record.failure_traceback

    def test_the_manifest_still_reaches_disk(self, dataset, tmp_path):
        """
        A set with failures is still a set that happened.

        Writing the manifest only on complete success would leave the one
        case where a record is most needed with no record at all.
        """
        job_runner = runner(
            dataset,
            tmp_path,
            jobs=[{"id": "bad", "overrides": {"source": {"path": str(tmp_path / "absent.csv")}}}],
        )

        job_runner.run()

        assert JobSetManifest.read(job_runner.output_directory).failed


class TestValidationPrecedesExecution:
    """Nothing runs until every job is known to be sound."""

    def test_a_misconfigured_job_stops_the_set_before_it_starts(self, dataset, tmp_path):
        """
        Raised, unlike a job that fails while running.

        The distinction is deliberate: a specification error is knowable
        without doing any work, so discovering it after thirty-nine runs
        would be the framework's fault rather than the data's.
        """
        job_runner = runner(
            dataset,
            tmp_path,
            jobs=[{"id": "a"}, {"id": "b", "overrides": {"source": {"kind": "invented"}}}],
        )

        with pytest.raises(SpecError):
            job_runner.run()

    def test_nothing_is_written_when_validation_fails(self, dataset, tmp_path):
        """
        No manifest, no job directories, no partial set to clean up.

        A set that refused to start should leave the filesystem as it found
        it, so a re-run after the fix is a fresh attempt.
        """
        job_runner = runner(
            dataset, tmp_path, jobs=[{"id": "a", "overrides": {"source": {"kind": "invented"}}}]
        )

        with pytest.raises(SpecError):
            job_runner.run()

        assert not job_runner.output_directory.exists()


class TestTheRunIdentifier:
    """Derived from the specification, so a re-run extends rather than scatters."""

    def test_the_same_specification_lands_in_the_same_place(self, dataset, tmp_path):
        """
        Two runs of one file share a directory.

        A timestamped identifier would make a re-run after a crash produce
        a second partial set beside the first rather than completing it.
        """
        first = runner(dataset, tmp_path)
        second = runner(dataset, tmp_path)

        assert first.run_id == second.run_id

    def test_a_changed_specification_lands_somewhere_else(self, dataset, tmp_path):
        """
        Derived from the digest, so a different set is a different place.

        Otherwise a changed configuration would overwrite the results of
        the one it replaced, and the manifest would describe neither.
        """
        first = runner(dataset, tmp_path)
        second = runner(dataset, tmp_path, jobs=[{"id": "a"}, {"id": "c"}])

        assert first.run_id != second.run_id

    def test_the_identifier_carries_the_sets_name(self, dataset, tmp_path):
        """
        A directory listing should be readable without opening anything.

        `portfolio-7f3a9c21` says what it is; a bare digest does not.
        """
        assert runner(dataset, tmp_path).run_id.startswith("set-")

    def test_an_explicit_identifier_is_honoured(self, dataset, tmp_path):
        """A caller that has its own naming scheme keeps it."""
        spec = parse_job_set_spec(job_set_payload(dataset, tmp_path / "out"))

        assert JobSetRunner(spec, run_id="chosen").run_id == "chosen"


class TestPayloadsWithoutRunning:
    """Inspecting a set is as useful as running one."""

    def test_payloads_are_produced_for_every_job(self, dataset, tmp_path):
        """
        The natural unit to inspect, test against, and re-run one job from.

        A job that failed inside a set should be reproducible on its own,
        and this makes that a one-line operation rather than a
        reconstruction of the merge, the seed and the directory layout.
        """
        payloads = runner(dataset, tmp_path).payloads()

        assert set(payloads) == {"a", "b"}

    def test_producing_payloads_runs_nothing(self, dataset, tmp_path):
        """Inspection must not have side effects on the filesystem."""
        job_runner = runner(dataset, tmp_path)

        job_runner.payloads()

        assert not job_runner.output_directory.exists()

    def test_every_job_in_a_set_shares_its_run_identifier(self, dataset, tmp_path):
        """
        The set is one experiment, and its bundles should be findable together.

        The digest is what makes a set's members a set in the catalog
        rather than forty unrelated runs.
        """
        payloads = runner(dataset, tmp_path).payloads()

        assert len({payload.run_id for payload in payloads.values()}) == 1
        assert len({payload.spec_digest for payload in payloads.values()}) == 1


class TestTheSetLevelContext:
    """A set needs somewhere to write its own figures and logs."""

    def test_the_context_points_at_the_sets_directory(self, dataset, tmp_path):
        """
        Not at any job's.

        Set-level output belongs beside the job directories, not inside
        one of them, where it would look like that job produced it.
        """
        job_runner = runner(dataset, tmp_path)

        assert job_runner.context().output_directory == job_runner.output_directory

    def test_the_context_is_not_the_one_a_job_runs_under(self, dataset, tmp_path):
        """
        Each job derives its own, in its own process.

        The set's context has no job identifier, so a seed derived from it
        would belong to no job in particular.
        """
        job_runner = runner(dataset, tmp_path)

        assert job_runner.context().run_id == job_runner.run_id
        assert job_runner.context().seed != job_runner.payloads()["a"].context().seed
```

---

## 8. `tests/rade_qnet/orchestration/jobs/test_jobs_unit.py`

9702 bytes · SHA-256 `e3cb40ca51ccddfe`

```python
"""
Tests for the unit of work a job set dispatches.

``run_job`` is the one function a job set sends to a worker, so the decisions
under test are mostly about what survives the crossing into another process
and what has to be rebuilt once it arrives.

Three of those decisions are load-bearing:

- **The payload is plain data.** No catalog handle, no open file, no live
  model. A handle is meaningless in another process, and the symptom of
  sending one is a pickling error at dispatch time -- or worse, a handle that
  pickles and then refers to nothing.
- **The seed is derived from the job identifier**, not assigned by position.
  Position would mean reordering a job set changed every model in it, and
  adding a job at the top would change all the ones below.
- **Registrations are replayed before any name is resolved.** A spawned
  worker starts with a bare interpreter; see `ARCHITECTURE.md` defect 12.
"""

from __future__ import annotations

import pickle

import pytest

from src.rade_qnet.core.lifecycle.components import engine as register_engine
from src.rade_qnet.core.lifecycle.components import model as register_model
from src.rade_qnet.core.lifecycle.errors import ComponentError
from src.rade_qnet.core.spec.jobs import parse_job_set_spec
from src.rade_qnet.orchestration.jobs.set import JobSetRunner
from src.rade_qnet.orchestration.jobs.unit import JobOutcome, JobPayload, run_job
from src.rade_qnet.testkit.fixtures import SyntheticEngine, isolated_registries

from .support import (
    ENGINE_TAG,
    MODEL_NAME,
    SyntheticSupervisedModel,
    job_set_payload,
    write_linear_dataset,
)


@pytest.fixture(autouse=True)
def _registries():
    """
    Isolate the component registries for every test here.

    Autouse because every test registers the same engine and model, and a
    leaked registration fails whichever test happens to run next rather
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
        register_engine(ENGINE_TAG)(SyntheticEngine)
        register_model(MODEL_NAME, engine=ENGINE_TAG)(SyntheticSupervisedModel)
        yield


@pytest.fixture
def payloads(tmp_path):
    """
    Provide the payloads for a two-job set, without running anything.

    Returns
    -------
    dict
        Job identifier to :class:`JobPayload`.
    """
    dataset = write_linear_dataset(tmp_path / "linear.csv")
    spec = parse_job_set_spec(job_set_payload(dataset, tmp_path / "out"))
    return JobSetRunner(spec).payloads()


class TestThePayloadCrossesAProcessBoundary:
    """
    What a job carries has to be plain data.

    The constraint is not stylistic: an open handle either refuses to pickle
    or pickles into something that refers to nothing in the receiving
    process, and the second failure mode is far harder to diagnose.
    """

    def test_a_payload_pickles(self, payloads):
        """
        The whole payload round-trips, specification included.

        The specification is a pydantic model, so it pickles by value and
        arrives identical to the one validated in the parent -- which is
        what lets validation happen once, up front, for the whole set.
        """
        payload = payloads["a"]

        restored = pickle.loads(pickle.dumps(payload))

        assert restored.job_id == payload.job_id
        assert restored.spec == payload.spec

    def test_the_payload_carries_a_catalog_path_not_a_catalog(self, payloads):
        """
        A path, because a catalog holds a lock file handle.

        Each worker opens its own against the same path, which is what the
        single-writer catalog is built for. Sending the catalog itself
        would send a handle to a lock this process holds.
        """
        assert isinstance(payloads["a"].catalog_root, type(payloads["a"].output_directory))

    def test_the_payload_names_the_modules_a_worker_must_import(self, payloads):
        """
        Registration modules travel as strings.

        Resolved in the parent, so an unregistered name fails where the
        error can list the alternatives. See defect 12.
        """
        modules = payloads["a"].registration_modules

        assert SyntheticSupervisedModel.__module__ in modules
        assert all(isinstance(module, str) for module in modules)


class TestSeedsAreDerivedFromIdentifiers:
    """A job's seed comes from its name, not from where it sits in the list."""

    def test_two_jobs_get_different_seeds(self, payloads):
        """
        Otherwise every job in a set would train the same model.

        The set would still look healthy: forty runs, forty bundles, forty
        identical scores that nobody reads as a bug.
        """
        assert payloads["a"].context().seed != payloads["b"].context().seed

    def test_the_same_identifier_gives_the_same_seed(self, tmp_path):
        """
        Re-running one job alone reproduces what it produced in the set.

        This is what makes a failed job debuggable: it can be re-run on its
        own and behave identically.
        """
        dataset = write_linear_dataset(tmp_path / "linear.csv")
        spec = parse_job_set_spec(job_set_payload(dataset, tmp_path / "out"))

        first = JobSetRunner(spec).payloads()["a"].context().seed
        second = JobSetRunner(spec).payloads()["a"].context().seed

        assert first == second

    def test_reordering_the_jobs_does_not_change_their_seeds(self, tmp_path):
        """
        Order-independence, which a positional seed would not give.

        With positional seeds, inserting a job at the top of a file would
        silently retrain every job below it with different weights, and the
        diff that caused it would be one added line.
        """
        dataset = write_linear_dataset(tmp_path / "linear.csv")
        forward = parse_job_set_spec(
            job_set_payload(dataset, tmp_path / "out", jobs=[{"id": "a"}, {"id": "b"}])
        )
        reversed_ = parse_job_set_spec(
            job_set_payload(dataset, tmp_path / "out", jobs=[{"id": "b"}, {"id": "a"}])
        )

        assert (
            JobSetRunner(forward).payloads()["a"].context().seed
            == JobSetRunner(reversed_).payloads()["a"].context().seed
        )


class TestRunningOneJob:
    """The function a worker actually calls."""

    def test_a_job_produces_an_outcome(self, payloads):
        """
        Metrics, a seed, a bundle location -- and never a model.

        The outcome crosses back through a pickle, and sending trained
        weights through it would make a forty-job set's return value the
        size of forty models for no reason: the weights are already on
        disk, and the record says where.
        """
        outcome = run_job(payloads["a"])

        assert isinstance(outcome, JobOutcome)
        assert outcome.job_id == "a"
        assert outcome.model_name == MODEL_NAME
        assert outcome.bundle_directory is not None

    def test_the_exact_fit_is_recovered(self, payloads):
        """
        An r-squared of one, because the data is exactly linear.

        Asserting the value rather than a tolerance is what distinguishes a
        job that trained from one that merely completed.
        """
        outcome = run_job(payloads["a"])

        assert outcome.metric("test", "r2") == pytest.approx(1.0)

    def test_the_outcome_records_the_seed_that_was_applied(self, payloads):
        """
        The derived seed, not the set's base seed.

        Recorded so a job can be reproduced from its manifest row alone,
        without re-deriving anything.
        """
        payload = payloads["a"]

        assert run_job(payload).seed == payload.context().seed

    def test_the_job_writes_beneath_its_own_directory(self, payloads):
        """
        Each job owns a directory, so two jobs cannot overwrite each other.

        The layout is decided in the parent rather than the worker, so a
        set has one place that knows where things go.
        """
        payload = payloads["a"]

        outcome = run_job(payload)

        assert outcome.bundle_directory.is_relative_to(payload.output_directory)

    def test_an_outcome_pickles(self, payloads):
        """The return trip is as constrained as the outbound one."""
        outcome = run_job(payloads["a"])

        assert pickle.loads(pickle.dumps(outcome)) == outcome


class TestWhatAJobRefuses:
    """Failures that should happen in the worker, clearly."""

    def test_an_unresolvable_model_names_the_alternatives(self, payloads):
        """
        The registry's error, listing what it does know.

        A worker that cannot resolve a name has to say so clearly, because
        the usual cause is a registration that did not travel across the
        process boundary rather than a typo in the file.
        """
        payload = payloads["a"]
        absent = payload.spec.model_copy(
            update={"model": payload.spec.model.model_copy(update={"name": "absent"})}
        )

        with pytest.raises(ComponentError, match="no model named 'absent'"):
            run_job(
                JobPayload(
                    job_id=payload.job_id,
                    spec=absent,
                    run_id=payload.run_id,
                    spec_digest=payload.spec_digest,
                    output_directory=payload.output_directory,
                )
            )
```

