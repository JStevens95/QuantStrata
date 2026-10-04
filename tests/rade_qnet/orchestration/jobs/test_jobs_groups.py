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
