"""
Tests for reading a portfolio and fingerprinting its snapshot.

Two decisions are under test.

**A portfolio is declared, not discovered.** The clusters are the ones a
manifest lists, not the subdirectories that happen to be present. A directory
listing silently changes meaning when somebody leaves a scratch folder
behind, and a job set that quietly grew a forty-first cluster is a result
nobody asked for.

**The snapshot is fingerprinted, cheaply.** Every result that looks
surprising is asked the same question first: was this trained on the data I
think it was? The digest covers what identifies the snapshot -- names,
universes, file sizes and timestamps -- rather than the P&L itself, because a
check expensive enough to be disabled is worth nothing.
"""

from __future__ import annotations

import json

import pytest

from src.rade_qnet.core.runtime.errors import SpecError
from src.rade_qnet.domains.pnl.portfolio import MANIFEST_FILENAME, Portfolio, read_portfolio


def write_portfolio(root, clusters=None):
    """
    Lay out a portfolio directory on disk.

    Parameters
    ----------
    root
        The portfolio directory, created if absent.
    clusters
        Manifest entries. Defaults to two clusters with distinguishable
        universes.

    Returns
    -------
    pathlib.Path
        The portfolio directory.
    """
    if clusters is None:
        clusters = [
            {
                "name": "FX__G10",
                "elementary_ids": ["EURUSD", "GBPUSD"],
                "target_ids": ["EURGBP"],
                "asset_class": "FX",
            },
            {
                "name": "RATES__USD",
                "elementary_ids": ["USD2Y", "USD5Y", "USD10Y"],
                "target_ids": ["USD7Y"],
                "asset_class": "RATES",
            },
        ]

    root.mkdir(parents=True, exist_ok=True)
    for entry in clusters:
        directory = root / entry.get("directory", entry["name"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "pnl.csv").write_text("date,value\n2024-01-01,1.0\n", encoding="utf-8")

    (root / MANIFEST_FILENAME).write_text(json.dumps({"clusters": clusters}), encoding="utf-8")
    return root


@pytest.fixture
def portfolio(tmp_path):
    """
    Provide a two-cluster portfolio read from disk.

    Returns
    -------
    Portfolio
        The portfolio.
    """
    return read_portfolio(write_portfolio(tmp_path / "book"))


class TestReadingAPortfolio:
    """What comes back from a well-formed directory."""

    def test_every_declared_cluster_is_read(self, portfolio):
        """Two entries in, two clusters out."""
        assert portfolio.names == ("FX__G10", "RATES__USD")

    def test_clusters_keep_manifest_order(self, tmp_path):
        """
        Not sorted.

        A job set's directory listing should match the file somebody
        wrote, so that reading one tells you about the other.
        """
        root = write_portfolio(
            tmp_path / "book",
            clusters=[{"name": "zeta"}, {"name": "alpha"}],
        )

        assert read_portfolio(root).names == ("zeta", "alpha")

    def test_a_clusters_universe_is_read(self, portfolio):
        """The identifiers, in the order the manifest gave them."""
        cluster = portfolio.cluster("FX__G10")

        assert cluster.universe.elementary_ids == ("EURUSD", "GBPUSD")
        assert cluster.universe.target_ids == ("EURGBP",)

    def test_unrecognised_fields_are_kept_as_attributes(self, portfolio):
        """
        Carried, not interpreted.

        Every institution partitions a book differently, and a fixed set
        of fields here would be wrong for the second one.
        """
        assert portfolio.cluster("FX__G10").attributes == {"asset_class": "FX"}

    def test_a_cluster_directory_defaults_to_its_name(self, portfolio):
        """The common layout needs no `directory` key."""
        assert portfolio.cluster("FX__G10").directory.name == "FX__G10"

    def test_a_cluster_directory_can_be_given_explicitly(self, tmp_path):
        """
        For a book whose directories are not named after its clusters.

        Common when the cluster name carries a desk prefix the filesystem
        does not.
        """
        root = write_portfolio(
            tmp_path / "book", clusters=[{"name": "FX__G10", "directory": "fx_g10"}]
        )

        assert read_portfolio(root).cluster("FX__G10").directory.name == "fx_g10"


class TestWhatIsRefused:
    """Problems that are knowable before any training starts."""

    def test_a_missing_manifest_says_what_was_expected(self, tmp_path):
        """
        Named, because "declared not discovered" is not obvious.

        A user who laid out directories and expected them to be found
        needs to be told about the manifest, not about a missing file.
        """
        (tmp_path / "book").mkdir()

        with pytest.raises(SpecError, match=MANIFEST_FILENAME):
            read_portfolio(tmp_path / "book")

    def test_unreadable_json_is_reported_as_such(self, tmp_path):
        """A truncated file should not surface as a type error."""
        root = tmp_path / "book"
        root.mkdir()
        (root / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

        with pytest.raises(SpecError, match=r"(?i)json"):
            read_portfolio(root)

    def test_a_manifest_without_clusters_is_refused(self, tmp_path):
        """An empty portfolio expands into an empty job set, which is not a run."""
        root = tmp_path / "book"
        root.mkdir()
        (root / MANIFEST_FILENAME).write_text(json.dumps({"clusters": []}), encoding="utf-8")

        with pytest.raises(SpecError, match=r"(?i)at least one"):
            read_portfolio(root)

    def test_a_cluster_pointing_nowhere_fails_at_read(self, tmp_path):
        """
        Here, not thirty-nine jobs later.

        A missing directory is knowable before any training starts, so
        discovering it partway through a set would be the framework's
        fault rather than the data's.
        """
        root = tmp_path / "book"
        write_portfolio(root, clusters=[{"name": "present"}])
        manifest = json.loads((root / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        manifest["clusters"].append({"name": "absent"})
        (root / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(SpecError, match="absent"):
            read_portfolio(root)

    def test_an_unknown_cluster_lookup_lists_the_known_ones(self, portfolio):
        """
        The usual cause is a typo in a cluster filter.

        Listing the alternatives turns that into a one-second fix.
        """
        with pytest.raises(KeyError, match="FX__G10"):
            portfolio.cluster("FX__G11")


class TestTheSnapshotFingerprint:
    """Which version of the book a run used."""

    def test_reading_the_same_directory_twice_agrees(self, tmp_path):
        """
        Stable, or it would be useless as provenance.

        A digest that changed on every read would mark every run as having
        used different data.
        """
        root = write_portfolio(tmp_path / "book")

        assert read_portfolio(root).fingerprint == read_portfolio(root).fingerprint

    def test_changing_the_data_changes_the_fingerprint(self, tmp_path):
        """
        The property the whole mechanism exists for.

        A refreshed directory must not look like the one a bundle was
        trained against.
        """
        root = write_portfolio(tmp_path / "book")
        before = read_portfolio(root).fingerprint

        (root / "FX__G10" / "pnl.csv").write_text("date,value\n2024-01-01,2.0\n", encoding="utf-8")

        assert read_portfolio(root).fingerprint != before

    def test_changing_the_universe_changes_the_fingerprint(self, tmp_path):
        """
        An added instrument is a different snapshot.

        The file contents may be untouched while the problem has changed,
        so the universe is part of the digest rather than only the data.
        """
        root = write_portfolio(tmp_path / "book")
        before = read_portfolio(root).fingerprint

        write_portfolio(
            root, clusters=[{"name": "FX__G10", "elementary_ids": ["EURUSD", "GBPUSD", "USDJPY"]}]
        )

        assert read_portfolio(root).fingerprint != before


class TestSelectingASubset:
    """Running part of a book."""

    def test_selection_keeps_the_named_clusters_in_order(self, portfolio):
        """The order given, not the manifest's, because the caller asked."""
        assert portfolio.select(["RATES__USD", "FX__G10"]).names == ("RATES__USD", "FX__G10")

    def test_selecting_nothing_keeps_everything(self, portfolio):
        """
        `None` means "all", so a caller need not branch.

        The common case is no filter at all.
        """
        assert portfolio.select(None) is portfolio

    def test_the_fingerprint_survives_selection(self, portfolio):
        """
        A subset of a snapshot was still trained on that snapshot.

        Recomputing would make two runs over different subsets of
        identical data look like runs over different data, which is the
        opposite of what provenance is for.
        """
        assert portfolio.select(["FX__G10"]).fingerprint == portfolio.fingerprint

    def test_selecting_an_unknown_cluster_fails(self, portfolio):
        """A filter naming a cluster that is not there is a mistake, not an empty set."""
        with pytest.raises(KeyError):
            portfolio.select(["absent"])


class TestDescribing:
    """What a log line and a progress display show."""

    def test_a_portfolio_summarises_itself(self, portfolio):
        """Cluster count, instrument count, and an abbreviated snapshot digest."""
        summary = portfolio.describe()

        assert "2 cluster(s)" in summary
        assert "7 instrument(s)" in summary

    def test_a_cluster_summarises_its_two_sides(self, portfolio):
        """Both counts, because one without the other says little."""
        assert portfolio.cluster("FX__G10").describe() == "FX__G10: 2 elementary, 1 target"

    def test_a_portfolio_is_iterable_and_sized(self, portfolio):
        """
        So a caller can loop without reaching for an attribute.

        The expansion in `clusters.py` reads better for it, and so does
        every ad-hoc script somebody writes against this.
        """
        assert len(portfolio) == 2
        assert [cluster.name for cluster in portfolio] == list(portfolio.names)
        assert isinstance(portfolio, Portfolio)
