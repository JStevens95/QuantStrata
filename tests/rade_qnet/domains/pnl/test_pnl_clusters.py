"""
Tests for expanding a portfolio into a job set.

The decision under test is a layering one. ``orchestration`` may not import
``domains`` or ``models``, so the job-set runner cannot know what a cluster
is. Expansion therefore happens here, and by the time the runner sees
anything there are only jobs -- which is what lets the same runner serve any
domain.

The other thing worth pinning down is the per-cluster override hook. It is
the mechanism behind the whole proposition that a portfolio is a job set
rather than a loop: a liquid cluster with abundant history supports a wider
model than a thin one, and forcing both to one configuration means
underfitting the first or overfitting the second.
"""

from __future__ import annotations

import json

import pytest

from src.rade_qnet.domains.pnl.clusters import (
    cluster_overrides,
    fingerprint_tag,
    job_set_for,
)
from src.rade_qnet.domains.pnl.portfolio import MANIFEST_FILENAME, read_portfolio

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
def portfolio(tmp_path):
    """
    Provide a two-cluster portfolio on disk.

    Returns
    -------
    Portfolio
        The portfolio.
    """
    root = tmp_path / "book"
    clusters = [
        {"name": "FX__G10", "elementary_ids": ["EURUSD"], "target_ids": ["EURGBP"]},
        {"name": "RATES__USD", "elementary_ids": ["USD2Y", "USD5Y"], "target_ids": ["USD7Y"]},
    ]
    root.mkdir(parents=True)
    for entry in clusters:
        (root / entry["name"]).mkdir()
    (root / MANIFEST_FILENAME).write_text(json.dumps({"clusters": clusters}), encoding="utf-8")
    return read_portfolio(root)


class TestExpansion:
    """A portfolio becomes a job set, one job per cluster."""

    def test_there_is_one_job_per_cluster(self, portfolio, tmp_path):
        """Two clusters in, two jobs out, named after them."""
        spec = job_set_for(portfolio, defaults=DEFAULTS, output_root=tmp_path / "out")

        assert spec.job_ids == ("FX__G10", "RATES__USD")

    def test_each_job_reads_its_own_clusters_directory(self, portfolio, tmp_path):
        """
        The one override every cluster job needs, whatever the model.

        Deliberately the only one: anything else would be this module
        deciding how a model should be configured, which the model
        declares.
        """
        spec = job_set_for(portfolio, defaults=DEFAULTS, output_root=tmp_path / "out")

        directory = spec.run_spec_for("FX__G10").source.params["directory"]

        assert directory == str(portfolio.cluster("FX__G10").directory)

    def test_shared_defaults_reach_every_job(self, portfolio, tmp_path):
        """
        Defaults are merged under each job's overrides.

        A per-cluster setting wins and everything it does not mention
        survives, which is the merge property the whole job-set design
        rests on.
        """
        spec = job_set_for(portfolio, defaults=DEFAULTS, output_root=tmp_path / "out")

        assert spec.run_spec_for("RATES__USD").training.epochs == 1

    def test_every_job_validates_at_expansion(self, portfolio, tmp_path):
        """
        Not at dispatch.

        A portfolio that expands into something misconfigured should fail
        where the error can name the cluster, rather than after some of
        its jobs have already trained.
        """
        spec = job_set_for(portfolio, defaults=DEFAULTS, output_root=tmp_path / "out")

        assert set(spec.validate_jobs()) == {"FX__G10", "RATES__USD"}

    def test_each_job_carries_its_clusters_description(self, portfolio, tmp_path):
        """
        So a manifest and a log line say what a job actually is.

        `RATES__USD` is a name; "3 elementary, 1 target" is the shape.
        """
        spec = job_set_for(portfolio, defaults=DEFAULTS, output_root=tmp_path / "out")

        assert spec.job("FX__G10").description == "FX__G10: 1 elementary, 1 target"


class TestPerClusterComplexity:
    """The reason a portfolio is a job set rather than a loop."""

    def test_the_hook_can_vary_the_model_per_cluster(self, portfolio, tmp_path):
        """
        A wide model for the liquid cluster, a narrow one for the thin.

        Forcing one configuration on both means underfitting the first or
        overfitting the second.
        """
        widths = {"FX__G10": 32, "RATES__USD": 8}

        spec = job_set_for(
            portfolio,
            defaults=DEFAULTS,
            output_root=tmp_path / "out",
            overrides_for=lambda cluster: {"model": {"params": {"units": widths[cluster.name]}}},
        )

        assert spec.run_spec_for("FX__G10").model.params["units"] == 32
        assert spec.run_spec_for("RATES__USD").model.params["units"] == 8

    def test_the_hook_does_not_lose_the_directory_override(self, portfolio, tmp_path):
        """
        The caller's fragment is merged over this module's, not instead of it.

        A hook returning a `source` fragment must not silently detach the
        job from its cluster's data.
        """
        spec = job_set_for(
            portfolio,
            defaults=DEFAULTS,
            output_root=tmp_path / "out",
            overrides_for=lambda cluster: {"source": {"transforms": {"sequence": {"length": 8}}}},
        )

        run = spec.run_spec_for("FX__G10")

        assert run.source.params["directory"] == str(portfolio.cluster("FX__G10").directory)
        assert run.source.transforms.sequence.length == 8

    def test_the_hook_can_see_cluster_attributes(self, portfolio, tmp_path):
        """
        Complexity is usually a function of what the cluster *is*.

        Passing the cluster rather than its name is what lets a rule be
        written against asset class or instrument count instead of a
        hand-maintained lookup.
        """
        spec = job_set_for(
            portfolio,
            defaults=DEFAULTS,
            output_root=tmp_path / "out",
            overrides_for=lambda cluster: {
                "model": {"params": {"units": 8 * cluster.universe.n_elementary}}
            },
        )

        assert spec.run_spec_for("RATES__USD").model.params["units"] == 16


class TestProvenance:
    """Which snapshot a set was expanded from."""

    def test_the_snapshot_digest_is_carried_as_a_tag(self, portfolio, tmp_path):
        """
        So it reaches every bundle in the set.

        A result that looks surprising six months from now is asked one
        question first: was this trained on the data I think it was.
        """
        spec = job_set_for(portfolio, defaults=DEFAULTS, output_root=tmp_path / "out")

        assert spec.tags == (fingerprint_tag(portfolio.fingerprint),)


class TestTheSetsOwnFields:
    """What the expansion decides, and what it leaves to the caller."""

    def test_the_set_is_named(self, portfolio, tmp_path):
        """
        A default that reads in a directory listing.

        `portfolio-7f3a9c21` says what it is; a bare digest does not.
        """
        spec = job_set_for(portfolio, defaults=DEFAULTS, output_root=tmp_path / "out")

        assert spec.name == "portfolio"

    def test_a_name_can_be_given(self, portfolio, tmp_path):
        """A caller running two books wants to tell them apart."""
        spec = job_set_for(portfolio, defaults=DEFAULTS, output_root=tmp_path / "out", name="eod")

        assert spec.name == "eod"

    def test_placement_is_left_to_the_policy_by_default(self, portfolio, tmp_path):
        """
        `auto`, which is the normal path.

        The expansion knows about clusters, not about how many cores the
        machine running it has.
        """
        spec = job_set_for(portfolio, defaults=DEFAULTS, output_root=tmp_path / "out")

        assert spec.placement.executor == "auto"

    def test_placement_can_be_declared(self, portfolio, tmp_path):
        """A caller that knows its machine can say so."""
        spec = job_set_for(
            portfolio,
            defaults=DEFAULTS,
            output_root=tmp_path / "out",
            placement={"executor": "processes", "workers": 4},
        )

        assert spec.placement.executor == "processes"
        assert spec.placement.workers == 4

    def test_a_selected_subset_expands_to_fewer_jobs(self, portfolio, tmp_path):
        """
        Running part of a book is running part of a book.

        The selection happens on the portfolio, before expansion, so the
        job set has no notion of a cluster that was filtered out.
        """
        spec = job_set_for(
            portfolio.select(["RATES__USD"]), defaults=DEFAULTS, output_root=tmp_path / "out"
        )

        assert spec.job_ids == ("RATES__USD",)


class TestTheOverrideFragment:
    """The minimal per-cluster fragment, on its own."""

    def test_it_names_only_the_source_directory(self, portfolio):
        """
        Minimal on purpose.

        Everything else is the model's declaration, and this module is in
        no position to know it.
        """
        cluster = portfolio.cluster("FX__G10")

        assert cluster_overrides(cluster) == {
            "source": {"params": {"directory": str(cluster.directory)}}
        }
