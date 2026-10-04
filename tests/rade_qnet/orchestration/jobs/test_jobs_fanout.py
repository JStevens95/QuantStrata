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
