"""
Tests for the defaults-and-overrides merge.

"Deep merge" means at least three different things in common usage, and the
one this framework needs is the one whose failure is silent: an override of a
nested field that discards its siblings produces a job configured plausibly
and wrongly, with no error and no symptom beyond a model that underperforms.

So every rule in `merge.py`'s table gets a test, and the nesting rule gets one
per depth rather than one in general. A merge that is correct at one level
and wrong at three is a real and common implementation, because the recursion
is usually the part that is written last.
"""

from __future__ import annotations

import pytest

from src.rade_xl.core.spec.merge import deep_merge, merge_all


class TestSiblingsSurvive:
    """The rule the whole module exists for, at each nesting depth."""

    def test_at_one_level(self):
        """An override of one key leaves the others alone."""
        merged = deep_merge({"units": 256, "layers": 3}, {"layers": 1})
        assert merged == {"units": 256, "layers": 1}

    def test_at_two_levels(self):
        """
        The depth at which a naive implementation starts to fail.

        A merge written as a single `dict.update` is correct at one level and
        wrong here: it replaces the whole `model` mapping, taking `units`
        with it.
        """
        merged = deep_merge(
            {"model": {"units": 256, "layers": 3}},
            {"model": {"layers": 1}},
        )
        assert merged == {"model": {"units": 256, "layers": 1}}

    def test_at_three_levels(self):
        """
        Because a recursion that only recurses once is a real implementation.

        The specification tree genuinely goes this deep --
        `source.transforms.sequence.length` is four -- so this is the shape
        of a real override rather than a contrived one.
        """
        merged = deep_merge(
            {"source": {"transforms": {"sequence": {"length": 20, "stride": 1}}}},
            {"source": {"transforms": {"sequence": {"length": 5}}}},
        )
        assert merged == {"source": {"transforms": {"sequence": {"length": 5, "stride": 1}}}}

    def test_siblings_survive_at_every_level_at_once(self):
        """
        One override, three levels, a sibling at each.

        The three tests above could all pass against an implementation that
        recursed correctly but dropped keys on the way back up. This one
        cannot.
        """
        merged = deep_merge(
            {
                "seed": 7,
                "source": {
                    "kind": "model",
                    "transforms": {"sequence": {"length": 20}, "scaling": {"method": "standard"}},
                },
            },
            {"source": {"transforms": {"sequence": {"length": 5}}}},
        )
        assert merged == {
            "seed": 7,
            "source": {
                "kind": "model",
                "transforms": {"sequence": {"length": 5}, "scaling": {"method": "standard"}},
            },
        }


class TestWhatReplaces:
    """Everything that is not two mappings."""

    def test_a_scalar_replaces(self):
        """There is nothing to merge."""
        assert deep_merge({"seed": 0}, {"seed": 7}) == {"seed": 7}

    def test_a_sequence_replaces_rather_than_concatenating(self):
        """
        The rule most often assumed to be the other way round.

        There is no identity to merge elements on, and the case that matters
        argues the same way: a job asking for `reports: [summary]` means
        those reports, not those plus whatever the defaults wanted.
        """
        merged = deep_merge({"reports": ["summary", "curves"]}, {"reports": ["summary"]})
        assert merged == {"reports": ["summary"]}

    def test_an_empty_sequence_replaces_too(self):
        """
        Which is how a job switches every report off.

        An implementation that treated an empty override as "no opinion"
        would make that impossible to express.
        """
        assert deep_merge({"reports": ["summary"]}, {"reports": []}) == {"reports": []}

    def test_an_explicit_none_overrides(self):
        """
        Because `None` is a value, not an absence.

        Several settings use `None` to mean "let the framework decide" --
        `device_index`, `threads_per_worker`, `workers`. A job resetting one
        to `None` is asking for that behaviour, and a merge that read `None`
        as "no opinion" would leave no way to ask.
        """
        assert deep_merge({"workers": 8}, {"workers": None}) == {"workers": None}

    def test_a_mapping_replaces_a_scalar(self):
        """
        Rather than being merged into it somehow.

        Guessing a resolution here would hide a configuration mistake; the
        merged value is the one the user most recently wrote, and validation
        downstream is what reports it.
        """
        assert deep_merge({"model": "ridge"}, {"model": {"name": "ridge"}}) == {
            "model": {"name": "ridge"}
        }

    def test_a_scalar_replaces_a_mapping(self):
        """The same rule, from the other side."""
        assert deep_merge({"model": {"name": "ridge"}}, {"model": "ridge"}) == {"model": "ridge"}


class TestKeySets:
    """Which keys end up in the result."""

    def test_a_key_only_in_the_override_is_added(self):
        """A job may configure something the defaults never mentioned."""
        assert deep_merge({"seed": 0}, {"name": "run"}) == {"seed": 0, "name": "run"}

    def test_a_key_only_in_the_base_is_kept(self):
        """Which is the point of having defaults at all."""
        assert deep_merge({"seed": 0, "name": "run"}, {"seed": 7}) == {"seed": 7, "name": "run"}

    def test_an_empty_override_changes_nothing(self):
        """A job with no overrides is the shared configuration exactly."""
        base = {"model": {"units": 256}, "seed": 0}
        assert deep_merge(base, {}) == base

    def test_an_empty_base_yields_the_override(self):
        """So a job set with no defaults still works."""
        assert deep_merge({}, {"seed": 7}) == {"seed": 7}


class TestIsolation:
    """
    The result shares no mutable container with either input.

    This matters more here than it usually would. The base is a job set's
    shared defaults and is merged once per job, so a result that aliased a
    nested mapping would let one job's later mutation reach every other job's
    configuration -- a bug that appears only with more than one job and only
    sometimes.
    """

    def test_the_base_is_not_modified(self):
        """Not even at depth."""
        base = {"model": {"units": 256}}
        deep_merge(base, {"model": {"units": 64}})
        assert base == {"model": {"units": 256}}

    def test_the_override_is_not_modified(self):
        """The same, from the other side."""
        override = {"model": {"units": 64}}
        deep_merge({"model": {"units": 256}}, override)
        assert override == {"model": {"units": 64}}

    def test_a_nested_mapping_is_not_shared_with_the_base(self):
        """
        An untouched branch is copied, not aliased.

        The subtle case: `source` is not mentioned by the override, so a
        shallow copy of the base would carry the base's own dictionary
        straight into the result.
        """
        base = {"source": {"kind": "model"}}
        merged = deep_merge(base, {"seed": 7})
        merged["source"]["kind"] = "tabular"
        assert base["source"]["kind"] == "model"

    def test_a_nested_sequence_is_not_shared_with_the_override(self):
        """The same property for the container type that replaces."""
        override = {"reports": ["summary"]}
        merged = deep_merge({}, override)
        merged["reports"].append("curves")
        assert override["reports"] == ["summary"]

    def test_a_string_survives_as_a_string(self):
        """
        A string is not treated as a sequence of characters.

        It is one, and copying it element-wise would turn a model name into
        a list of single-character strings.
        """
        assert deep_merge({}, {"model": "ridge"}) == {"model": "ridge"}


class TestMergeAll:
    """Folding several layers, so precedence reads in argument order."""

    def test_the_rightmost_layer_wins(self):
        """Framework defaults, then set defaults, then the job."""
        assert merge_all({"a": 1, "b": 1}, {"b": 2, "c": 2}, {"c": 3}) == {"a": 1, "b": 2, "c": 3}

    def test_nesting_is_respected_across_layers(self):
        """
        Each layer merges deeply, not only the last.

        A fold implemented with `dict.update` would pass the test above and
        fail this one.
        """
        merged = merge_all(
            {"model": {"units": 256, "layers": 3, "dropout": 0.1}},
            {"model": {"layers": 1}},
            {"model": {"dropout": 0.0}},
        )
        assert merged == {"model": {"units": 256, "layers": 1, "dropout": 0.0}}

    def test_no_layers_is_an_empty_mapping(self):
        """Rather than an error, so a caller need not special-case it."""
        assert merge_all() == {}

    def test_one_layer_is_a_copy_of_it(self):
        """Equal, and not the same object."""
        layer = {"model": {"units": 256}}
        merged = merge_all(layer)
        assert merged == layer
        assert merged["model"] is not layer["model"]


class TestAgainstRealSpecFragments:
    """
    The case the module was written for, end to end.

    Asserted as a merge rather than through validation, because this module
    may not import the run spec -- but the shapes are the real ones, so a
    change to the schema that broke this would be visible here.
    """

    @pytest.fixture
    def defaults(self):
        """Return a job set's shared defaults, in the real schema's shape."""
        return {
            "model": {"name": "hybrid_gnn_rnn", "units": 256, "gnn_layers": 3},
            "source": {
                "kind": "model",
                "transforms": {
                    "sequence": {"length": 20},
                    "reduction": {"method": "basis_selection", "fit_on": "train"},
                },
            },
            "training": {"engine": "torch", "epochs": 200},
            "reports": {"enabled": ["summary", "curves"]},
        }

    def test_a_sparse_cluster_gets_a_smaller_model_and_keeps_everything_else(self, defaults):
        """
        The motivating case from the architecture document.

        A cluster with thin history is given a narrower, shallower network.
        Every other decision -- the window, the basis policy, the epoch
        budget, the reports -- must come through untouched, because the user
        said nothing about them.
        """
        merged = deep_merge(defaults, {"model": {"units": 64, "gnn_layers": 1}})

        assert merged["model"] == {"name": "hybrid_gnn_rnn", "units": 64, "gnn_layers": 1}
        assert merged["source"]["transforms"]["sequence"]["length"] == 20
        assert merged["source"]["transforms"]["reduction"]["fit_on"] == "train"
        assert merged["training"] == {"engine": "torch", "epochs": 200}

    def test_a_job_pointing_at_its_own_data_keeps_the_shared_transforms(self, defaults):
        """
        The override every job has, and the one most likely to be mishandled.

        `source.params` is a sibling of `source.transforms`, so setting the
        former must not disturb the latter -- and a job set where every job
        silently lost its transforms would still train, and would train on
        unscaled, unreduced data.
        """
        merged = deep_merge(defaults, {"source": {"params": {"directory": "clusters/EURUSD"}}})

        assert merged["source"]["params"] == {"directory": "clusters/EURUSD"}
        assert merged["source"]["kind"] == "model"
        assert merged["source"]["transforms"]["sequence"]["length"] == 20

    def test_the_defaults_are_reusable_across_jobs(self, defaults):
        """
        Merged once per job, so the first job must not disturb the second.

        The aliasing bug this guards against is invisible with one job and
        intermittent with forty, which is the worst combination there is.
        """
        first = deep_merge(defaults, {"model": {"units": 64}})
        second = deep_merge(defaults, {"model": {"gnn_layers": 1}})

        assert first["model"]["units"] == 64
        assert second["model"]["units"] == 256
        assert second["model"]["gnn_layers"] == 1
        assert defaults["model"]["units"] == 256
