"""
Tests for the run specification and its single parsing entry point.

``parse_run_spec`` is the only way configuration enters the framework, and that
is what makes two things possible: the ``task`` discriminator can be defaulted
in one place rather than relying on schema magic, and a pydantic
``ValidationError`` can be converted into one ``SpecError`` whose message is
actionable. A user who mistyped a YAML key should see one framework error, not
a pydantic traceback.
"""

from __future__ import annotations

import json

import pytest

from src.rade_qnet.core.lifecycle.errors import SpecError
from src.rade_qnet.core.provenance.hashing import digest_spec
from src.rade_qnet.core.spec.run import (
    ComponentRef,
    ReinforcementRunSpec,
    SupervisedRunSpec,
    dump_run_spec,
    load_run_spec,
    parse_run_spec,
)


class TestComponentRef:
    """Both natural YAML spellings are accepted."""

    def test_a_bare_string_is_accepted(self):
        """
        ``model: ridge`` is how a user writes a model with no parameters.

        Requiring ``{name: ridge}`` would be correct and unpleasant.
        """
        spec = parse_run_spec({"model": "ridge"}, origin="test")
        assert spec.model.name == "ridge"
        assert spec.model.params == {}

    def test_inline_parameters_are_collected(self):
        """
        ``{name: ridge, alpha: 1.0}`` keeps parameters at the top level.

        Nesting them under ``params`` is how the data is stored, not how a
        user wants to type it.
        """
        spec = parse_run_spec({"model": {"name": "ridge", "alpha": 1.0}}, origin="test")
        assert spec.model.params == {"alpha": 1.0}

    def test_the_explicit_form_is_still_accepted(self):
        """The normalised form must survive a round trip through itself."""
        spec = parse_run_spec({"model": {"name": "ridge", "params": {"alpha": 1.0}}}, origin="test")
        assert spec.model.params == {"alpha": 1.0}

    def test_a_mapping_without_a_name_is_rejected(self):
        """There is no way to guess which component was meant."""
        with pytest.raises(SpecError, match="name"):
            parse_run_spec({"model": {"alpha": 1.0}}, origin="test")

    def test_describe_renders_without_the_pydantic_repr(self):
        """
        A component reference appears in reports a human reads.

        The default repr renders as ``name='ridge' params={}``, which reads as
        a debugging artefact in a summary page.
        """
        assert ComponentRef(name="ridge").describe() == "ridge"
        assert "alpha=1.0" in ComponentRef(name="ridge", params={"alpha": 1.0}).describe()


class TestTaskDiscrimination:
    """The task selects which run spec is built."""

    def test_the_task_defaults_to_supervised(self):
        """
        Defaulted at the entry point, not in the schema.

        A discriminated union cannot take a default tag from a member's field
        default, so the single parsing entry point supplies it -- which keeps
        the behaviour in one readable place rather than in schema machinery.
        """
        assert isinstance(parse_run_spec({"model": "demo"}, origin="test"), SupervisedRunSpec)

    def test_the_reinforcement_task_resolves(self):
        """The other member of the union."""
        spec = parse_run_spec(
            {"model": "demo", "task": "reinforcement", "environment": "hedging"}, origin="test"
        )
        assert isinstance(spec, ReinforcementRunSpec)

    def test_an_unknown_task_is_rejected(self):
        """A misspelled task fails at load."""
        with pytest.raises(SpecError):
            parse_run_spec({"model": "demo", "task": "supervize"}, origin="test")


class TestRequiredFields:
    """A top-level spec legitimately requires what it cannot default."""

    def test_the_model_is_required(self):
        """
        There is no sensible default model.

        This is the refinement to the "every spec constructs with defaults"
        rule: it applies to specs reached through a ``default_factory``, not
        to the top-level spec, which must name what to run.
        """
        with pytest.raises(SpecError, match="model"):
            parse_run_spec({}, origin="test")

    def test_everything_else_defaults(self):
        """
        Naming a model is enough to get a complete, valid spec.

        Which is what makes the first customisation tier -- spec only -- real.
        """
        spec = parse_run_spec({"model": "demo"}, origin="test")
        assert spec.hardware is not None
        assert spec.reports is not None
        assert spec.source is not None
        assert spec.training is not None


class TestValidationMessages:
    """Every failure arrives as one SpecError naming the field."""

    def test_an_unknown_top_level_key_is_rejected(self):
        """
        The second diagnosed defect's counterpart.

        A misspelled key that was ignored means the run trains with a default
        and reports plausible numbers.
        """
        with pytest.raises(SpecError, match="sed"):
            parse_run_spec({"model": "demo", "sed": 3}, origin="test")

    def test_the_error_names_the_offending_field_path(self):
        """
        A dotted path, so a deep failure is locatable.

        Without it, "split fractions must sum to less than one" does not say
        which of several nested splits is at fault.
        """
        with pytest.raises(SpecError) as caught:
            parse_run_spec(
                {
                    "model": "demo",
                    "source": {
                        "kind": "tabular",
                        "path": "x.csv",
                        "split": {
                            "kind": "chronological",
                            "validation_fraction": 0.8,
                            "test_fraction": 0.8,
                        },
                    },
                },
                origin="test",
            )
        assert "split" in str(caught.value)

    def test_the_origin_appears_in_the_message(self):
        """
        The message says which file was wrong.

        A job set loads many specs; an error that does not name its source
        sends the reader to the wrong file.
        """
        with pytest.raises(SpecError, match=r"my_config\.yaml"):
            parse_run_spec({}, origin="my_config.yaml")

    def test_a_non_mapping_payload_is_rejected(self):
        """
        A YAML file containing a list is a common mistake.

        Reported as a spec error, because the fix is the user's.
        """
        with pytest.raises(SpecError):
            parse_run_spec(["model", "demo"], origin="test")


class TestCrossFieldValidation:
    """Combinations that cannot work are rejected at load."""

    def test_a_sequence_window_with_an_explicit_split_is_rejected(self):
        """
        The framework cannot insert a boundary gap into caller-given indices.

        With a window longer than one scenario, a sample ending just after a
        split boundary contains training scenarios. For a generated split the
        framework adds clearance; for an explicit split only the caller can.
        """
        with pytest.raises(SpecError, match="sequence"):
            parse_run_spec(
                {
                    "model": "demo",
                    "source": {
                        "kind": "tabular",
                        "path": "x.csv",
                        "split": {
                            "kind": "explicit",
                            "train": [0, 1, 2],
                            "validation": [3],
                            "test": [4],
                        },
                        "transforms": {"sequence": {"length": 20}},
                    },
                },
                origin="test",
            )

    def test_a_single_scenario_window_with_an_explicit_split_is_accepted(self):
        """
        A window of one cannot straddle a boundary.

        So the restriction applies only where it is needed.
        """
        spec = parse_run_spec(
            {
                "model": "demo",
                "source": {
                    "kind": "tabular",
                    "path": "x.csv",
                    "split": {
                        "kind": "explicit",
                        "train": [0, 1, 2],
                        "validation": [3],
                        "test": [4],
                    },
                    "transforms": {"sequence": {"length": 1}},
                },
            },
            origin="test",
        )
        assert spec.source.transforms.sequence.length == 1


class TestImmutabilityAndRoundTrip:
    """A spec is frozen and survives serialisation exactly."""

    def test_the_spec_is_frozen(self):
        """So the bundle's record is a record of what happened."""
        spec = parse_run_spec({"model": "demo"}, origin="test")
        with pytest.raises(Exception, match="frozen"):
            spec.seed = 99

    def test_round_trip_is_exact(self):
        """
        The first diagnosed defect, at the top level.

        A spec that does not round-trip cannot be recorded in a bundle.
        """
        spec = parse_run_spec(
            {
                "model": {"name": "demo", "alpha": 1.0},
                "seed": 99,
                "tags": ["nightly"],
                "hardware": {"device": "cpu", "determinism": "warn"},
                "training": {"engine": "torch", "epochs": 7},
            },
            origin="test",
        )
        reloaded = parse_run_spec(json.loads(spec.model_dump_json()), origin="test")
        assert reloaded == spec

    def test_the_digest_is_stable_across_reparse(self):
        """
        A re-parsed spec recognises itself.

        Which is what lets a bundle claim to record the configuration that
        produced it, and what makes a step cache hit.
        """
        payload = {"model": "demo", "seed": 3}
        assert digest_spec(parse_run_spec(payload, origin="a")) == digest_spec(
            parse_run_spec(payload, origin="b")
        )

    def test_key_order_does_not_change_the_digest(self):
        """Reordering two keys in a YAML file is not a change to the run."""
        first = parse_run_spec({"model": "demo", "seed": 3}, origin="test")
        second = parse_run_spec({"seed": 3, "model": "demo"}, origin="test")
        assert digest_spec(first) == digest_spec(second)


class TestFileIo:
    """Loading and dumping, with every user-facing failure named."""

    def test_a_yaml_file_round_trips(self, tmp_path):
        """The normal path."""
        path = tmp_path / "run.yaml"
        spec = parse_run_spec({"model": "demo", "seed": 5}, origin="test")
        dump_run_spec(spec, path)
        assert load_run_spec(path) == spec

    def test_a_json_file_round_trips(self, tmp_path):
        """
        JSON is accepted as well as YAML.

        A bundle stores JSON, so reloading a bundle's spec uses the same
        reader as loading a hand-written configuration.
        """
        path = tmp_path / "run.json"
        spec = parse_run_spec({"model": "demo", "seed": 5}, origin="test")
        dump_run_spec(spec, path)
        assert load_run_spec(path) == spec

    def test_a_missing_file_is_a_spec_error(self):
        """
        Every "you pointed us at the wrong file" case is the user's to fix.

        So all of them report as a spec error rather than as an OSError.
        """
        with pytest.raises(SpecError):
            load_run_spec("/nonexistent/path/run.yaml")

    def test_a_file_containing_a_list_is_a_spec_error(self, tmp_path):
        """A common YAML mistake, reported in terms the user can act on."""
        path = tmp_path / "run.yaml"
        path.write_text("- model: demo\n", encoding="utf-8")
        with pytest.raises(SpecError):
            load_run_spec(path)

    def test_malformed_yaml_is_a_spec_error(self, tmp_path):
        """Not a yaml library exception leaking through."""
        path = tmp_path / "run.yaml"
        path.write_text("model: [unclosed\n", encoding="utf-8")
        with pytest.raises(SpecError):
            load_run_spec(path)

    def test_an_unknown_suffix_is_rejected_on_dump(self, tmp_path):
        """
        Writing a spec to an unrecognised format would be unreadable later.

        Better to refuse than to write something that cannot be loaded.
        """
        spec = parse_run_spec({"model": "demo"}, origin="test")
        with pytest.raises(SpecError):
            dump_run_spec(spec, tmp_path / "run.txt")

    def test_the_loaded_file_name_appears_in_a_validation_error(self, tmp_path):
        """A job set's failure must name the file that caused it."""
        path = tmp_path / "broken.yaml"
        path.write_text("seed: 3\n", encoding="utf-8")
        with pytest.raises(SpecError, match=r"broken\.yaml"):
            load_run_spec(path)
