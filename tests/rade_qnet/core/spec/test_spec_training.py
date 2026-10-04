"""
Tests for the training specification.

Two structural decisions are pinned down here.

``TrainingSpec`` discriminates on ``engine``, so asking for ``epochs`` with the
XGBoost engine fails at load rather than being ignored. The alternative -- one
training spec with every backend's fields on it -- means most fields are
meaningless for any given run, and nothing can tell you which.

``RlTrainingSpec`` is deliberately **not** in that union. It is selected by
``task`` one level up, because reinforcement learning differs in what it
*means* rather than in which library runs it: an epoch over a fixed dataset has
no counterpart when experience is generated as training proceeds.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from src.rade_qnet.core.spec.training import (
    CheckpointSpec,
    EarlyStoppingSpec,
    RlTrainingSpec,
    SchedulerSpec,
    SklearnTrainingSpec,
    TorchTrainingSpec,
    TrainingSpec,
    XGBoostTrainingSpec,
)

_TRAINING_ADAPTER = TypeAdapter(TrainingSpec)


class TestDefaults:
    """Nested specs are bare-constructible; the top level is not."""

    @pytest.mark.parametrize(
        "spec_type", [EarlyStoppingSpec, CheckpointSpec, SchedulerSpec, TorchTrainingSpec]
    )
    def test_each_spec_constructs_with_defaults(self, spec_type):
        """
        The refined rule from the phase definition of done.

        Anything reached through another spec's ``default_factory`` must be
        constructible bare, or the outer spec cannot be built at all.
        """
        assert spec_type() is not None

    def test_early_stopping_is_off_by_default(self):
        """
        Stopping early is a choice, not a default.

        A user who did not ask for it should get the epoch budget they
        configured.
        """
        assert EarlyStoppingSpec().enabled is False

    def test_the_best_checkpoint_is_restored_by_default(self):
        """
        The safe default.

        Without restoring, the reported metrics and the saved weights
        describe different models -- which is why ``FitOutcome`` records
        whether it happened.
        """
        assert CheckpointSpec().restore_best is True


class TestEngineUnion:
    """The union discriminates on ``engine``."""

    @pytest.mark.parametrize(
        ("engine", "expected"),
        [
            ("torch", TorchTrainingSpec),
            ("xgboost", XGBoostTrainingSpec),
            ("sklearn", SklearnTrainingSpec),
        ],
    )
    def test_the_engine_selects_the_member(self, engine, expected):
        """Each backend gets its own fields, and only its own."""
        assert isinstance(_TRAINING_ADAPTER.validate_python({"engine": engine}), expected)

    def test_an_unknown_engine_is_rejected(self):
        """A misspelled engine fails at load."""
        with pytest.raises(ValidationError):
            _TRAINING_ADAPTER.validate_python({"engine": "pytorch"})

    def test_a_torch_field_is_rejected_for_xgboost(self):
        """
        The payoff of discriminating.

        ``epochs`` means nothing to a boosted-tree fit. Accepting and ignoring
        it would let a user tune a field with no effect and never find out.
        """
        with pytest.raises(ValidationError):
            _TRAINING_ADAPTER.validate_python({"engine": "xgboost", "epochs": 50})

    def test_batch_size_is_not_a_training_field(self):
        """
        Batch size belongs to the loader.

        It describes how data is delivered, not how the model is optimised,
        and putting it in both places is how the two end up disagreeing.
        """
        assert "batch_size" not in TorchTrainingSpec.model_fields


class TestTorchTraining:
    """Field-level and cross-field validation for the gradient engine."""

    def test_zero_epochs_is_rejected(self):
        """A run that trains for no epochs produces an untrained model."""
        with pytest.raises(ValidationError):
            TorchTrainingSpec(epochs=0)

    def test_a_non_positive_learning_rate_is_rejected(self):
        """
        Zero learns nothing; negative ascends the loss.

        Both complete without error, which is exactly why they are rejected
        here instead.
        """
        with pytest.raises(ValidationError):
            TorchTrainingSpec(learning_rate=0.0)

    def test_a_clashing_monitor_direction_is_rejected(self):
        """
        Early stopping and checkpointing must agree on what "better" means.

        If one minimises and the other maximises the same metric, they select
        different epochs as best -- so training stops on one judgement and
        the saved weights reflect the other.
        """
        with pytest.raises(ValidationError, match="monitor"):
            TorchTrainingSpec(
                early_stopping={"enabled": True, "monitor": "val_loss", "mode": "min"},
                checkpoint={"monitor": "val_loss", "mode": "max"},
            )

    def test_agreeing_monitor_directions_are_accepted(self):
        """The legitimate configuration."""
        spec = TorchTrainingSpec(
            early_stopping={"enabled": True, "monitor": "val_loss", "mode": "min"},
            checkpoint={"monitor": "val_loss", "mode": "min"},
        )
        assert spec.early_stopping.enabled is True

    def test_different_metrics_may_use_different_directions(self):
        """
        The check is per metric, not global.

        Stopping on a loss while checkpointing on an accuracy is coherent,
        and a blanket rule would reject it.
        """
        spec = TorchTrainingSpec(
            early_stopping={"enabled": True, "monitor": "val_loss", "mode": "min"},
            checkpoint={"monitor": "val_accuracy", "mode": "max"},
        )
        assert spec.checkpoint.mode == "max"


class TestReinforcementTraining:
    """RL is selected by task, not by engine."""

    def test_it_is_not_a_member_of_the_engine_union(self):
        """
        Asking for it by engine does not work, and should not.

        It is selected one level up by ``task``, because the difference is
        what training *means*, not which library performs it.
        """
        with pytest.raises(ValidationError):
            _TRAINING_ADAPTER.validate_python({"engine": "rl"})

    def test_it_constructs_with_defaults(self):
        """Reached through the reinforcement run spec's default factory."""
        assert RlTrainingSpec() is not None

    def test_an_on_policy_batch_larger_than_the_update_is_rejected(self):
        """
        An on-policy learner must not reuse stale experience.

        A batch larger than the experience collected between updates can only
        be filled by sampling data the algorithm assumes is fresh, which
        breaks its correctness guarantee without any error.
        """
        with pytest.raises(ValidationError, match="batch_size"):
            RlTrainingSpec(learner="ppo", steps_per_update=128, batch_size=256)

    def test_an_on_policy_batch_within_the_update_is_accepted(self):
        """The legitimate configuration."""
        assert RlTrainingSpec(learner="ppo", steps_per_update=2048, batch_size=64) is not None

    def test_an_off_policy_learner_may_exceed_the_update_size(self):
        """
        The check applies only to on-policy learners.

        An off-policy learner samples from a replay buffer by design, so a
        batch larger than one update's collection is normal rather than a
        mistake.
        """
        assert RlTrainingSpec(learner="dqn", steps_per_update=1, batch_size=256) is not None


class TestStrictness:
    """The usual spec guarantees."""

    def test_an_unknown_key_is_rejected(self):
        """A typo fails at load."""
        with pytest.raises(ValidationError):
            TorchTrainingSpec(epoch=50)

    def test_the_spec_is_frozen(self):
        """A tuning trial must build a new spec rather than mutate one."""
        spec = TorchTrainingSpec()
        with pytest.raises(ValidationError):
            spec.epochs = 10

    def test_round_trip_is_exact(self):
        """
        Including the nested callback specs.

        The seventh diagnosed defect was a tuning loop using
        ``dataclasses.replace`` on a spec that was not a dataclass; a spec
        that round-trips can be rebuilt instead.
        """
        spec = TorchTrainingSpec(
            epochs=25,
            learning_rate=3e-4,
            early_stopping={"enabled": True, "patience": 5},
            checkpoint={"restore_best": False},
            scheduler={"kind": "cosine"},
        )
        assert TorchTrainingSpec.model_validate(spec.model_dump()) == spec

    def test_the_union_round_trips_through_json(self):
        """
        The discriminator survives serialisation.

        If ``engine`` were dropped on dump, the reloaded spec could not be
        resolved back to its member, and the bundle's record of the run would
        be unusable.
        """
        spec = _TRAINING_ADAPTER.validate_python({"engine": "xgboost"})
        payload = _TRAINING_ADAPTER.dump_python(spec, mode="json")
        assert _TRAINING_ADAPTER.validate_python(payload) == spec
