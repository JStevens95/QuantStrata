"""
Tests for the Torch engine.

The engine is the composition of everything else in the package, so the
headline test here is the conformance suite: the same seven clauses any
backend must satisfy, run against the real thing. That is deliberately the
same code a user writing their own engine runs, because a suite that only the
built-in engine passes is a suite that encodes the built-in engine's
accidents.

The rest of the tests cover the engine's own decisions, which are mostly about
*ordering* and *refusing*. The stage order in ``prepare`` is a contract: move
to the device, compile, then distribute, then build the optimiser. Each step
depends on the previous one having happened, and getting it wrong produces a
run that trains nothing while reporting a normal curve -- defect 6.

One behaviour here was a genuine inconsistency found while building this
phase. ``fit`` warned that it would proceed without a validation split and
then raised from the checkpoint, because the default monitor is ``val_loss``.
It now substitutes ``train_loss`` and says so, which is what the warning
already implied.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.rade_xl.core.runtime.errors import EngineError
from src.rade_xl.core.runtime.seeding import seed_everything
from src.rade_xl.core.spec.hardware import HardwareSpec
from src.rade_xl.core.spec.training import (
    CheckpointSpec,
    EarlyStoppingSpec,
    TorchTrainingSpec,
)
from src.rade_xl.engines.torch.engine import TorchEngine
from src.rade_xl.testkit.conformance import check_engine
from src.rade_xl.testkit.fixtures import SyntheticTensorSource, make_signature


class Net(nn.Module):
    """
    A small model whose forward signature matches a batch's keys.

    Parameters
    ----------
    n_features
        Input width.
    """

    def __init__(self, n_features: int = 4) -> None:
        """Build the stack."""
        super().__init__()
        self.layer = nn.Linear(n_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        return self.layer(features)


class LazyNet(nn.Module):
    """A model whose first layer infers its own input width."""

    def __init__(self) -> None:
        """Build the lazy stack."""
        super().__init__()
        self.layer = nn.LazyLinear(1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""
        return self.layer(features)


def make_source(n_samples: int = 64, *, seed: int = 0) -> SyntheticTensorSource:
    """
    Build a source over a learnable linear relationship.

    Parameters
    ----------
    n_samples
        Rows.
    seed
        Seed for the features.

    Returns
    -------
    SyntheticTensorSource
        The source.
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, 4)).astype(np.float32)
    targets = (features @ np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)).reshape(-1, 1)
    return SyntheticTensorSource(features=features, targets=targets, batch_size=16)


def training_spec(**kwargs) -> TorchTrainingSpec:
    """
    Build a short training spec.

    Parameters
    ----------
    **kwargs
        Overrides for the spec's fields.

    Returns
    -------
    TorchTrainingSpec
        The spec.
    """
    return TorchTrainingSpec(**{"epochs": 3, "learning_rate": 0.05, **kwargs})


def prepared(engine: TorchEngine, model: nn.Module, **kwargs):
    """
    Prepare a model for fitting on the CPU.

    Parameters
    ----------
    engine
        The engine.
    model
        The materialised model.
    **kwargs
        Overrides for the hardware or training spec.

    Returns
    -------
    ModelHandle
        The prepared handle.
    """
    return engine.prepare(
        model,
        hardware=kwargs.pop("hardware", HardwareSpec(device="cpu")),
        training=kwargs.pop("training", training_spec()),
        **kwargs,
    )


class TestConformance:
    """The seven clauses any backend must satisfy, run against the real one."""

    def test_the_torch_engine_conforms(self, tmp_path):
        """
        The headline test for the whole package.

        The same suite a user runs against their own engine: fitting changes
        the caller's model, predictions align with the source's rows, weights
        round-trip through a file, the unwrapped model is not a wrapper, and
        fitting is a strict improvement. Run on the built-in engine too,
        because a suite that only an external engine runs will encode this
        engine's accidents as requirements.
        """
        report = check_engine(
            TorchEngine(),
            model_factory=Net,
            source_factory=make_source,
            signature=make_signature(n_features=4, dtype="float32"),
            directory=tmp_path,
            training=training_spec(epochs=5),
            hardware=HardwareSpec(device="cpu"),
        )
        assert report.passed, report.summary()


class TestCapabilities:
    """What the engine declares, which the pipeline checks before it runs."""

    def test_the_engine_declares_its_name(self):
        """
        So a run manifest records which backend produced the result.

        Two runs of the same config on different backends are different runs,
        and nothing else in the output says so.
        """
        assert TorchEngine().capabilities().name == "torch"

    def test_lazy_materialisation_is_declared(self):
        """
        Because the pipeline reads the declaration, not the model.

        It uses the declaration to decide whether to run the materialise
        stage at all, so an engine that quietly supported lazy parameters
        without saying so would never have the stage run for it.
        """
        assert TorchEngine().capabilities().supports_lazy_materialisation

    def test_distribution_is_declared(self):
        """So a pipeline can refuse a distributed config on a tree backend."""
        assert TorchEngine().capabilities().supports_distributed

    def test_the_declared_accelerators_match_the_machine(self):
        """
        Read from torch rather than hardcoded.

        A hardcoded CUDA claim would make the pipeline accept a GPU config on
        a machine with no GPU, and the degradation would then happen two
        stages later than the check that was meant to catch it.
        """
        assert "cpu" in TorchEngine().capabilities().accelerators


class TestMaterialise:
    """Driven by the signature, before anything consumes the parameters."""

    def test_a_lazy_model_gains_its_parameters(self):
        """
        Which is what makes the optimiser built next non-empty.

        An optimiser over unmaterialised parameters tracks nothing: it
        constructs, steps, and updates nothing, and the loss curve is flat in
        a way that reads as a learning-rate fault.
        """
        model = TorchEngine().materialise(LazyNet(), make_signature(n_features=4, dtype="float32"))
        assert sum(parameter.numel() for parameter in model.parameters()) > 0

    def test_a_non_torch_model_is_refused(self):
        """
        Naming the type, because the cause is usually a misrouted model.

        A model built for one backend handed to another otherwise fails inside
        a forward pass, where the message is about tensors.
        """
        with pytest.raises(EngineError):
            TorchEngine().materialise(object(), make_signature(n_features=4))


class TestPrepare:
    """The stage order, which is the contract."""

    def test_the_handle_records_the_resolved_device(self):
        """Not the requested one, so a degraded run is visible in the output."""
        handle = prepared(TorchEngine(), Net())
        assert handle.device == "cpu"

    def test_the_optimiser_is_built_over_materialised_parameters(self):
        """
        Which is the ordering half of defect 6.

        Built before materialisation, the optimiser holds an empty parameter
        group: it constructs without complaint and steps without effect.
        """
        handle = prepared(TorchEngine(), Net())
        optimiser = handle.extras["optimiser"]
        assert sum(len(group["params"]) for group in optimiser.param_groups) > 0

    def test_a_model_with_no_trainable_parameters_is_refused(self):
        """
        Because training it would run to completion changing nothing.

        This is the condition defect 6 produces, caught at the stage that can
        still explain it -- the message names both causes, a frozen model and
        a missing materialise stage.
        """
        model = Net()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with pytest.raises(EngineError, match="trainable parameters"):
            prepared(TorchEngine(), model)

    def test_the_unwrapped_model_is_the_plain_module(self):
        """
        So a checkpoint carries parameter names with no wrapper prefix.

        A bundle holding ``module.`` or ``_orig_mod.`` keys can only be loaded
        by reconstructing the same wrapper, which the reader has no way to
        know about.
        """
        handle = prepared(TorchEngine(), Net())
        assert isinstance(handle.unwrapped, Net)

    def test_a_non_torch_training_spec_is_refused(self):
        """
        Before any data is read.

        A training spec for another backend names fields this engine does not
        have, and reading one of them at epoch zero fails with an attribute
        error that says nothing about backends.
        """
        with pytest.raises(EngineError):
            TorchEngine().prepare(Net(), hardware=HardwareSpec(device="cpu"), training=object())


class TestFit:
    """Training, and the two numbers it must report honestly."""

    def test_fitting_reduces_the_training_loss(self):
        """
        On a target that is an exact linear function of the features.

        Reachable by the model, so a correctly wired engine has to get closer
        to it; a failure here is wiring, not optimisation.
        """
        engine = TorchEngine()
        signature = make_signature(n_features=4, dtype="float32")
        handle = prepared(engine, engine.materialise(Net(), signature))
        outcome = engine.fit(
            handle,
            {"train": make_source()},
            training_spec(epochs=10),
        )
        assert outcome.history[-1].train_loss < outcome.history[0].train_loss

    def test_the_caller_s_model_is_the_one_that_was_trained(self):
        """
        Not a copy, which is the single most expensive thing to get wrong.

        A copy-training engine completes, reports a falling loss, writes a
        full report and persists a bundle -- holding the untrained weights.
        Every downstream number is then a measurement of random
        initialisation.
        """
        engine = TorchEngine()
        signature = make_signature(n_features=4, dtype="float32")
        model = engine.materialise(Net(), signature)
        before = model.layer.weight.detach().clone()

        engine.fit(
            prepared(engine, model),
            {"train": make_source()},
            training_spec(),
        )
        assert not torch.equal(model.layer.weight, before)

    def test_a_validation_split_is_used_when_present(self):
        """
        So the history carries the number early stopping selects on.

        A validation source accepted and ignored would make every validation
        loss absent, and every validation-monitored callback a no-op.
        """
        engine = TorchEngine()
        signature = make_signature(n_features=4, dtype="float32")
        outcome = engine.fit(
            prepared(engine, engine.materialise(Net(), signature)),
            {"train": make_source(), "validation": make_source(32, seed=1)},
            training_spec(),
        )
        assert outcome.history[0].val_loss is not None


class TestFitWithoutValidation:
    """The inconsistency found while building this phase, and its fix."""

    def test_a_run_with_no_validation_split_still_fits(self):
        """
        It warned that it would proceed, then raised from the checkpoint.

        The default monitor is ``val_loss``, so the warning was followed
        immediately by an error contradicting it. A user configuring no
        validation fraction got a message saying it was fine and then a
        failure.
        """
        engine = TorchEngine()
        signature = make_signature(n_features=4, dtype="float32")
        outcome = engine.fit(
            prepared(engine, engine.materialise(Net(), signature)),
            {"train": make_source()},
            training_spec(
                early_stopping=EarlyStoppingSpec(enabled=True, patience=2),
                checkpoint=CheckpointSpec(enabled=True),
            ),
        )
        assert outcome.history

    def test_the_monitor_falls_back_to_the_training_loss(self):
        """
        Which is the only metric such a run produces.

        Substituted rather than disabling the callbacks, because a run with no
        validation split still benefits from stopping when its training loss
        stops moving -- and the outcome has to name the metric that was
        actually used.
        """
        engine = TorchEngine()
        signature = make_signature(n_features=4, dtype="float32")
        outcome = engine.fit(
            prepared(engine, engine.materialise(Net(), signature)),
            {"train": make_source()},
            training_spec(checkpoint=CheckpointSpec(enabled=True)),
        )
        assert outcome.monitor == "train_loss"

    def test_the_substitution_is_warned_about(self, caplog):
        """
        Because the run is no longer doing what the config asked for.

        Silently monitoring a different metric would make a checkpoint
        selected on training loss indistinguishable from one selected on
        validation loss, and the first is a much weaker claim.
        """
        engine = TorchEngine()
        signature = make_signature(n_features=4, dtype="float32")
        with caplog.at_level("WARNING"):
            engine.fit(
                prepared(engine, engine.materialise(Net(), signature)),
                {"train": make_source()},
                training_spec(checkpoint=CheckpointSpec(enabled=True)),
            )
        assert "train_loss" in caplog.text


class TestPredict:
    """One row out per row in, in the source's order."""

    def test_one_prediction_per_sample(self):
        """
        Which is what makes a metric's denominator right.

        A short prediction array against a full target vector is either a
        broadcast or a silent truncation, and both produce a number.
        """
        engine = TorchEngine()
        signature = make_signature(n_features=4, dtype="float32")
        handle = prepared(engine, engine.materialise(Net(), signature))
        source = make_source(50)
        assert engine.predict(handle, source).shape[0] == source.n_samples

    def test_predictions_are_a_numpy_array(self):
        """
        So nothing downstream of the engine imports torch.

        That is what lets ``analysis`` and ``orchestration`` stay
        backend-agnostic: a metric computed over a tensor would tie the whole
        reporting layer to one backend.
        """
        engine = TorchEngine()
        signature = make_signature(n_features=4, dtype="float32")
        handle = prepared(engine, engine.materialise(Net(), signature))
        assert isinstance(engine.predict(handle, make_source()), np.ndarray)

    def test_two_passes_over_an_ordered_source_agree(self):
        """
        Because scoring makes two passes and pairs them row for row.

        This is the property the alignment bug violated: a reshuffling source
        gives two differently ordered passes, every metric still computes, and
        a correctly trained model reports a negative r-squared on its own
        training split.
        """
        engine = TorchEngine()
        signature = make_signature(n_features=4, dtype="float32")
        handle = prepared(engine, engine.materialise(Net(), signature))
        source = make_source()
        assert np.array_equal(engine.predict(handle, source), engine.predict(handle, source))


class TestReproducibility:
    """
    Two runs of one configuration, which is a promise the framework makes.

    This is the capstone on the seeder in ``engines/torch/seeding.py``. Before
    it existed these tests failed: ``seed_everything`` covered Python and
    NumPy and left Torch untouched, so every weight initialisation differed
    between two runs that both reported the same seed. Nothing in the output
    said so -- both runs completed and the scores simply disagreed.

    The negative test matters as much as the positive one. A seed that is
    accepted and ignored makes two runs identical for the wrong reason, and a
    sensitivity study across seeds would conclude the model was perfectly
    stable.
    """

    def _fit_once(self, seed: int, tmp_path) -> list[torch.Tensor]:
        """
        Seed, fit from scratch, and return the resulting weights.

        Parameters
        ----------
        seed
            Applied through ``core`` rather than through Torch directly, so
            the test exercises the registration and not just the function.
        tmp_path
            Checkpoint directory.

        Returns
        -------
        list of torch.Tensor
            Detached copies of every parameter, in a stable order.
        """
        seed_everything(seed, determinism="strict")
        engine = TorchEngine()
        handle = prepared(
            engine,
            engine.materialise(Net(), make_signature(n_features=4, dtype="float32")),
        )
        engine.fit(handle, {"train": make_source()}, training_spec(epochs=3))
        # `unwrapped` rather than the wrapped model, because that is the one a
        # checkpoint holds -- so it is the one whose reproducibility matters.
        return [parameter.detach().clone() for parameter in handle.unwrapped.parameters()]

    def test_one_seed_gives_identical_weights(self, tmp_path):
        """
        Bit for bit, not merely close.

        ``allclose`` would pass on a run whose sampling differed but whose
        optimiser happened to converge to a similar place, which is exactly
        the failure being ruled out. The fit is short enough that an
        unseeded initialisation could not have converged away from it.
        """
        first = self._fit_once(4321, tmp_path)
        second = self._fit_once(4321, tmp_path)
        assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))

    def test_a_different_seed_gives_different_weights(self, tmp_path):
        """
        So the seed is being applied rather than accepted and discarded.

        Without this, the test above would also pass on an engine that
        initialised every model identically regardless of seed, and a study
        sweeping seeds would measure nothing while looking conclusive.
        """
        first = self._fit_once(4321, tmp_path)
        other = self._fit_once(8765, tmp_path)
        assert not all(torch.equal(left, right) for left, right in zip(first, other, strict=True))
