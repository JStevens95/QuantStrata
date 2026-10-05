# `tranql/models/rade/rade_qnet/tests/engines/torch`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 44 | 1965 | `ce84b941ee165fc1` |
| 2 | `test_torch_engine.py` | 520 | 19622 | `f3327525ec28b6c7` |
| 3 | `test_torch_loaders.py` | 244 | 9192 | `caf4130a6e746d13` |
| 4 | `test_torch_materialise.py` | 295 | 11057 | `f7fbb31f07001372` |

---

## 1. `tranql/models/rade/rade_qnet/tests/engines/torch/__init__.py`

1965 bytes · SHA-256 `ce84b941ee165fc1`

```python
"""
Tests for ``rade_qnet.engines.torch`` -- the PyTorch engine.

Tests here use tiny models on the CPU and assert exact or tightly-toleranced
numbers, so the suite stays fast enough to run on every change. Behaviour that
genuinely needs a GPU or multiple processes is marked and skipped when the
hardware is absent, rather than being left untested and unmentioned.

Planned modules
---------------
``test_torch_engine.py``
    Build, materialise, fit, checkpoint and predict against the engine
    contract.  [Phase 2]
``test_torch_loops.py``
    ``fit_epochs`` and ``fit_steps``: early stopping fires at the right epoch,
    the best checkpoint is the one restored, and a resumed run continues from
    the right step.  [Phase 2]
``test_torch_callbacks.py``
    Callback ordering and interaction, particularly early stopping together
    with learning-rate scheduling.  [Phase 2]
``test_torch_losses.py``
    Each loss against hand-computed values.  [Phase 2]
``test_torch_hardware.py``
    Device and precision resolution from a spec, including the fallback when
    requested hardware is unavailable.  [Phase 2]
``test_torch_materialise.py``
    Lazy parameters acquire concrete shapes before an optimiser, a checkpoint
    or a distributed wrapper touches them -- the ordering defect this engine
    exists to fix.  [Phase 2]
``test_torch_checkpoint.py``
    Checkpoints round-trip as state dictionaries, and load without executing
    pickled code.  [Phase 2]
``test_torch_loaders.py``
    Collation, worker configuration, and static inputs uploaded once rather
    than compared per sample.  [Phase 2]
``test_torch_distributed.py``
    Setup and teardown ordering. Skipped unless multiple devices are present.
    [Phase 2]
``test_torch_predictor.py``
    Batched inference, and that the precompute path gives results identical to
    the plain path.  [Phase 5]
``test_torch_risk.py``
    Differentiable risk measures against analytic values.  [Phase 7]
"""
```

---

## 2. `tranql/models/rade/rade_qnet/tests/engines/torch/test_torch_engine.py`

19622 bytes · SHA-256 `f3327525ec28b6c7`

```python
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

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import EngineError
from tranql.models.rade.rade_qnet.rade_qnet.core.provenance.seeding import seed_everything
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.hardware import HardwareSpec
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.training import (
    CheckpointSpec,
    EarlyStoppingSpec,
    TorchTrainingSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.engines.torch.engine import TorchEngine
from tranql.models.rade.rade_qnet.rade_qnet.testkit.conformance import check_engine
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import (
    SyntheticTensorSource,
    make_signature,
)


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
```

---

## 3. `tranql/models/rade/rade_qnet/tests/engines/torch/test_torch_loaders.py`

9192 bytes · SHA-256 `caf4130a6e746d13`

```python
"""
Tests for the batch-to-device bridge -- defect 4.

Defect 4 is that static inputs were merged into every sample during dataset
indexing, and the collate function then ran ``torch.equal`` across the batch
for each static key and returned ``values[0]``. So a graph adjacency matrix
was compared against itself ``batch_size`` times per batch, every batch, every
epoch, to confirm something true by construction.

The important part of the fix is that it is *not a behavioural change*: the
old collation already returned one copy, and the network already received
exactly one. Static inputs now travel outside the batch stream, which delivers
the same tensors to the same place and deletes the comparison.

So the tests here establish two things. That a static input is uploaded once
-- asserted as object identity across batches, which is the only way to tell
one upload from several. And that the tensors reaching the model are the same
ones, so the refactor is a deletion rather than a change.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tranql.models.rade.rade_qnet.rade_qnet.core.contract.data import TARGET_KEY
from tranql.models.rade.rade_qnet.rade_qnet.core.contract.signature import (
    InputSignature,
    TensorSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.engines.torch.loaders import (
    StaticInputs,
    to_device_batches,
    to_tensor,
)
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import SyntheticTensorSource

CPU = torch.device("cpu")


@pytest.fixture
def source():
    """
    Provide a bounded source with no static inputs.

    Returns
    -------
    SyntheticTensorSource
        The source.
    """
    rng = np.random.default_rng(0)
    return SyntheticTensorSource(
        features=rng.normal(size=(50, 4)).astype(np.float32),
        targets=rng.normal(size=(50, 1)).astype(np.float32),
        batch_size=16,
    )


@pytest.fixture
def graph_source():
    """
    Provide a source carrying a static adjacency matrix.

    Returns
    -------
    SyntheticTensorSource
        The source.
    """
    rng = np.random.default_rng(1)
    return SyntheticTensorSource(
        features=rng.normal(size=(50, 4)).astype(np.float32),
        targets=rng.normal(size=(50, 1)).astype(np.float32),
        batch_size=16,
        static={"adjacency": np.eye(4, dtype=np.float32)},
    )


class TestToTensor:
    """Conversion, with the dtype decided by the signature."""

    def test_an_array_becomes_a_tensor_of_the_declared_dtype(self):
        """
        Not of whatever numpy happened to produce.

        A float64 array meeting float32 weights is a hard error, so the
        signature has to be what decides.
        """
        tensor = to_tensor(np.zeros((2, 3), dtype=np.float64), dtype=torch.float32, device=CPU)
        assert tensor.dtype == torch.float32

    def test_a_tensor_is_accepted_unchanged_in_shape(self):
        """So a source that already produces tensors is not penalised."""
        tensor = to_tensor(torch.zeros(2, 3), dtype=torch.float32, device=CPU)
        assert tensor.shape == (2, 3)


class TestStaticInputsAreUploadedOnce:
    """Defect 4, stated as object identity."""

    def test_the_same_tensor_object_appears_in_every_batch(self, graph_source):
        """
        One upload, not one per batch.

        Identity rather than equality is the whole point: equality would hold
        even if the tensor were re-created for every batch, which is precisely
        the cost being removed.
        """
        signature = graph_source.signature
        static = StaticInputs.from_source(graph_source, signature=signature, device=CPU)

        seen = [
            inputs["adjacency"]
            for inputs, _ in to_device_batches(
                graph_source, signature=signature, device=CPU, static=static
            )
        ]
        assert len({id(tensor) for tensor in seen}) == 1

    def test_a_static_input_is_not_batched(self, graph_source):
        """
        It keeps its declared shape, with no batch axis prepended.

        A 4 by 4 adjacency collated per sample would arrive as 16 by 4 by 4,
        which is the shape the old implementation built and then threw away.
        """
        signature = graph_source.signature
        static = StaticInputs.from_source(graph_source, signature=signature, device=CPU)
        inputs, _ = next(
            iter(to_device_batches(graph_source, signature=signature, device=CPU, static=static))
        )
        assert inputs["adjacency"].shape == (4, 4)

    def test_a_model_with_no_static_inputs_pays_nothing(self, source):
        """
        Which is the common case and must not be a special case.

        A batch for a tabular model carries exactly its dynamic input.
        """
        inputs, _ = next(iter(to_device_batches(source, signature=source.signature, device=CPU)))
        assert set(inputs) == {"features"}


class TestBatchContents:
    """What arrives at the model, and in what form."""

    def test_the_target_is_separated_from_the_inputs(self, source):
        """
        Because the model takes the inputs and the loss takes the target.

        A target left among the inputs is passed to ``forward`` as a keyword
        argument, and the failure names the argument rather than the cause.
        """
        inputs, target = next(
            iter(to_device_batches(source, signature=source.signature, device=CPU))
        )
        assert TARGET_KEY not in inputs
        assert target.shape == (16, 1)

    def test_every_batch_is_yielded(self, source):
        """
        Including the short final one, unless the source dropped it.

        Silently discarding it would make the loop train on fewer samples than
        the source reported, and the denominator of every metric wrong by
        less than one batch -- never enough to notice.
        """
        batches = list(to_device_batches(source, signature=source.signature, device=CPU))
        assert sum(target.shape[0] for _, target in batches) == 50

    def test_the_values_survive_the_conversion(self, source):
        """
        The refactor is a deletion, so the numbers must be unchanged.

        Asserted against the source's own arrays rather than against a
        recomputation, because the claim is specifically that nothing happens
        to them on the way.
        """
        inputs, _ = next(iter(to_device_batches(source, signature=source.signature, device=CPU)))
        assert torch.equal(inputs["features"], torch.from_numpy(source.features[:16]))


class TestValidation:
    """A batch that does not match its signature is reported, not guessed at."""

    def test_a_missing_declared_input_is_reported(self):
        """
        Naming the input, because ``forward`` will not.

        A missing keyword argument raises from inside the model, where the
        message mentions a parameter name and nothing about the signature that
        was supposed to supply it.
        """
        rng = np.random.default_rng(2)
        signature = InputSignature(
            dynamic={
                "features": TensorSpec(shape=(None, 4), dtype="float32"),
                "absent": TensorSpec(shape=(None, 2), dtype="float32"),
            },
            target=TensorSpec(shape=(None, 1), dtype="float32"),
        )
        source = SyntheticTensorSource(
            features=rng.normal(size=(20, 4)).astype(np.float32),
            targets=rng.normal(size=(20, 1)).astype(np.float32),
            signature_override=signature,
        )
        with pytest.raises(ContractError):
            list(to_device_batches(source, signature=signature, device=CPU))

    def test_only_the_first_batch_is_validated_by_default(self, source):
        """
        Because every batch has the same keys by construction.

        Validating all of them would put a dictionary comparison in the inner
        loop of training, which is a real cost for a check that cannot newly
        fail after the first batch.
        """
        batches = list(
            to_device_batches(source, signature=source.signature, device=CPU, validate_first=True)
        )
        assert len(batches) == 4


class TestStaticInputsHelper:
    """The holder itself."""

    def test_an_empty_holder_merges_nothing(self, source):
        """So the no-static path has no branch of its own downstream."""
        static = StaticInputs.from_source(source, signature=source.signature, device=CPU)
        assert not static
        assert set(static.merge_into({"features": torch.zeros(2, 4)})) == {"features"}

    def test_the_description_names_the_inputs_and_their_shapes(self, graph_source):
        """
        So a run log says the adjacency was uploaded, and how big it was.

        A static input is the one thing in a batch nobody sees in the loss
        curve, so if it is wrong the log is where it has to be visible.
        """
        static = StaticInputs.from_source(
            graph_source, signature=graph_source.signature, device=CPU
        )
        assert "adjacency" in str(static.describe())
```

---

## 4. `tranql/models/rade/rade_qnet/tests/engines/torch/test_torch_materialise.py`

11057 bytes · SHA-256 `f7fbb31f07001372`

```python
"""
Tests for lazy-parameter materialisation -- defect 6.

A model with lazily shaped parameters has *no parameters at all* until it has
seen one batch. ``torch.nn.LazyLinear`` is the common case: its weight is an
``UninitializedParameter`` with no shape until the first forward pass infers
the input width.

Handing such a model to an optimiser produces an optimiser tracking an empty
parameter group. It constructs without complaint, ``step()`` succeeds, and
nothing is ever updated. The loss curve is perfectly flat, which reads as a
learning-rate problem and sends an investigation in the wrong direction for as
long as it takes someone to count the parameters.

Wrapping it for distributed training is worse: the wrapper either crashes deep
inside the distributed library or, on some versions, synchronises an empty
parameter set -- so every rank trains independently and the run silently is
not distributed at all.

The fix is that materialisation happens first, driven by the input signature
rather than by a batch. That is why :class:`InputSignature` exists: the dummy
forward needs exact shapes and dtypes with no data present.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tranql.models.rade.rade_qnet.rade_qnet.core.contract.signature import (
    InputSignature,
    TensorSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import EngineError
from tranql.models.rade.rade_qnet.rade_qnet.engines.torch.materialise import (
    count_parameters,
    dummy_batch,
    has_lazy_parameters,
    materialise,
    torch_dtype,
)
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import make_signature


class LazyNet(nn.Module):
    """A model whose first layer infers its own input width."""

    def __init__(self) -> None:
        """Build the lazy stack."""
        super().__init__()
        self.layer = nn.LazyLinear(8)
        self.head = nn.Linear(8, 1)

    def forward(self, features):
        """Run the forward pass."""
        return self.head(torch.relu(self.layer(features)))


class EagerNet(nn.Module):
    """A model whose shapes are fixed at construction."""

    def __init__(self, n_features: int = 4) -> None:
        """Build the eager stack."""
        super().__init__()
        self.layer = nn.Linear(n_features, 1)

    def forward(self, features):
        """Run the forward pass."""
        return self.layer(features)


class StaticNet(nn.Module):
    """A model consuming a static input alongside its dynamic one."""

    def __init__(self) -> None:
        """Build the stack."""
        super().__init__()
        self.layer = nn.LazyLinear(1)

    def forward(self, features, adjacency):
        """Mix the features through the adjacency before projecting."""
        return self.layer(features @ adjacency)


class TestDetection:
    """Telling a lazy model from an eager one, before anything consumes it."""

    def test_a_lazy_model_is_detected(self):
        """So the pipeline knows materialisation has work to do."""
        assert has_lazy_parameters(LazyNet())

    def test_an_eager_model_is_not(self):
        """
        So a model that never needed materialising pays nothing for it.

        Most models are eager, and a dummy forward pass on every one of them
        would be a cost for no benefit.
        """
        assert not has_lazy_parameters(EagerNet())

    def test_a_lazy_model_reports_no_parameters(self):
        """
        The defect, stated as the number it turns on.

        This is what an optimiser built before materialisation would see, and
        it is why such an optimiser updates nothing.
        """
        assert count_parameters(LazyNet()) == 0

    def test_a_materialised_model_reports_its_real_parameters(self):
        """
        And this is what it sees afterwards.

        4 by 8 weights plus 8 biases, then 8 by 1 plus 1: 73 in total,
        asserted exactly rather than as "more than zero", because a wrong
        inferred width would also be more than zero.
        """
        model = materialise(LazyNet(), make_signature(n_features=4, dtype="float32"))
        assert count_parameters(model) == (4 * 8 + 8) + (8 * 1 + 1)


class TestTheDummyBatch:
    """Synthesised from the signature, with no data present."""

    def test_every_declared_input_appears(self):
        """
        Including static inputs, because the forward pass takes them too.

        A dummy batch missing one produces a ``TypeError`` from inside
        ``forward``, which says nothing about the signature.
        """
        signature = InputSignature(
            dynamic={"features": TensorSpec(shape=(None, 4), dtype="float32")},
            static={"adjacency": TensorSpec(shape=(4, 4), dtype="float32")},
            target=TensorSpec(shape=(None, 1), dtype="float32"),
        )
        batch = dummy_batch(signature)
        assert set(batch) >= {"features", "adjacency"}

    def test_wildcard_dimensions_become_concrete(self):
        """
        A ``None`` in a shape is the batch axis, and a tensor cannot have one.

        Two rows rather than one, so a shape bug that collapses the batch axis
        is visible.
        """
        batch = dummy_batch(make_signature(n_features=4, dtype="float32"))
        assert batch["features"].shape[0] > 1
        assert batch["features"].shape[1] == 4

    def test_the_dtype_matches_the_declaration(self):
        """
        Because a float64 input meeting float32 weights is a hard error.

        Which is the right behaviour, but it has to be the *signature* that
        decides, not whatever numpy happened to default to.
        """
        batch = dummy_batch(make_signature(n_features=4, dtype="float32"))
        assert batch["features"].dtype == torch.float32

    def test_integer_inputs_are_filled_with_zeros(self):
        """
        Because an integer input is almost always an index.

        Ones would be a valid index into an embedding table of size two or
        more and would fail on a table of size one; zero is valid for any
        non-empty table.
        """
        signature = InputSignature(
            dynamic={"tokens": TensorSpec(shape=(None, 3), dtype="int64")},
            target=TensorSpec(shape=(None, 1), dtype="float32"),
        )
        assert torch.all(dummy_batch(signature)["tokens"] == 0)

    def test_float_inputs_are_filled_with_ones(self):
        """
        Zeros would hide a shape error in any multiplicative layer.

        A zero matrix multiplied by anything is zero whatever the shapes
        happen to be, so a dummy pass over zeros can succeed where real data
        would not.
        """
        batch = dummy_batch(make_signature(n_features=4, dtype="float32"))
        assert torch.all(batch["features"] == 1.0)


class TestMaterialise:
    """The pass itself, and what it reports when it cannot run."""

    def test_an_eager_model_is_returned_unchanged(self):
        """
        No dummy pass is run, which is the cheap path and the common one.

        Returning the same object rather than a copy means nothing downstream
        has to care whether materialisation was needed.
        """
        model = EagerNet()
        assert materialise(model, make_signature(n_features=4, dtype="float32")) is model

    def test_a_model_with_static_inputs_is_materialised(self):
        """
        The case the signature's static group exists for.

        A forward pass needing an adjacency matrix cannot be run from the
        dynamic inputs alone, so a signature that did not declare static
        inputs would make this model unmaterialisable.
        """
        signature = InputSignature(
            dynamic={"features": TensorSpec(shape=(None, 4), dtype="float32")},
            static={"adjacency": TensorSpec(shape=(4, 4), dtype="float32")},
            target=TensorSpec(shape=(None, 1), dtype="float32"),
        )
        model = materialise(StaticNet(), signature)
        assert count_parameters(model) > 0

    def test_materialisation_is_idempotent(self):
        """
        Because a pipeline stage may be re-run, and a tuning sweep will be.

        A second dummy pass that re-initialised the weights would discard
        whatever the first one established.
        """
        signature = make_signature(n_features=4, dtype="float32")
        once = materialise(LazyNet(), signature)
        before = count_parameters(once)
        assert count_parameters(materialise(once, signature)) == before

    def test_a_signature_that_does_not_fit_the_model_is_reported(self):
        """
        With the signature printed, because that is the thing to fix.

        A bare ``RuntimeError`` from inside a matrix multiply does not say
        which declared shape was wrong, and the shapes are not visible in the
        traceback.
        """
        signature = InputSignature(
            dynamic={
                "features": TensorSpec(shape=(None, 4), dtype="float32"),
                "unexpected": TensorSpec(shape=(None, 2), dtype="float32"),
            },
            target=TensorSpec(shape=(None, 1), dtype="float32"),
        )
        with pytest.raises(EngineError, match="input signature"):
            materialise(LazyNet(), signature)

    def test_materialisation_does_not_leave_gradients_behind(self):
        """
        The dummy pass is an initialisation, not a training step.

        A gradient left on a parameter is added to the first real batch's
        gradient, so the first update is computed partly from synthetic ones.
        """
        model = materialise(LazyNet(), make_signature(n_features=4, dtype="float32"))
        assert all(
            parameter.grad is None or torch.all(parameter.grad == 0)
            for parameter in model.parameters()
        )


class TestDtypeMapping:
    """Signature dtype strings to torch dtypes."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("float32", torch.float32),
            ("float64", torch.float64),
            ("int64", torch.int64),
        ],
    )
    def test_the_common_dtypes_map(self, name, expected):
        """So a signature can be written in numpy's vocabulary."""
        assert torch_dtype(name) == expected

    def test_an_unknown_dtype_is_reported_with_the_alternatives(self):
        """
        Rather than defaulting to float32.

        A silent default would make an integer index column into a float one,
        and an embedding lookup against a float index fails much later with a
        message about indices.
        """
        with pytest.raises(EngineError):
            torch_dtype("complex256")

    def test_numpy_dtype_names_are_accepted(self):
        """
        Because that is what a signature derived from arrays will carry.

        ``str(array.dtype)`` is the natural way to build a signature from
        data, and it must not need translating at every call site.
        """
        assert torch_dtype(str(np.zeros(1, dtype=np.float32).dtype)) == torch.float32
```

