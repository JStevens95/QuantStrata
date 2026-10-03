"""
Tests for the engine interface itself.

``Engine`` is the seam that makes the framework backend-agnostic, and the two
design decisions worth pinning down are both about what it does *not* contain.

It is a ``Protocol``, so a backend conforms by having the right methods rather
than by inheriting. That matters because an engine is the piece most likely to
be written outside this repository -- by someone wrapping a library the
framework has never heard of -- and an abstract base class would require them
to import from here at class-definition time.

And it declares its capabilities as *data* rather than as methods that raise.
The pipeline can then refuse an impossible combination before reading any
data, rather than discovering at epoch three that the engine cannot
checkpoint. ``isinstance`` against the protocol routes; the conformance suite
in ``testkit`` verifies the behaviour behind it. The two are deliberately
different jobs: a structural check cannot tell a real ``fit`` from one that
returns an empty history.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from src.rade_xl.engines.base import Engine, EngineCapabilities, ModelHandle
from src.rade_xl.testkit.fixtures import SyntheticEngine


class TestEngineCapabilities:
    """Declared as data, so a pipeline can check before it runs."""

    def test_the_defaults_describe_the_simplest_usable_engine(self):
        """
        Epochs, validation and checkpointing on; the rest off.

        A new backend should not have to opt into the features every engine
        has, and should have to opt into the ones most do not -- which is
        distribution and lazy materialisation.
        """
        capabilities = EngineCapabilities(name="minimal")
        assert capabilities.supports_epochs
        assert capabilities.supports_validation_during_fit
        assert capabilities.supports_checkpointing
        assert not capabilities.supports_distributed
        assert not capabilities.supports_lazy_materialisation

    def test_there_is_no_payload_field(self):
        """
        Because there is one payload type, so there is nothing to route on.

        A second one existed until Phase 6, for one-shot engines wanting a
        whole matrix rather than a stream. Nothing ever read the flag, and
        the mismatch it was meant to bridge did not exist: this framework's
        data layer imports no training library, so a batch is already NumPy.
        Pinned as an absence because the field reading as supported was the
        defect -- see the Phase 6 charter, section 8.1.
        """
        assert not hasattr(EngineCapabilities(name="minimal"), "payload")

    def test_the_cpu_is_always_a_declared_accelerator(self):
        """
        So every engine can run somewhere.

        An engine declaring only CUDA could not be tested on a laptop, and in
        practice that means it is only ever tested in production.
        """
        assert "cpu" in EngineCapabilities(name="minimal").accelerators

    def test_the_description_is_a_single_log_line(self):
        """
        Naming the engine and what it can do.

        Written once at the top of a run, it is what makes a later "the engine
        cannot do that" message explicable.
        """
        described = EngineCapabilities(name="torch", supports_distributed=True).describe()
        assert "torch" in described
        assert "\n" not in described

    def test_capabilities_are_immutable(self):
        """
        Because they are a declaration, not a running state.

        A pipeline that could edit them would be able to talk itself into an
        unsupported path, and the record of what the engine claimed would no
        longer be the claim it was checked against.
        """
        capabilities = EngineCapabilities(name="torch")
        with pytest.raises(FrozenInstanceError):
            capabilities.name = "other"


class TestModelHandle:
    """What ``prepare`` returns, and why it holds two references."""

    def test_the_wrapped_and_unwrapped_models_are_both_held(self):
        """
        Because the two are used for different things.

        Training runs through the wrapper, so the gradients synchronise;
        checkpointing goes through the unwrapped model, so the saved parameter
        names have no wrapper prefix. Deriving either one on demand means
        unwrapping in several places and getting it wrong in one of them.
        """
        model = object()
        handle = ModelHandle(model=model, unwrapped=model)
        assert handle.model is model
        assert handle.unwrapped is model

    def test_the_resolved_hardware_is_recorded_not_the_request(self):
        """
        So a report states what the run got, not what it asked for.

        A request for CUDA that degraded to the CPU is the single most likely
        explanation for a throughput regression, and the request alone cannot
        reveal it.
        """
        model = object()
        handle = ModelHandle(model=model, unwrapped=model, device="cpu", precision="fp32")
        assert handle.device == "cpu"

    def test_whether_the_run_is_actually_distributed_is_recorded(self):
        """
        Taken from the wrapping that happened, not the spec that asked.

        A distributed config run in a single process is correct but not
        parallel, and nothing else in the output distinguishes the two.
        """
        model = object()
        assert not ModelHandle(model=model, unwrapped=model).is_distributed

    def test_engine_specific_apparatus_lives_in_extras(self):
        """
        So the handle stays backend-agnostic.

        An optimiser field would be meaningless for a tree backend, and a
        typed union of every backend's apparatus would mean adding a backend
        requires editing ``core``.
        """
        model = object()
        handle = ModelHandle(model=model, unwrapped=model, extras={"optimiser": "sgd"})
        assert handle.extras["optimiser"] == "sgd"

    def test_a_handle_is_immutable(self):
        """
        Because it is the record of a completed preparation.

        Mutating the device on a prepared handle would make it describe a
        placement that never happened.
        """
        model = object()
        with pytest.raises(FrozenInstanceError):
            ModelHandle(model=model, unwrapped=model).device = "cuda"

    def test_the_description_names_the_device_and_the_precision(self):
        """
        The two facts that explain a run's throughput.

        Logged at preparation, before any epoch has run, so a slow run is
        explicable from its first few lines.
        """
        model = object()
        described = ModelHandle(
            model=model, unwrapped=model, device="cpu", precision="bf16"
        ).describe()
        assert "cpu" in described
        assert "bf16" in described


class TestTheProtocol:
    """Structural conformance, which is what lets a backend be external."""

    def test_an_engine_written_elsewhere_conforms_without_inheriting(self):
        """
        The point of a protocol over an abstract base class.

        Someone wrapping a library this framework has never heard of should
        not have to import from here at class-definition time -- that turns a
        structural requirement into a packaging one.
        """
        assert isinstance(SyntheticEngine(), Engine)

    def test_a_class_missing_a_method_does_not_conform(self):
        """
        So the check is worth making.

        A protocol that accepted anything would let a half-written engine
        reach the fit stage, where the failure is an ``AttributeError`` with
        no indication of what the object was meant to be.
        """

        class Incomplete:
            """An engine with nothing but a name."""

            def capabilities(self) -> EngineCapabilities:
                """Return the declaration."""
                return EngineCapabilities(name="incomplete")

        assert not isinstance(Incomplete(), Engine)

    def test_conformance_does_not_imply_correctness(self):
        """
        Which is why the conformance *suite* exists alongside the protocol.

        This object has every method and is structurally an engine. Its
        ``fit`` returns nothing useful and its ``predict`` returns zeros. A
        structural check cannot tell it from a real engine, and the behaviour
        it is missing is precisely what the testkit's ``check_engine``
        verifies.
        """

        class Hollow:
            """An engine whose every method is a plausible no-op."""

            def capabilities(self) -> EngineCapabilities:
                """Return the declaration."""
                return EngineCapabilities(name="hollow")

            def materialise(self, model, signature):
                """Return the model untouched."""
                del signature
                return model

            def prepare(self, model, *, hardware, training, static=None):
                """Wrap the model in a handle and change nothing."""
                del hardware, training, static
                return ModelHandle(model=model, unwrapped=model)

            def fit(self, handle, sources, training, *, on_epoch_end=None):
                """Return nothing at all, having trained nothing."""
                del handle, sources, training, on_epoch_end

            def predict(self, handle, source):
                """Return zeros of a plausible shape."""
                del handle
                return np.zeros(source.n_samples)

            def save_weights(self, handle, path: Path) -> None:
                """Write nothing."""
                del handle, path

            def load_weights(self, model, path: Path):
                """Return the model untouched."""
                del path
                return model

        assert isinstance(Hollow(), Engine)

    def test_the_real_engine_conforms(self):
        """
        Checked here as well as in the conformance suite.

        This one is cheap and runs without torch, so a broken signature on
        the test engine is caught before the expensive suite runs.
        """
        assert isinstance(SyntheticEngine(), Engine)
