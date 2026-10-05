# `tests/rade_qnet/engines`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 15 | 581 | `17c5d2fc92ec9189` |
| 2 | `test_engine_layout.py` | 261 | 10110 | `ccec6580fbb253cf` |
| 3 | `test_engines_base.py` | 261 | 10333 | `99dda0aeda3cf4ad` |
| 4 | `test_engines_loaders.py` | 180 | 7017 | `69e2fdf28f2f1d6e` |

---

## 1. `tests/rade_qnet/engines/__init__.py`

581 bytes · SHA-256 `17c5d2fc92ec9189`

```python
"""
Tests for ``rade_qnet.engines`` -- the training-library adapters.

As with sources, there is one shared contract suite that every engine is run
through. It is the suite that keeps the engine interface honest: a one-shot
tree fit and a multi-epoch gradient loop must both satisfy it, and if only the
gradient loop can, then the interface is a PyTorch interface with a generic
name.

Planned modules
---------------
``test_engines_base.py``
    The ``Engine`` protocol and capability flags, plus the shared conformance
    suite each adapter is parametrised into.  [Phase 1]
"""
```

---

## 2. `tests/rade_qnet/engines/test_engine_layout.py`

10110 bytes · SHA-256 `ccec6580fbb253cf`

```python
"""
The engine package convention, enforced.

``models/`` has had an enforced layout since Phase 3, and engines have not,
with the predictable result: ``xgboost`` is one module, ``sklearn`` was two
under a name (``adapters.py``) that described nothing, and ``torch`` had grown
to eleven flat files mixing five unrelated concerns.  Someone adding a fourth
backend had no template to copy and no vocabulary to obey.

This module closes that gap.  The sanctioned names are the four verbs an
engine performs and the three parts of *fit* big enough to need their own
package, and the rule is the same one ``test_model_layout.py`` applies to
models: ``engine.py`` is required, everything else is optional, and a name
outside the set is a failing test rather than a convention nobody mentioned.

Why a closed set is the load-bearing half
------------------------------------------
Requiring ``engine.py`` prevents nothing -- a backend obviously has one.  What
keeps the tree navigable is *forbidding the rest*, because drift happens one
plausible ``utils.py`` at a time and each addition is individually defensible.
Closing the set means a genuinely new concern has to be a reviewed extension
of the vocabulary here, argued once for every engine, rather than a judgement
call made quietly inside one of them.

What the progression across engines says
-----------------------------------------
The vocabulary is deliberately *optional* down to a single file, so the shape
of an engine package reports what its library actually owns.  ``xgboost`` is
one module because a boosted fit is a single call with no loop, no learner and
no device to choose.  ``sklearn`` is one module for the same reason.  Only
``torch`` carries ``training/``, ``learners/`` and ``hardware/``, because only
PyTorch makes the framework supply those.  A reader can therefore tell what a
backend does from ``ls`` alone, which is the whole return on the convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The one file an engine package must contain. It holds the class satisfying
#: :class:`~rade_qnet.engines.base.Engine`, and it is required because a
#: package here that does not implement that protocol is not an engine.
REQUIRED_FILES = frozenset({"__init__.py", "engine.py"})

#: Every other name an engine package may use, and nothing else. Each one is
#: a verb the engine performs, in the order a run performs them: build the
#: model, feed it, predict with it. Fitting is the fourth, and it is large
#: enough to be a package rather than a module -- see below.
OPTIONAL_FILES = frozenset(
    {
        "materialise.py",  # spec + signature -> an unfitted native model
        "loaders.py",  # a BatchSource -> whatever this library consumes
        "predictor.py",  # a fitted model -> predictions
    }
)

#: Sub-packages an engine may contain. These are the three parts of *fit* that
#: earn their own vocabulary, and they are separate from one another because
#: they change for different reasons: a new algorithm touches ``learners``
#: only, a new stopping rule touches ``training`` only, and a new accelerator
#: touches ``hardware`` only.
OPTIONAL_DIRECTORIES = frozenset(
    {
        "training",  # when a fit happens: drivers, callbacks, losses, checkpoints
        "learners",  # what one update means, one module per algorithm
        "hardware",  # where it runs, and whether it runs the same way twice
    }
)

#: Module names permitted inside ``training/``. Closed for the same reason the
#: top level is: this is where a fit's machinery accumulates fastest.
TRAINING_MODULES = frozenset(
    {
        "__init__.py",
        "loops.py",
        "callbacks.py",
        "losses.py",
        "checkpoint.py",
        "risk.py",
    }
)

#: Module names permitted inside ``hardware/``.
HARDWARE_MODULES = frozenset(
    {
        "__init__.py",
        "devices.py",
        "distributed.py",
        "determinism.py",
    }
)

#: Files directly under ``engines/`` that are not engine packages. ``base.py``
#: holds the protocols; ``loaders.py`` holds the drain shared by every
#: one-shot engine, hoisted here so that xgboost does not have to import from
#: sklearn to reach it.
FRAMEWORK_MODULES = frozenset({"__init__.py", "base.py", "loaders.py"})

ENGINES_ROOT = Path(__file__).resolve().parents[3] / "src" / "rade_qnet" / "engines"


def engine_packages() -> Iterator[Path]:
    """
    Yield every engine package directory.

    Yields
    ------
    pathlib.Path
        One directory per engine, in a stable alphabetical order so a
        failure names the same package on every machine.
    """
    for child in sorted(ENGINES_ROOT.iterdir()):
        if child.is_dir() and child.name != "__pycache__":
            yield child


def names_in(directory: Path) -> frozenset[str]:
    """
    Return the file and directory names in ``directory``, ignoring caches.

    Parameters
    ----------
    directory
        The directory to list.

    Returns
    -------
    frozenset of str
        Names, not paths, so a failure message is readable.
    """
    return frozenset(
        child.name
        for child in directory.iterdir()
        if child.name not in {"__pycache__", ".ruff_cache"}
    )


PACKAGES = list(engine_packages())


class TestTheEngineVocabulary:
    """Every engine package draws its filenames from one closed set."""

    @pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.name)
    def test_an_engine_package_has_an_engine_module(self, package: Path) -> None:
        """
        ``engine.py`` exists, because that is what makes it an engine.

        Parameters
        ----------
        package
            One engine package directory.
        """
        missing = REQUIRED_FILES - names_in(package)
        assert not missing, f"{package.name} is missing {sorted(missing)}"

    @pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.name)
    def test_an_engine_package_contains_nothing_unsanctioned(self, package: Path) -> None:
        """
        No file or directory is named anything outside the vocabulary.

        This is the check that keeps the rest honest. Without it the required
        file is a floor and the package drifts anyway.

        Parameters
        ----------
        package
            One engine package directory.
        """
        allowed = REQUIRED_FILES | OPTIONAL_FILES | OPTIONAL_DIRECTORIES
        unexpected = names_in(package) - allowed
        assert not unexpected, (
            f"{package.name} contains {sorted(unexpected)}, which is outside the engine "
            f"vocabulary {sorted(allowed)}. Either rename it to one of those, or extend "
            f"the vocabulary here and say in the charter what the new name means."
        )

    @pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.name)
    def test_a_sub_package_draws_from_its_own_closed_set(self, package: Path) -> None:
        """
        ``training/`` and ``hardware/`` are closed too.

        These are where a fit's machinery accumulates fastest, so leaving them
        open would move the drift one level down rather than prevent it.

        Parameters
        ----------
        package
            One engine package directory.
        """
        for directory, permitted in (
            ("training", TRAINING_MODULES),
            ("hardware", HARDWARE_MODULES),
        ):
            path = package / directory
            if not path.is_dir():
                continue
            unexpected = names_in(path) - permitted
            assert not unexpected, (
                f"{package.name}/{directory} contains {sorted(unexpected)}, outside "
                f"{sorted(permitted)}"
            )

    def test_engines_itself_holds_only_protocols_and_the_shared_drain(self) -> None:
        """
        Nothing accumulates directly under ``engines/``.

        The package root is the first place a "just this one helper" lands,
        and a helper there is reachable by every engine, which is how one
        backend's convenience becomes every backend's dependency.
        """
        loose = frozenset(
            child.name
            for child in ENGINES_ROOT.iterdir()
            if child.is_file() and child.suffix == ".py"
        )
        assert loose == FRAMEWORK_MODULES, (
            f"engines/ holds {sorted(loose)}; expected exactly {sorted(FRAMEWORK_MODULES)}"
        )


class TestWhatTheShapeReports:
    """The layout is informative, not merely tidy."""

    def test_only_torch_needs_the_fit_sub_packages(self) -> None:
        """
        A one-shot engine has no loop, no learner and no device policy.

        If this ever fails for ``sklearn`` or ``xgboost`` it is worth asking
        why: a boosted fit acquiring a ``training/`` package means the engine
        has started reimplementing a loop the framework already owns.
        """
        with_training = {p.name for p in PACKAGES if (p / "training").is_dir()}
        assert with_training == {"torch"}

    def test_no_engine_imports_another(self) -> None:
        """
        Backends are siblings, not a hierarchy.

        ``xgboost`` used to import ``sklearn.adapters`` for the drain that
        turns a batch stream into one matrix. The drain is pure NumPy and
        belongs to neither, so it was hoisted to ``engines/loaders.py``. The
        rule matters because a cross-engine import makes one backend's
        presence a condition of another's, which is exactly what installing
        only xgboost is supposed to avoid.
        """
        names = {p.name for p in PACKAGES}
        for package in PACKAGES:
            for module in package.rglob("*.py"):
                text = module.read_text()
                for other in names - {package.name}:
                    assert f"..{other}." not in text, (
                        f"{module} imports from the {other} engine; hoist the shared part "
                        f"to engines/loaders.py instead"
                    )
```

---

## 3. `tests/rade_qnet/engines/test_engines_base.py`

10333 bytes · SHA-256 `99dda0aeda3cf4ad`

```python
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

from src.rade_qnet.engines.base import Engine, EngineCapabilities, ModelHandle
from src.rade_qnet.testkit.fixtures import SyntheticEngine


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
```

---

## 4. `tests/rade_qnet/engines/test_engines_loaders.py`

7017 bytes · SHA-256 `69e2fdf28f2f1d6e`

```python
"""
Tests for the scikit-learn adapters.

The adapters are where the framework's streaming view of data meets a library
that wants all of it at once, and nearly every way that can go wrong is
silent. A column order that differs between fit and predict produces a model
that scores well on its own split and badly on anything else. A flatten that
picks the wrong axis produces a matrix of the right size and the wrong
meaning. Neither raises.

So the tests here are mostly about *order* and *shape*, and they assert on
values rather than on sizes wherever a size would also pass by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.core.lifecycle.errors import EngineError
from src.rade_qnet.engines.loaders import drain, flatten, reject_static
from src.rade_qnet.testkit.fixtures import SyntheticTensorSource


def source(n_samples: int = 20, n_features: int = 3, **kwargs: object) -> SyntheticTensorSource:
    """
    Build a small source with recognisable values.

    Feature ``(i, j)`` holds ``i * 10 + j``, so a row, a column or an
    ordering mistake is visible in the number itself rather than only in a
    shape.

    Parameters
    ----------
    n_samples
        Row count.
    n_features
        Column count.
    **kwargs
        Passed through to the fixture.

    Returns
    -------
    SyntheticTensorSource
        A re-iterable source.
    """
    grid = np.arange(n_samples)[:, None] * 10 + np.arange(n_features)[None, :]
    return SyntheticTensorSource(
        features=grid.astype(np.float64),
        targets=np.arange(n_samples, dtype=np.float64)[:, None],
        batch_size=7,
        **kwargs,  # type: ignore[arg-type]
    )


class TestFlattening:
    """A matrix is what scikit-learn takes, whatever arrived."""

    def test_a_matrix_is_returned_unchanged(self) -> None:
        """Two dimensions already are the required shape."""
        array = np.arange(12, dtype=np.float64).reshape(4, 3)
        assert flatten(array) is array

    def test_a_sequence_window_is_flattened_to_one_row_per_sample(self) -> None:
        """Samples are preserved; everything after them collapses."""
        array = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert flatten(array).shape == (2, 12)

    def test_flattening_keeps_each_sample_s_values_together(self) -> None:
        """
        The collapse is within a row, not across rows.

        A transposed flatten gives a matrix of exactly the right shape whose
        rows are mixtures of different samples. Nothing downstream can detect
        that, so it is pinned here on values.
        """
        array = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert np.array_equal(flatten(array)[0], np.arange(12, dtype=np.float64))

    def test_a_column_vector_is_a_matrix_already(self) -> None:
        """A single feature is still two-dimensional."""
        assert flatten(np.zeros((5, 1))).shape == (5, 1)


class TestDraining:
    """Every batch, concatenated, in the order the source yielded them."""

    def test_the_row_count_is_the_sample_count(self) -> None:
        """Nothing is dropped at a batch boundary."""
        assert drain(source(n_samples=20)).n_samples == 20

    def test_the_rows_arrive_in_the_source_s_order(self) -> None:
        """
        Order is the contract that lets predictions be compared to targets.

        The final batch here is short -- twenty samples in sevens -- which is
        the case where an adapter that pre-allocates gets this wrong.
        """
        drained = drain(source(n_samples=20))
        assert np.array_equal(drained.features[:, 0], np.arange(20) * 10)

    def test_the_target_is_one_dimensional(self) -> None:
        """
        scikit-learn warns on a column vector and silently changes behaviour.

        A ``(n, 1)`` target makes several estimators treat the problem as
        multi-output, which changes the shape of ``predict`` and breaks the
        comparison against targets downstream.
        """
        assert drain(source()).target is not None
        assert drain(source()).target.ndim == 1  # type: ignore[union-attr]

    def test_features_and_target_stay_aligned(self) -> None:
        """Row ``i`` of the matrix belongs to element ``i`` of the target."""
        drained = drain(source(n_samples=20))
        assert drained.target is not None
        assert np.array_equal(drained.features[:, 0] / 10, drained.target)

    def test_a_source_without_targets_is_allowed_when_not_required(self) -> None:
        """Inference has no target and must still drain."""
        batches = source()
        drained = drain(batches, require_target=False)
        assert drained.n_samples == 20

    def test_feature_names_describe_the_flattened_columns(self) -> None:
        """
        One name per column, so a coefficient can be attributed.

        A single wide input is the common case and the one a coefficient
        table is read for, so it must be named rather than skipped.

        Underscores rather than bracket subscripts: XGBoost refuses feature
        names containing brackets, and both engines share these adapters.
        """
        drained = drain(source(n_features=3))
        assert drained.feature_names == ("features_0", "features_1", "features_2")


class TestRefusals:
    """What the adapter will not quietly accept."""

    def test_static_inputs_are_refused(self) -> None:
        """
        A graph cannot be flattened into a design matrix.

        Dropping it would be the dangerous alternative: the run would
        succeed, the model would be missing an input it was specified with,
        and the only evidence would be a worse metric.
        """
        with pytest.raises(EngineError, match="static"):
            reject_static({"adjacency": np.zeros((3, 3))})

    def test_no_static_inputs_is_not_a_refusal(self) -> None:
        """The common case passes through silently."""
        reject_static({})

    def test_an_empty_source_is_refused(self) -> None:
        """
        Zero rows reaches scikit-learn as an obscure error from inside it.

        Raising here names the actual problem, which is a source or a split
        that produced nothing.
        """
        empty = SyntheticTensorSource(features=np.zeros((0, 3)), targets=np.zeros((0, 1)))
        with pytest.raises(EngineError, match="no batches"):
            drain(empty)

    def test_a_missing_target_is_refused_when_required(self) -> None:
        """Fitting without a target is a specification error, not a crash."""

        class Untargeted(SyntheticTensorSource):
            """A source whose batches carry inputs only."""

            def batches(self):
                for batch in super().batches():
                    yield {k: v for k, v in batch.items() if k != "target"}

        untargeted = Untargeted(features=np.zeros((8, 3)), targets=np.zeros((8, 1)))
        with pytest.raises(EngineError, match="target"):
            drain(untargeted)
```

