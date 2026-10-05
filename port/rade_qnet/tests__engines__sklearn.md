# `tranql/models/rade/rade_qnet/tests/engines/sklearn`

2 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 10 | 304 | `b0a1ae83d8b2a9fc` |
| 2 | `test_sklearn_engine.py` | 277 | 10652 | `d92376e80569861b` |

---

## 1. `tranql/models/rade/rade_qnet/tests/engines/sklearn/__init__.py`

304 bytes · SHA-256 `b0a1ae83d8b2a9fc`

```python
"""
Tests for ``rade_qnet.engines.sklearn`` -- the scikit-learn engine.

Planned modules
---------------
``test_sklearn_engine.py``
    Any estimator exposing ``fit`` and ``predict`` is trainable through the
    engine contract, and its persisted form reloads and predicts identically.
    [Phase 6]
"""
```

---

## 2. `tranql/models/rade/rade_qnet/tests/engines/sklearn/test_sklearn_engine.py`

10652 bytes · SHA-256 `d92376e80569861b`

```python
"""
Tests for the scikit-learn engine.

The headline is the conformance suite, run against the real engine. That is
the whole point of this phase: the seven clauses were written against Torch,
and if a closed-form estimator with no epochs, no optimiser and no device
cannot pass them, the clauses encoded Torch's accidents rather than the
contract.

The rest of these tests cover what this engine decides for itself, which is
mostly about *honesty*. It reports one epoch because it had one, not because
one is a convenient default. It declares no capability it does not have, and
the pipeline reads those declarations rather than the engine's name. And it
refuses hardware it cannot provide instead of quietly running on the CPU,
because a GPU run that is secretly a CPU run is discovered from the clock.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.linear_model import Lasso, Ridge

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import get_engine
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import EngineError
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.hardware import HardwareSpec
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.training import SklearnTrainingSpec
from tranql.models.rade.rade_qnet.rade_qnet.engines.sklearn import SklearnEngine
from tranql.models.rade.rade_qnet.rade_qnet.testkit.conformance import check_engine
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import (
    SyntheticTensorSource,
    make_signature,
)

if TYPE_CHECKING:
    from pathlib import Path

N_FEATURES = 4
N_SAMPLES = 64


def prepared(engine: SklearnEngine, model: object) -> object:
    """
    Prepare a model with the settings every test here shares.

    Parameters
    ----------
    engine
        The engine under test.
    model
        The estimator to wrap.

    Returns
    -------
    ModelHandle
        Ready to fit.
    """
    return engine.prepare(
        model,
        hardware=HardwareSpec(device="cpu"),
        training=SklearnTrainingSpec(),
    )


def learnable_source() -> SyntheticTensorSource:
    """
    Build a source whose target is a linear function of its features.

    Linear on purpose: a ridge regression that cannot fit a linear target
    has a bug, so a failure here is never about the problem being hard.

    Returns
    -------
    SyntheticTensorSource
        A re-iterable source over sixty-four samples.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(size=(N_SAMPLES, N_FEATURES))
    weights = np.array([1.5, -2.0, 0.5, 3.0])
    return SyntheticTensorSource(
        features=features,
        targets=(features @ weights + 0.25)[:, None],
        batch_size=16,
    )


class TestTheContract:
    """The clauses every engine must satisfy, whatever is underneath."""

    def test_the_engine_passes_the_conformance_suite(self, tmp_path: Path) -> None:
        """
        The same seven clauses the Torch engine passes.

        Written as one test rather than seven because the suite is the unit
        a user writing their own engine runs; splitting it here would test
        something the framework does not actually offer.
        """
        report = check_engine(
            SklearnEngine(),
            model_factory=lambda: Ridge(alpha=0.1),
            source_factory=learnable_source,
            signature=make_signature(n_features=N_FEATURES),
            directory=tmp_path,
            training=SklearnTrainingSpec(),
            hardware=HardwareSpec(device="cpu"),
        )
        assert report.passed, report.describe()

    def test_the_engine_is_registered_under_its_name(self) -> None:
        """A specification names an engine by string and must resolve it."""
        assert get_engine("sklearn") is SklearnEngine


class TestWhatItClaims:
    """Capabilities are read by the pipeline, so they must be true."""

    def test_no_capability_is_claimed(self) -> None:
        """
        No epochs, no AMP, no compilation and no devices exist here.

        Every ``supports_`` flag is swept rather than named one by one, so
        that a capability added to the dataclass later and defaulted to
        true is caught here instead of in a run that silently takes a path
        this engine cannot follow.
        """
        claimed = asdict(SklearnEngine().capabilities())
        supported = {k: v for k, v in claimed.items() if k.startswith("supports_")}
        assert supported and not any(supported.values()), supported

    def test_the_only_accelerator_offered_is_the_cpu(self) -> None:
        """Trees and linear models run on one kind of hardware."""
        assert SklearnEngine().capabilities().accelerators == ("cpu",)

    def test_the_history_reports_the_one_pass_that_happened(self) -> None:
        """
        One record, not zero and not the configured epoch count.

        Zero would leave a report with an empty curve and no explanation.
        The configured count would be a fabrication: a closed-form fit does
        not iterate, and a flat curve of fifty identical points implies a
        model that stopped improving rather than one that never looped.
        """
        engine = SklearnEngine()
        handle = prepared(engine, Ridge(alpha=0.1))
        outcome = engine.fit(handle, {"train": learnable_source()}, SklearnTrainingSpec())
        assert len(outcome.history) == 1

    def test_the_single_epoch_is_reported_as_the_best(self) -> None:
        """
        There is nothing to restore, so the epoch that ran is the best one.

        A ``restored_best`` of false would tell a reader that checkpointing
        was configured and did not fire, which is not what happened.
        """
        engine = SklearnEngine()
        handle = prepared(engine, Ridge(alpha=0.1))
        outcome = engine.fit(handle, {"train": learnable_source()}, SklearnTrainingSpec())
        assert outcome.restored_best


class TestFitting:
    """What fitting does, and to whom."""

    def test_fitting_mutates_the_caller_s_model(self) -> None:
        """
        The handle's model is the object the caller passed, fitted in place.

        An engine that fits a copy produces a correct ``FitOutcome`` and a
        bundle containing an untrained model. Nothing downstream notices
        until the served predictions are wrong.
        """
        model = Ridge(alpha=0.1)
        engine = SklearnEngine()
        handle = prepared(engine, model)
        engine.fit(handle, {"train": learnable_source()}, SklearnTrainingSpec())
        assert hasattr(model, "coef_")

    def test_the_fit_recovers_the_underlying_weights(self) -> None:
        """
        A sanity check that the columns are not scrambled.

        The conformance suite checks that fitting improves on not fitting,
        which a permuted design matrix also satisfies. This checks the
        actual coefficients, which it does not.
        """
        model = Ridge(alpha=1e-6)
        engine = SklearnEngine()
        handle = prepared(engine, model)
        engine.fit(handle, {"train": learnable_source()}, SklearnTrainingSpec())
        assert np.allclose(model.coef_.ravel(), [1.5, -2.0, 0.5, 3.0], atol=1e-6)

    def test_a_validation_split_is_scored_as_well(self) -> None:
        """
        Validation is scored even though it cannot influence a closed fit.

        It changes nothing about the model, which is why an engine might be
        tempted to skip it -- but every comparison in a report is against
        validation, so an absent number is an absent row.
        """
        engine = SklearnEngine()
        handle = prepared(engine, Ridge(alpha=0.1))
        outcome = engine.fit(
            handle,
            {"train": learnable_source(), "validation": learnable_source()},
            SklearnTrainingSpec(),
        )
        assert outcome.history[0].val_loss is not None


class TestPersistence:
    """A saved model must come back as the same model."""

    def test_weights_round_trip_into_a_fresh_estimator(self, tmp_path: Path) -> None:
        """Predictions are identical, not merely close."""
        engine = SklearnEngine()
        trained = Ridge(alpha=0.1)
        handle = prepared(engine, trained)
        engine.fit(handle, {"train": learnable_source()}, SklearnTrainingSpec())
        before = engine.predict(handle, learnable_source())

        path = tmp_path / "weights.bin"
        engine.save_weights(handle, path)
        restored = prepared(engine, engine.load_weights(Ridge(alpha=0.1), path))

        assert np.array_equal(before, engine.predict(restored, learnable_source()))

    def test_loading_into_a_different_kind_of_estimator_is_refused(self, tmp_path: Path) -> None:
        """
        A ridge's weights are not a forest's, and the shapes may well agree.

        Without this check the load succeeds, the handle holds an estimator
        of the saved type rather than the requested one, and the run reports
        a model it is not using.
        """
        engine = SklearnEngine()
        handle = prepared(engine, Ridge(alpha=0.1))
        engine.fit(handle, {"train": learnable_source()}, SklearnTrainingSpec())
        path = tmp_path / "weights.bin"
        engine.save_weights(handle, path)

        with pytest.raises(EngineError, match=r"Ridge|Lasso"):
            engine.load_weights(Lasso(alpha=0.1), path)


class TestRefusals:
    """Where silence would be more dangerous than an error."""

    def test_a_gpu_request_is_refused(self) -> None:
        """
        scikit-learn has no GPU, so the request cannot be honoured.

        Falling back to the CPU is the tempting behaviour and the wrong one:
        the run completes, and the only symptom is that it took far longer
        than the requester budgeted for.
        """
        engine = SklearnEngine()
        with pytest.raises(EngineError, match=r"cuda|device|cpu"):
            engine.prepare(
                Ridge(alpha=0.1),
                hardware=HardwareSpec(device="cuda"),
                training=SklearnTrainingSpec(),
            )

    def test_an_object_that_cannot_fit_is_refused(self) -> None:
        """
        Duck typing is checked once, here, rather than failing mid-fit.

        A model that reaches ``fit`` without the methods fails after the
        data has been built and drained, which on a real source is the
        expensive part of the run.
        """
        engine = SklearnEngine()
        with pytest.raises(EngineError, match=r"fit|predict|estimator"):
            prepared(engine, object())
```

