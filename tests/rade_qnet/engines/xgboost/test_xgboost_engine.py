"""
Tests for the XGBoost engine.

This engine is the awkward one, and that is why it is here. It is neither
iterative in the way Torch is nor closed-form in the way scikit-learn is: it
loops internally, over a budget the caller sets, and stops early on its own
terms. Nothing in the framework drives that loop.

So the questions these tests ask are mostly about *who owns the loop*. Early
stopping is XGBoost's, not the framework's, because reimplementing it would
mean training to the full budget and discarding rounds. The history is
XGBoost's own ``evals_result`` rather than a curve the framework assembled.
And prediction has to honour ``best_iteration``, because a booster that
stopped early still contains the rounds after the best one and will happily
use them if asked plainly -- which produces a served model measurably worse
than the one the report described, with no error anywhere.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

# Skipped rather than failed when the library is absent, so that a
# contributor on a minimal install sees a skip and a green suite. XGBoost is
# an optional dependency, and a test suite that cannot run without every
# optional dependency installed is a test suite that stops being run.
#
# Checked with ``find_spec`` rather than ``importorskip`` on purpose.
# ``importorskip`` would import ``xgboost`` here, directly, ahead of the
# engine package -- and that is precisely the import order that deadlocks
# against Torch's OpenMP runtime (§8.2). The engine package below imports
# Torch first; reaching around it reintroduces the hang.
if find_spec("xgboost") is None:  # pragma: no cover - depends on the host
    pytest.skip("xgboost is not installed", allow_module_level=True)

from src.rade_qnet.core.lifecycle.components import get_engine
from src.rade_qnet.core.lifecycle.errors import EngineError
from src.rade_qnet.core.spec.hardware import HardwareSpec
from src.rade_qnet.core.spec.training import XGBoostTrainingSpec

# ``xgboost`` itself comes from the engine package rather than from a
# direct import, for the ordering reason above.
from src.rade_qnet.engines.xgboost import BoosterModel, XGBoostEngine
from src.rade_qnet.engines.xgboost.engine import xgb
from src.rade_qnet.testkit.conformance import check_engine
from src.rade_qnet.testkit.fixtures import (
    SyntheticTensorSource,
    make_signature,
)

N_FEATURES = 4
N_SAMPLES = 128


def booster(**params: object) -> BoosterModel:
    """
    Build an unfitted booster holder with shallow, fast trees.

    Parameters
    ----------
    **params
        Booster settings, merged over the defaults.

    Returns
    -------
    BoosterModel
        Unfitted.
    """
    return BoosterModel({"max_depth": 3, **params})


def training(**kwargs: object) -> XGBoostTrainingSpec:
    """
    Build a short training specification.

    Parameters
    ----------
    **kwargs
        Overrides.

    Returns
    -------
    XGBoostTrainingSpec
        A specification with a small round budget.
    """
    return XGBoostTrainingSpec(**{"n_estimators": 20, **kwargs})  # type: ignore[arg-type]


def learnable_source(seed: int = 0) -> SyntheticTensorSource:
    """
    Build a source a shallow tree can fit.

    Parameters
    ----------
    seed
        Seeds the features, so that two calls can differ when a test wants
        a genuinely unseen split.

    Returns
    -------
    SyntheticTensorSource
        A re-iterable source.
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(N_SAMPLES, N_FEATURES))
    target = np.sin(features[:, 0]) + features[:, 1] ** 2 - features[:, 2]
    return SyntheticTensorSource(features=features, targets=target[:, None], batch_size=32)


def prepared(engine: XGBoostEngine, model: object, **kwargs: object) -> object:
    """
    Prepare a model with the settings these tests share.

    Parameters
    ----------
    engine
        The engine under test.
    model
        The booster holder.
    **kwargs
        Overrides for hardware or training.

    Returns
    -------
    ModelHandle
        Ready to fit.
    """
    return engine.prepare(
        model,
        hardware=kwargs.pop("hardware", HardwareSpec(device="cpu")),  # type: ignore[arg-type]
        training=kwargs.pop("training", training()),
        **kwargs,  # type: ignore[arg-type]
    )


class TestTheContract:
    """The clauses every engine must satisfy."""

    def test_the_engine_passes_the_conformance_suite(self, tmp_path: Path) -> None:
        """The same seven clauses Torch and scikit-learn pass."""
        report = check_engine(
            XGBoostEngine(),
            model_factory=booster,
            source_factory=learnable_source,
            signature=make_signature(n_features=N_FEATURES),
            directory=tmp_path,
            training=training(),
            hardware=HardwareSpec(device="cpu"),
        )
        assert report.passed, report.describe()

    def test_the_engine_is_registered_under_its_name(self) -> None:
        """A specification names an engine by string and must resolve it."""
        assert get_engine("xgboost") is XGBoostEngine


class TestTheOpenMpWorkaround:
    """
    The one thing in this package that is about another engine.

    On macOS the Homebrew ``libomp`` XGBoost links and the ``libomp`` the
    Torch wheel bundles are two images of the same library. Whichever loads
    first wins; the loser's thread pool deadlocks on its first parallel
    region, with no error and no traceback. Importing Torch first avoids it.

    This is tested because the symptom is a hang, and a hang is the one
    failure a test suite cannot report usefully: it looks like a slow
    machine until somebody samples the process. A deleted import would be
    found weeks later by a user, so it is pinned here instead.
    """

    def test_importing_the_package_imports_torch_first(self) -> None:
        """
        Torch is in ``sys.modules`` once this package has been imported.

        Asserted on the end state rather than on the order, because order
        cannot be observed after the fact -- and the end state is what the
        loader actually cares about, since the first import is the one that
        decides which image owns the process.
        """
        assert "src.rade_qnet.engines.xgboost" in sys.modules
        assert "torch" in sys.modules

    def test_the_guard_is_still_in_the_source(self) -> None:
        """
        The import survives an import-sorter or an unused-import sweep.

        It looks exactly like a mistake: an unrelated library imported for
        no visible reason and never referenced. The comment above it says
        so, and this test is the second line of defence.
        """
        source = Path("src/rade_qnet/engines/xgboost/__init__.py").read_text()
        assert 'find_spec("torch")' in source
        assert source.index("import torch") < source.index("from .engine")


class TestWhatItClaims:
    """Capabilities are read by the pipeline, so they must be true."""

    def test_checkpointing_is_not_claimed(self) -> None:
        """
        Early stopping is XGBoost's own and keeps no framework checkpoint.

        Claiming it would make the pipeline's checkpoint callback fire and
        write a file nothing reads, which is worse than not claiming it:
        a restore would then appear to be available and would restore
        nothing.
        """
        assert not XGBoostEngine().capabilities().supports_checkpointing

    def test_no_torch_capability_is_claimed(self) -> None:
        """Mixed precision, compilation and distribution are all absent."""
        claimed = asdict(XGBoostEngine().capabilities())
        supported = {k: v for k, v in claimed.items() if k.startswith("supports_")}
        assert supported and not any(supported.values()), supported


class TestFitting:
    """One call to ``xgb.train``, and what it must report back."""

    def test_the_booster_is_attached_to_the_caller_s_model(self) -> None:
        """
        ``xgb.train`` returns a booster rather than mutating one.

        That is why :class:`BoosterModel` exists. If the engine kept the
        returned booster to itself, the handle's model would stay unfitted
        and the bundle would be saved from it.
        """
        model = booster()
        engine = XGBoostEngine()
        engine.fit(prepared(engine, model), {"train": learnable_source()}, training())
        assert model.is_fitted

    def test_the_history_has_one_record_per_round(self) -> None:
        """
        The curve is XGBoost's ``evals_result``, not a fabrication.

        A history shorter than the budget means early stopping fired; one
        longer means the engine is counting something other than rounds.
        """
        engine = XGBoostEngine()
        outcome = engine.fit(
            prepared(engine, booster()),
            {"train": learnable_source()},
            training(n_estimators=20),
        )
        assert len(outcome.history) == 20

    def test_the_training_loss_falls(self) -> None:
        """Boosting that does not reduce training error is not boosting."""
        engine = XGBoostEngine()
        outcome = engine.fit(prepared(engine, booster()), {"train": learnable_source()}, training())
        assert outcome.history[-1].train_loss < outcome.history[0].train_loss

    def test_a_validation_split_is_watched(self) -> None:
        """Every record carries the number early stopping would act on."""
        engine = XGBoostEngine()
        outcome = engine.fit(
            prepared(engine, booster()),
            {"train": learnable_source(), "validation": learnable_source(seed=1)},
            training(),
        )
        assert all(record.val_loss is not None for record in outcome.history)


class TestEarlyStopping:
    """XGBoost's, deliberately, rather than the framework's."""

    def test_stopping_early_shortens_the_history(self) -> None:
        """
        A patience of one on noise stops well short of the budget.

        Validation here is drawn from a different seed, so it is genuinely
        unseen and the booster's improvement on it stops being real quickly.
        """
        engine = XGBoostEngine()
        outcome = engine.fit(
            prepared(engine, booster()),
            {"train": learnable_source(), "validation": learnable_source(seed=7)},
            training(n_estimators=200, early_stopping_rounds=1),
        )
        assert len(outcome.history) < 200

    def test_stopping_early_is_reported_as_a_restore(self) -> None:
        """
        The booster's best iteration is what prediction uses.

        That is a restore in every sense the framework cares about, so
        reporting it as one keeps the field meaning the same thing across
        engines -- which is the whole reason the field is on the contract
        rather than on the Torch engine.
        """
        engine = XGBoostEngine()
        outcome = engine.fit(
            prepared(engine, booster()),
            {"train": learnable_source(), "validation": learnable_source(seed=7)},
            training(n_estimators=200, early_stopping_rounds=1),
        )
        assert outcome.restored_best

    def test_prediction_uses_the_best_iteration_not_the_last(self) -> None:
        """
        The rounds after the best one are still in the booster.

        Predicting without an iteration range uses them. The run then serves
        a model worse than the one its own report described, and nothing
        raises -- the symptom is a validation metric that cannot be
        reproduced from the saved bundle.
        """
        engine = XGBoostEngine()
        model = booster()
        handle = prepared(engine, model)
        engine.fit(
            handle,
            {"train": learnable_source(), "validation": learnable_source(seed=7)},
            training(n_estimators=200, early_stopping_rounds=1),
        )
        assert model.booster is not None
        best = model.booster.best_iteration
        # The booster kept going for at least one round past the best, so
        # an engine ignoring ``best_iteration`` would give a different
        # answer here. If it did not, the test proves nothing and says so.
        assert best + 1 < model.booster.num_boosted_rounds()

        unseen = learnable_source(seed=7)
        matrix = xgb.DMatrix(
            unseen.features,
            feature_names=[f"features_{i}" for i in range(N_FEATURES)],
        )
        every_round = model.booster.predict(matrix)
        assert not np.array_equal(
            engine.predict(handle, learnable_source(seed=7)).ravel(), every_round
        )


class TestPersistence:
    """A saved booster must come back as the same booster."""

    def test_weights_round_trip_exactly(self, tmp_path: Path) -> None:
        """Predictions are identical, not merely close."""
        engine = XGBoostEngine()
        model = booster()
        handle = prepared(engine, model)
        engine.fit(handle, {"train": learnable_source()}, training())
        before = engine.predict(handle, learnable_source())

        path = tmp_path / "weights.bin"
        engine.save_weights(handle, path)
        restored = prepared(engine, engine.load_weights(booster(), path))

        assert np.array_equal(before, engine.predict(restored, learnable_source()))

    def test_the_saved_payload_is_json(self, tmp_path: Path) -> None:
        """
        The format is the engine's decision, not the filename's.

        The framework names every weights file ``weights.bin``, and XGBoost
        guesses its format from the extension -- it would guess UBJSON and
        warn. Saying ``json`` explicitly means the on-disk format is a
        property of this engine rather than of a constant three packages
        away that nobody would think to look at before renaming.
        """
        engine = XGBoostEngine()
        handle = prepared(engine, booster())
        engine.fit(handle, {"train": learnable_source()}, training())
        path = tmp_path / "weights.bin"
        engine.save_weights(handle, path)

        assert json.loads(path.read_text())["learner"] is not None


class TestRefusals:
    """Where silence would be more dangerous than an error."""

    def test_an_unfitted_booster_cannot_predict(self) -> None:
        """There is no booster to ask, so asking is a programming error."""
        engine = XGBoostEngine()
        with pytest.raises(EngineError, match=r"fit|unfitted|not been trained"):
            engine.predict(prepared(engine, booster()), learnable_source())

    def test_another_engine_s_training_spec_is_refused(self) -> None:
        """
        A Torch spec here means the specification names the wrong engine.

        Reading what overlaps and ignoring the rest would run with silently
        different settings -- an epoch budget treated as a round budget, and
        a learning-rate schedule dropped entirely.
        """
        engine = XGBoostEngine()
        with pytest.raises(EngineError, match=r"XGBoostTrainingSpec|engine"):
            engine.prepare(booster(), hardware=HardwareSpec(device="cpu"), training=object())

    def test_static_inputs_are_refused(self) -> None:
        """A graph has no column in a design matrix."""
        engine = XGBoostEngine()
        with pytest.raises(EngineError, match="static"):
            engine.fit(
                prepared(engine, booster()),
                {
                    "train": SyntheticTensorSource(
                        features=np.zeros((16, N_FEATURES)),
                        targets=np.zeros((16, 1)),
                        static={"adjacency": np.zeros((4, 4))},
                    )
                },
                training(),
            )
