# `tranql/models/rade/rade_qnet/rade_qnet/engines/sklearn`

2 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 33 | 1378 | `9687c52982db50ec` |
| 2 | `engine.py` | 541 | 19332 | `3da30705e4f82de9` |

---

## 1. `tranql/models/rade/rade_qnet/rade_qnet/engines/sklearn/__init__.py`

1378 bytes · SHA-256 `9687c52982db50ec`

```python
"""
The scikit-learn engine.

Exists to make simple models genuinely cheap.  A ridge regression should cost
a few lines and still receive the full lifecycle -- leakage-aware splits,
versioned bundles, the standard metrics and reports, and fan-out across a job
set.  A framework that only pays off for large models is a framework people
work around.

Modules
-------
``engine.py``
    Wraps any estimator exposing ``fit`` and ``predict``, persisting it via
    ``joblib`` alongside the framework's own manifest.
``adapters.py``
    Turns a stream of batches into the single matrix a one-shot fit needs.
    Shared with the XGBoost engine, so there is one definition of how a
    source becomes a matrix and one definition of the row order that
    results.

Importing this package registers its components
-----------------------------------------------
Importing ``rade_qnet.engines.sklearn`` registers :class:`SklearnEngine` under
the name ``"sklearn"``, which is what lets a specification name it as a
string.  Eager rather than lazy, for the reason given in the Torch engine's
package docstring: a registry populated only once somebody happens to have
imported the right module is the classic source of "no engine named
'sklearn'" from a configuration that is perfectly correct.
"""

from .engine import ENGINE_NAME, SklearnEngine

__all__ = ["ENGINE_NAME", "SklearnEngine"]
```

---

## 2. `tranql/models/rade/rade_qnet/rade_qnet/engines/sklearn/engine.py`

19332 bytes · SHA-256 `3da30705e4f82de9`

```python
"""
The scikit-learn engine: a closed-form fit behind the same contract.

What this engine is for
-----------------------
It exists so that a ridge regression costs a few lines and still receives the
full lifecycle — a leakage-aware split, a versioned bundle, the standard
metrics and reports, fan-out across a job set, evaluation and inference from
the saved bundle. A framework that only pays off for large models is one
people work around for small ones, and then maintain two of.

It is also half of this phase's honesty test. Almost every concept the Torch
engine introduced is absent here: no epoch, no optimiser, no step, no device,
no gradient. If the ``Engine`` protocol can be satisfied without any of them,
the protocol describes training rather than describing PyTorch.

What it deliberately does not do
---------------------------------
**It does not loop.** ``fit`` drains the training split and makes one call. A
closed-form estimator reaches its answer in that call, and a loop around it
would repeat identical work and report a learning curve that was flat by
construction.

**It does not checkpoint.** There is no intermediate state worth keeping, so
the capability is declared absent and the pipeline skips the stage rather
than being told later that it had no effect.

**It does not pretend about hardware.** Scikit-learn runs on the CPU in
``fp32``. A specification asking for ``bf16`` on a GPU is refused in
``prepare``, not accepted and ignored — see §3.4 of the Phase 6 charter for
why the third option is the dangerous one.

Why ``joblib`` rather than a parameter payload
-----------------------------------------------
The Phase 2 contract says implementations must write a parameter payload, not
a pickled model object. This engine writes a ``joblib`` file, which is a
pickle, and that is a deliberate deviation with a bounded justification.

The rule was written against a specific failure: ``torch.load`` with
``weights_only=False`` over a whole ``nn.Module``, which made every saved
model fragile to a refactor of the class and made loading one equivalent to
executing whatever was inside it. Scikit-learn has no state-dict equivalent —
an estimator's fitted attributes are not enumerable in any supported way —
and ``joblib`` is the library's own documented format. Reimplementing it per
estimator type would be strictly worse and would break on the first estimator
nobody anticipated.

What is recovered is the half of the rule that mattered: :meth:`load_weights`
refuses a payload whose estimator class does not match the model it was
handed. A silent mismatch was the real danger, and that is checked rather
than trusted.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import joblib
import numpy as np

from ...core.contract.result import EpochRecord, FitOutcome
from ...core.lifecycle.components import engine as register_engine
from ...core.lifecycle.errors import EngineError
from ...core.provenance.logging import get_logger
from ...core.spec.training import SklearnTrainingSpec
from ..base import EngineCapabilities, ModelHandle
from ..loaders import drain, reject_static

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from ...core.contract.data import TensorLike
    from ...core.contract.signature import InputSignature
    from ...core.contract.source import BatchSource
    from ...core.spec.hardware import HardwareSpec

__all__ = ["ENGINE_NAME", "SklearnEngine"]

_LOGGER = get_logger(__name__)

#: The name a specification uses to ask for this engine.
ENGINE_NAME = "sklearn"

#: The single record a closed-form fit reports.
#:
#: One rather than zero, because ``best_epoch`` and every curve renderer
#: assume a history with something in it, and an empty history would make
#: "the best epoch" a question with no answer rather than an obvious one.
_THE_ONLY_EPOCH = 0

#: What this engine runs on, and the only thing it runs on.
_DEVICE = "cpu"
_PRECISION = "fp32"


@register_engine(ENGINE_NAME)
class SklearnEngine:
    """
    Drives any estimator exposing ``fit`` and ``predict``.

    Deliberately not restricted to a list of known estimator classes. The
    duck-typed check is scikit-learn's own convention, and honouring it means
    this engine also drives an estimator from ``sklearn``-compatible
    libraries, or one a user wrote, without a change here.
    """

    def capabilities(self) -> EngineCapabilities:
        """
        Declare what this engine supports, which is deliberately little.

        Returns
        -------
        EngineCapabilities
            Everything optional is absent, which is the honest answer and
            the one that lets the pipeline skip stages rather than run them
            into a no-op.
        """
        return EngineCapabilities(
            name=ENGINE_NAME,
            supports_epochs=False,
            supports_validation_during_fit=False,
            supports_checkpointing=False,
            supports_distributed=False,
            supports_lazy_materialisation=False,
            accelerators=(_DEVICE,),
        )

    def materialise(self, model: object, signature: InputSignature) -> object:
        """
        Return the model unchanged, because its parameters are not lazy.

        A no-op rather than an error. The stage exists for models whose
        shapes depend on their first batch, and an estimator's do not: it
        learns them during ``fit`` and has none before.

        Parameters
        ----------
        model
            The untrained estimator.
        signature
            The declared interface, unused.

        Returns
        -------
        object
            The model, unchanged.
        """
        del signature
        return model

    def prepare(
        self,
        model: object,
        *,
        hardware: HardwareSpec,
        training: object,
        static: Mapping[str, TensorLike] | None = None,
    ) -> ModelHandle:
        """
        Check the request is one this engine can honour, and wrap the model.

        There is no device to move to and no optimiser to build, so this
        stage is entirely about refusing things early. Every check here
        catches a specification that would otherwise train successfully and
        mean something other than what it says.

        Parameters
        ----------
        model
            The estimator.
        hardware
            Device, precision, compilation and distribution settings.
        training
            Must be a
            :class:`~rade_qnet.core.spec.training.SklearnTrainingSpec`.
        static
            Static inputs, which this engine cannot consume.

        Returns
        -------
        ModelHandle
            The estimator, with no apparatus because it needs none.

        Raises
        ------
        EngineError
            If the training spec is another engine's, if the estimator does
            not expose ``fit`` and ``predict``, or if the hardware
            specification asks for something trees and linear models do not
            have.
        """
        self._training_spec(training)
        _require_estimator(model)
        reject_static(static or {})
        _reject_unusable_hardware(hardware)

        handle = ModelHandle(
            model=model,
            unwrapped=model,
            device=_DEVICE,
            precision=_PRECISION,
            is_distributed=False,
            static={},
        )
        _LOGGER.debug("prepared %s", handle.describe())
        return handle

    def fit(
        self,
        handle: ModelHandle,
        sources: Mapping[str, BatchSource],
        training: object,
        *,
        on_epoch_end: object = None,
    ) -> FitOutcome:
        """
        Fit the estimator in one call and report it as a single epoch.

        The validation split is scored afterwards rather than during, so the
        record carries a ``val_loss`` that is comparable with a neural
        network's final epoch. Scoring it costs one extra prediction pass and
        makes the two engines' histories readable on the same axes, which is
        the point of this phase's comparison.

        Parameters
        ----------
        handle
            The prepared estimator.
        sources
            One source per split. ``train`` is required.
        training
            The engine's training spec, for ``fit_params``.
        on_epoch_end
            Invoked with the single record, if given. Called even though
            there is only one, so a hook that counts epochs sees the same
            thing it would from any other engine.

        Returns
        -------
        FitOutcome
            A one-record history.

        Raises
        ------
        EngineError
            If the training split is absent or cannot be drained.
        """
        spec = self._training_spec(training)
        train = sources.get("train")
        if train is None:
            raise EngineError(
                f"no 'train' source was supplied; this engine received "
                f"{sorted(sources)} and has nothing to fit on"
            )

        started = time.perf_counter()
        split = drain(train)
        _LOGGER.info("fitting %s on %s", type(handle.unwrapped).__name__, split.describe())
        # Fitted in place, so the caller's model is the fitted one. An engine
        # that fitted a copy would satisfy every shape check and leave the
        # handle -- and therefore the bundle -- holding untrained weights.
        handle.unwrapped.fit(split.features, split.target, **dict(spec.fit_params))

        train_loss = _mean_squared_error(
            np.asarray(handle.unwrapped.predict(split.features)), split.target
        )
        record = EpochRecord(
            epoch=_THE_ONLY_EPOCH,
            train_loss=train_loss,
            val_loss=self._validation_loss(handle, sources.get("validation")),
            seconds=time.perf_counter() - started,
        )
        if callable(on_epoch_end):
            on_epoch_end(record)

        return FitOutcome(
            history=(record,),
            monitor="val_loss" if record.val_loss is not None else "train_loss",
            best_epoch=_THE_ONLY_EPOCH,
            best_monitor_value=(
                record.val_loss if record.val_loss is not None else record.train_loss
            ),
            stopped_early=False,
            # True, and not a white lie: with one fit there is exactly one set
            # of parameters, so the ones in hand are by definition the best
            # ones. Reporting False would suggest the saved weights and the
            # reported metrics might describe different models, which is the
            # question this field exists to answer.
            restored_best=True,
            total_seconds=record.seconds,
        )

    def predict(self, handle: ModelHandle, source: BatchSource) -> NDArray[np.floating]:
        """
        Predict over a source, in the source's own row order.

        Parameters
        ----------
        handle
            The fitted estimator.
        source
            Batches to predict over. Must be bounded.

        Returns
        -------
        numpy.ndarray
            One row per sample, in source order, in the model's own output
            space. Inverting the target transform is the caller's job.

        Raises
        ------
        EngineError
            If the source is unbounded or the estimator is not fitted.
        """
        split = drain(source, require_target=False)
        try:
            values = handle.unwrapped.predict(split.features)
        except Exception as error:
            raise EngineError(
                f"{type(handle.unwrapped).__name__}.predict failed [{type(error).__name__}] {error}"
            ) from error
        return np.asarray(values, dtype=np.float64).reshape(split.n_samples, -1)

    def save_weights(self, handle: ModelHandle, path: Path) -> None:
        """
        Write the fitted estimator in scikit-learn's own format.

        Parameters
        ----------
        handle
            The fitted estimator. ``unwrapped`` is saved, as the contract
            requires, though this engine never wraps anything.
        path
            Destination file, inside a staging directory.
        """
        joblib.dump(handle.unwrapped, path)

    def load_weights(self, model: object, path: Path) -> object:
        """
        Load an estimator saved by :meth:`save_weights`.

        Returns the loaded object rather than copying attributes into the
        model it was handed, because an estimator's fitted attributes are not
        enumerable in any supported way. The contract permits this: it says
        the caller must use the return value rather than assume the argument
        was mutated, which exists for precisely this case.

        Parameters
        ----------
        model
            A freshly built estimator of the same class. Used to check the
            payload rather than to receive it.
        path
            File written by :meth:`save_weights`.

        Returns
        -------
        object
            The fitted estimator.

        Raises
        ------
        EngineError
            If the file cannot be read, or holds an estimator of a different
            class than the model expected. The second case is the one that
            matters: a bundle opened against the wrong architecture would
            otherwise predict confidently from a model nobody asked for.
        """
        try:
            loaded = joblib.load(path)
        except Exception as error:
            raise EngineError(
                f"could not read the estimator at {path} [{type(error).__name__}] {error}"
            ) from error

        if type(loaded) is not type(model):
            raise EngineError(
                f"{path} holds a {type(loaded).__name__} but the bundle was "
                f"opened against a {type(model).__name__}. Loading it anyway "
                f"would produce confident predictions from a model the "
                f"specification did not describe"
            )
        return loaded

    def _validation_loss(self, handle: ModelHandle, source: BatchSource | None) -> float | None:
        """
        Score the validation split after fitting, if there is one.

        Parameters
        ----------
        handle
            The fitted estimator.
        source
            The validation source, or ``None``.

        Returns
        -------
        float or None
            Mean squared error, or ``None`` when there is nothing to score.
        """
        if source is None:
            return None
        split = drain(source)
        predictions = np.asarray(handle.unwrapped.predict(split.features))
        return _mean_squared_error(predictions, split.target)

    @staticmethod
    def _training_spec(training: object) -> SklearnTrainingSpec:
        """
        Narrow the protocol's opaque training argument to this engine's spec.

        Parameters
        ----------
        training
            Whatever the pipeline passed through.

        Returns
        -------
        SklearnTrainingSpec
            The narrowed spec.

        Raises
        ------
        EngineError
            If it belongs to another engine, which means the specification
            names one engine and configures another.
        """
        if not isinstance(training, SklearnTrainingSpec):
            raise EngineError(
                f"the {ENGINE_NAME!r} engine needs a SklearnTrainingSpec and "
                f"received a {type(training).__name__}; the specification's "
                f"training block names a different engine"
            )
        return training


def _require_estimator(model: object) -> None:
    """
    Check the model is something this engine can drive.

    Duck-typed rather than an ``isinstance`` test against
    ``sklearn.base.BaseEstimator``, which is scikit-learn's own convention
    and means this engine also drives a compatible estimator from another
    library or one a user wrote.

    Parameters
    ----------
    model
        The candidate estimator.

    Raises
    ------
    EngineError
        If it does not expose both ``fit`` and ``predict``.
    """
    missing = [name for name in ("fit", "predict") if not callable(getattr(model, name, None))]
    if missing:
        raise EngineError(
            f"{type(model).__name__} does not expose {missing}, so the "
            f"{ENGINE_NAME!r} engine cannot drive it. An estimator needs a "
            f"callable fit and predict"
        )


def _reject_unusable_hardware(hardware: HardwareSpec) -> None:
    """
    Refuse hardware settings this engine has no way to honour.

    The distinction drawn here is between *cannot* and *this machine happens
    not to*. An absent GPU degrades to the CPU elsewhere in the framework,
    because the run would otherwise have succeeded and the user's intent is
    served by running it. Asking a closed-form estimator for mixed precision
    is different in kind: there is no such thing for it, so accepting the
    setting would mean the specification says something untrue about the
    model it produced.

    Parameters
    ----------
    hardware
        The requested hardware specification.

    Raises
    ------
    EngineError
        If precision, compilation or distribution were requested.
    """
    unusable = [
        message
        for condition, message in (
            (
                hardware.precision != _PRECISION,
                f"precision={hardware.precision!r} (this engine computes in {_PRECISION})",
            ),
            (hardware.compile_model, "compile_model=True (there is no graph to compile)"),
            (
                hardware.distributed != "none",
                f"distributed={hardware.distributed!r} (a single fit cannot be sharded)",
            ),
        )
        if condition
    ]
    if unusable:
        raise EngineError(
            f"the {ENGINE_NAME!r} engine cannot honour {unusable}. Refused "
            f"rather than ignored: a run that quietly dropped these would "
            f"complete, report plausible metrics, and leave the specification "
            f"describing a model that was never trained"
        )
    if hardware.device not in {"auto", _DEVICE}:
        raise EngineError(
            f"the {ENGINE_NAME!r} engine cannot honour device="
            f"{hardware.device!r}; it runs on the {_DEVICE} and nothing else. "
            f"Use 'auto' or {_DEVICE!r}"
        )


def _mean_squared_error(predictions: NDArray[np.floating], targets: NDArray[np.floating]) -> float:
    """
    Return the mean squared error, for the fit history.

    Computed here rather than imported from ``analysis.metrics``, which is a
    one-way dependency this package may not take: ``engines`` sits below
    ``analysis``. The duplication is three lines and the alternative is a
    cycle.

    Parameters
    ----------
    predictions, targets
        Equal-length arrays.

    Returns
    -------
    float
        The mean squared error.
    """
    difference = np.ravel(predictions) - np.ravel(targets)
    return float(np.mean(difference * difference))
```

