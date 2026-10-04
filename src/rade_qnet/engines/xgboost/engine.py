"""
The XGBoost engine: the hardest test of the engine contract.

Why this engine is the interesting one
---------------------------------------
Every concept the Torch engine introduced is absent here. There is no epoch
loop, no optimiser, no gradient, no device, no learning rate schedule, no
checkpoint. A booster is trained by one call to ``xgboost.train`` and comes
back finished.

If the ``Engine`` protocol can accommodate that and still produce the same
bundle, the same metrics and the same reports as a neural network, then it
describes *training* rather than describing PyTorch with a generic name. That
claim had never been tested, because until this phase there was one engine,
and an interface with one implementation is not an interface.

The translation this file performs
-----------------------------------
Three things have to be mapped, and each has a wrong answer that would look
fine::

    boosting round   ->  EpochRecord     not "no history", which would make
                                         best_epoch meaningless and leave the
                                         curve renderer with nothing to draw

    native early     ->  stopped_early   not a framework reimplementation,
    stopping             + best_epoch    which cannot see inside the booster
                                         and would fight the one that can

    booster JSON     ->  save_weights    not a pickle, which would make the
                                         saved model fragile to a refactor of
                                         a class it does not even contain

The first is the one worth dwelling on. A boosting round and an epoch are the
same thing for every purpose this framework has: a unit of progress carrying
a training loss, optionally a validation loss, and an index that can be
plotted. Reporting them as such means ``analysis.visuals`` renders a boosting
curve with no knowledge that it is one, and a tree model's learning curve sits
on the same axes as a network's. Inventing a separate "boosting history"
concept would have meant a second renderer, a second report, and a second
thing for a reader to learn.

What is declared absent
-----------------------
``supports_epochs`` is false even though the history has many records, and
that is not a contradiction. The flag asks whether training proceeds in passes
*the framework can observe between* and intervene in -- which is what
framework-level early stopping and checkpointing need. XGBoost's rounds happen
inside one library call, so the framework sees them only afterwards, as a
report. Declaring true would invite a monitored callback that never fires.

Why the library is imported at module scope
--------------------------------------------
``xgboost`` is an optional dependency and this module is where the
optionality lives. Importing it here means a host that never trains trees
never imports it and never pays for it, and a host that does gets a clear
``ImportError`` at the import rather than an obscure failure mid-run. On
macOS the wheel additionally needs an OpenMP runtime that pip does not
install, which is a second reason not to make it a hard requirement.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import xgboost as xgb

from ...core.contract.result import EpochRecord, FitOutcome
from ...core.lifecycle.components import engine as register_engine
from ...core.lifecycle.errors import EngineError
from ...core.provenance.logging import get_logger
from ...core.spec.training import XGBoostTrainingSpec
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

__all__ = ["ENGINE_NAME", "BoosterModel", "XGBoostEngine"]

_LOGGER = get_logger(__name__)

#: The name a specification uses to ask for this engine.
ENGINE_NAME = "xgboost"

#: What this engine runs on. XGBoost can use a GPU, and this engine does not
#: expose that yet: ``device="cuda"`` changes the tree method and therefore
#: the numbers, so offering it without a parity test would be offering a
#: silent change of model. Listed as a known gap rather than a limitation.
_DEVICE = "cpu"
_PRECISION = "fp32"

#: The split XGBoost watches for early stopping, and the key it reports under.
_WATCHED = "validation"
_TRAIN = "train"


class BoosterModel:
    """
    A booster and the settings it will be trained with.

    Every other engine receives a model object that already exists. A booster
    does not: ``xgboost.train`` *returns* one, so there is nothing to hand to
    ``prepare`` and nothing to save until after the fit.

    This class is the small adapter that makes the lifecycle work anyway. A
    model definition returns one of these carrying its hyper-parameters, the
    engine fills in the booster during ``fit``, and ``save_weights`` writes
    the booster out. It is the minimum needed to let a thing that is created
    by training pass through stages that assume a thing that is created
    before it.

    Parameters
    ----------
    params
        Booster parameters, as XGBoost names them. Held rather than merged
        into the training spec because they describe what the model *is* --
        its depth, its regularisation -- which is the model's business, while
        the spec carries how long to train it for.

    Attributes
    ----------
    booster
        ``None`` until fitted. The attribute exists from construction so that
        a reader can see that an unfitted model is a real state rather than
        an error, and so ``load_weights`` has somewhere to put a booster it
        read from disk.
    """

    __slots__ = ("booster", "params")

    def __init__(self, params: Mapping[str, object] | None = None) -> None:
        self.params: dict[str, object] = dict(params or {})
        self.booster: xgb.Booster | None = None

    @property
    def is_fitted(self) -> bool:
        """
        Whether this model has a booster to predict with.

        Returns
        -------
        bool
            True once trained or loaded.
        """
        return self.booster is not None

    def describe(self) -> str:
        """
        Return a compact summary, for logs.

        Returns
        -------
        str
            For example ``BoosterModel(fitted, 120 tree(s))``.
        """
        if self.booster is None:
            return "BoosterModel(unfitted)"
        return f"BoosterModel(fitted, {self.booster.num_boosted_rounds()} round(s))"


@register_engine(ENGINE_NAME)
class XGBoostEngine:
    """
    Trains gradient-boosted trees behind the framework's engine contract.

    One call, many rounds, reported as a history the rest of the framework
    cannot distinguish from a neural network's.
    """

    def capabilities(self) -> EngineCapabilities:
        """
        Declare what this engine supports.

        Returns
        -------
        EngineCapabilities
            ``supports_epochs`` is false despite the multi-record history:
            the flag asks whether the framework can intervene *between*
            passes, and these happen inside one library call.
        """
        return EngineCapabilities(
            name=ENGINE_NAME,
            supports_epochs=False,
            # False for the same reason: the booster scores the validation
            # split every round, but it does so internally. Nothing the
            # framework provides can observe or act on it mid-fit.
            supports_validation_during_fit=False,
            supports_checkpointing=False,
            supports_distributed=False,
            supports_lazy_materialisation=False,
            accelerators=(_DEVICE,),
        )

    def materialise(self, model: object, signature: InputSignature) -> object:
        """
        Return the model unchanged; a booster has no lazy shapes.

        Parameters
        ----------
        model
            The untrained model.
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
        Check the request can be honoured and wrap the model.

        Parameters
        ----------
        model
            A :class:`BoosterModel`.
        hardware
            Device, precision, compilation and distribution settings.
        training
            Must be an
            :class:`~rade_qnet.core.spec.training.XGBoostTrainingSpec`.
        static
            Static inputs, which this engine cannot consume.

        Returns
        -------
        ModelHandle
            The model, with no apparatus because it needs none.

        Raises
        ------
        EngineError
            If the spec belongs to another engine, the model is not a
            :class:`BoosterModel`, or the hardware asks for something trees
            do not have.
        """
        self._training_spec(training)
        if not isinstance(model, BoosterModel):
            raise EngineError(
                f"the {ENGINE_NAME!r} engine drives a BoosterModel and received "
                f"a {type(model).__name__}. A booster is produced by training "
                f"rather than constructed before it, so the model a definition "
                f"returns has to be the holder rather than the booster itself"
            )
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
        Train in one call and report every boosting round as an epoch.

        Parameters
        ----------
        handle
            The prepared model.
        sources
            One source per split. ``train`` is required; ``validation``
            enables the library's early stopping.
        training
            The engine's training spec.
        on_epoch_end
            Invoked with each record *after* training, not during. The
            booster does not surrender control mid-fit, so a hook that
            expects live updates will see them all at once -- which is why
            :meth:`capabilities` declares no epoch support, and why that
            declaration is worth reading before relying on the hook.

        Returns
        -------
        FitOutcome
            One record per boosting round.

        Raises
        ------
        EngineError
            If the training split is absent or cannot be drained.
        """
        spec = self._training_spec(training)
        model = handle.unwrapped
        train = sources.get(_TRAIN)
        if train is None:
            raise EngineError(
                f"no {_TRAIN!r} source was supplied; this engine received "
                f"{sorted(sources)} and has nothing to fit on"
            )

        started = time.perf_counter()
        matrices = {name: _matrix(drain(source)) for name, source in sources.items()}
        watchlist = [(matrices[_TRAIN], _TRAIN)]
        if _WATCHED in matrices:
            watchlist.append((matrices[_WATCHED], _WATCHED))

        evaluations: dict[str, dict[str, list[float]]] = {}
        model.booster = xgb.train(
            {**_booster_params(spec), **model.params},
            matrices[_TRAIN],
            num_boost_round=spec.n_estimators,
            evals=watchlist,
            evals_result=evaluations,
            # The library's own, which runs per round inside the booster with
            # access to state the framework cannot see. Disabled when there is
            # no validation split, because stopping on training error selects
            # the most overfitted round available.
            early_stopping_rounds=(spec.early_stopping_rounds if _WATCHED in matrices else None),
            verbose_eval=False,
        )

        history = _history_from(evaluations, seconds=time.perf_counter() - started)
        if callable(on_epoch_end):
            for record in history:
                on_epoch_end(record)

        return self._outcome(model.booster, history, watched=_WATCHED in matrices, spec=spec)

    def predict(self, handle: ModelHandle, source: BatchSource) -> NDArray[np.floating]:
        """
        Predict over a source, in the source's own row order.

        Parameters
        ----------
        handle
            The fitted model.
        source
            Batches to predict over. Must be bounded.

        Returns
        -------
        numpy.ndarray
            One row per sample, in source order.

        Raises
        ------
        EngineError
            If the model has not been fitted, or the source is unbounded.
        """
        model = handle.unwrapped
        if not isinstance(model, BoosterModel) or model.booster is None:
            raise EngineError(
                "this model has no booster: it has neither been fitted nor "
                "loaded from a bundle. Predicting would require inventing one"
            )
        split = drain(source, require_target=False)
        # `iteration_range` to the best round, so a model stopped early
        # predicts with the rounds that earned the reported metric rather
        # than with the extra ones that followed it downhill.
        best = getattr(model.booster, "best_iteration", None)
        values = model.booster.predict(
            _matrix(split),
            iteration_range=(0, best + 1) if best is not None else (0, 0),
        )
        return np.asarray(values, dtype=np.float64).reshape(split.n_samples, -1)

    def save_weights(self, handle: ModelHandle, path: Path) -> None:
        """
        Write the booster in XGBoost's own JSON format.

        Not a pickle, which satisfies the Phase 2 rule rather than deviating
        from it as the scikit-learn engine must. The format is documented and
        stable, so a bundle written here is readable by anyone with the
        library and no copy of this framework -- which is a property worth
        having when a model outlives the platform that trained it.

        Written through ``save_raw`` rather than ``save_model`` because the
        bundle's weights file is named ``weights.bin`` by ``storage``, and
        ``save_model`` picks its format from the extension. Handed an
        unfamiliar one it guesses, says so in a warning, and writes UBJSON --
        which works, and means the format on disk is decided by a filename
        this engine does not choose. Naming it here makes the bundle's
        contents a property of the engine rather than of a constant three
        packages away.

        Parameters
        ----------
        handle
            The fitted model.
        path
            Destination file, inside a staging directory.

        Raises
        ------
        EngineError
            If the model was never fitted. Writing an empty bundle would
            produce something that loads and cannot predict.
        """
        model = handle.unwrapped
        if not isinstance(model, BoosterModel) or model.booster is None:
            raise EngineError(
                "there is no booster to save: the model was never fitted. A "
                "bundle written now would load successfully and fail at the "
                "first prediction"
            )
        path.write_bytes(model.booster.save_raw(raw_format="json"))

    def load_weights(self, model: object, path: Path) -> object:
        """
        Load a booster into a freshly built model.

        Parameters
        ----------
        model
            A fresh :class:`BoosterModel` of the same configuration.
        path
            File written by :meth:`save_weights`.

        Returns
        -------
        object
            The model, holding the loaded booster.

        Raises
        ------
        EngineError
            If the model is not a :class:`BoosterModel`, or the file cannot
            be read as a booster.
        """
        if not isinstance(model, BoosterModel):
            raise EngineError(
                f"expected a BoosterModel to load into and received a {type(model).__name__}"
            )
        booster = xgb.Booster()
        try:
            # From bytes rather than from the path, for the same reason
            # `save_weights` writes them: the loader would otherwise infer a
            # format from an extension that says nothing about one.
            booster.load_model(bytearray(path.read_bytes()))
        except Exception as error:
            raise EngineError(
                f"could not read a booster from {path} [{type(error).__name__}] {error}"
            ) from error
        model.booster = booster
        return model

    @staticmethod
    def _outcome(
        booster: xgb.Booster,
        history: tuple[EpochRecord, ...],
        *,
        watched: bool,
        spec: XGBoostTrainingSpec,
    ) -> FitOutcome:
        """
        Assemble the result, reading the best round from the booster.

        Taken from the booster rather than recomputed from the history,
        because the booster is the authority on which round it stopped at and
        a second opinion is a second thing that can be wrong.

        Parameters
        ----------
        booster
            The trained booster.
        history
            One record per round.
        watched
            Whether a validation split was supplied.
        spec
            The training spec, for the configured round budget.

        Returns
        -------
        FitOutcome
            History, best round and whether it stopped early.
        """
        best = getattr(booster, "best_iteration", None)
        if best is None and history:
            best = len(history) - 1
        stopped_early = bool(
            watched and spec.early_stopping_rounds is not None and len(history) < spec.n_estimators
        )
        return FitOutcome(
            history=history,
            monitor="val_loss" if watched else "train_loss",
            best_epoch=best,
            best_monitor_value=(
                _monitor_value(history[best], watched=watched)
                if best is not None and best < len(history)
                else None
            ),
            stopped_early=stopped_early,
            # True because `predict` honours `best_iteration`, so the model
            # that serves is the model that earned the reported metric. An
            # engine that reported True while predicting with every round
            # would be describing a different model than it saved.
            restored_best=True,
            total_seconds=sum(record.seconds for record in history),
        )

    @staticmethod
    def _training_spec(training: object) -> XGBoostTrainingSpec:
        """
        Narrow the protocol's opaque training argument to this engine's spec.

        Parameters
        ----------
        training
            Whatever the pipeline passed through.

        Returns
        -------
        XGBoostTrainingSpec
            The narrowed spec.

        Raises
        ------
        EngineError
            If it belongs to another engine.
        """
        if not isinstance(training, XGBoostTrainingSpec):
            raise EngineError(
                f"the {ENGINE_NAME!r} engine needs an XGBoostTrainingSpec and "
                f"received a {type(training).__name__}; the specification's "
                f"training block names a different engine"
            )
        return training


def _matrix(split: object) -> xgb.DMatrix:
    """
    Build the library's own matrix type from a drained split.

    Parameters
    ----------
    split
        A :class:`~rade_qnet.engines.loaders.DrainedSplit`.

    Returns
    -------
    xgboost.DMatrix
        Features, target where present, and column names where they could be
        matched -- so an importance table names instruments rather than
        integers.
    """
    return xgb.DMatrix(
        split.features,
        label=split.target if split.target.size else None,
        feature_names=list(split.feature_names) if split.feature_names else None,
    )


def _booster_params(spec: XGBoostTrainingSpec) -> dict[str, object]:
    """
    Translate the framework's training spec into booster parameters.

    A translation and nothing more: every name on the right is XGBoost's own,
    which is why the spec uses those names too. A spec that invented its own
    vocabulary would make a user who knows the library learn a mapping, and
    would make this function a place where the mapping could be wrong.

    Parameters
    ----------
    spec
        The validated training spec.

    Returns
    -------
    dict
        Parameters for ``xgboost.train``.
    """
    return {
        "max_depth": spec.max_depth,
        "eta": spec.learning_rate,
        "subsample": spec.subsample,
        "colsample_bytree": spec.colsample_bytree,
        "min_child_weight": spec.min_child_weight,
        "lambda": spec.reg_lambda,
        "objective": spec.objective,
        "device": _DEVICE,
    }


def _history_from(
    evaluations: Mapping[str, Mapping[str, list[float]]], *, seconds: float
) -> tuple[EpochRecord, ...]:
    """
    Turn the library's evaluation log into the framework's history.

    Parameters
    ----------
    evaluations
        What ``xgboost.train`` wrote into ``evals_result``: split name to
        metric name to one value per round.
    seconds
        Total wall time, divided evenly across rounds. Approximate and
        labelled as such: the library does not report per-round timing, and
        the alternative is leaving every record at zero, which would make a
        duration report silently wrong rather than evenly smoothed.

    Returns
    -------
    tuple of EpochRecord
        One record per boosting round.
    """
    train_losses = _first_metric(evaluations.get(_TRAIN, {}))
    validation_losses = _first_metric(evaluations.get(_WATCHED, {}))
    per_round = seconds / len(train_losses) if train_losses else 0.0

    return tuple(
        EpochRecord(
            epoch=index,
            train_loss=float(value),
            val_loss=(float(validation_losses[index]) if index < len(validation_losses) else None),
            seconds=per_round,
        )
        for index, value in enumerate(train_losses)
    )


def _first_metric(scores: Mapping[str, list[float]]) -> list[float]:
    """
    Return the first metric the booster recorded for a split.

    The booster records one entry per evaluation metric, and with a single
    objective there is exactly one. Taking the first rather than looking up a
    name means this does not have to know that ``reg:squarederror`` reports
    ``rmse`` -- a mapping that would need extending for every objective.

    Parameters
    ----------
    scores
        Metric name to one value per round.

    Returns
    -------
    list of float
        The per-round values, or an empty list if nothing was recorded.
    """
    for values in scores.values():
        return list(values)
    return []


def _monitor_value(record: EpochRecord, *, watched: bool) -> float:
    """
    Return the value the best round is best by.

    Parameters
    ----------
    record
        The best round's record.
    watched
        Whether a validation split was supplied.

    Returns
    -------
    float
        The validation loss when there was one, otherwise the training loss.
    """
    if watched and record.val_loss is not None:
        return record.val_loss
    return record.train_loss


def _reject_unusable_hardware(hardware: HardwareSpec) -> None:
    """
    Refuse hardware settings this engine has no way to honour.

    Parameters
    ----------
    hardware
        The requested hardware specification.

    Raises
    ------
    EngineError
        If precision, compilation or distribution were requested, or a
        device other than the CPU.
    """
    unusable = [
        message
        for condition, message in (
            (
                hardware.precision != _PRECISION,
                f"precision={hardware.precision!r} (a booster has no mixed-precision mode)",
            ),
            (hardware.compile_model, "compile_model=True (there is no graph to compile)"),
            (
                hardware.distributed != "none",
                f"distributed={hardware.distributed!r} (use XGBoost's own "
                f"distributed training, which this engine does not yet expose)",
            ),
        )
        if condition
    ]
    if unusable:
        raise EngineError(
            f"the {ENGINE_NAME!r} engine cannot honour {unusable}. Refused "
            f"rather than ignored: a run that quietly dropped these would "
            f"complete and report metrics for a model the specification does "
            f"not describe"
        )
    if hardware.device not in {"auto", _DEVICE}:
        raise EngineError(
            f"the {ENGINE_NAME!r} engine runs on the {_DEVICE} only. XGBoost "
            f"can use a GPU, and exposing it here would change the tree method "
            f"and therefore the numbers, so it is withheld until there is a "
            f"parity test rather than offered as a silent change of model"
        )
