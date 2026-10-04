"""
Turn a stream of batches into the single matrix a one-shot fit needs.

Why a one-shot engine is handed a stream at all
-----------------------------------------------
It looks backwards. A tree does not iterate, so handing it an iterator and
asking it to put the pieces back together is work that only exists because
of how the data arrived.

The alternative was a second payload type carrying the whole split at once,
and it was built in Phase 1 and retired in Phase 6. The short version is that
the mismatch it bridged does not exist: ``sources`` imports no training
library, so a batch is already a mapping of NumPy arrays, and the
concatenation here is one copy that XGBoost's ``DMatrix`` construction was
going to make anyway. The long version, including why the second payload
would have forked the lifecycle, is in the Phase 6 charter §3.1.

What matters is that this file is the *only* place the conversion happens.
Both one-shot engines import it, so there is one definition of how a stream
becomes a matrix and one definition of the row order that results — which is
the property every downstream attribution depends on, and the one that would
be quietly violated if two engines each grew their own.

Why order is the whole contract
-------------------------------
``Engine.predict`` promises predictions in source order, because that is what
lets a number be traced back to a scenario and an instrument. The order here
is the order the source yields, and nothing sorts, shuffles or regroups. A
training split's source may well be shuffled — that is its business — which
is why scoring uses :func:`~rade_qnet.orchestration.stages.scoring.
scoring_source` to obtain a stable view first. This module does not try to
fix that, because a module that silently reordered its input would make the
stable-order machinery above it untestable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.contract.data import TARGET_KEY
from ..core.lifecycle.errors import EngineError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from ..core.contract.source import BatchSource

__all__ = ["DrainedSplit", "drain", "flatten", "reject_static"]

#: Rank of the matrix a one-shot estimator consumes.
_MATRIX_RANK = 2


class DrainedSplit:
    """
    One split, materialised as a feature matrix and a target vector.

    Deliberately not a frozen dataclass in ``core``: it never crosses a
    pipeline boundary and never reaches a contract. It exists for the few
    lines between draining a source and calling ``fit``, and giving it a
    home in ``core`` would reintroduce exactly the second payload type this
    phase retired.

    Parameters
    ----------
    features
        Two-dimensional, samples by features, in source order.
    target
        One-dimensional, one value per sample, in the same order. Flattened
        because every estimator this engine drives predicts a scalar, and a
        column vector produces a ``DataConversionWarning`` on every fit.
    feature_names
        Column names where the signature supplied them. Carried because tree
        models report importances positionally, and an importance table keyed
        by index is nearly useless.
    """

    __slots__ = ("feature_names", "features", "target")

    def __init__(
        self,
        features: NDArray[np.float64],
        target: NDArray[np.float64],
        feature_names: tuple[str, ...] | None = None,
    ) -> None:
        self.features = features
        self.target = target
        self.feature_names = feature_names

    @property
    def n_samples(self) -> int:
        """
        Number of rows.

        Returns
        -------
        int
            The row count, which both arrays agree on by construction.
        """
        return int(self.features.shape[0])

    def describe(self) -> str:
        """
        Return a compact summary, for logs.

        Returns
        -------
        str
            For example ``240 x 12``.
        """
        return f"{self.n_samples} x {self.features.shape[1]}"


def flatten(array: NDArray[np.floating]) -> NDArray[np.float64]:
    """
    Reduce a batch's input to two dimensions, samples by features.

    A sequence model's input arrives as ``(samples, timesteps, features)``,
    and a one-shot estimator has no notion of a time axis. Flattening every
    axis after the first turns one timestep-feature pair into one column,
    which is the standard way to put a windowed problem in front of a tree:
    the model loses the knowledge that column 7 and column 19 are the same
    quantity at different lags, and gains the ability to be fitted at all.

    Stating that plainly matters, because it is the main reason a tabular
    baseline underperforms a recurrent model on the same data, and reading
    that difference as "the recurrent architecture is better" when it is
    partly "the baseline was handed a worse representation" would be the
    wrong conclusion to draw from this phase's comparison.

    Parameters
    ----------
    array
        An input of rank one or more.

    Returns
    -------
    numpy.ndarray
        Two-dimensional, with the leading axis preserved.
    """
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.ndim == _MATRIX_RANK:
        return values
    return values.reshape(values.shape[0], -1)


def drain(source: BatchSource, *, require_target: bool = True) -> DrainedSplit:
    """
    Consume a bounded source once and return the whole split.

    Parameters
    ----------
    source
        The split's source. Must be bounded: a one-shot fit over an endless
        stream has no point at which to start.
    require_target
        Whether a missing target is an error. True when fitting, false when
        predicting -- a source built for inference has no labels, and
        demanding them would make a trained model unusable on live data.

    Returns
    -------
    DrainedSplit
        Features, target and column names, in source order.

    Raises
    ------
    EngineError
        If the source is unbounded, yields nothing, yields batches whose
        inputs disagree, or omits a target that was required.
    """
    if source.steps_per_epoch is None:
        raise EngineError(
            "a one-shot engine needs a bounded source: this one reports no "
            "step count, so there is no last batch and nothing to fit against. "
            "An unbounded source belongs to an interactive learner"
        )

    # Checked here as well as in the engines' ``prepare``, because this is
    # the function that actually drops them: ``_input_names`` reads only the
    # signature's dynamic inputs, so a static one would vanish into a run
    # that succeeds with an input missing. The engines check early to fail
    # before the data is built; this checks where the loss would happen.
    reject_static(source.static)

    names = _input_names(source)
    columns: list[list[NDArray[np.float64]]] = [[] for _ in names]
    targets: list[NDArray[np.float64]] = []

    for index, batch in enumerate(source.batches()):
        for position, name in enumerate(names):
            if name not in batch:
                raise EngineError(
                    f"batch {index} is missing the {name!r} input that the "
                    f"source's signature declares; the feature matrix would "
                    f"have a different width than the one before it"
                )
            columns[position].append(flatten(batch[name]))
        if TARGET_KEY in batch:
            targets.append(np.ravel(np.asarray(batch[TARGET_KEY], dtype=np.float64)))

    if not any(columns) or not columns[0]:
        raise EngineError(
            f"the source yielded no batches, so there is nothing to fit. It "
            f"reports {source.steps_per_epoch} step(s), which means the "
            f"count and the iteration disagree"
        )

    features = np.hstack([np.vstack(parts) for parts in columns])
    target = np.concatenate(targets) if targets else np.empty(0, dtype=np.float64)

    if require_target and target.size == 0:
        raise EngineError(
            f"the source yields no {TARGET_KEY!r}, so there is nothing to fit "
            f"towards. A source without labels can be predicted over but not "
            f"trained on"
        )
    if target.size and target.size != features.shape[0]:
        raise EngineError(
            f"the source yielded {features.shape[0]} feature row(s) but "
            f"{target.size} target value(s). A model fitted on a misaligned "
            f"pair learns a permutation and reports a plausible loss while "
            f"doing it"
        )

    return DrainedSplit(features, target, _feature_names(source, features.shape[1]))


def _input_names(source: BatchSource) -> tuple[str, ...]:
    """
    Return the dynamic inputs to concatenate, in a fixed order.

    Sorted rather than taken in declaration order, so that two runs of one
    configuration produce the same column layout. A model whose columns
    depend on dictionary ordering would have weights that silently stop
    matching their own feature names.

    Parameters
    ----------
    source
        The split's source.

    Returns
    -------
    tuple of str
        Input names, excluding the target.

    Raises
    ------
    EngineError
        If the signature declares no dynamic inputs.
    """
    names = tuple(sorted(name for name in source.signature.dynamic if name != TARGET_KEY))
    if not names:
        raise EngineError(
            "the source's signature declares no dynamic inputs, so there are "
            "no features to fit on. A model whose inputs are entirely static "
            "has one sample, not a dataset"
        )
    return names


def _feature_names(source: BatchSource, n_columns: int) -> tuple[str, ...] | None:
    """
    Name the matrix's columns where the signature makes that possible.

    An input wider than one column is expanded into one name per column,
    ``features_0``, ``features_1`` and so on, using the widths the signature
    declares. Naming only the case where every input happens to be a single
    scalar would leave the common case -- one wide feature block --
    permanently unnamed, which is the case a coefficient table is read for.

    The separator is an underscore rather than a bracket subscript because
    XGBoost rejects ``[``, ``]`` and ``<`` in feature names outright, and
    these adapters are shared by both engines. A naming scheme one of them
    refuses is not a naming scheme.

    The widths come from the signature rather than from the matrix, so they
    are checked against it: if the two disagree, something upstream flattened
    differently than declared, and ``None`` is returned. A positional
    importance table is unhelpful; a *mislabelled* one is worse, because it
    is read with confidence.

    Parameters
    ----------
    source
        The split's source.
    n_columns
        Columns the drained matrix actually has.

    Returns
    -------
    tuple of str or None
        One name per column, or ``None`` if they cannot be matched.
    """
    dynamic = source.signature.dynamic
    names: list[str] = []
    for name in _input_names(source):
        # Everything after the sample axis is what flattening collapsed, so
        # its product is the column count this input contributes.
        width = int(np.prod(dynamic[name].shape[1:])) if dynamic[name].shape[1:] else 1
        names.extend([name] if width == 1 else [f"{name}_{i}" for i in range(width)])
    return tuple(names) if len(names) == n_columns else None


def reject_static(static: Mapping[str, object]) -> None:
    """
    Refuse static inputs, which a one-shot estimator cannot consume.

    Parameters
    ----------
    static
        Static inputs carried on the handle.

    Raises
    ------
    EngineError
        If any are present. A graph or an entity table has no column in a
        feature matrix, and quietly dropping it would fit the model on less
        than the specification asked for -- which produces a worse model and
        no indication that anything was lost.
    """
    if static:
        raise EngineError(
            f"this engine cannot consume static inputs, and the data build "
            f"supplied {sorted(static)}. A one-shot estimator has one feature "
            f"matrix and no second channel to put them in. Refused rather "
            f"than dropped, because a dropped input produces a quietly worse "
            f"model that still trains and still reports a loss"
        )
