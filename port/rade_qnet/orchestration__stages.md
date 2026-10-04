# `src/rade_qnet/orchestration/stages`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 49 | 2391 | `d4555a56cb073632` |
| 2 | `reload.py` | 273 | 9485 | `625a003bdd93daa9` |
| 3 | `resolve.py` | 125 | 4829 | `52a0bf45099efed5` |
| 4 | `scoring.py` | 450 | 16206 | `2c240ae5ec2ff681` |
| 5 | `search.py` | 245 | 7996 | `f3ffe459960cd03e` |

---

## 1. `src/rade_qnet/orchestration/stages/__init__.py`

2391 bytes · SHA-256 `d4555a56cb073632`

```python
"""
The work a pipeline stage does, when more than one pipeline does it.

Why this is not part of ``pipelines``
--------------------------------------
It was, and that was the problem.  ``pipelines`` held five ``Pipeline``
subclasses and four modules that were not pipelines at all, so a folder named
for a noun contained things that were not that noun and ``ls`` stopped being
an answer to "what can this framework do?".

The split is along a real line rather than a tidy one.  A *pipeline* is a
named lifecycle a user can invoke: train, evaluate, infer, tune, reinforce.
A *stage* is a step inside one, and the four modules here are the steps that
more than one lifecycle performs identically.  Evaluation and inference both
reload a bundle; training, evaluation and tuning all turn a prepared dataset
into something scoreable.

Shared as functions, not as a base class
-----------------------------------------
That choice is older than this package and worth restating, because the
alternative looks cheaper every time.  ``EvaluatePipeline`` and
``InferPipeline`` share their reload path, and a common base class would
express that in fewer lines.  It would also mean that changing how inference
handles an unseen entity could silently change what evaluation reports, with
nothing in either file to suggest it might.  A function has one caller
relationship and it is visible at the call site.

Modules
-------
``resolve.py``
    Which pipeline class runs a given model's lifecycle, honouring a model's
    own overrides.  Read by the public API and by a job unit.
``reload.py``
    A model, its fitted state and its signature, back from a bundle
    directory.  Read by evaluation and inference, and the one place that
    refuses an interactive bundle to a supervised reader by name.
``scoring.py``
    Turning a prepared dataset into a ``BatchSource`` for one split,
    collecting targets, and aligning predictions back to the rows they came
    from.  Read by training, evaluation and tuning.
``search.py``
    Proposing hyper-parameter trials from a tuning specification, by grid or
    at random, and expanding a flat trial into nested specification blocks.
    Read by tuning only -- it lives here because it is a *stage*, not because
    it is shared, and because leaving it in ``tune.py`` made that module the
    largest pipeline by a third.
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/orchestration/stages/reload.py`

9485 bytes · SHA-256 `625a003bdd93daa9`

```python
"""
Opening a saved bundle and turning it back into something runnable.

A bundle on disk is a directory of files. Evaluation, inference and any later
comparison all need the same thing from it: the specification it was trained
from, the model definition that interprets its weights, the fitted state its
transforms live in, and the lineage that records how its data was split.

Shared as a module rather than as a pipeline base class. Two pipelines
calling the same function stay in step; two pipelines inheriting a common
base diverge the first time one of them needs a stage the other does not --
and evaluation and inference differ in exactly that way, because one has
targets and the other does not.

Why a bundle is not self-describing
-----------------------------------
The manifest records a model *name*, not a Python import path. Phase 1 chose
that deliberately: a recorded class path means renaming a class breaks every
bundle written before the rename. The cost is that loading a bundle requires
the model's package to be importable, so that the name resolves through the
registry.

That cost is not worth engineering away, because it is not really a cost.
Weights are meaningless without the architecture that interprets them, so any
process that can use a bundle has the model code anyway. What *is* worth
getting right is the error. A registry miss from a bundle load should say
that the bundle names a model this process has not imported -- which is the
same fault as Phase 4's defect 12, reached from the other direction -- rather
than reporting a bare lookup failure that leaves the reader guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...core.authoring.definition import PredictorDefinition
from ...core.authoring.supervised import RebuildableDataModule
from ...core.lifecycle.components import MODELS, ComponentError, get_model
from ...core.lifecycle.errors import BundleError
from ...core.provenance.logging import get_logger
from ...core.spec.run import SupervisedRunSpec
from ...storage.bundle import (
    load_fitted_state,
    load_lineage,
    load_signature,
    load_spec,
    open_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ...core.contract.bundle import SavedBundle
    from ...core.contract.data import DataLineage
    from ...core.contract.signature import InputSignature
    from ...core.contract.state import FittedState

__all__ = ["LoadedBundle", "load_bundle"]

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedBundle:
    """
    Everything a pipeline needs to put a saved model back to work.

    Parameters
    ----------
    saved
        The located bundle: its directory and verified manifest.
    spec
        The run specification the model was trained from.
    definition
        The model definition, resolved from the manifest's model name.
    state
        The fitted state, to be applied and never re-fitted.
    lineage
        The data lineage, supplying the split that must not be re-derived.
    signature
        The input signature the model was built against, used to check that
        rebuilt data still presents the interface the weights expect.
    """

    saved: SavedBundle
    spec: SupervisedRunSpec
    definition: PredictorDefinition
    state: FittedState
    lineage: DataLineage
    signature: InputSignature

    @property
    def model_name(self) -> str:
        """
        Return the name the model is registered under.

        Returns
        -------
        str
            From the manifest, so it is what was recorded rather than what
            the resolved class happens to be called now.
        """
        return self.saved.manifest.model_name

    def describe(self) -> str:
        """
        Return a one-line description, for a log and a report header.

        Returns
        -------
        str
            Model name, bundle version and the split sizes it was trained
            against.
        """
        sizes = {name: len(values) for name, values in self.lineage.split_indices.items()}
        return (
            f"{self.model_name} v{self.saved.manifest.version} "
            f"(spec {self.saved.manifest.spec_digest[:8]}, splits {sizes})"
        )


def load_bundle(directory: Path, *, verify: bool = True) -> LoadedBundle:
    """
    Open a saved bundle and resolve everything needed to run it again.

    Parameters
    ----------
    directory
        The bundle directory.
    verify
        Whether to re-hash every file against the manifest. Defaults to
        true, because a bundle about to produce numbers someone will act on
        is exactly the case where the check is worth its cost.

    Returns
    -------
    LoadedBundle
        The specification, definition, fitted state, lineage and signature.

    Raises
    ------
    BundleError
        If the bundle is missing or inconsistent, if its model is not
        registered in this process, or if its definition cannot supply the
        fitted-state type needed to read ``fitted_state/``.
    """
    saved = open_bundle(directory, verify=verify)
    spec = load_spec(saved)

    if not isinstance(spec, SupervisedRunSpec):
        raise BundleError(
            f"the bundle at {directory} holds a {type(spec).__name__}; only "
            f"supervised runs can be evaluated or served through this path"
        )

    definition = _definition_for(saved.manifest.model_name, directory)
    state = load_fitted_state(saved, _state_type_for(definition, spec, directory))
    lineage = load_lineage(saved)
    signature = load_signature(saved)

    loaded = LoadedBundle(
        saved=saved,
        spec=spec,
        definition=definition,
        state=state,
        lineage=lineage,
        signature=signature,
    )
    _LOGGER.info("loaded %s", loaded.describe())
    return loaded


def _definition_for(model_name: str, directory: Path) -> PredictorDefinition:
    """
    Resolve a manifest's model name into an instantiated definition.

    Parameters
    ----------
    model_name
        The name recorded in the manifest.
    directory
        The bundle directory, named in errors so the reader knows which
        bundle could not be opened.

    Returns
    -------
    PredictorDefinition
        A fresh instance of the registered definition.

    Raises
    ------
    BundleError
        If the name is not registered here, or is registered to something
        that is not a predictor.
    """
    try:
        model_type = get_model(model_name)
    except ComponentError as error:
        raise BundleError(
            f"the bundle at {directory} was trained by a model named "
            f"{model_name!r}, which is not registered in this process. A "
            f"model registers when its package is imported, so import the "
            f"package that defines it before loading the bundle. "
            f"Registered here: {sorted(MODELS.names()) or 'nothing'}"
        ) from error

    definition = model_type()
    if not isinstance(definition, PredictorDefinition):
        raise BundleError(
            f"the bundle at {directory} names the model {model_name!r}, which "
            f"is registered as a {type(definition).__name__}; only a predictor "
            f"can be evaluated or served through this path"
        )
    return definition


def _state_type_for(
    definition: PredictorDefinition, spec: SupervisedRunSpec, directory: Path
) -> type[FittedState]:
    """
    Ask a definition for the concrete type its fitted state loads into.

    The type is not recorded on disk, by the same decision that keeps class
    paths out of the manifest. The data module declares it, so the data
    module is asked -- which is why loading a bundle needs the model package
    and not merely the framework.

    Parameters
    ----------
    definition
        The resolved model definition.
    spec
        The bundle's run specification, needed because a definition may
        choose its data module from the spec.
    directory
        The bundle directory, named in errors.

    Returns
    -------
    type[FittedState]
        The class to read ``fitted_state/`` into.

    Raises
    ------
    BundleError
        If the definition cannot supply a data module that declares one.
    """
    data_module = getattr(definition, "data_module", None)
    if data_module is None:
        raise BundleError(
            f"the model {type(definition).__name__} has no data_module(), so "
            f"nothing can say what type the fitted state in {directory} loads "
            f"into. A model that builds its data by a bespoke route must "
            f"expose the module that knows how to rebuild it"
        )

    module = data_module(spec)
    if not isinstance(module, RebuildableDataModule):
        raise BundleError(
            f"{type(definition).__name__}.data_module() returned a "
            f"{type(module).__name__}, which has no rebuild(); the bundle at "
            f"{directory} can be opened but its model cannot be re-fed the "
            f"way it was trained, so any metric from it would be meaningless"
        )

    state_type = getattr(module, "state_type", None)
    if not isinstance(state_type, type):
        raise BundleError(
            f"{type(module).__name__} declares no state_type, so the fitted "
            f"state in {directory} cannot be read back"
        )
    return state_type
```

---

## 3. `src/rade_qnet/orchestration/stages/resolve.py`

4829 bytes · SHA-256 `52a0bf45099efed5`

```python
"""
Pick the pipeline class a model wants for a given lifecycle stage.

Why this module exists
----------------------
A model declares its overrides on its framework definition::

    class HybridGnnRnnModel:
        pipelines = MappingProxyType({"train": HybridTrainPipeline, ...})

That declaration had, until this module, no reader. Every override was
reachable only by importing the subclass and instantiating it by hand,
which is what the phase examples did -- so the overrides worked, were
tested, and were nevertheless unreachable through ``api.train`` and every
other documented entry point. A user following the documentation got the
framework's base pipeline and a flagship model quietly missing its graph
diagnostics.

That is the worst shape a defect can take: the feature exists, its tests
pass, and the only thing wrong is that nothing connects it. This module is
the connection, and it is deliberately one function so there is exactly one
place where the lookup can be got wrong.

Why the override must be a subclass
------------------------------------
The caller has already decided which lifecycle it is running, and has a
contract with that pipeline's return type: ``api.evaluate`` promises an
:class:`~rade_qnet.core.contract.result.EvaluationResult`, and a model that
returned something else would break a caller that never asked for a custom
pipeline in the first place. Requiring a subclass is what makes "the model
customises training" different from "the model replaces training".

The check also catches the likeliest mistake by far, which is a model
listing a class under the wrong key -- a tuning pipeline under ``"eval"``
reads perfectly well in a dictionary literal and is nonsense at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.lifecycle.errors import ComponentError
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["pipeline_for"]

_LOGGER = get_logger(__name__)

#: The lifecycle keys a model may declare. Checked so that a typo -- a
#: model declaring ``"evaluate"`` where the framework reads ``"eval"`` --
#: is reported rather than silently ignored, which would present as the
#: override simply not running.
LIFECYCLES: frozenset[str] = frozenset({"train", "eval", "infer", "tune"})


def pipeline_for[PipelineT: type](
    definition: object, lifecycle: str, default: PipelineT
) -> PipelineT:
    """
    Return the pipeline class to run, honouring the model's override.

    Parameters
    ----------
    definition
        The model's framework definition, or ``None`` when the caller has
        none -- a search over a model that is resolved per trial, for
        instance. A definition that declares no overrides is as ordinary
        as one that declares some.
    lifecycle
        Which pipeline is wanted: ``train``, ``eval``, ``infer`` or
        ``tune``.
    default
        The framework's own pipeline for that lifecycle, returned when the
        model declares no override.

    Returns
    -------
    type
        The override if the model declares one, otherwise ``default``.

    Raises
    ------
    ComponentError
        If the model declares an override under an unknown lifecycle key,
        or one that is not a subclass of the framework's pipeline.
    """
    if lifecycle not in LIFECYCLES:
        raise ComponentError(
            f"{lifecycle!r} is not a lifecycle; expected one of {sorted(LIFECYCLES)}"
        )

    overrides: Mapping[str, type] = getattr(definition, "pipelines", None) or {}
    unknown = set(overrides) - LIFECYCLES
    if unknown:
        raise ComponentError(
            f"{type(definition).__name__} declares pipeline override(s) under "
            f"{sorted(unknown)}, which no lifecycle reads. The override would "
            f"never run, and the model would appear to work while silently "
            f"using the framework's pipeline. Expected keys: {sorted(LIFECYCLES)}"
        )

    override = overrides.get(lifecycle)
    if override is None:
        return default

    if not (isinstance(override, type) and issubclass(override, default)):
        raise ComponentError(
            f"{type(definition).__name__} declares {override!r} for the "
            f"{lifecycle!r} lifecycle, but it is not a subclass of "
            f"{default.__name__}. A pipeline that does not extend the "
            f"framework's own has no obligation to return what the caller was "
            f"promised, which turns a model's customisation into a broken "
            f"contract for every caller that never asked for one"
        )

    _LOGGER.debug(
        "using %s's %r pipeline override %s",
        type(definition).__name__,
        lifecycle,
        override.__name__,
    )
    return override
```

---

## 4. `src/rade_qnet/orchestration/stages/scoring.py`

16206 bytes · SHA-256 `2c240ae5ec2ff681`

```python
"""
Turning a fitted model and a data bundle into scored metrics.

Extracted from :class:`~rade_qnet.orchestration.pipelines.train.TrainPipeline`
when evaluation arrived, because of a requirement that reads as a one-line
test and is actually a design constraint:

    Re-evaluating a saved bundle must reproduce the metrics recorded in it.

Two implementations of scoring cannot satisfy that for long. They would agree
on the day they were written and then drift -- one gains a guard, the other
gains a different one, and the first anybody hears of it is a re-evaluation
that disagrees with a bundle by a fraction nobody can account for. So there
is one implementation, and both pipelines call it.

What makes scoring harder than it looks
---------------------------------------
Every helper here exists because of a failure that produces a *number* rather
than an error:

- **Two passes must line up.** Scoring iterates a source twice, once for the
  forward pass and once for the targets. A shuffled source reorders between
  them, so each prediction is compared against an unrelated target. The counts
  still match and every metric still computes; the only symptom is a model
  that appears to score at random on the split it was trained on.
  :func:`scoring_source` routes through
  :class:`~rade_qnet.core.contract.source.OrderedSource`, and verifies rather
  than assumes.

- **The source decides which rows exist.** A sequence model discards the first
  ``length - 1`` rows of each split, because no complete window ends there.
  Targets read from the dataset would include them, misaligning every metric
  by a few rows -- degrading a score rather than breaking it.
  :func:`collect_targets` reads through the source for that reason.

- **Shapes that broadcast.** NumPy will expand an ``(n, 1)`` against an
  ``(n,)`` into an ``(n, n)``, and the mean of that is a plausible number
  with no meaning. :func:`align` checks instead.

- **Units.** Metrics are computed after
  :meth:`~rade_qnet.core.contract.state.FittedState.inverse_transform_targets`,
  so they are in the target's original units. An error in standardised space
  is not a quantity anyone can act on, and two runs' standardised errors --
  each scaled by its own training mean -- are not comparable to each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...analysis.metrics.regression import baseline_metrics, regression_metrics
from ...core.contract.data import TensorBatchData
from ...core.contract.result import EvalResult
from ...core.contract.source import BatchSource, OrderedSource
from ...core.lifecycle.errors import ContractError
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from ...core.contract.data import DataBundle

__all__ = [
    "EVALUATED_SPLITS",
    "FEATURE_MATRIX_RANK",
    "TARGET_KEY",
    "TRAIN_SPLIT",
    "align",
    "collect_targets",
    "feature_matrix",
    "order_is_stable",
    "score_splits",
    "scoring_source",
    "source_for",
    "static_inputs",
]

_LOGGER = get_logger(__name__)

#: The split a model learns from, and the one whose statistics define the
#: baseline. Named rather than written inline because three functions need it
#: to mean the same thing.
TRAIN_SPLIT = "train"

#: The splits worth scoring, in report order. Training is included on purpose:
#: a model that scores well on train and badly on validation has overfitted,
#: and that is a different problem from one that scores badly on both. Without
#: the training score the two are indistinguishable in the saved bundle.
EVALUATED_SPLITS: tuple[str, ...] = ("train", "validation", "test")

#: The batch key holding the observed values.
TARGET_KEY = "target"

#: The rank a feature array must have for the flat quality metrics to mean
#: anything: one row per sample, one column per feature.
FEATURE_MATRIX_RANK = 2


def score_splits(
    data: DataBundle[object],
    predict: Callable[[BatchSource], NDArray[np.floating]],
    *,
    splits: tuple[str, ...] = EVALUATED_SPLITS,
) -> dict[str, EvalResult]:
    """
    Score a model on every available split, in the target's original units.

    The one implementation of scoring, called both at the end of training and
    when a saved bundle is re-evaluated. Takes the forward pass as a callable
    rather than an engine and a handle, so that the caller decides how
    predictions are produced -- training already holds a prepared handle,
    while evaluation has just rebuilt one -- without this function needing to
    know about either.

    Parameters
    ----------
    data
        The data bundle, supplying the sources and the fitted state that
        inverts the target transform.
    predict
        Runs a forward pass over one source and returns its raw output, in
        the model's own output space. Inverting is done here, so a caller
        that inverted too would invert twice.
    splits
        Which splits to score, in report order. Any that the bundle does not
        carry are skipped rather than raising: a run configured without a
        validation split is a legitimate configuration, not an error.

    Returns
    -------
    dict
        Split name to result, each with its metrics and a naive baseline.
    """
    # The training targets, in original units, for the `mean` baseline.
    # Deliberately the training mean rather than the scored split's: using
    # the latter would give the baseline information the model did not have,
    # making it unbeatable and therefore useless as a reference.
    train_source = scoring_source(data, TRAIN_SPLIT)
    train_targets = collect_targets(data, train_source, split=TRAIN_SPLIT)

    evaluations: dict[str, EvalResult] = {}
    for name in splits:
        if name not in data.splits:
            continue
        source = scoring_source(data, name)
        predictions = data.state.inverse_transform_targets(
            np.asarray(predict(source), dtype=np.float64)
        )
        targets = collect_targets(data, source, split=name)
        predictions, targets = align(predictions, targets, split=name)

        evaluations[name] = EvalResult(
            split=name,
            metrics=regression_metrics(predictions, targets),
            n_samples=int(targets.shape[0]),
            in_original_units=True,
            baseline_metrics=baseline_metrics(
                targets, strategy="mean", train_targets=train_targets
            ),
        )
        _LOGGER.info(
            "scored %s on %d sample(s): %s",
            name,
            evaluations[name].n_samples,
            {key: round(value, 6) for key, value in evaluations[name].metrics.items()},
        )
    return evaluations


def static_inputs(data: DataBundle[object]) -> Mapping[str, object]:
    """
    Return the training split's static inputs.

    Parameters
    ----------
    data
        The data bundle.

    Returns
    -------
    Mapping
        Input name to tensor. Empty for a model with no static inputs, which
        is most of them.
    """
    return source_for(data, TRAIN_SPLIT).static


def source_for(data: DataBundle[object], split: str) -> BatchSource:
    """
    Return one split's batch source.

    Parameters
    ----------
    data
        The data bundle.
    split
        Split name.

    Returns
    -------
    BatchSource
        The split's source.

    Raises
    ------
    ContractError
        If the split's payload is not a batch source. The tensor engines
        consume :class:`~rade_qnet.core.contract.data.TensorBatchData`, whose
        ``loader`` is the source; anything else means the model's data
        build produced a payload for a different engine.
    """
    payload = data.split(split)
    if isinstance(payload, TensorBatchData):
        loader = payload.loader
        if isinstance(loader, BatchSource):
            return loader
        raise ContractError(
            f"the {split!r} split's loader is a {type(loader).__name__}, which "
            f"does not satisfy BatchSource; a gradient engine needs a source "
            f"it can re-iterate and ask for a batch count"
        )
    if isinstance(payload, BatchSource):
        return payload
    raise ContractError(
        f"the {split!r} split carries a {type(payload).__name__}, which is "
        f"neither TensorBatchData nor a BatchSource. A gradient engine cannot "
        f"consume it; check that the model's data build targets this engine"
    )


def scoring_source(data: DataBundle[object], split: str) -> BatchSource:
    """
    Return one split's source in a form safe to traverse twice.

    Scoring takes two passes over a split -- one for the forward pass and
    one to collect the targets -- and pairs the results row for row. A
    training source reshuffles between passes, so pairing them directly
    compares each prediction against an unrelated target. The sample
    counts still match and every metric still computes, so the only
    symptom is a model that appears to score at random on the split it was
    trained on.

    Routes through
    :class:`~rade_qnet.core.contract.source.OrderedSource` where the source
    offers it, and otherwise verifies that the source is already stable
    rather than assuming it.

    Parameters
    ----------
    data
        The data bundle.
    split
        Split name.

    Returns
    -------
    BatchSource
        A source whose passes line up.

    Raises
    ------
    ContractError
        If the source's order varies between passes and it offers no
        ordered view. Raised rather than scored, because a plausible wrong
        number is worse than no number.
    """
    source = source_for(data, split)
    if isinstance(source, OrderedSource):
        source = source.ordered()

    if not order_is_stable(source):
        raise ContractError(
            f"the {split!r} source yields its samples in a different order on "
            f"each pass, and does not implement OrderedSource.ordered(). "
            f"Scoring needs two passes to line up, so every metric for this "
            f"split would be computed against mismatched targets. Implement "
            f"ordered() to return a stable-order view"
        )
    return source


def collect_targets(
    data: DataBundle[object], source: BatchSource, *, split: str
) -> NDArray[np.float64]:
    """
    Collect one split's targets, in the target's original units.

    Read by iterating the source rather than from the dataset directly,
    for a reason worth stating: the source decides which rows are usable.
    A sequence model discards the first ``length - 1`` rows of every split
    because no complete window ends there, and targets taken from the
    dataset would include them -- silently misaligning every metric by a
    few rows in a way that degrades a score rather than breaking it.

    Parameters
    ----------
    data
        The data bundle, for the fitted state that inverts the target.
    source
        The split's source. Taken as a parameter rather than re-derived
        from ``data``, because the caller has already resolved it to a
        stable-order view and re-deriving would hand back the shuffled
        training source -- which is exactly the misalignment
        :func:`scoring_source` exists to prevent.
    split
        Split name, for messages.

    Returns
    -------
    numpy.ndarray
        Targets, one row per sample, inverse-transformed.

    Raises
    ------
    ContractError
        If a batch has no target, or the source yielded no batches.
    """
    collected: list[NDArray[np.float64]] = []
    for batch in source.batches():
        if TARGET_KEY not in batch:
            raise ContractError(
                f"a batch from the {split!r} source has keys {sorted(batch)} and "
                f"no {TARGET_KEY!r}; a split cannot be scored without targets"
            )
        collected.append(np.asarray(batch[TARGET_KEY], dtype=np.float64))

    if not collected:
        raise ContractError(
            f"the {split!r} source yielded no batches, so it cannot be scored. "
            f"A batch size larger than the split with drop_last set produces "
            f"this"
        )
    return data.state.inverse_transform_targets(np.concatenate(collected, axis=0))


def feature_matrix(data: DataBundle[object]) -> NDArray[np.float64] | None:
    """
    Collect the training split's features as one matrix, if it is flat.

    Returns ``None`` rather than raising for a source whose features are
    not a flat sample-by-feature matrix -- a sequence model's windows, a
    graph model's node table. Quality metrics for those are the model's own
    business, and a framework that guessed would report a completeness
    figure over the wrong axis.

    Read through the ordered view, because one of the quality metrics is
    order-dependent: staleness counts rows that repeat the row before them,
    and over a shuffled pass that is a measure of nothing. Completeness and
    coverage would survive the shuffle, which is what makes this the kind
    of mistake that produces three plausible numbers and one meaningless
    one.

    Parameters
    ----------
    data
        The data bundle.

    Returns
    -------
    numpy.ndarray or None
        The feature matrix, or ``None`` if the shape is not flat.
    """
    source = scoring_source(data, TRAIN_SPLIT)
    blocks: list[NDArray[np.float64]] = []
    for batch in source.batches():
        for name in sorted(batch):
            if name == TARGET_KEY:
                continue
            array = np.asarray(batch[name], dtype=np.float64)
            if array.ndim != FEATURE_MATRIX_RANK:
                return None
            blocks.append(array)
    if not blocks:
        return None
    return np.concatenate(blocks, axis=0)


def order_is_stable(source: BatchSource) -> bool:
    """
    Return whether two passes over a source yield targets in the same order.

    Compares the targets rather than the features because they are the smaller
    array by a wide margin -- one column against a window of many -- and any
    reordering that would misalign a metric reorders both.

    Parameters
    ----------
    source
        The source to check.

    Returns
    -------
    bool
        True if two passes agree. True for an unbounded source, which cannot
        be checked this way and is never scored by this pipeline anyway.
    """
    if source.steps_per_epoch is None:
        return True
    passes = [
        np.concatenate(
            [np.ravel(np.asarray(batch[TARGET_KEY])) for batch in source.batches()],
            axis=0,
        )
        for _ in range(2)
    ]
    first, second = passes
    return first.shape == second.shape and bool(np.array_equal(first, second))


def align(
    predictions: NDArray[np.float64], targets: NDArray[np.float64], *, split: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Reduce predictions and targets to matching one-dimensional arrays.

    Parameters
    ----------
    predictions
        Model output, in original units.
    targets
        Observed values, in original units.
    split
        Split name, for the message.

    Returns
    -------
    tuple
        The two arrays, flattened.

    Raises
    ------
    ContractError
        If the sample counts disagree. Checked rather than broadcast, because
        NumPy will happily broadcast a ``(n, 1)`` against an ``(n,)`` into an
        ``(n, n)`` and the resulting metric is a number that looks plausible.
    """
    flat_predictions = np.ravel(predictions)
    flat_targets = np.ravel(targets)
    if flat_predictions.shape != flat_targets.shape:
        raise ContractError(
            f"on the {split!r} split the model produced {flat_predictions.size} "
            f"prediction(s) for {flat_targets.size} target(s). Equal counts are "
            f"required; a mismatch usually means the source dropped rows the "
            f"target collection kept, or the model's output has an extra axis"
        )
    return flat_predictions, flat_targets
```

---

## 5. `src/rade_qnet/orchestration/stages/search.py`

7996 bytes · SHA-256 `f3ffe459960cd03e`

```python
"""
Proposing points in a search space.

Separated from the pipeline that consumes them because proposing and
evaluating are independent concerns, and keeping them apart means a sampler
can be tested exhaustively without training anything -- which matters, since
the properties worth asserting about a sampler (it is reproducible, it stays
in bounds, it does not repeat a grid point) are all cheap to check and all
expensive to discover from a search that merely finished.

Reproducibility
---------------
Every sampler draws from a seeded generator created once per search, not per
trial. A per-trial generator seeded by the trial number would also be
reproducible, and would be worse: the sequences for adjacent trials would be
correlated in whatever way the underlying algorithm correlates adjacent
seeds, and a search would explore less of the space than its budget suggests
while looking exactly like one that explored more.

Dotted paths
------------
A dimension names ``training.learning_rate``; a run specification nests. The
expansion happens here, once, in :func:`expand`, and the result goes through
the same validated merge a job set's overrides do -- so a path that does not
exist fails at trial construction rather than being explored as though it
were a real knob. That is the structural cure for defect 7.
"""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np

from ...core.lifecycle.errors import SpecError
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from ...core.spec.tune import Dimension, TuneSpec

__all__ = ["expand", "propose"]

_LOGGER = get_logger(__name__)


def propose(spec: TuneSpec) -> list[dict[str, Any]]:
    """
    Produce the flat proposals for a whole search, in trial order.

    Produced up front rather than one at a time. A search that drew its next
    point only when the previous one finished would be a prerequisite for
    adaptive sampling, which this does not do -- and producing them all now
    buys two things that matter more today: the proposals can be logged and
    compared before anything trains, and a grid search can say honestly how
    many distinct points it actually has.

    Parameters
    ----------
    spec
        The search specification.

    Returns
    -------
    list of dict
        One flat mapping of dotted path to value per trial.
    """
    if spec.sampler == "grid":
        return _grid(spec)
    return _random(spec)


def expand(flat: Mapping[str, Any]) -> dict[str, Any]:
    """
    Turn a flat mapping of dotted paths into the nested shape a spec expects.

    Parameters
    ----------
    flat
        Dotted path to value, such as ``{"training.learning_rate": 0.01}``.

    Returns
    -------
    dict
        The nested equivalent, such as
        ``{"training": {"learning_rate": 0.01}}``.

    Raises
    ------
    SpecError
        If two paths disagree about whether a segment is a mapping --
        ``training`` and ``training.learning_rate`` in the same space, say.
        One of the two would overwrite the other depending on iteration
        order, so the search would explore a different space on a different
        day.
    """
    nested: dict[str, Any] = {}
    for path, value in flat.items():
        segments = path.split(".")
        cursor = nested
        for segment in segments[:-1]:
            existing = cursor.setdefault(segment, {})
            if not isinstance(existing, dict):
                raise SpecError(
                    f"the search path {path!r} needs {segment!r} to be a mapping, "
                    f"but another path in the same space sets it to a value. "
                    f"Vary one or the other, not both"
                )
            cursor = existing
        leaf = segments[-1]
        if isinstance(cursor.get(leaf), dict):
            raise SpecError(
                f"the search path {path!r} sets a value, but another path in the "
                f"same space treats it as a mapping. Vary one or the other"
            )
        cursor[leaf] = value
    return nested


def _random(spec: TuneSpec) -> list[dict[str, Any]]:
    """
    Draw independent points from the space.

    Parameters
    ----------
    spec
        The search specification.

    Returns
    -------
    list of dict
        One proposal per trial.
    """
    # One generator for the whole search, for the reason in the module
    # docstring.
    generator = np.random.default_rng(spec.seed)
    proposals = [
        {dimension.path: _draw(dimension, generator) for dimension in spec.space.dimensions}
        for _ in range(spec.trials)
    ]
    _LOGGER.info(
        "proposed %d random trial(s) over %d dimension(s): %s",
        len(proposals),
        len(spec.space.dimensions),
        list(spec.space.paths),
    )
    return proposals


def _grid(spec: TuneSpec) -> list[dict[str, Any]]:
    """
    Enumerate the product of the axes, truncated to the trial budget.

    Parameters
    ----------
    spec
        The search specification.

    Returns
    -------
    list of dict
        One proposal per grid point, at most ``spec.trials`` of them.
    """
    points: Iterator[tuple[Any, ...]] = product(
        *(dimension.values for dimension in spec.space.dimensions)
    )
    proposals = [
        dict(zip(spec.space.paths, values, strict=True))
        for _, values in zip(range(spec.trials), points, strict=False)
    ]

    total = spec.grid_size
    if total > spec.trials:
        # Said rather than silently truncated. A grid that ran two thirds of
        # its points and reported a best trial reads exactly like a complete
        # one, and the conclusion drawn from it would be wrong in a way
        # nothing in the output hints at.
        _LOGGER.warning(
            "the grid holds %d point(s) but the budget is %d trial(s); the last "
            "%d point(s) will not be explored, so the reported best trial is the "
            "best of a partial grid",
            total,
            spec.trials,
            total - spec.trials,
        )
    else:
        _LOGGER.info("proposed the complete grid of %d point(s)", len(proposals))
    return proposals


def _draw(dimension: Dimension, generator: np.random.Generator) -> object:
    """
    Draw one value from one axis.

    Parameters
    ----------
    dimension
        The axis.
    generator
        The search's generator.

    Returns
    -------
    object
        A value from the enumeration, or a float from the range.
    """
    if not dimension.is_continuous:
        return _choice(dimension.values, generator)

    low, high = float(dimension.low or 0.0), float(dimension.high or 0.0)
    if dimension.log:
        # Uniform in the exponent, so each decade gets an equal share of the
        # budget. Sampling uniformly in the value instead would put nine
        # tenths of the trials in the top decade of a range like 1e-4 to
        # 1e-1, which is almost never what the range was meant to express.
        return float(math.exp(generator.uniform(math.log(low), math.log(high))))
    return float(generator.uniform(low, high))


def _choice(values: Sequence[object], generator: np.random.Generator) -> object:
    """
    Pick one of an enumeration's values.

    Indexed rather than passed to ``Generator.choice`` directly, because that
    coerces its input to an array -- which would turn an integer axis into
    ``numpy.int64`` and a mixed axis into strings. Either would then be
    merged into a specification and fail validation for a reason that has
    nothing to do with what the user wrote.

    Parameters
    ----------
    values
        The enumeration.
    generator
        The search's generator.

    Returns
    -------
    object
        One value, with its original Python type.
    """
    return values[int(generator.integers(len(values)))]
```

