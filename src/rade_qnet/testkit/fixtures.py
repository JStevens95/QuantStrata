"""
Synthetic building blocks for testing the framework and user models.

Everything here is deterministic, small, and free of I/O. The point is that a
test of the train pipeline should not need PyTorch, a dataset, or a
filesystem -- it should need a source that yields batches and a model that
consumes them.

:func:`isolated_registries` deserves particular attention
---------------------------------------------------------
The component registries are module-level, so they are shared by every test in
a process. A test that registers a model called ``demo`` and does not clean up
makes the *next* test's registration raise a duplicate-name error -- and
because it depends on collection order, the failure appears in whichever test
happens to run second, not in the one at fault. Those are among the most
expensive failures to diagnose.

:func:`isolated_registries` snapshots and restores all four registries, which
turns that whole class of problem into something that cannot happen.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ..core.contract.bundle import Manifest, ModelBundle
from ..core.contract.data import (
    SPLIT_NAMES,
    Batch,
    DataBundle,
    DataLineage,
    SplitIndices,
    TensorBatchData,
    TensorLike,
)
from ..core.contract.result import EpochRecord, EvalResult, FitOutcome, TrainingResult
from ..core.contract.signature import InputSignature, SpaceSpec, TensorSpec
from ..core.contract.source import BatchSource
from ..core.contract.state import FittedState
from ..core.runtime.components import ENGINES, LEARNERS, MODELS, REPORTS
from ..core.runtime.context import RunContext
from ..core.runtime.errors import EngineError
from ..core.runtime.hooks import PipelineHook
from ..core.spec.run import RunSpec, parse_run_spec
from ..engines.base import EngineCapabilities, ModelHandle
from ..sources.environment import StepOutcome
from ..storage.runs.catalog import InMemoryCatalog

__all__ = [
    "PLACEHOLDER_DIGEST",
    "LinearModel",
    "RecordingHook",
    "StandardisingState",
    "SyntheticArraySource",
    "SyntheticEngine",
    "SyntheticEnvironment",
    "SyntheticTensorSource",
    "isolated_registries",
    "make_lineage",
    "make_model_bundle",
    "make_run_context",
    "make_run_spec",
    "make_signature",
    "make_split_indices",
    "make_tensor_bundle",
    "make_training_result",
]

#: Default problem size. Deliberately tiny: a fixture exists to make a test
#: fast, and a hundred samples exercises every code path that ten thousand
#: would.
DEFAULT_N_SAMPLES = 100
DEFAULT_N_FEATURES = 4

#: Stand-in for a real SHA-256 digest. All zeros so that it is unmistakably
#: synthetic if it ever turns up in a manifest somebody is reading: a digest
#: that looked plausible would be worse than one that obviously is not.
PLACEHOLDER_DIGEST = "0" * 64


@contextmanager
def isolated_registries(*, empty: bool = False) -> Iterator[None]:
    """
    Restore every component registry when the block exits.

    See the module docstring for why this matters. Use it in any test that
    registers a component, including indirectly by importing a module that
    does.

    When to pass ``empty``
    ----------------------
    By default the block starts from whatever is registered, which is
    whatever the test session happened to import. That is fine for a test
    that only *adds* a name.

    It is not fine for a test that wants to *claim* one. A stand-in engine
    registered under a shipped engine's name succeeds or fails depending on
    whether some earlier test imported the real one, which makes the suite
    pass or fail on collection order. Pass ``empty=True`` and the block
    starts with no models and no engines, so the test owns every name it
    uses and the result does not depend on what ran before it.

    Reports and learners are *not* emptied, deliberately. Those two are
    catalogues the pipeline renders and resolves from rather than names a
    specification claims, so clearing them would not isolate a test -- it
    would change what the pipeline does, and a test asserting that a report
    was written would fail because the report no longer exists to write.

    Parameters
    ----------
    empty
        Start with the model and engine registries cleared.

    Yields
    ------
    None
        The block runs with the registries restorable.

    Examples
    --------
    >>> from rade_qnet.core.runtime.components import MODELS
    >>> with isolated_registries():
    ...     MODELS.register("temporary", object)
    >>> "temporary" in MODELS
    False
    """
    snapshots = [
        (registry, registry.snapshot()) for registry in (MODELS, ENGINES, LEARNERS, REPORTS)
    ]
    if empty:
        for registry, _ in snapshots:
            if registry in (MODELS, ENGINES):
                registry.restore({})
    try:
        yield
    finally:
        for registry, snapshot in snapshots:
            registry.restore(snapshot)


class StandardisingState(FittedState):
    """
    A fitted state that standardises and un-standardises a target.

    The simplest state that is not trivial: it holds two fitted numbers and
    has a real inverse, so a test of
    :meth:`FittedState.inverse_transform_targets` is testing something.

    Also the fixture that catches the "metrics in the wrong units" bug. A
    pipeline that forgets to invert will report a mean absolute error about
    :attr:`scale` times too small, and a test asserting against a known value
    will notice.

    Parameters
    ----------
    mean
        Target mean, fitted on the training split only.
    scale
        Target standard deviation, fitted on the training split only.
    """

    def __init__(self, mean: float = 0.0, scale: float = 1.0) -> None:
        """
        Store the fitted statistics.

        Parameters
        ----------
        mean
            Target mean.
        scale
            Target standard deviation. Must be positive.
        """
        if scale <= 0.0:
            message = f"scale must be positive, received {scale}"
            raise ValueError(message)
        self.mean = float(mean)
        self.scale = float(scale)

    @classmethod
    def fit(cls, train_targets: NDArray[np.floating]) -> Self:
        """
        Fit the statistics on training targets.

        Parameters
        ----------
        train_targets
            Targets from the training split *only*. Fitting on all scenarios
            would leak the held-out period's distribution into training, which
            is the error this signature makes awkward to commit.

        Returns
        -------
        Self
            The fitted state.
        """
        values = np.asarray(train_targets, dtype=np.float64).reshape(-1)
        scale = float(np.std(values))
        return cls(mean=float(np.mean(values)), scale=scale if scale > 0.0 else 1.0)

    def transform(self, targets: NDArray[np.floating]) -> NDArray[np.float64]:
        """
        Standardise targets.

        Parameters
        ----------
        targets
            Raw targets.

        Returns
        -------
        numpy.ndarray
            Standardised targets.
        """
        return (np.asarray(targets, dtype=np.float64) - self.mean) / self.scale

    def save(self, directory: Path) -> None:
        """
        Write the two statistics as a single array.

        Parameters
        ----------
        directory
            Destination directory.
        """
        np.save(directory / "standardiser.npy", np.array([self.mean, self.scale]))

    @classmethod
    def load(cls, directory: Path) -> Self:
        """
        Read the statistics back.

        Parameters
        ----------
        directory
            Directory written by :meth:`save`.

        Returns
        -------
        Self
            The loaded state.
        """
        mean, scale = np.load(directory / "standardiser.npy")
        return cls(mean=float(mean), scale=float(scale))

    def inverse_transform_targets(self, predictions: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Return standardised predictions to original target units.

        Parameters
        ----------
        predictions
            Predictions in standardised space.

        Returns
        -------
        numpy.ndarray
            Predictions in original units.
        """
        return np.asarray(predictions, dtype=np.float64) * self.scale + self.mean

    def describe(self) -> dict[str, object]:
        """
        Return the fitted statistics, for reports.

        Returns
        -------
        dict
            The mean and scale.
        """
        return {"type": type(self).__name__, "mean": self.mean, "scale": self.scale}

    def __eq__(self, other: object) -> bool:
        """Return whether another state holds the same statistics."""
        if not isinstance(other, StandardisingState):
            return NotImplemented
        return (self.mean, self.scale) == (other.mean, other.scale)

    def __hash__(self) -> int:
        """Return a hash of the fitted statistics."""
        return hash((self.mean, self.scale))


@dataclass(slots=True)
class SyntheticTensorSource:
    """
    A re-iterable batch source over in-memory arrays.

    Satisfies :class:`~rade_qnet.core.contract.source.BatchSource`. Batches are
    plain dictionaries of numpy arrays, which is enough for any
    engine-agnostic test and avoids a training-library dependency.

    Parameters
    ----------
    features
        Feature array, samples by features.
    targets
        Target array, one row per sample.
    batch_size
        Samples per batch. The final batch may be shorter.
    static
        Static inputs, delivered once rather than per batch. Reflected in the
        derived signature, since a source that delivers an input its own
        signature does not declare is internally inconsistent.
    unbounded
        Cycle forever and report ``steps_per_epoch`` as ``None``. This is how
        an interactive source behaves, and it is the only field that
        distinguishes the two cases as far as a training loop is concerned --
        so a framework that cannot produce one cannot test half its remit.
    signature_override
        Use a specific signature rather than deriving one from the arrays.

    Notes
    -----
    Re-iterable and deterministic: :meth:`batches` yields the same batches
    every time. That is the property the conformance suite checks, because a
    source backed by a bare generator silently yields nothing from the second
    epoch onwards and the symptom is a training curve that flatlines.
    """

    features: NDArray[np.floating]
    targets: NDArray[np.floating]
    batch_size: int = 16
    static: Mapping[str, TensorLike] = field(default_factory=dict)
    unbounded: bool = False
    signature_override: InputSignature | None = None

    @property
    def signature(self) -> InputSignature:
        """The declared interface of the batches this source yields."""
        if self.signature_override is not None:
            return self.signature_override
        # The dtype is read off the arrays rather than defaulted, so a source
        # built from float32 arrays does not declare float64 and send a
        # gradient engine a dtype its weights will refuse.
        return make_signature(
            n_features=int(self.features.shape[1]),
            static={
                name: TensorSpec(shape=np.shape(value), dtype=str(np.asarray(value).dtype))
                for name, value in self.static.items()
            },
            dtype=str(self.features.dtype),
        )

    @property
    def steps_per_epoch(self) -> int | None:
        """
        Number of batches per pass, including a short final batch.

        ``None`` when unbounded, which is what tells a loop to take its
        stopping condition from the training specification instead.
        """
        if self.unbounded:
            return None
        return -(-self.features.shape[0] // self.batch_size)

    @property
    def n_samples(self) -> int:
        """Number of samples per pass."""
        return int(self.features.shape[0])

    def batches(self) -> Iterator[Batch]:
        """
        Yield batches, in a fixed order.

        One pass over the arrays when bounded; the same pass repeated forever
        when unbounded.

        Yields
        ------
        Batch
            A mapping with ``features`` and ``target``.
        """
        while True:
            for start in range(0, self.features.shape[0], self.batch_size):
                stop = start + self.batch_size
                yield {
                    "features": self.features[start:stop],
                    "target": self.targets[start:stop],
                }
            if not self.unbounded:
                return


@dataclass(slots=True)
class SyntheticArraySource:
    """
    A whole-split source for the one-shot engines.

    Satisfies :class:`~rade_qnet.core.contract.source.BatchSource` by yielding
    exactly one batch containing everything, which is how a tree model wants
    its data and is a legitimate degenerate case of the same protocol -- the
    demonstration that the protocol genuinely spans both engine styles.

    Parameters
    ----------
    features
        Feature array, samples by features.
    targets
        Target array.
    """

    features: NDArray[np.floating]
    targets: NDArray[np.floating]

    @property
    def signature(self) -> InputSignature:
        """The declared interface of the single batch."""
        return make_signature(n_features=int(self.features.shape[1]))

    @property
    def static(self) -> Mapping[str, TensorLike]:
        """No static inputs."""
        return {}

    @property
    def steps_per_epoch(self) -> int:
        """One batch per pass."""
        return 1

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return int(self.features.shape[0])

    def batches(self) -> Iterator[Batch]:
        """
        Yield the whole split as one batch.

        Yields
        ------
        Batch
            A mapping with ``features`` and ``target``.
        """
        yield {"features": self.features, "target": self.targets}


class RecordingHook(PipelineHook):
    """
    A hook that records every event it receives.

    Lets a test assert on the *sequence* of lifecycle events, not just the
    final result -- which is how you verify that ``on_run_end`` fires after a
    failure, or that a stage was skipped rather than silently succeeding.

    Attributes
    ----------
    events
        Every event, in order, as ``(event_name, *details)`` tuples.
    """

    def __init__(self) -> None:
        """Start with an empty event list."""
        self.events: list[tuple[object, ...]] = []

    def on_run_start(self, run_id: str, spec_digest: str) -> None:
        """Record the run start."""
        self.events.append(("run_start", run_id, spec_digest))

    def on_run_end(self, run_id: str, *, succeeded: bool) -> None:
        """Record the run end."""
        self.events.append(("run_end", run_id, succeeded))

    def on_stage_start(self, stage: str) -> None:
        """Record a stage start."""
        self.events.append(("stage_start", stage))

    def on_stage_end(self, stage: str, *, seconds: float) -> None:
        """Record a stage end."""
        # The duration is deliberately not recorded: it varies between runs,
        # so an assertion on the event list would be flaky if it were.
        del seconds
        self.events.append(("stage_end", stage))

    def on_stage_error(self, stage: str, error: BaseException) -> None:
        """Record a stage failure."""
        self.events.append(("stage_error", stage, type(error).__name__))

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """Record an epoch end."""
        self.events.append(("epoch_end", epoch, dict(metrics)))

    def on_metrics(self, stage: str, metrics: Mapping[str, float]) -> None:
        """Record published metrics."""
        self.events.append(("metrics", stage, dict(metrics)))

    def on_artifact(self, name: str, path: str) -> None:
        """Record a published artifact."""
        # The path is deliberately not recorded: it contains a temporary
        # directory, so an assertion on the event list would not be portable.
        del path
        self.events.append(("artifact", name))

    def names(self) -> tuple[str, ...]:
        """
        Return just the event names, for a concise assertion.

        Returns
        -------
        tuple of str
            Event names, in order.
        """
        return tuple(str(event[0]) for event in self.events)


def make_signature(
    *,
    n_features: int = DEFAULT_N_FEATURES,
    static: Mapping[str, TensorSpec] | None = None,
    dtype: str = "float64",
) -> InputSignature:
    """
    Build a plain tabular signature.

    Parameters
    ----------
    n_features
        Number of feature columns.
    static
        Static input specs, if the test needs any.
    dtype
        Element type for both the features and the target.

        Defaults to ``float64`` because numpy does, and overriding it to
        ``float32`` is necessary for any test that reaches a gradient engine:
        torch parameters are ``float32`` by default, and a ``float64`` input
        meeting a ``float32`` weight matrix is a hard error rather than a
        promotion.

    Returns
    -------
    InputSignature
        A signature with one dynamic input and a scalar target.
    """
    return InputSignature(
        dynamic={"features": TensorSpec(shape=(None, n_features), dtype=dtype)},
        static=dict(static or {}),
        target=TensorSpec(shape=(None, 1), dtype=dtype),
    )


def make_split_indices(
    *,
    n_scenarios: int = DEFAULT_N_SAMPLES,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> SplitIndices:
    """
    Build chronological, disjoint, contiguous split indices.

    Chronological rather than random, because that is what a time-series
    problem requires and a fixture that modelled it the easy way would let a
    leaking pipeline pass its tests.

    Parameters
    ----------
    n_scenarios
        Total scenarios.
    validation_fraction
        Fraction held out for validation, taken from the end of training.
    test_fraction
        Fraction held out for test, taken from the very end.

    Returns
    -------
    SplitIndices
        Disjoint train, validation and test indices.
    """
    n_test = int(n_scenarios * test_fraction)
    n_validation = int(n_scenarios * validation_fraction)
    n_train = n_scenarios - n_validation - n_test
    indices = np.arange(n_scenarios, dtype=np.int64)
    return SplitIndices(
        train=indices[:n_train],
        validation=indices[n_train : n_train + n_validation],
        test=indices[n_train + n_validation :],
    )


def make_lineage(
    *,
    splits: SplitIndices | None = None,
    n_entities: int | None = None,
    notes: Mapping[str, str] | None = None,
    spec_digest: str = PLACEHOLDER_DIGEST,
) -> DataLineage:
    """
    Build a lineage record for a synthetic dataset.

    Parameters
    ----------
    splits
        Split indices to record. Defaults to :func:`make_split_indices`.
    n_entities
        Number of entities, if the problem has an entity axis.
    notes
        Free-form annotations.
    spec_digest
        Digest of the specification this data was built from. Defaults to an
        obviously-synthetic placeholder. Overridable because the lineage is
        where the spec digest originates -- :func:`rade_qnet.storage.write_bundle`
        copies it into the manifest rather than recomputing it -- so a test
        proving that link needs to be able to set a recognisable value.

    Returns
    -------
    DataLineage
        A complete lineage record.
    """
    resolved = splits or make_split_indices()
    total = sum(resolved.sizes.values())
    return DataLineage(
        source_fingerprint="synthetic",
        spec_digest=spec_digest,
        split_indices=resolved.as_lineage(),
        n_scenarios=total,
        n_entities=n_entities,
        framework_version="0.1.0.dev0",
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
        notes=dict(notes or {}),
    )


def make_tensor_bundle(
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_features: int = DEFAULT_N_FEATURES,
    batch_size: int = 16,
    seed: int = 0,
    static: Mapping[str, TensorLike] | None = None,
    splits: Sequence[str] = SPLIT_NAMES,
) -> DataBundle[TensorBatchData]:
    """
    Build a data bundle for the gradient engines.

    The target is a known linear function of the features plus noise, so a
    test can assert that a model learned *something* without depending on a
    particular model.

    The standardiser is fitted on the training split only. That is not
    incidental: a fixture that fitted it on everything would make a leaking
    pipeline pass.

    Parameters
    ----------
    n_samples
        Total samples.
    n_features
        Feature columns.
    batch_size
        Samples per batch.
    seed
        Seed for the generated data.
    static
        Static inputs to attach to each split.
    splits
        Which splits to include. Narrowing this is how a caller tests the
        no-validation-split path -- the case where early stopping has nothing
        to monitor, which is a real configuration and a common source of
        downstream failures.

    Returns
    -------
    DataBundle
        The requested splits of
        :class:`~rade_qnet.core.contract.data.TensorBatchData`.
    """
    requested = _requested_splits(splits)
    features, targets, indices = _synthesise(n_samples, n_features, seed)
    state = StandardisingState.fit(targets[indices.train])
    standardised = state.transform(targets)

    payloads: dict[str, TensorBatchData] = {}
    for name in requested:
        rows = indices[name]
        if rows.size == 0:
            continue
        payloads[name] = TensorBatchData(
            loader=SyntheticTensorSource(
                features=features[rows],
                targets=standardised[rows],
                batch_size=batch_size,
            ),
            static=dict(static or {}),
            n_samples=int(rows.size),
            n_batches=-(-int(rows.size) // batch_size),
        )

    return DataBundle(
        splits=payloads,
        signature=make_signature(n_features=n_features),
        state=state,
        lineage=make_lineage(splits=indices),
    )


def make_run_spec(overrides: Mapping[str, object] | None = None) -> RunSpec:
    """
    Build a valid run specification.

    Goes through :func:`~rade_qnet.core.spec.run.parse_run_spec` rather than
    constructing the model directly, so a fixture exercises the same
    validation and the same discriminated-union resolution a real
    configuration file does.

    Parameters
    ----------
    overrides
        Keys merged over the minimal specification.

    Returns
    -------
    RunSpec
        A validated specification.
    """
    payload: dict[str, object] = {"model": "synthetic", "seed": 7}
    payload.update(overrides or {})
    return parse_run_spec(payload, origin="testkit.fixtures")


def make_run_context(
    output_directory: Path,
    *,
    hooks: Sequence[PipelineHook] = (),
    run_id: str = "test-run",
    seed: int = 7,
    catalog: object | None = None,
    job_id: str | None = None,
) -> RunContext:
    """
    Build a run context backed by an in-memory catalog and no tracker.

    Parameters
    ----------
    output_directory
        Where the run may write, normally a ``tmp_path``.
    hooks
        Observers to attach.
    run_id
        Run identifier.
    seed
        Base seed.
    catalog
        Catalog to attach. Defaults to a fresh
        :class:`~rade_qnet.storage.runs.catalog.InMemoryCatalog`; pass one explicitly
        when the test needs to inspect what was registered.
    job_id
        Job identifier, for a context standing in for a job-set member.

    Returns
    -------
    RunContext
        A context suitable for a pipeline test.
    """
    return RunContext(
        run_id=run_id,
        spec_digest=PLACEHOLDER_DIGEST,
        output_directory=output_directory,
        seed=seed,
        job_id=job_id,
        hooks=tuple(hooks),
        catalog=catalog if catalog is not None else InMemoryCatalog(),
        tracker=None,
    )


def make_training_result(
    *,
    n_epochs: int = 5,
    with_validation: bool = True,
    with_evaluation: bool = True,
) -> TrainingResult:
    """
    Build a plausible training result.

    Parameters
    ----------
    n_epochs
        Number of epoch records, with a monotonically falling loss.
    with_validation
        Whether to record a validation loss per epoch, and to evaluate the
        validation split alongside the test split.
    with_evaluation
        Whether to include split evaluations at all.

    Returns
    -------
    TrainingResult
        A complete result.
    """
    history = tuple(
        EpochRecord(
            epoch=index,
            train_loss=1.0 / (index + 1),
            val_loss=1.05 / (index + 1) if with_validation else None,
            learning_rate=1e-3,
            seconds=0.1,
        )
        for index in range(n_epochs)
    )
    best = n_epochs - 1 if n_epochs else None
    fit = FitOutcome(
        history=history,
        monitor="val_loss" if with_validation else "train_loss",
        best_epoch=best,
        best_monitor_value=history[-1].val_loss if (history and with_validation) else None,
        restored_best=True,
        total_seconds=0.1 * n_epochs,
    )
    # Validation is scored slightly better than test, which is the ordering a
    # real run usually shows and the one a report's column order should make
    # visible. A fixture with a single split would leave the multi-split
    # rendering path untested.
    scored = ("validation", "test") if with_validation else ("test",)
    evaluations = (
        {
            name: EvalResult(
                split=name,
                metrics={"mae": 0.25 + offset, "rmse": 0.31 + offset},
                n_samples=15,
                in_original_units=True,
                baseline_metrics={"mae": 0.40, "rmse": 0.52},
            )
            for offset, name in ((0.05 * rank, name) for rank, name in enumerate(scored))
        }
        if with_evaluation
        else {}
    )
    return TrainingResult(fit=fit, evaluations=evaluations, seed=7)


def make_model_bundle(
    *,
    with_manifest: bool = False,
    result: TrainingResult | None = None,
    model: object | None = None,
    state: FittedState | None = None,
    spec: RunSpec | None = None,
    lineage: DataLineage | None = None,
) -> ModelBundle:
    """
    Build an in-memory model bundle.

    Parameters
    ----------
    with_manifest
        Whether to attach a manifest, as if the bundle had been written.
    result
        Training result to carry. Defaults to :func:`make_training_result`.
    model
        Stand-in for the trained model. Supplying a recognisable object is
        how a test asserts the bundle carried it through untouched, which is
        the whole contract for this field: ``core`` never inspects it.
    state
        Fitted state to carry. Defaults to a standardiser with a real,
        checkable inverse.
    spec
        Specification to carry. Defaults to :func:`make_run_spec`.
    lineage
        Lineage to carry. Defaults to :func:`make_lineage`. Worth overriding
        together with ``spec`` when a test needs the bundle's recorded spec
        digest to match the spec it actually came from.

    Returns
    -------
    ModelBundle
        A bundle suitable for testing a report or a bundle writer.
    """
    manifest = None
    if with_manifest:
        manifest = Manifest(
            model_name="synthetic",
            engine="testkit",
            version=1,
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            framework_version="0.1.0.dev0",
            spec_digest=PLACEHOLDER_DIGEST,
        )
    return ModelBundle(
        model=object() if model is None else model,
        state=state or StandardisingState(mean=0.5, scale=2.0),
        signature=make_signature(),
        spec=spec or make_run_spec(),
        lineage=lineage or make_lineage(),
        result=result or make_training_result(),
        manifest=manifest,
    )


def _requested_splits(splits: Sequence[str]) -> tuple[str, ...]:
    """
    Validate a requested split selection and return it in canonical order.

    Parameters
    ----------
    splits
        Split names a caller asked for.

    Returns
    -------
    tuple of str
        The requested names, ordered as :data:`SPLIT_NAMES` orders them.

    Raises
    ------
    ValueError
        If a name is unrecognised, or if ``train`` is absent. Caught here
        rather than in :class:`DataBundle` so the message points at the
        fixture call, which is where the mistake is.
    """
    unknown = sorted(set(splits) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"unknown split name(s) {unknown}; expected a subset of {SPLIT_NAMES}")
    if "train" not in splits:
        raise ValueError("a data bundle always needs a 'train' split")
    return tuple(name for name in SPLIT_NAMES if name in splits)


def _synthesise(
    n_samples: int,
    n_features: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], SplitIndices]:
    """
    Generate features, a linearly related target, and chronological splits.

    Parameters
    ----------
    n_samples
        Total samples.
    n_features
        Feature columns.
    seed
        Seed for the generator.

    Returns
    -------
    tuple
        Features, targets and split indices.
    """
    # A seeded Generator rather than the legacy global functions, so two
    # fixtures built in one test cannot perturb each other's draws.
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(n_samples, n_features))
    weights = np.linspace(1.0, 2.0, n_features)
    targets = features @ weights + generator.normal(scale=0.1, size=n_samples)
    return features, targets, make_split_indices(n_scenarios=n_samples)


@dataclass(slots=True)
class LinearModel:
    """
    A least-squares linear model, as an engine-native object.

    The counterpart to :class:`SyntheticEngine`: something concrete for it to
    materialise, fit, save and reload. Deliberately a plain dataclass rather
    than anything resembling a neural network, because the point of a
    framework test is to exercise the framework.

    Parameters
    ----------
    n_features
        Width of the input the model expects.
    coefficients
        Fitted weights, or ``None`` before fitting. ``None`` rather than zeros
        is what makes this model *lazily shaped* in the same sense a
        ``LazyLinear`` layer is, so the pipeline's materialise stage has
        something real to do.
    intercept
        Fitted intercept.
    """

    n_features: int
    coefficients: NDArray[np.float64] | None = None
    intercept: float = 0.0

    @property
    def is_materialised(self) -> bool:
        """Whether the parameters exist yet."""
        return self.coefficients is not None

    def predict(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        """
        Apply the model.

        Parameters
        ----------
        features
            Samples by features.

        Returns
        -------
        numpy.ndarray
            One prediction per row.

        Raises
        ------
        EngineError
            If called before the parameters exist.
        """
        if self.coefficients is None:
            raise EngineError(
                "this model has no parameters yet; the engine's materialise step "
                "must run before a forward pass"
            )
        return np.asarray(features, dtype=np.float64) @ self.coefficients + self.intercept


@dataclass(slots=True)
class SyntheticEngine:
    """
    A deterministic engine that needs no training library.

    Satisfies :class:`~rade_qnet.engines.base.Engine` structurally, so the
    conformance suite holds it to the same definition as
    :class:`~rade_qnet.engines.torch.engine.TorchEngine` -- and so a test of the
    *pipeline* can run end to end in milliseconds without importing PyTorch.

    Fits by closed-form least squares, which matters more than it sounds: the
    solution is exact and reproducible, so a pipeline test can assert on the
    metrics themselves rather than on a tolerance. A test that can only check
    "the loss went down" cannot tell a correct pipeline from one that scores
    the wrong split.

    Parameters
    ----------
    epochs_reported
        How many epoch records :meth:`fit` fabricates. The fit is one shot --
        least squares has no epochs -- but a pipeline consumes a
        :class:`~rade_qnet.core.contract.result.FitOutcome` either way, and the
        curve reports and the history CSV need records to render. This is the
        same accommodation a real boosted-tree engine makes by reporting one
        record per round.
    """

    epochs_reported: int = 3

    def capabilities(self) -> EngineCapabilities:
        """
        Declare what this engine supports.

        Returns
        -------
        EngineCapabilities
            A minimal, honest set: no checkpointing, no distribution, no
            mixed precision.
        """
        return EngineCapabilities(
            name="synthetic",
            supports_epochs=False,
            supports_validation_during_fit=True,
            supports_checkpointing=False,
            supports_distributed=False,
            supports_lazy_materialisation=True,
            accelerators=("cpu",),
        )

    def materialise(self, model: object, signature: InputSignature) -> object:
        """
        Give the model its parameters, sized from the signature alone.

        Parameters
        ----------
        model
            A :class:`LinearModel`.
        signature
            The declared interface. The width is taken from here rather than
            from a batch, which is the property that lets the pipeline's
            materialise stage run before any data reaches the model.

        Returns
        -------
        object
            The same model, now materialised.
        """
        linear = _as_linear(model)
        if not linear.is_materialised:
            width = int(signature.dynamic["features"].shape[-1])
            linear.n_features = width
            linear.coefficients = np.zeros(width, dtype=np.float64)
        return linear

    def prepare(
        self,
        model: object,
        *,
        hardware: object,
        training: object,
        static: Mapping[str, TensorLike] | None = None,
    ) -> ModelHandle:
        """
        Wrap the model in a handle.

        Parameters
        ----------
        model
            The materialised model.
        hardware
            Ignored: this engine runs on the CPU and nowhere else.
        training
            Ignored: least squares has nothing to configure.
        static
            Static inputs, carried through so a test can assert they were
            delivered once rather than per batch.

        Returns
        -------
        ModelHandle
            The handle.

        Raises
        ------
        EngineError
            If the model was never materialised. Refused rather than
            materialised here, because silently covering for a missing stage
            is how the ordering that defect 6 is about gets lost.
        """
        del hardware, training
        linear = _as_linear(model)
        if not linear.is_materialised:
            raise EngineError(
                "prepare() received a model with no parameters; materialise() "
                "must run first, which is why the pipeline makes it a stage"
            )
        return ModelHandle(
            model=linear,
            unwrapped=linear,
            device="cpu",
            precision="fp32",
            static=dict(static or {}),
        )

    def fit(
        self,
        handle: ModelHandle,
        sources: Mapping[str, BatchSource],
        training: object,
        *,
        on_epoch_end: object = None,
    ) -> FitOutcome:
        """
        Solve for the least-squares coefficients over the training source.

        Parameters
        ----------
        handle
            The prepared model.
        sources
            Must contain ``train``; ``validation`` is scored if present.
        training
            Ignored.
        on_epoch_end
            Ignored.

        Returns
        -------
        FitOutcome
            A history whose losses are the real training and validation
            errors of the solved model, repeated across the reported epochs.

        Raises
        ------
        EngineError
            If there is no training source.
        """
        del training, on_epoch_end
        if "train" not in sources:
            raise EngineError(f"fit() needs a 'train' source; received {sorted(sources)}")

        linear = _as_linear(handle.unwrapped)
        features, targets = _collect(sources["train"])

        # A column of ones, so the intercept is solved jointly rather than by
        # centring first -- which keeps this exact for a target that was not
        # centred by the data build.
        design = np.column_stack([features, np.ones(features.shape[0])])
        solution, *_ = np.linalg.lstsq(design, targets, rcond=None)
        linear.coefficients = np.asarray(solution[:-1], dtype=np.float64)
        linear.intercept = float(solution[-1])

        train_loss = _mean_squared_error(linear.predict(features), targets)
        validation_loss = None
        if "validation" in sources:
            held_features, held_targets = _collect(sources["validation"])
            validation_loss = _mean_squared_error(linear.predict(held_features), held_targets)

        history = tuple(
            EpochRecord(
                epoch=index,
                train_loss=train_loss,
                val_loss=validation_loss,
                metrics={"grad_norm_mean": 0.0, "grad_norm_max": 0.0},
                learning_rate=1.0,
                seconds=0.0,
            )
            for index in range(self.epochs_reported)
        )
        return FitOutcome(
            history=history,
            best_epoch=self.epochs_reported - 1,
            best_monitor_value=validation_loss if validation_loss is not None else train_loss,
            restored_best=True,
        )

    def predict(self, handle: ModelHandle, source: BatchSource) -> NDArray[np.floating]:
        """
        Run the model over a source.

        Parameters
        ----------
        handle
            The fitted model.
        source
            The source to predict over.

        Returns
        -------
        numpy.ndarray
            One prediction per sample, in the order the source yielded them.
        """
        features, _ = _collect(source)
        return _as_linear(handle.unwrapped).predict(features)

    def save_weights(self, handle: ModelHandle, path: Path) -> None:
        """
        Write the coefficients.

        Parameters
        ----------
        handle
            The fitted model.
        path
            Destination file.
        """
        linear = _as_linear(handle.unwrapped)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Through an open handle rather than by passing the path, because
        # `np.savez` given a path appends `.npz` to it -- so the file the
        # bundle's manifest records would not be the file that exists, and the
        # bundle would fail to reopen with a confusing "missing weights".
        with path.open("wb") as stream:
            np.savez(
                stream,
                coefficients=linear.coefficients,
                intercept=np.array([linear.intercept], dtype=np.float64),
            )

    def load_weights(self, model: object, path: Path) -> object:
        """
        Load coefficients into a fresh model.

        Parameters
        ----------
        model
            A :class:`LinearModel` to load into.
        path
            A file written by :meth:`save_weights`.

        Returns
        -------
        object
            The model, with the saved parameters.
        """
        linear = _as_linear(model)
        with np.load(path) as arrays:
            linear.coefficients = np.asarray(arrays["coefficients"], dtype=np.float64)
            linear.intercept = float(arrays["intercept"][0])
        linear.n_features = int(linear.coefficients.size)
        return linear


def _as_linear(model: object) -> LinearModel:
    """
    Narrow an opaque model object to a :class:`LinearModel`.

    Parameters
    ----------
    model
        The engine-native model.

    Returns
    -------
    LinearModel
        The same object.

    Raises
    ------
    EngineError
        If it is something else. An engine receiving a model it cannot drive
        is a wiring fault, and naming both types is what makes it a two-second
        fix instead of a traceback from inside a matrix multiply.
    """
    if not isinstance(model, LinearModel):
        raise EngineError(
            f"SyntheticEngine drives a LinearModel; received a {type(model).__name__}"
        )
    return model


def _collect(
    source: BatchSource,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Drain one pass of a source into a feature matrix and a target vector.

    Parameters
    ----------
    source
        The source to drain. Must be bounded: this engine has no notion of an
        unbounded stream, which is exactly what ``steps_per_epoch is None``
        warns a consumer about.

    Returns
    -------
    tuple
        Features as samples by features, and targets as a flat vector.

    Raises
    ------
    EngineError
        If the source is unbounded, or yielded no batches.
    """
    if source.steps_per_epoch is None:
        raise EngineError(
            "SyntheticEngine cannot consume an unbounded source: a closed-form "
            "solve needs the whole split at once. An unbounded source belongs "
            "with an engine driven by fit_steps"
        )

    feature_blocks: list[NDArray[np.float64]] = []
    target_blocks: list[NDArray[np.float64]] = []
    for batch in source.batches():
        feature_blocks.append(np.asarray(batch["features"], dtype=np.float64))
        target_blocks.append(np.ravel(np.asarray(batch["target"], dtype=np.float64)))

    if not feature_blocks:
        raise EngineError("the source yielded no batches, so there is nothing to fit")
    return (
        np.concatenate(feature_blocks, axis=0),
        np.concatenate(target_blocks, axis=0),
    )


def _mean_squared_error(predictions: NDArray[np.floating], targets: NDArray[np.floating]) -> float:
    """
    Return the mean squared error.

    Parameters
    ----------
    predictions
        Predicted values.
    targets
        Observed values.

    Returns
    -------
    float
        The error.
    """
    residuals = np.ravel(predictions) - np.ravel(targets)
    return float(np.mean(np.square(residuals)))


#: Default size of the synthetic environment's observation.
DEFAULT_N_OBSERVATIONS = 3

#: Default number of discrete actions the synthetic environment accepts.
DEFAULT_N_ACTIONS = 2

#: Default episode length for the synthetic environment. Short, so that a
#: test collecting one small batch still sees an episode end -- which is the
#: part of a collector most worth exercising.
DEFAULT_EPISODE_LENGTH = 4


class SyntheticEnvironment:
    """
    A tiny deterministic environment, for exercising the interactive path.

    Satisfies
    :class:`~rade_qnet.sources.environment.protocol.Environment`. The
    observation is the step index broadcast to the observation width, so a
    test can read a batch back and see exactly which transitions it holds and
    where an episode restarted. The reward is one per step, so an episode's
    return is its length -- which makes an assertion about episode statistics
    a statement about arithmetic rather than about the environment.

    The action genuinely affects nothing. That is deliberate: this fixture is
    for testing the *plumbing*, and an environment whose dynamics depended on
    the action would make every such test depend on which action a policy
    happened to choose.

    Parameters
    ----------
    n_observations
        Width of the observation vector.
    n_actions
        Number of discrete actions accepted.
    episode_length
        Steps before the episode terminates.
    """

    def __init__(
        self,
        *,
        n_observations: int = DEFAULT_N_OBSERVATIONS,
        n_actions: int = DEFAULT_N_ACTIONS,
        episode_length: int = DEFAULT_EPISODE_LENGTH,
    ) -> None:
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.episode_length = episode_length
        self.step_index = 0
        self.resets = 0

    @property
    def observation_space(self) -> SpaceSpec:
        """
        What the policy sees.

        Returns
        -------
        SpaceSpec
            A bounded continuous space.
        """
        return SpaceSpec(
            kind="box",
            shape=(self.n_observations,),
            dtype="float32",
            low=0.0,
            high=float(self.episode_length),
        )

    @property
    def action_space(self) -> SpaceSpec:
        """
        What the policy must produce.

        Returns
        -------
        SpaceSpec
            A finite space of ``n_actions`` actions.
        """
        return SpaceSpec(kind="discrete", n=self.n_actions)

    def reset(self, *, seed: int | None = None) -> NDArray[np.float32]:
        """
        Start a new episode.

        Parameters
        ----------
        seed
            Accepted and unused: this environment is deterministic, so a
            seed would change nothing and pretending otherwise would make a
            reproducibility test pass for the wrong reason.

        Returns
        -------
        numpy.ndarray
            The zero observation.
        """
        del seed
        self.resets += 1
        self.step_index = 0
        return np.zeros(self.n_observations, dtype=np.float32)

    def step(self, action: object) -> StepOutcome:
        """
        Advance one step.

        Parameters
        ----------
        action
            Accepted and unused; see the class docstring.

        Returns
        -------
        StepOutcome
            A unit reward, terminating at the episode length.
        """
        del action
        self.step_index += 1
        return StepOutcome(
            observation=np.full(self.n_observations, self.step_index, dtype=np.float32),
            reward=1.0,
            terminated=self.step_index >= self.episode_length,
        )
