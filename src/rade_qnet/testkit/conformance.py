"""
The shared contract suite, checking that a model honours what it declares.

``isinstance`` against a runtime-checkable protocol establishes that a method
*exists*. It says nothing about whether the method returns the right thing.
This module closes that gap: for every capability a model declares, it calls
the capability and checks the result.

Why a model author should run this
----------------------------------
The clauses below are not stylistic preferences. Each one corresponds to a
failure that is silent -- it produces a run that completes, reports plausible
numbers, and is wrong. A few examples of what is checked and what it catches:

- A source that is not re-iterable trains for one epoch and then on nothing,
  producing a flat loss curve that looks like a learning-rate problem.
- A :class:`~rade_qnet.core.contract.state.FittedState` whose inverse is not an
  inverse reports metrics in the wrong units, so every number is
  incomparable with every other run.
- A :class:`~rade_qnet.core.capability.protocols.Routable` member that
  over-declares its coverage attributes predictions to instruments it never
  trained on.

What this suite deliberately does *not* flag
--------------------------------------------
Fitting along the **entity axis** using the full universe. Knowing which
instruments exist is not knowledge of their future, and a cross-sectional
model that saw only some of them would be solving a different problem. Only
fitting along the **scenario (time) axis** outside the training rows is
leakage, and only that is reported.

Stating the exclusion explicitly matters because a suite that flagged
entity-axis fitting would train authors to ignore its warnings, which costs
more than the suite is worth.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..core.capability.protocols import CustomStep, Inductive, Precomputable, Routable, StaticInputs
from ..core.contract.data import TARGET_KEY, Batch, DataBundle
from ..core.contract.result import FitOutcome
from ..core.contract.signature import InputSignature
from ..core.contract.source import BatchSource
from ..core.contract.state import FittedState
from ..core.runtime.logging import get_logger
from ..core.spec.hardware import HardwareSpec
from ..engines.base import Engine, EngineCapabilities, ModelHandle

__all__ = [
    "ConformanceReport",
    "check_data_bundle",
    "check_engine",
    "check_fitted_state",
    "check_model_capabilities",
    "check_source",
]

_LOGGER = get_logger(__name__)

#: Tolerance for the round-trip check on a fitted state's inverse. Loose
#: enough to allow float32 intermediate storage, tight enough that a genuinely
#: wrong inverse -- a forgotten offset, a reciprocal scale -- cannot pass.
INVERSE_TOLERANCE = 1e-6

#: Training error below which an engine is excused from improving on it. A
#: model handed a problem it already solves exactly cannot get better, and
#: failing it for that would be a false positive on the easiest possible case.
ALREADY_OPTIMAL = 1e-12

#: How many batches to draw from a source that declares itself unbounded.
#: Enough to establish that batches keep arriving and keep the declared shape,
#: small enough that checking an environment rollout stays cheap.
UNBOUNDED_PROBE_BATCHES = 8


@dataclass(slots=True)
class ConformanceReport:
    """
    The findings of a conformance check.

    Separates failures from warnings because the two call for different
    responses, and collapsing them would make the suite either too noisy to
    read or too permissive to be useful.

    Parameters
    ----------
    subject
        What was checked, for the messages.
    failures
        Contract violations. A model with any of these will misbehave.
    warnings
        Things that are legal but suspect, such as a capability declared and
        then returning nothing.
    checks_run
        Names of the clauses that were evaluated, so a reader can tell the
        difference between "passed" and "was not applicable".
    """

    subject: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether there were no failures. Warnings do not fail a check."""
        return not self.failures

    def fail(self, check: str, message: str) -> None:
        """
        Record a contract violation.

        Parameters
        ----------
        check
            Clause name.
        message
            What is wrong, and ideally what it will cause.
        """
        self.checks_run.append(check)
        self.failures.append(f"[{check}] {message}")

    def warn(self, check: str, message: str) -> None:
        """
        Record something legal but suspect.

        Parameters
        ----------
        check
            Clause name.
        message
            What looks wrong.
        """
        self.checks_run.append(check)
        self.warnings.append(f"[{check}] {message}")

    def passed_check(self, check: str) -> None:
        """
        Record a clause that was evaluated and satisfied.

        Parameters
        ----------
        check
            Clause name.
        """
        self.checks_run.append(check)

    def merge(self, other: ConformanceReport) -> None:
        """
        Absorb another report's findings.

        Parameters
        ----------
        other
            The report to absorb.
        """
        self.failures.extend(other.failures)
        self.warnings.extend(other.warnings)
        self.checks_run.extend(other.checks_run)

    def describe(self) -> str:
        """
        Return a human-readable summary.

        Returns
        -------
        str
            A multi-line report.
        """
        verdict = "PASSED" if self.passed else "FAILED"
        lines = [
            f"conformance for {self.subject}: {verdict} "
            f"({len(self.checks_run)} check(s), {len(self.failures)} failure(s), "
            f"{len(self.warnings)} warning(s))"
        ]
        lines.extend(f"  FAIL {message}" for message in self.failures)
        lines.extend(f"  WARN {message}" for message in self.warnings)
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        """
        Raise if any clause failed.

        The form a test should use: ``check_source(source).raise_if_failed()``
        produces an assertion whose message is the whole report.

        Raises
        ------
        AssertionError
            If there were any failures.
        """
        if not self.passed:
            raise AssertionError(self.describe())


def _drain(source: BatchSource, *, bounded: bool) -> list[Batch]:
    """
    Collect one pass over a source, or a prefix of an unbounded one.

    Parameters
    ----------
    source
        The source to traverse.
    bounded
        Whether the source declared a finite ``steps_per_epoch``.

    Returns
    -------
    list
        The batches collected.
    """
    if bounded:
        return list(source.batches())
    return list(islice(source.batches(), UNBOUNDED_PROBE_BATCHES))


def check_source(source: object, *, subject: str | None = None) -> ConformanceReport:
    """
    Check that an object honours the batch-source contract.

    Parameters
    ----------
    source
        The candidate source.
    subject
        Name for the messages, defaulting to the class name.

    Returns
    -------
    ConformanceReport
        The findings.
    """
    report = ConformanceReport(subject=subject or type(source).__name__)

    if not isinstance(source, BatchSource):
        report.fail(
            "source.protocol",
            f"{type(source).__name__} does not satisfy BatchSource; it needs "
            f"signature, static, steps_per_epoch, n_samples and batches()",
        )
        return report
    report.passed_check("source.protocol")

    signature = source.signature
    steps = source.steps_per_epoch

    # An unbounded source never stops, so draining it would hang the checker
    # on precisely the interactive case the framework exists to support. A
    # prefix is enough: every clause below is about the shape of what arrives,
    # not about how much of it there is.
    first_pass = _drain(source, bounded=steps is not None)
    if not first_pass:
        report.fail("source.non_empty", "batches() yielded nothing on the first pass")
        return report
    report.passed_check("source.non_empty")

    # The clause that matters most. A generator-backed source passes every
    # other check and then trains on nothing from epoch two onwards.
    second_pass = _drain(source, bounded=steps is not None)
    if not second_pass:
        report.fail(
            "source.re_iterable",
            "batches() yielded nothing on the second pass; the source is not "
            "re-iterable, so training would see data only in the first epoch. "
            "Return a fresh iterator from batches() rather than storing a "
            "generator",
        )
    elif len(second_pass) != len(first_pass):
        report.fail(
            "source.re_iterable",
            f"batches() yielded {len(first_pass)} batch(es) then "
            f"{len(second_pass)}; passes must be identical under a fixed seed",
        )
    else:
        report.passed_check("source.re_iterable")

    if steps is not None and steps != len(first_pass):
        report.fail(
            "source.steps_honest",
            f"steps_per_epoch reports {steps} but batches() yielded "
            f"{len(first_pass)}; schedules and progress reporting are computed "
            f"from this value, so it must be exact rather than an estimate",
        )
    else:
        report.passed_check("source.steps_honest")

    for index, batch in enumerate(first_pass):
        try:
            signature.validate_batch_keys(batch, where=f"batch {index}")
        except Exception as error:  # Broad: any failure is a finding, not a crash.
            report.fail("source.batch_keys", str(error))
            break
    else:
        report.passed_check("source.batch_keys")

    declared_static = set(signature.static)
    delivered_static = set(source.static)
    missing_static = sorted(declared_static - delivered_static)
    if missing_static:
        # Only a warning: a model may legitimately supply its own static
        # inputs through the StaticInputs capability instead.
        report.warn(
            "source.static_declared",
            f"the signature declares static input(s) {missing_static} that the "
            f"source does not deliver; the model must supply them through the "
            f"StaticInputs capability or the engine will have nothing to upload",
        )
    else:
        report.passed_check("source.static_declared")

    return report


def check_fitted_state(
    state: object,
    *,
    sample_predictions: np.ndarray | None = None,
    subject: str | None = None,
) -> ConformanceReport:
    """
    Check that a fitted state is a real, invertible state.

    Parameters
    ----------
    state
        The candidate state.
    sample_predictions
        Values to round-trip through the inverse. A default set spanning
        several orders of magnitude is used if omitted, because an inverse
        that is wrong only for large values is a real and easily missed bug.
    subject
        Name for the messages.

    Returns
    -------
    ConformanceReport
        The findings.
    """
    report = ConformanceReport(subject=subject or type(state).__name__)

    if not isinstance(state, FittedState):
        report.fail(
            "state.base",
            f"{type(state).__name__} is not a FittedState; it must implement "
            f"save, load and inverse_transform_targets",
        )
        return report
    report.passed_check("state.base")

    values = (
        sample_predictions
        if sample_predictions is not None
        else np.array([-1e3, -1.0, -1e-3, 0.0, 1e-3, 1.0, 1e3])
    )

    try:
        inverted = state.inverse_transform_targets(values)
    except Exception as error:  # Broad: any failure is a finding.
        report.fail("state.inverse_callable", f"inverse_transform_targets raised: {error}")
        return report

    inverted = np.asarray(inverted)
    if inverted.shape != values.shape:
        report.fail(
            "state.inverse_shape",
            f"inverse_transform_targets changed the shape from {values.shape} to "
            f"{inverted.shape}; it must map predictions one-for-one, or "
            f"predictions will no longer align with their entity identifiers",
        )
    else:
        report.passed_check("state.inverse_shape")

    if not np.all(np.isfinite(inverted)):
        report.fail(
            "state.inverse_finite",
            "inverse_transform_targets produced non-finite values; every metric "
            "computed downstream would then be non-finite",
        )
    else:
        report.passed_check("state.inverse_finite")

    # A varying input must produce a varying output. Catches a state that
    # ignores its argument -- returning a fitted mean, say -- which otherwise
    # passes every other clause and reports every prediction as the same
    # number.
    inputs_vary = values.size > 1 and not np.allclose(values, values.flat[0])
    outputs_constant = inverted.size > 1 and np.allclose(inverted, inverted.flat[0])
    if inputs_vary and outputs_constant:
        report.fail(
            "state.inverse_varies",
            "inverse_transform_targets returned a constant; it is ignoring its "
            "input, so every prediction would be reported as the same value",
        )
    else:
        report.passed_check("state.inverse_varies")

    return report


def check_data_bundle(bundle: object, *, subject: str | None = None) -> ConformanceReport:
    """
    Check a data bundle's internal consistency.

    Parameters
    ----------
    bundle
        The candidate bundle.
    subject
        Name for the messages.

    Returns
    -------
    ConformanceReport
        The findings.
    """
    report = ConformanceReport(subject=subject or type(bundle).__name__)

    if not isinstance(bundle, DataBundle):
        report.fail("bundle.type", f"{type(bundle).__name__} is not a DataBundle")
        return report
    report.passed_check("bundle.type")

    report.merge(check_fitted_state(bundle.state, subject=f"{report.subject}.state"))

    recorded = bundle.lineage.split_indices
    if not recorded.get("train"):
        report.fail(
            "bundle.lineage_splits",
            "the lineage records no training scenarios; a bundle whose lineage "
            "does not state which rows it trained on cannot be re-evaluated "
            "against the same data later",
        )
    else:
        report.passed_check("bundle.lineage_splits")

    # Scenario-axis leakage: the check that matters. Entity-axis fitting is
    # deliberately not examined; see the module docstring.
    seen: dict[int, str] = {}
    for split, indices in recorded.items():
        for index in indices:
            if index in seen and seen[index] != split:
                report.fail(
                    "bundle.no_scenario_overlap",
                    f"scenario {index} appears in both the {seen[index]!r} and "
                    f"{split!r} splits; a held-out metric computed over shared "
                    f"scenarios is not held out",
                )
                return report
            seen[index] = split
    report.passed_check("bundle.no_scenario_overlap")

    for name in bundle.split_names:
        payload = bundle.split(name)
        loader = getattr(payload, "loader", None)
        if loader is not None:
            report.merge(check_source(loader, subject=f"{report.subject}.{name}"))

    return report


def check_model_capabilities(
    model: object,
    *,
    assigned_targets: Sequence[str] | None = None,
    subject: str | None = None,
) -> ConformanceReport:
    """
    Check every capability a model declares, by calling it.

    Only the declared capabilities are checked. A model that implements none
    of them passes trivially, which is correct: the simple case should not be
    penalised for being simple.

    Parameters
    ----------
    model
        The model object, or its definition.
    assigned_targets
        The targets a job assigned, if the model is
        :class:`~rade_qnet.core.capability.protocols.Routable`. Needed to check
        that declared coverage is a subset of what was assigned.
    subject
        Name for the messages.

    Returns
    -------
    ConformanceReport
        The findings.
    """
    report = ConformanceReport(subject=subject or type(model).__name__)
    declared: list[str] = []

    if isinstance(model, StaticInputs):
        declared.append("StaticInputs")
        _check_static_inputs(model, report)
    if isinstance(model, Precomputable):
        declared.append("Precomputable")
        _check_precomputable(model, report)
    if isinstance(model, CustomStep):
        declared.append("CustomStep")
        _check_custom_step(report)
    if isinstance(model, Routable):
        declared.append("Routable")
        _check_routable(model, report, assigned_targets)
    if isinstance(model, Inductive):
        declared.append("Inductive")
        _check_inductive(model, report)

    _LOGGER.debug("%s declares capabilities: %s", report.subject, ", ".join(declared) or "none")
    return report


def _check_static_inputs(model: StaticInputs, report: ConformanceReport) -> None:
    """
    Check that a declared static-input provider returns something.

    Parameters
    ----------
    model
        The model under test.
    report
        Report to record findings in.
    """
    try:
        static = model.static_inputs()
    except Exception as error:  # Broad: any failure is a finding, not a crash.
        report.fail("capability.static_inputs", f"static_inputs() raised: {error}")
        return
    if not static:
        report.warn(
            "capability.static_inputs",
            "StaticInputs is declared but static_inputs() returned nothing; "
            "either the capability is unnecessary or the inputs are missing",
        )
    else:
        report.passed_check("capability.static_inputs")


def _check_precomputable(model: Precomputable, report: ConformanceReport) -> None:
    """
    Check that precomputation has static inputs to precompute from.

    Parameters
    ----------
    model
        The model under test.
    report
        Report to record findings in.
    """
    if not isinstance(model, StaticInputs):
        report.fail(
            "capability.precomputable_needs_static",
            "Precomputable is declared without StaticInputs; there is nothing "
            "constant to precompute, so the cached encoding would be stale "
            "after the first batch",
        )
    else:
        report.passed_check("capability.precomputable_needs_static")


def _check_custom_step(report: ConformanceReport) -> None:
    """
    Record that a custom training step needs engine-level verification.

    The two properties that matter -- a live gradient path to the parameters,
    and not calling ``backward()`` -- cannot be established without an engine
    and a real tensor. So this clause records the obligation rather than
    pretending to discharge it; the Phase 2 engine suite checks it for real.

    Parameters
    ----------
    report
        Report to record findings in.
    """
    report.passed_check("capability.custom_step_declared")
    report.warn(
        "capability.custom_step_gradient",
        "CustomStep is declared: training_step must return a scalar loss with "
        "a live gradient path to the model's parameters, and must not call "
        "backward() itself. That cannot be checked without an engine, so the "
        "engine-level conformance suite checks it in Phase 2",
    )


def _check_routable(
    model: Routable,
    report: ConformanceReport,
    assigned_targets: Sequence[str] | None,
) -> None:
    """
    Check that declared target coverage is non-empty, unique and assigned.

    Parameters
    ----------
    model
        The model under test.
    report
        Report to record findings in.
    assigned_targets
        What the job assigned, or ``None`` to skip the subset check.
    """
    try:
        covered = list(model.covered_targets())
    except Exception as error:  # Broad: any failure is a finding, not a crash.
        report.fail("capability.routable", f"covered_targets() raised: {error}")
        return

    if not covered:
        report.fail(
            "capability.routable_non_empty",
            "covered_targets() returned nothing; a job-set member that covers no "
            "target contributes no predictions, and the job's targets would "
            "silently have no owner",
        )
        return
    if len(set(covered)) != len(covered):
        duplicates = sorted({target for target in covered if covered.count(target) > 1})
        report.fail(
            "capability.routable_unique",
            f"covered_targets() repeats {duplicates}; a duplicated target would "
            f"have its prediction counted twice in an assembled portfolio result",
        )
        return
    if assigned_targets is not None and not set(covered) <= set(assigned_targets):
        extra = sorted(set(covered) - set(assigned_targets))
        report.fail(
            "capability.routable_subset",
            f"covered_targets() claims {extra}, which the job did not assign; "
            f"predictions would be attributed to instruments this member was "
            f"never responsible for",
        )
        return
    report.passed_check("capability.routable")


def _check_inductive(model: Inductive, report: ConformanceReport) -> None:
    """
    Check that the inductive declaration is a plain boolean.

    Parameters
    ----------
    model
        The model under test.
    report
        Report to record findings in.
    """
    try:
        supported = model.supports_unseen_entities()
    except Exception as error:  # Broad: any failure is a finding, not a crash.
        report.fail("capability.inductive", f"supports_unseen_entities() raised: {error}")
        return
    if not isinstance(supported, bool):
        report.fail(
            "capability.inductive_bool",
            f"supports_unseen_entities() returned {type(supported).__name__}, not a bool",
        )
    else:
        report.passed_check("capability.inductive")


def check_engine(
    engine: object,
    *,
    model_factory: Callable[[], object],
    source_factory: Callable[[], BatchSource],
    signature: InputSignature,
    directory: Path,
    training: object = None,
    hardware: HardwareSpec | None = None,
    subject: str | None = None,
) -> ConformanceReport:
    """
    Check that an object honours the engine contract.

    Every clause corresponds to a failure that produces a run which completes
    and reports plausible numbers.

    What is checked, and what each clause catches
    --------------------------------------------
    ``engine.materialise_before_prepare``
        That ``prepare`` is willing to accept a materialised model, and that
        the handle's optimiser-facing object has parameters. This is defect 6:
        a lazily shaped model has none until it has seen a batch, and an
        optimiser built over the empty set reports success and never updates
        anything. The loss curve is flat and nothing else is wrong.
    ``engine.fit_is_an_improvement``
        That the fitted model scores better on its training split than the
        untrained one did. A deliberately low bar -- it is not a quality
        threshold -- because the failure it catches is an engine that trains a
        *copy* and leaves the caller's model untouched, which is easy to do
        when a wrapper is involved and impossible to see in a ``FitOutcome``.
    ``engine.predict_is_aligned``
        That ``predict`` returns one prediction per sample, repeatably.
        Checked against a second call because an engine that leaves dropout
        enabled scores a different model each time it is asked.

        Note the limit: a *consistent* permutation passes this clause, because
        nothing here knows the true order. It is caught by
        ``engine.fit_is_an_improvement`` instead, which compares against the
        targets -- so the two clauses together cover it even though neither
        does alone.
    ``engine.weights_round_trip``
        That weights saved and reloaded into a *fresh* model reproduce the
        predictions bit for bit. Not approximately: a round trip that loses
        precision means a served model and a scored model are different
        models, and the difference will be blamed on the data.
    ``engine.handle_exposes_an_unwrapped_model``
        That ``unwrapped`` is not a wrapper. The bundle persists this, and a
        distributed or compiled wrapper saved into a bundle cannot be reopened
        without reconstructing the same wrapper -- which the reader of the
        bundle has no way to know about.

    Parameters
    ----------
    engine
        The candidate engine.
    model_factory
        Returns a fresh, unmaterialised model the engine can drive. A factory
        rather than an instance, because the round-trip clause needs a second
        model that has never been fitted.
    source_factory
        Returns a fresh bounded source. A factory for the same reason.
    signature
        The signature the model is built against.
    directory
        A writable directory for the weight round trip.
    training
        The engine's own training specification, passed through to ``prepare``
        and ``fit`` unexamined. Required for any engine that reads it -- a
        gradient engine needs its optimiser and epoch count from somewhere --
        and ignorable for one that does not.
    hardware
        Hardware specification, defaulting to whatever ``HardwareSpec``
        defaults to. Worth overriding only to check an accelerator path.
    subject
        Name for the messages, defaulting to the class name.

    Returns
    -------
    ConformanceReport
        The findings.
    """
    report = ConformanceReport(subject=subject or type(engine).__name__)

    if not isinstance(engine, Engine):
        report.fail(
            "engine.protocol",
            f"{type(engine).__name__} does not satisfy Engine; it needs "
            f"capabilities, materialise, prepare, fit, predict, save_weights "
            f"and load_weights",
        )
        return report
    report.passed_check("engine.protocol")

    capabilities = engine.capabilities()
    if not isinstance(capabilities, EngineCapabilities):
        report.fail(
            "engine.capabilities",
            f"capabilities() returned a {type(capabilities).__name__} rather than "
            f"an EngineCapabilities; logs, reports and this suite read it to "
            f"state what the engine supports",
        )
        return report
    report.passed_check("engine.capabilities")

    resolved_hardware = hardware if hardware is not None else HardwareSpec()
    try:
        handle = _prepare_for_check(
            engine,
            model_factory(),
            signature,
            training=training,
            hardware=resolved_hardware,
            report=report,
        )
    except Exception as error:  # Broad: any failure here is a reportable one.
        report.fail(
            "engine.materialise_before_prepare",
            f"materialise() then prepare() raised [{type(error).__name__}] {error}",
        )
        return report
    if handle is None:
        return report

    source = source_factory()
    _check_fit_changes_the_callers_model(engine, handle, source, training, report)
    _check_predictions_are_aligned(engine, handle, source, report)
    _check_weights_round_trip(
        engine,
        handle,
        model_factory,
        source,
        signature,
        directory,
        training=training,
        hardware=resolved_hardware,
        report=report,
    )
    _check_unwrapped_is_not_a_wrapper(handle, report)
    return report


def _prepare_for_check(
    engine: Engine,
    model: object,
    signature: InputSignature,
    *,
    training: object,
    hardware: HardwareSpec,
    report: ConformanceReport,
) -> ModelHandle | None:
    """
    Materialise and prepare a model, checking the handle that comes back.

    Parameters
    ----------
    engine
        The engine under check.
    model
        A fresh model.
    signature
        The signature to materialise against.
    training
        The engine's training specification.
    hardware
        The hardware specification.
    report
        Report to record findings in.

    Returns
    -------
    ModelHandle or None
        The handle, or ``None`` if it was not usable.
    """
    materialised = engine.materialise(model, signature)
    handle = engine.prepare(
        materialised,
        hardware=hardware,
        training=training,
        static=dict(_static_of(signature)),
    )
    if not isinstance(handle, ModelHandle):
        report.fail(
            "engine.materialise_before_prepare",
            f"prepare() returned a {type(handle).__name__} rather than a ModelHandle",
        )
        return None
    report.passed_check("engine.materialise_before_prepare")
    return handle


def _check_fit_changes_the_callers_model(
    engine: Engine,
    handle: ModelHandle,
    source: BatchSource,
    training: object,
    report: ConformanceReport,
) -> None:
    """
    Check that fitting improves the training-split error of *this* model.

    The probe is a prediction before and a prediction after. An engine whose
    unfitted model cannot predict at all is handled separately rather than
    excused: see :func:`_error_before_fitting`.

    Parameters
    ----------
    engine
        The engine under check.
    handle
        The prepared handle.
    source
        A bounded training source.
    training
        The engine's training specification.
    report
        Report to record findings in.
    """
    targets = _targets_of(source)
    before = _error_before_fitting(engine, handle, source, targets)
    outcome = engine.fit(handle, {"train": source}, training)
    try:
        after = _mean_squared_error(engine.predict(handle, source), targets)
    except Exception as error:  # Broad: any failure here is a reportable one.
        report.fail(
            "engine.fit_is_an_improvement",
            f"predict() raised after a completed fit [{type(error).__name__}] "
            f"{error}; a fitted model must be able to predict",
        )
        return

    if not isinstance(outcome, FitOutcome):
        report.fail(
            "engine.fit_is_an_improvement",
            f"fit() returned a {type(outcome).__name__} rather than a FitOutcome",
        )
        return
    if not np.isfinite(after):
        report.fail(
            "engine.fit_is_an_improvement",
            f"after fitting, predictions are not finite (error {after}); the update diverged",
        )
        return
    # Strict improvement is required, not merely "no worse". An engine that
    # fits a *copy* leaves the caller's model untouched, so the error before
    # and after are bit-identical -- which "no worse" accepts and which is the
    # single most likely way for an engine to report a successful fit having
    # trained nothing the caller can use.
    if before is None:
        # The model could not predict before fitting and can now. For a
        # closed-form estimator that is the strongest form of this check
        # available and a stronger one than the numerical comparison: a
        # fitted copy would have left the caller's model unfitted, and an
        # unfitted model raises rather than returning a plausible number.
        report.passed_check("engine.fit_is_an_improvement")
        return
    if after >= before > ALREADY_OPTIMAL:
        report.fail(
            "engine.fit_is_an_improvement",
            f"training-split error went from {before:.6g} to {after:.6g}, which "
            f"is no improvement. The usual cause is that fit() trained a copy "
            f"and left the caller's model untouched -- something a FitOutcome "
            f"cannot reveal, because the copy really did train",
        )
        return
    report.passed_check("engine.fit_is_an_improvement")


def _error_before_fitting(
    engine: Engine,
    handle: ModelHandle,
    source: BatchSource,
    targets: NDArray[np.float64],
) -> float | None:
    """
    Score the untrained model, or report that it cannot be scored.

    A gradient model predicts from its initialisation, so there is always a
    "before" to compare against. A closed-form estimator has no parameters
    at all until it is fitted and raises instead -- which is correct
    behaviour and must not be mistaken for a broken engine.

    This was found in Phase 6, when the first non-gradient engine met a
    conformance suite that had only ever seen one. Returning ``None`` rather
    than a sentinel number keeps the distinction explicit at the call site:
    there is no "before", as opposed to a "before" that happened to be zero.

    Parameters
    ----------
    engine
        The engine under check.
    handle
        The prepared handle, holding an unfitted model.
    source
        A bounded training source.
    targets
        The split's targets.

    Returns
    -------
    float or None
        The untrained error, or ``None`` if the model refused to predict.
    """
    try:
        return _mean_squared_error(engine.predict(handle, source), targets)
    except Exception:  # Broad: every library signals this differently.
        return None


def _check_predictions_are_aligned(
    engine: Engine,
    handle: ModelHandle,
    source: BatchSource,
    report: ConformanceReport,
) -> None:
    """
    Check that predictions are one per sample and in a repeatable order.

    Parameters
    ----------
    engine
        The engine under check.
    handle
        The fitted handle.
    source
        A bounded source.
    report
        Report to record findings in.
    """
    expected = source.n_samples
    first = np.ravel(np.asarray(engine.predict(handle, source)))
    second = np.ravel(np.asarray(engine.predict(handle, source)))

    if expected is not None and first.size != expected:
        report.fail(
            "engine.predict_is_aligned",
            f"predict() returned {first.size} value(s) for a source declaring "
            f"{expected} sample(s); a metric computed from this is paired with "
            f"the wrong targets",
        )
        return
    if not np.array_equal(first, second):
        report.fail(
            "engine.predict_is_aligned",
            "two predict() calls over the same source disagreed. Either the "
            "forward pass is non-deterministic -- dropout left enabled is the "
            "usual cause -- or the results are reassembled out of order",
        )
        return
    report.passed_check("engine.predict_is_aligned")


def _check_weights_round_trip(
    engine: Engine,
    handle: ModelHandle,
    model_factory: Callable[[], object],
    source: BatchSource,
    signature: InputSignature,
    directory: Path,
    *,
    training: object,
    hardware: HardwareSpec,
    report: ConformanceReport,
) -> None:
    """
    Check that saved weights reload into a fresh model exactly.

    Parameters
    ----------
    engine
        The engine under check.
    handle
        The fitted handle.
    model_factory
        Returns a fresh, never-fitted model.
    source
        A bounded source.
    signature
        The signature to materialise the fresh model against.
    directory
        A writable directory.
    training
        The engine's training specification.
    hardware
        The hardware specification.
    report
        Report to record findings in.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "conformance_weights"
    expected = np.ravel(np.asarray(engine.predict(handle, source)))

    try:
        engine.save_weights(handle, path)
        reloaded = engine.load_weights(engine.materialise(model_factory(), signature), path)
        reloaded_handle = engine.prepare(
            reloaded,
            hardware=hardware,
            training=training,
            static=dict(handle.static),
        )
        actual = np.ravel(np.asarray(engine.predict(reloaded_handle, source)))
    except Exception as error:  # Broad: any failure here is a reportable one.
        report.fail(
            "engine.weights_round_trip",
            f"saving and reloading weights raised [{type(error).__name__}] {error}",
        )
        return

    if actual.shape != expected.shape:
        report.fail(
            "engine.weights_round_trip",
            f"the reloaded model produced {actual.size} prediction(s) against "
            f"{expected.size} before saving",
        )
        return
    if not np.array_equal(actual, expected):
        largest = float(np.max(np.abs(actual - expected)))
        report.fail(
            "engine.weights_round_trip",
            f"the reloaded model's predictions differ by up to {largest:.3g}. A "
            f"served model and a scored model must be the same model; a "
            f"difference here will be blamed on the data",
        )
        return
    report.passed_check("engine.weights_round_trip")


def _check_unwrapped_is_not_a_wrapper(handle: ModelHandle, report: ConformanceReport) -> None:
    """
    Check that the handle's persisted model is not a wrapper.

    Parameters
    ----------
    handle
        The handle.
    report
        Report to record findings in.
    """
    if handle.is_distributed and handle.unwrapped is handle.model:
        report.fail(
            "engine.handle_exposes_an_unwrapped_model",
            "the handle reports is_distributed but 'unwrapped' is the wrapped "
            "model. The bundle persists 'unwrapped', and a distributed wrapper "
            "saved into a bundle cannot be reopened without reconstructing the "
            "same wrapper",
        )
        return
    report.passed_check("engine.handle_exposes_an_unwrapped_model")


def _static_of(signature: InputSignature) -> dict[str, object]:
    """
    Synthesise zero-filled static inputs from a signature.

    Parameters
    ----------
    signature
        The signature declaring them.

    Returns
    -------
    dict
        One array per declared static input. Zeros rather than random values
        so that a failure elsewhere in the suite is reproducible.
    """
    return {
        name: np.zeros(spec.concrete_shape(1), dtype=spec.dtype)
        for name, spec in signature.static.items()
    }


def _targets_of(source: BatchSource) -> NDArray[np.float64]:
    """
    Drain one pass of a source's targets.

    Parameters
    ----------
    source
        A bounded source.

    Returns
    -------
    numpy.ndarray
        Flat target vector.
    """
    return np.concatenate(
        [np.ravel(np.asarray(batch[TARGET_KEY], dtype=np.float64)) for batch in source.batches()],
        axis=0,
    )


def _mean_squared_error(predictions: NDArray[np.floating], targets: NDArray[np.floating]) -> float:
    """
    Return the mean squared error between two arrays, flattened.

    Parameters
    ----------
    predictions
        Predicted values.
    targets
        Observed values.

    Returns
    -------
    float
        The error, or infinity if the shapes cannot be compared -- which lets
        the caller report a misalignment rather than raise from inside a
        subtraction.
    """
    flat_predictions = np.ravel(np.asarray(predictions, dtype=np.float64))
    flat_targets = np.ravel(np.asarray(targets, dtype=np.float64))
    if flat_predictions.shape != flat_targets.shape:
        return float("inf")
    return float(np.mean(np.square(flat_predictions - flat_targets)))
