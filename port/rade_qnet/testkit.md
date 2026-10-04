# `src/rade_qnet/testkit`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 42 | 2099 | `be147ade92895e66` |
| 2 | `conformance.py` | 1189 | 40460 | `f34d858b31058be2` |
| 3 | `fixtures.py` | 1494 | 47524 | `a10dd886a237dbfa` |
| 4 | `parity.py` | 1009 | 36692 | `366dcea95f2cd3ac` |

---

## 1. `src/rade_qnet/testkit/__init__.py`

2099 bytes · SHA-256 `be147ade92895e66`

```python
"""
Tools that prove a model conforms to the framework's contracts.

A framework that only documents its contracts will see them violated.  This
package makes conformance executable: a model author imports a suite, points it
at their model, and gets a verdict.  It is shipped as part of the package (not
hidden in the test tree) because model authors outside this repository need it
too.

Modules
-------
``conformance.py``
    The contract suite.  Where ``isinstance`` against a protocol only proves a
    method exists, this calls it and checks the result: that a source is
    re-iterable and honest about its size, that a fitted state's inverse is a
    real inverse, that a bundle's splits do not overlap along the scenario
    axis, and that every declared capability behaves.  It deliberately does
    *not* flag entity-axis fitting over the full universe, which is legitimate.
    [Phase 1 delivered, extended per phase]
``fixtures.py``
    Small synthetic sources, specs, bundles and hooks.  Deterministic and fast,
    so a pipeline test needs no real data, no network and no training library.
    ``isolated_registries`` is the one every test that registers a component
    should use.  [Phase 1, delivered]
``parity.py``
    Compares a run against a golden fixture captured from the original
    implementation, at a tolerance chosen per level.  Exact for the fitted
    state and the batch contents, where a difference means different
    arithmetic; looser for the forward pass and the training curve, where
    reassociated floating point legitimately moves the last bits.  The effort
    is in the diagnostic rather than the verdict: a failure names the array,
    the worst element, both values and the mismatch count, because "arrays
    differ" across a refactor of several thousand lines is a day of
    bisection.  [Phase 0+3, delivered]

Not part of the production runtime
----------------------------------
Nothing in the framework's training path imports this package.  It may depend
on ``pytest``, which the rest of ``rade_qnet`` may not.
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/testkit/conformance.py`

40460 bytes · SHA-256 `f34d858b31058be2`

```python
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
- A :class:`~rade_qnet.core.authoring.capabilities.Routable` member that
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

from ..core.authoring.capabilities import (
    CustomStep,
    Inductive,
    Precomputable,
    Routable,
    StaticInputs,
)
from ..core.contract.data import TARGET_KEY, Batch, DataBundle
from ..core.contract.result import FitOutcome
from ..core.contract.signature import InputSignature
from ..core.contract.source import BatchSource
from ..core.contract.state import FittedState
from ..core.provenance.logging import get_logger
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
        :class:`~rade_qnet.core.authoring.capabilities.Routable`. Needed to check
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
```

---

## 3. `src/rade_qnet/testkit/fixtures.py`

47524 bytes · SHA-256 `a10dd886a237dbfa`

```python
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
from ..core.lifecycle.components import ENGINES, LEARNERS, MODELS, REPORTS
from ..core.lifecycle.context import RunContext
from ..core.lifecycle.errors import EngineError
from ..core.lifecycle.hooks import PipelineHook
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
    >>> from rade_qnet.core.lifecycle.components import MODELS
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
```

---

## 4. `src/rade_qnet/testkit/parity.py`

36692 bytes · SHA-256 `366dcea95f2cd3ac`

```python
"""
Compares a refactored run against a golden fixture captured from the original.

The refactor's whole correctness claim is that the flagship model in
``rade_qnet`` produces the same results as the flagship model in ``rade_ml_pt``.
That claim is backed either by an artifact captured before anything changed,
or by an assertion. This module is the half that does the comparing; the
fixture it compares against is written by
``examples/rade_qnet/phase0_capture_baseline.py`` running the *unmodified*
original.

Why the diagnostic matters more than the verdict
------------------------------------------------
A parity failure says a refactor of several thousand lines produced a
different number somewhere. "Arrays differ" turns that into a day of
bisection. ``compare_arrays`` therefore reports the array's name, the index
of the worst element, both values at that index, and how many elements
mismatched out of how many -- which is usually enough to name the cause
without opening a debugger. A single mismatch in a 48,000-element array is a
boundary condition; half the array mismatching is a transpose or an ordering
difference.

Five levels, and why the tolerances widen
-----------------------------------------
``state`` and ``tensors`` are compared **exactly**. They are deterministic
NumPy computations over identical inputs, so there is no floating-point
excuse available: a difference means different arithmetic, which is a bug.

``forward`` allows ``atol=1e-6``. The original's weights are loaded into the
new model and both run on one batch, so the arithmetic is the same but may be
*reassociated* -- a fused kernel, a different reduction order -- which
perturbs the last bits without changing the computation.

``curve`` allows ``rtol=1e-3``. Several epochs of training accumulate that
perturbation. A tighter tolerance here produces a test that fails
occasionally and is therefore disabled, which is worse than a looser one that
still means something.

``job_set`` returns to exact. Nothing about *where* work runs should change
arithmetic, so a difference between a sequential and a parallel run means
state is being shared or seeded per-worker incorrectly.

The ordering trap
-----------------
Basis selection returns an *ordered* list, and that order fixes column
positions in every array downstream. A refactor that selects the same
instruments in a different order passes a set comparison and then fails
everything after it, with a symptom far from the cause. ``compare_state``
therefore compares the basis as a sequence, and
``test_reordered_basis_fails`` exists to keep it that way.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.lifecycle.errors import ContractError
from ..core.provenance.logging import get_logger

__all__ = [
    "ADJACENCY_VALUE_ATOL",
    "CURVE_RTOL",
    "FORWARD_ATOL",
    "Comparison",
    "GoldenFixture",
    "ParityReport",
    "compare_arrays",
    "compare_curve",
    "compare_forward",
    "compare_job_set",
    "compare_state",
    "compare_tensors",
    "load_golden",
]

#: Where fixtures live, relative to the repository root. Resolved from this
#: file rather than from the working directory, so a test passes regardless of
#: where pytest was invoked from.
_FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "rade_qnet" / "golden"

#: Tolerances per level, as justified in the module docstring. Named constants
#: rather than defaults scattered across call sites, so that loosening one is
#: a visible edit in one place rather than an argument someone added.
FORWARD_ATOL = 1e-6
CURVE_RTOL = 1e-3

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Comparison:
    """
    The outcome of comparing one named array or value.

    Parameters
    ----------
    name
        What was compared, used verbatim in the message so a failure names
        the artifact rather than its position in some list.
    passed
        Whether it matched within tolerance.
    detail
        The diagnostic. Empty when it passed, because a report of fifty
        passing lines buries the one that failed.
    n_mismatched
        How many elements differed. Zero for a non-array comparison.
    n_total
        How many elements were compared. The ratio is what distinguishes a
        boundary condition from a systematic difference.
    worst_deviation
        Largest absolute difference found, or ``0.0``. Carried separately
        from the message so a caller can assert on it.
    """

    name: str
    passed: bool
    detail: str = ""
    n_mismatched: int = 0
    n_total: int = 0
    worst_deviation: float = 0.0


@dataclass
class ParityReport:
    """
    The findings for one parity level.

    Holds every comparison rather than stopping at the first failure. One
    difference is a clue; the set of differences is usually the diagnosis --
    every array failing points somewhere entirely different from one array
    failing.

    Parameters
    ----------
    level
        Which level this is, for the messages.
    comparisons
        One entry per artifact compared, in the order compared.
    """

    level: str
    comparisons: list[Comparison] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether every comparison matched."""
        return all(comparison.passed for comparison in self.comparisons)

    @property
    def failures(self) -> list[Comparison]:
        """Only the comparisons that failed, in the order compared."""
        return [comparison for comparison in self.comparisons if not comparison.passed]

    @property
    def worst_deviation(self) -> float:
        """
        The largest absolute difference across every comparison.

        Reported even on a pass, because a level that passes at 9e-7 against
        a tolerance of 1e-6 is worth knowing about before it starts failing.
        """
        return max((comparison.worst_deviation for comparison in self.comparisons), default=0.0)

    def add(self, comparison: Comparison) -> None:
        """
        Record one comparison.

        Parameters
        ----------
        comparison
            The outcome to record.
        """
        self.comparisons.append(comparison)

    def summary(self) -> str:
        """
        Render a human-readable verdict.

        Returns
        -------
        str
            One line on a pass. On a failure, a line per failed comparison,
            which is what gets attached to the assertion so the diagnosis is
            in the test output rather than behind a debugger.
        """
        if self.passed:
            return (
                f"parity level {self.level!r}: {len(self.comparisons)} comparison(s) "
                f"passed, worst deviation {self.worst_deviation:.3e}"
            )
        lines = [
            f"parity level {self.level!r}: {len(self.failures)} of "
            f"{len(self.comparisons)} comparison(s) FAILED"
        ]
        lines.extend(f"  - {comparison.detail}" for comparison in self.failures)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class GoldenFixture:
    """
    A loaded fixture, addressed by artifact name rather than by file path.

    Parameters
    ----------
    name
        The fixture's directory name.
    directory
        Where it was loaded from.
    manifest
        What was captured, from which commit, with which config and seed.
        Carried so a parity failure can report the baseline's provenance,
        which is the first thing to check when a long-passing test starts
        failing.
    """

    name: str
    directory: Path
    manifest: Mapping[str, Any]

    def array(self, relative_path: str) -> NDArray[Any]:
        """
        Load one ``.npy`` array from the fixture.

        Parameters
        ----------
        relative_path
            Path within the fixture directory, including the suffix.

        Returns
        -------
        numpy.ndarray
            The stored array.

        Raises
        ------
        ContractError
            If the file is absent. Raised rather than returning ``None``
            because an incomplete fixture silently skipping a comparison is
            how a parity suite comes to pass everything.
        """
        path = self.directory / relative_path
        if not path.is_file():
            raise ContractError(
                f"the golden fixture {self.name!r} has no file {relative_path!r} "
                f"(looked in {self.directory}). The fixture is incomplete, so this "
                f"comparison would be skipped rather than run -- re-capture it with "
                f"examples/rade_qnet/phase0_capture_baseline.py"
            )
        # `allow_pickle=False` is the default and is relied upon: a fixture is
        # data, and a fixture that can execute code on load is a fixture that
        # could hide a difference rather than reveal one.
        return np.load(path)

    def arrays(self, relative_path: str) -> dict[str, NDArray[Any]]:
        """
        Load every array from one ``.npz`` archive in the fixture.

        Returned eagerly as a dictionary rather than as the lazy handle
        ``numpy.load`` gives, because that handle holds the file open and a
        caller who kept one past the end of a test would leave a descriptor
        behind on every comparison.

        Parameters
        ----------
        relative_path
            Path within the fixture directory, including the suffix.

        Returns
        -------
        dict
            Array name to array.

        Raises
        ------
        ContractError
            If the archive is absent, for the reason given in
            :meth:`array`.
        """
        path = self.directory / relative_path
        if not path.is_file():
            raise ContractError(
                f"the golden fixture {self.name!r} has no file {relative_path!r} "
                f"under {self.directory}; re-capture it rather than skipping the "
                f"comparison"
            )
        # `allow_pickle=False` is the default and is left so deliberately:
        # a fixture is an ordinary file on disk, and unpickling one would
        # execute whatever it contained.
        with np.load(path) as archive:
            return {name: archive[name] for name in archive.files}

    def json(self, relative_path: str) -> object:
        """
        Load one JSON document from the fixture.

        Parameters
        ----------
        relative_path
            Path within the fixture directory, including the suffix.

        Returns
        -------
        object
            The parsed document. Typed as ``object`` rather than ``Any``
            because a fixture holds whatever was captured, and ``Any`` would
            silence the type checker at every call site rather than at this
            one.

        Raises
        ------
        ContractError
            If the file is absent.
        """
        path = self.directory / relative_path
        if not path.is_file():
            raise ContractError(
                f"the golden fixture {self.name!r} has no file {relative_path!r} "
                f"(looked in {self.directory}). The fixture is incomplete, so this "
                f"comparison would be skipped rather than run -- re-capture it with "
                f"examples/rade_qnet/phase0_capture_baseline.py"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    def has(self, relative_path: str) -> bool:
        """
        Whether an artifact is present.

        Parameters
        ----------
        relative_path
            Path within the fixture directory.

        Returns
        -------
        bool
            True when the file exists.
        """
        return (self.directory / relative_path).is_file()


def load_golden(name: str, *, root: Path | None = None) -> GoldenFixture:
    """
    Read a fixture and its manifest.

    Parameters
    ----------
    name
        Fixture directory name, such as ``hybrid_gnn_rnn``.
    root
        Override the fixture root. Only for testing this module against a
        temporary fixture; production callers pass the name alone.

    Returns
    -------
    GoldenFixture
        The loaded fixture.

    Raises
    ------
    ContractError
        If the directory or its manifest is missing, naming the path looked
        at. A fixture that is simply absent must not read as a pass.
    """
    directory = (root or _FIXTURE_ROOT) / name
    if not directory.is_dir():
        raise ContractError(
            f"no golden fixture named {name!r} at {directory}. Capture one with "
            f"examples/rade_qnet/phase0_capture_baseline.py before running parity tests"
        )

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ContractError(
            f"the golden fixture at {directory} has no manifest.json, so there is no "
            f"record of which commit, config or seed produced it. A fixture without "
            f"provenance cannot be a baseline -- re-capture it"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _LOGGER.debug("loaded golden fixture %r captured at %s", name, manifest.get("captured_at"))
    return GoldenFixture(name=name, directory=directory, manifest=manifest)


def _compare_structure(
    actual: NDArray[Any], expected: NDArray[Any], *, name: str
) -> Comparison | None:
    """
    Check shape and dtype, which are reported as themselves or not at all.

    Separated from the value comparison because the two failures call for
    completely different investigations. A shape mismatch reported as a value
    difference sends the reader looking for a numerical bug when the cause is
    a transpose; a dtype mismatch silently upcast hides a real narrowing.

    Parameters
    ----------
    actual
        What the refactored code produced.
    expected
        What the fixture recorded.
    name
        Artifact name for the message.

    Returns
    -------
    Comparison or None
        A failure when the structure differs, otherwise ``None`` to signal
        that the values are worth comparing.
    """
    if actual.shape != expected.shape:
        return Comparison(
            name=name,
            passed=False,
            detail=(
                f"{name}: shape mismatch, got {actual.shape} expected {expected.shape}. "
                f"This is a structural difference, not a numerical one -- check for a "
                f"transpose, a dropped axis, or a different element count before "
                f"looking at values"
            ),
            n_total=expected.size,
        )

    if actual.dtype != expected.dtype:
        # Reported rather than coerced. float32 where float64 was captured is
        # a real narrowing that costs precision downstream, and an integer
        # index array that became a float is usually a bug in how it was
        # built.
        return Comparison(
            name=name,
            passed=False,
            detail=(
                f"{name}: dtype mismatch, got {actual.dtype} expected {expected.dtype}. "
                f"Not upcast automatically, because a narrowed float loses precision "
                f"downstream and an index array that became floating-point is a "
                f"construction bug"
            ),
            n_total=expected.size,
        )
    return None


def _compare_exactly(actual: NDArray[Any], expected: NDArray[Any], *, name: str) -> Comparison:
    """
    Compare non-numeric arrays element by element.

    Strings and objects have no meaningful tolerance, so the first differing
    element is reported rather than the worst one.

    Parameters
    ----------
    actual
        What the refactored code produced.
    expected
        What the fixture recorded.
    name
        Artifact name for the message.

    Returns
    -------
    Comparison
        The outcome.
    """
    mismatched = np.asarray(actual != expected)
    n_mismatched = int(mismatched.sum())
    if n_mismatched == 0:
        return Comparison(name=name, passed=True, n_total=expected.size)

    first = int(np.argmax(mismatched.ravel()))
    return Comparison(
        name=name,
        passed=False,
        detail=(
            f"{name}: {n_mismatched} of {expected.size} element(s) differ; "
            f"first at flat index {first}: got {actual.ravel()[first]!r}, "
            f"expected {expected.ravel()[first]!r}"
        ),
        n_mismatched=n_mismatched,
        n_total=expected.size,
    )


def compare_arrays(
    actual: NDArray[Any],
    expected: NDArray[Any],
    *,
    name: str,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> Comparison:
    """
    Compare two arrays and produce a diagnostic good enough to act on.

    Shape and dtype are checked before values, and reported as themselves. A
    shape mismatch reported as a value difference sends the reader looking for
    a numerical bug when the cause is a transpose or a missing axis; a dtype
    mismatch silently upcast hides a real narrowing.

    Parameters
    ----------
    actual
        What the refactored code produced.
    expected
        What the fixture recorded.
    name
        The artifact's name, used verbatim in the message.
    atol
        Absolute tolerance. The default of zero means exact, which is correct
        for levels 1, 2 and 5.
    rtol
        Relative tolerance, applied as ``rtol * abs(expected)``.

    Returns
    -------
    Comparison
        The outcome, with a diagnostic when it failed.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    structural = _compare_structure(actual, expected, name=name)
    if structural is not None:
        return structural

    if expected.size == 0:
        return Comparison(name=name, passed=True, n_total=0)

    # Non-numeric arrays (strings, objects) have no meaningful tolerance, so
    # they are compared for equality element by element.
    if expected.dtype.kind not in "fciub":
        return _compare_exactly(actual, expected, name=name)

    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    allowed = atol + rtol * np.abs(expected.astype(np.float64))

    # NaN needs handling in both directions, and the subtraction above gets
    # one of them silently wrong. A fixture that recorded NaN and a run that
    # reproduces it have matched, so `both_nan` is excused. But `nan - 2.0` is
    # `nan`, and `nan > tolerance` is *False*, so a refactor that started
    # producing NaN where the original produced a number would otherwise be
    # reported as a pass -- the most dangerous direction this harness has.
    actual_nan = np.isnan(actual.astype(np.float64))
    expected_nan = np.isnan(expected.astype(np.float64))
    both_nan = actual_nan & expected_nan
    only_one_nan = actual_nan ^ expected_nan
    mismatched = ((difference > allowed) & ~both_nan) | only_one_nan
    n_mismatched = int(mismatched.sum())

    finite = difference[np.isfinite(difference)]
    worst_deviation = float(finite.max()) if finite.size else 0.0

    if n_mismatched == 0:
        return Comparison(
            name=name,
            passed=True,
            n_total=expected.size,
            worst_deviation=worst_deviation,
        )

    # The worst *mismatching* element, not the worst overall: with a relative
    # tolerance the largest absolute difference may be the one that passed.
    masked = np.where(mismatched, difference, -np.inf)
    worst_flat = int(np.argmax(masked))
    worst_index = np.unravel_index(worst_flat, expected.shape)
    index_text = worst_index[0] if expected.ndim == 1 else worst_index

    tolerance_text = "exact" if atol == 0.0 and rtol == 0.0 else f"atol={atol:g}, rtol={rtol:g}"
    return Comparison(
        name=name,
        passed=False,
        detail=(
            f"{name}: {n_mismatched} of {expected.size} element(s) differ "
            f"({tolerance_text}); worst at index {index_text}: "
            f"got {actual[worst_index]!r}, expected {expected[worst_index]!r}, "
            f"difference {difference[worst_index]:.6e}"
        ),
        n_mismatched=n_mismatched,
        n_total=expected.size,
        worst_deviation=worst_deviation,
    )


def _compare_sequence(actual: Sequence[Any], expected: Sequence[Any], *, name: str) -> Comparison:
    """
    Compare two ordered sequences, treating order as part of the value.

    Used for the selected basis, where the order fixes column positions in
    every array downstream. A set comparison here would pass on a reordering
    and fail everything after it, with the symptom far from the cause -- so
    the difference is reported as a reordering when the membership matches,
    which names the likely fix directly.

    Parameters
    ----------
    actual
        The sequence produced.
    expected
        The sequence captured.
    name
        Artifact name for the message.

    Returns
    -------
    Comparison
        The outcome.
    """
    actual = list(actual)
    expected = list(expected)

    if actual == expected:
        return Comparison(name=name, passed=True, n_total=len(expected))

    if sorted(map(str, actual)) == sorted(map(str, expected)):
        differing = [
            position
            for position, (left, right) in enumerate(zip(actual, expected, strict=True))
            if left != right
        ]
        return Comparison(
            name=name,
            passed=False,
            detail=(
                f"{name}: same {len(expected)} member(s) in a DIFFERENT ORDER, first "
                f"differing at position {differing[0]}: got {actual[differing[0]]!r}, "
                f"expected {expected[differing[0]]!r}. Order fixes column positions in "
                f"every array downstream, so this alone will fail every later "
                f"comparison -- fix the ordering before investigating anything else"
            ),
            n_mismatched=len(differing),
            n_total=len(expected),
        )

    missing = sorted(set(map(str, expected)) - set(map(str, actual)))
    extra = sorted(set(map(str, actual)) - set(map(str, expected)))
    return Comparison(
        name=name,
        passed=False,
        detail=(
            f"{name}: membership differs, got {len(actual)} expected {len(expected)}; "
            f"missing {missing[:5]}, unexpected {extra[:5]}"
        ),
        n_mismatched=len(missing) + len(extra),
        n_total=len(expected),
    )


def compare_state(actual: Mapping[str, Any], golden: GoldenFixture) -> ParityReport:
    """
    Level 1 — the fitted state, compared exactly.

    Scaler statistics, the selected basis *with its order*, the encoded
    attribute matrix, the sparse graph and the universe. All are
    deterministic NumPy computations over identical inputs, so any difference
    is a difference in arithmetic.

    The index arrays are compared too, and they are the subtle ones: the
    original recomputes them **after** dimensionality reduction, so a refactor
    carrying pre-reduction indices forward produces plausible arrays that are
    wrong in every downstream stage.

    Parameters
    ----------
    actual
        The produced state, keyed by the fixture's artifact names without
        their directory or suffix -- ``scaler_mean``, ``selected_basis``,
        ``combined_features``, and so on.
    golden
        The loaded fixture.

    Returns
    -------
    ParityReport
        One comparison per artifact.
    """
    report = ParityReport(level="1-state")

    # Compared as an ordered sequence, never as a set. See `_compare_sequence`.
    if "selected_basis" in actual:
        report.add(
            _compare_sequence(
                actual["selected_basis"],
                golden.json("level1_state/selected_basis.json"),
                name="selected_basis",
            )
        )

    for key, relative_path in _LEVEL1_ARRAYS.items():
        if key not in actual:
            continue
        if not golden.has(relative_path):
            continue
        report.add(
            compare_arrays(
                np.asarray(actual[key]),
                golden.array(relative_path),
                name=key,
                atol=_LEVEL1_ATOL.get(key, 0.0),
            )
        )

    if "universe" in actual and golden.has("level1_state/universe.json"):
        expected_universe = golden.json("level1_state/universe.json")
        for field_name in ("elementary_ids", "target_ids"):
            if field_name in expected_universe:
                report.add(
                    _compare_sequence(
                        actual["universe"].get(field_name, []),
                        expected_universe[field_name],
                        name=f"universe.{field_name}",
                    )
                )
    return report


#: Level 1 artifacts that are plain arrays, mapped from the key a caller
#: supplies to the file the capture wrote. Kept as data rather than as a chain
#: of `if` statements so that adding an artifact is a one-line change and the
#: capture script can be checked against the same list.
_LEVEL1_ARRAYS: Mapping[str, str] = {
    "scaler_mean": "level1_state/scaler_mean.npy",
    "scaler_scale": "level1_state/scaler_scale.npy",
    "target_scaler_mean": "level1_state/target_scaler_mean.npy",
    "target_scaler_scale": "level1_state/target_scaler_scale.npy",
    "combined_features": "level1_state/combined_features.npy",
    "adjacency_indices": "level1_state/adjacency_indices.npy",
    "adjacency_values": "level1_state/adjacency_values.npy",
    "adjacency_shape": "level1_state/adjacency_shape.npy",
    # Post-reduction, and that is the whole point. The original recomputes
    # these as 0..n_e and n_e..n_e+n_t *after* the basis is selected, so a
    # refactor that carries the pre-reduction indices forward produces arrays
    # that look right and index the wrong columns.
    "elementary_idx": "level1_state/elementary_idx.npy",
    "target_idx": "level1_state/target_idx.npy",
}

#: One float32 ULP at a magnitude of one. The single artifact that is not
#: compared exactly, and the reason is worth stating in full.
#:
#: The baseline found graph neighbours with scikit-learn, whose brute-force
#: euclidean kernel works in float32 and computes the distance from the
#: expanded `|a|^2 - 2ab + |b|^2` rather than from the explicit difference.
#: That expansion cancels: the baseline's distances carry about 2e-07 of
#: error where float64 would carry 1e-16. The port computes the explicit
#: difference in float64 instead, which is both more accurate and free of
#: scikit-learn, and the two therefore disagree in the last float32 bit of
#: each edge weight.
#:
#: Matching bit for bit would mean reproducing the accumulation order of a
#: Cython float32 GEMM -- not stable across scikit-learn versions or BLAS
#: builds, and achievable only by keeping scikit-learn in the data path that
#: the port exists to remove. Being less accurate on purpose, against a
#: moving target, is the worse trade.
#:
#: What is *not* relaxed is the graph's structure. `adjacency_indices` is
#: still compared exactly, and `TestTheGraphIsRobust` measures the margin
#: between the furthest kept neighbour and the nearest dropped one: it is
#: four orders of magnitude above this noise, so no edge is at risk of
#: flipping. The weights differ in their last bit; the graph does not differ.
ADJACENCY_VALUE_ATOL = 1e-7

_LEVEL1_ATOL: Mapping[str, float] = {"adjacency_values": ADJACENCY_VALUE_ATOL}

#: The same tolerance, keyed by the name the batches use. A batch carries
#: the graph verbatim from the fitted state, so the artifact that carries a
#: tolerance at level 1 carries the identical one here -- and must, because
#: nothing between the two stages touches it.
_LEVEL2_ATOL: Mapping[str, float] = {"adjacency_values": ADJACENCY_VALUE_ATOL}


def compare_tensors(
    actual: Mapping[str, Mapping[str, Any]],
    golden: GoldenFixture,
    *,
    allow_missing: Collection[str] = (),
) -> ParityReport:
    """
    Level 2 — batch contents, compared exactly.

    Confirms that moving static inputs out of per-sample collation into a
    separate field changed nothing. The original merged every static tensor
    into every sample and then compared across the batch to recover the one
    copy; the refactor delivers that one copy directly. The network receives
    the same object either way, and this level is what says so.

    Parameters
    ----------
    actual
        Split name to a mapping of tensor name to array, for the batches the
        fixture captured.
    golden
        The loaded fixture.
    allow_missing
        Tensor names the fixture carries that the refactor deliberately does
        not produce. An input the original passed into every batch and never
        read is the motivating case: dropping it is a finding to record, not
        a failure, but it has to be named here so that dropping a second one
        is still a failure.

    Returns
    -------
    ParityReport
        One comparison per tensor per captured batch, plus the split indices.
    """
    report = ParityReport(level="2-tensors")

    if golden.has("level2_tensors/split_indices.json"):
        expected_splits = golden.json("level2_tensors/split_indices.json")
        for split, expected_indices in expected_splits.items():
            if split not in actual:
                continue
            produced = actual[split].get("__indices__")
            if produced is None:
                continue
            report.add(
                compare_arrays(
                    np.asarray(produced, dtype=np.int64),
                    np.asarray(expected_indices, dtype=np.int64),
                    name=f"{split}.split_indices",
                )
            )

    for split, tensors in actual.items():
        relative_path = f"level2_tensors/{split}_batch_000.npz"
        if not golden.has(relative_path):
            continue
        with np.load(golden.directory / relative_path) as stored:
            for tensor_name in sorted(stored.files):
                if tensor_name in allow_missing:
                    # Declared by the caller as deliberately not delivered.
                    # Named one at a time rather than enabled wholesale, so
                    # dropping a second input is a failure that has to be
                    # argued for rather than one the suite absorbs.
                    continue
                if tensor_name not in tensors:
                    report.add(
                        Comparison(
                            name=f"{split}.{tensor_name}",
                            passed=False,
                            detail=(
                                f"{split}.{tensor_name}: present in the fixture but "
                                f"absent from the produced batch. A key that "
                                f"disappeared is either a renamed input or one the "
                                f"refactor stopped delivering; pass it in "
                                f"`allow_missing` if the omission is intended"
                            ),
                        )
                    )
                    continue
                report.add(
                    compare_arrays(
                        np.asarray(tensors[tensor_name]),
                        stored[tensor_name],
                        name=f"{split}.{tensor_name}",
                        atol=_LEVEL2_ATOL.get(tensor_name, 0.0),
                    )
                )
    return report


def compare_forward(
    actual: NDArray[Any], golden: GoldenFixture, *, atol: float = FORWARD_ATOL
) -> ParityReport:
    """
    Level 3 — the forward pass, at ``atol=1e-6``.

    The original's weights are loaded into the refactored model and both are
    run on one fixed batch. The arithmetic is the same but may be
    reassociated by a fused kernel or a different reduction order, which
    perturbs the last bits without changing the computation. A difference
    much larger than the tolerance is a structural difference, not a
    numerical one.

    Parameters
    ----------
    actual
        The refactored model's output for the captured batch.
    golden
        The loaded fixture.
    atol
        Absolute tolerance.

    Returns
    -------
    ParityReport
        A single comparison.
    """
    report = ParityReport(level="3-forward")
    report.add(
        compare_arrays(
            np.asarray(actual, dtype=np.float64),
            golden.array("level3_forward/outputs.npy").astype(np.float64),
            name="forward_outputs",
            atol=atol,
        )
    )
    return report


def compare_curve(
    actual: Mapping[str, Sequence[float]], golden: GoldenFixture, *, rtol: float = CURVE_RTOL
) -> ParityReport:
    """
    Level 4 — the training curve, at ``rtol=1e-3``.

    Several epochs accumulate the last-bit perturbation level 3 tolerates, so
    the tolerance is relative and loose. It is deliberately not tighter: a
    gate that fails one run in ten gets disabled, and a disabled gate proves
    nothing. If this level is flaky at ``1e-3``, the response is to shorten
    the captured run rather than to loosen the tolerance further.

    Parameters
    ----------
    actual
        Curve name to per-epoch values, such as ``train_loss`` and
        ``val_loss``.
    golden
        The loaded fixture.
    rtol
        Relative tolerance.

    Returns
    -------
    ParityReport
        One comparison per curve.
    """
    report = ParityReport(level="4-training")
    expected_curves = golden.json("level4_training/curve.json")

    for curve_name, expected_values in sorted(expected_curves.items()):
        if curve_name not in actual:
            report.add(
                Comparison(
                    name=curve_name,
                    passed=False,
                    detail=(
                        f"{curve_name}: captured in the fixture but not produced. "
                        f"A missing curve is a missing metric, not a passing one"
                    ),
                )
            )
            continue
        report.add(
            compare_arrays(
                np.asarray(actual[curve_name], dtype=np.float64),
                np.asarray(expected_values, dtype=np.float64),
                name=curve_name,
                rtol=rtol,
            )
        )
    return report


def compare_job_set(
    sequential: Mapping[str, NDArray[Any]], parallel: Mapping[str, NDArray[Any]]
) -> ParityReport:
    """
    Level 5 — sequential against parallel, compared exactly.

    Nothing about *where* work runs should change arithmetic. A difference
    here means state is shared between workers or seeds are derived
    per-worker in a way that depends on placement, and both produce results
    that cannot be reproduced on a differently sized machine.

    Compared against each other rather than against a fixture: the claim is
    that the two agree, and pinning either one to a stored artifact would
    make the test fail for the unrelated reason that the model improved.

    Parameters
    ----------
    sequential
        Artifact name to array, from the sequential run.
    parallel
        The same, from the parallel run.

    Returns
    -------
    ParityReport
        One comparison per artifact, plus a check that the two runs produced
        the same set of artifacts.
    """
    report = ParityReport(level="5-job-set")

    only_sequential = sorted(set(sequential) - set(parallel))
    only_parallel = sorted(set(parallel) - set(sequential))
    if only_sequential or only_parallel:
        report.add(
            Comparison(
                name="artifacts",
                passed=False,
                detail=(
                    f"the two runs produced different artifacts: "
                    f"{only_sequential} only sequentially, {only_parallel} only in "
                    f"parallel. A job that ran in one placement and not the other is "
                    f"a scheduling bug, not a numerical one"
                ),
            )
        )

    for name in sorted(set(sequential) & set(parallel)):
        report.add(
            compare_arrays(np.asarray(parallel[name]), np.asarray(sequential[name]), name=name)
        )
    return report
```

