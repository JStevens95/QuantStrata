# `tests/rade_qnet/core/lifecycle`

7 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 33 | 1441 | `41150f68fedf6616` |
| 2 | `test_lifecycle_components.py` | 299 | 10383 | `7e93e52fcab13c55` |
| 3 | `test_lifecycle_context.py` | 309 | 11010 | `c187dac84db6bf6d` |
| 4 | `test_lifecycle_errors.py` | 105 | 3905 | `58c2af643b46fd7a` |
| 5 | `test_lifecycle_hooks.py` | 115 | 4449 | `7732a3679d7e16f5` |
| 6 | `test_lifecycle_pipeline.py` | 327 | 11275 | `d07c5906200a2d10` |
| 7 | `test_lifecycle_registry.py` | 127 | 4747 | `801deef65d258f62` |

---

## 1. `tests/rade_qnet/core/lifecycle/__init__.py`

1441 bytes · SHA-256 `41150f68fedf6616`

```python
"""
Tests for ``rade_qnet.core.lifecycle`` -- pipeline execution machinery.

This sub-tree is where the framework's debuggability is defended. A failing
stage must produce an error naming that stage, a hook must fire in a
predictable order, a seed must reproduce a run exactly, and a spec hash must be
stable across processes -- otherwise the cache silently misses and tuning
re-runs work it already did.

Planned modules
---------------
``test_lifecycle_context.py``
    ``RunContext`` construction, directory layout and logger binding.
    [Phase 1]
``test_lifecycle_pipeline.py``
    The ``step()`` runner: timing, event emission, cache hits and misses, and
    failure wrapped in a ``StageError`` that names the stage.  [Phase 1]
``test_lifecycle_hooks.py``
    Hook ordering, and the guarantee that a misbehaving hook cannot abort a
    run.  [Phase 1]
``test_lifecycle_components.py``
    Registration and lookup by name, duplicate detection and the error raised
    for an unknown name.  [Phase 1]
``test_lifecycle_seeding.py``
    Reproducibility under each determinism level.  [Phase 1]
``test_lifecycle_hashing.py``
    Hash stability: equal specs hash equally, key order does not matter, and
    the hash survives a fresh interpreter.  [Phase 1]
``test_lifecycle_logging.py``
    Contextual identifiers surviving a process boundary.  [Phase 1]
``test_lifecycle_errors.py``
    The error hierarchy and its messages.  [Phase 1]
"""
```

---

## 2. `tests/rade_qnet/core/lifecycle/test_lifecycle_components.py`

10383 bytes · SHA-256 `7e93e52fcab13c55`

```python
"""
Tests for the component registry.

Resolution is **by name**, not by importable dotted path. That is the decision
under test: a spec saying ``model: hybrid_gnn_rnn`` keeps working when the
class moves between modules, whereas a spec naming
``rade_qnet.models.hybrid.HybridModel`` becomes invalid the first time anyone
reorganises a package -- and every bundle written against it becomes
unloadable.
"""

from __future__ import annotations

import pytest

from src.rade_qnet.core.lifecycle.components import (
    ENGINES,
    LEARNERS,
    MODELS,
    REPORTS,
    Registry,
    engine,
    get_engine,
    get_learner,
    get_model,
    get_report,
    import_registrations,
    learner,
    model,
    registration_modules,
    report,
)
from src.rade_qnet.core.lifecycle.errors import ComponentError
from src.rade_qnet.testkit.fixtures import isolated_registries


@pytest.fixture(autouse=True)
def _isolate():
    """
    Restore every registry after each test.

    The registries are module-level and shared by the whole process. Without
    this, a test that registers ``demo`` makes the *next* test's registration
    fail with a duplicate-name error -- a failure that depends on collection
    order and appears in the wrong test.
    """
    with isolated_registries():
        yield


class TestDecorators:
    """The decorators register and annotate in one step."""

    def test_the_model_decorator_registers_and_returns_the_class(self):
        """
        The decorator must return the class, not a wrapper.

        Returning something else would break inheritance and ``isinstance``
        for every decorated model.
        """

        @model("demo", engine="torch")
        class Demo:
            """A model definition."""

        assert get_model("demo") is Demo
        assert Demo.component_name == "demo"
        assert Demo.component_engine == "torch"

    def test_the_engine_decorator_registers(self):
        """Engines resolve by the name a spec uses."""

        @engine("demo_engine")
        class DemoEngine:
            """An engine."""

        assert get_engine("demo_engine") is DemoEngine
        assert DemoEngine.component_name == "demo_engine"

    def test_the_learner_decorator_records_its_engine(self):
        """
        A learner declares which engine it belongs to.

        A PPO learner written against the Torch engine cannot be driven by the
        XGBoost one, and recording the pairing lets that be rejected at
        resolution rather than at the first backward pass.
        """

        @learner("demo_learner", engine="torch")
        class DemoLearner:
            """A learner."""

        assert get_learner("demo_learner") is DemoLearner
        assert DemoLearner.component_engine == "torch"

    def test_the_report_decorator_registers(self):
        """Reports are enabled by name in the spec."""

        @report("demo_report")
        class DemoReport:
            """A report."""

        assert get_report("demo_report") is DemoReport

    def test_each_registry_has_its_own_namespace(self):
        """
        One name may be used in two registries without collision.

        A model and a report may both reasonably be called ``summary``.
        """

        @model("shared", engine="torch")
        class SharedModel:
            """A model."""

        @report("shared")
        class SharedReport:
            """A report."""

        assert get_model("shared") is SharedModel
        assert get_report("shared") is SharedReport


class TestModuleRegistries:
    """The four shared registries exist and are distinct."""

    @pytest.mark.parametrize(
        ("registry", "kind"),
        [(MODELS, "model"), (ENGINES, "engine"), (LEARNERS, "learner"), (REPORTS, "report")],
    )
    def test_each_registry_declares_its_kind(self, registry, kind):
        """
        The kind appears in the repr, and therefore in error messages.

        Singular rather than plural, because it reads as "unknown model 'x'"
        at the point where a user sees it.
        """
        assert f"kind='{kind}'" in repr(registry)

    def test_the_four_registries_are_distinct_objects(self):
        """
        Sharing one registry would let a report shadow a model.

        Distinct objects are what give each kind its own namespace.
        """
        assert len({id(MODELS), id(ENGINES), id(LEARNERS), id(REPORTS)}) == 4

    def test_the_summary_report_is_registered_by_importing_it(self):
        """
        Importing a module with a decorated class registers it.

        This is how a model becomes available to a spec: the model's module is
        imported, and the decorator does the rest.
        """
        # Deliberately deferred: the import *is* the subject of this test.
        # Hoisting it to the module top would make the registration happen at
        # collection time, so the test would pass without demonstrating
        # anything about when registration occurs.
        from src.rade_qnet.analysis.reports.summary import SummaryReport  # noqa: PLC0415

        assert get_report("summary") is SummaryReport


class TestReplayingRegistrationsInAWorker:
    """
    Recording which module registered a component.

    What lets a worker rebuild the registry its parent had.

    A spawned worker starts with a bare interpreter. A component registered
    purely as an import side effect is absent there, and it *appears* to work
    anyway because spawn re-imports the main module -- so the failure arrives
    the first time the entry point changes rather than when the mistake is
    made. See `ARCHITECTURE.md` defect 12.
    """

    def test_registering_records_the_components_own_module(self):
        """
        The module recorded is the component's, not the registry's.

        A decorator runs inside the module being imported, so the decorated
        class's `__module__` is exactly the import a worker has to replay.
        """
        registry = Registry[type]("widget")

        class Widget:
            pass

        registry.register("widget", Widget)

        assert registry.entry("widget").defining_module == Widget.__module__

    def test_registration_modules_round_trips_through_an_import(self, tmp_path, monkeypatch):
        """
        What the parent records is what the worker needs to import.

        The round trip is the property that matters: a name resolved in one
        process yields a module string that makes the same name resolvable
        in another, with nothing in between but the string.
        """
        module = tmp_path / "rade_qnet_probe_round_trip.py"
        module.write_text(
            "from src.rade_qnet.core.lifecycle.components import REPORTS\n"
            "class Probe:\n"
            "    pass\n"
            "REPORTS.register('round_trip', Probe)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        import_registrations(["rade_qnet_probe_round_trip"])

        assert registration_modules((REPORTS, "round_trip")) == ("rade_qnet_probe_round_trip",)

    def test_duplicate_modules_appear_once(self):
        """
        Two names from one module produce one import.

        A worker importing the same module twice is harmless but the list is
        also a record of what a run depends on, and a duplicate reads as a
        second dependency.
        """
        registry = Registry[type]("widget")

        class First:
            pass

        class Second:
            pass

        registry.register("first", First)
        registry.register("second", Second)

        assert registration_modules((registry, "first"), (registry, "second")) == (
            First.__module__,
        )

    def test_an_unregistered_name_fails_in_the_caller(self):
        """
        Resolution happens here, not in the worker.

        This is the point of looking the names up eagerly: the error can list
        what *is* available, whereas the same mistake discovered inside a
        worker surfaces as a process that died.
        """
        with pytest.raises(ComponentError, match="no model named 'absent'"):
            registration_modules((MODELS, "absent"))

    def test_importing_a_module_performs_its_registration(self, tmp_path, monkeypatch):
        """
        The import is what makes the name resolvable.

        Written as a module on disk rather than reusing a package already
        imported by this process, because an import that has already run is
        a no-op the second time -- which would let this pass without the
        registration ever happening.
        """
        module = tmp_path / "rade_qnet_probe_registration.py"
        module.write_text(
            "from src.rade_qnet.core.lifecycle.components import REPORTS\n"
            "class Probe:\n"
            "    pass\n"
            "REPORTS.register('probe', Probe)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        import_registrations(["rade_qnet_probe_registration"])

        assert get_report("probe").__name__ == "Probe"

    def test_importing_registrations_is_safe_to_repeat(self, tmp_path, monkeypatch):
        """
        Idempotent, so the sequential path can call it unconditionally.

        In a parent that already imported everything this is a dictionary
        lookup, which is why the runner does not branch on executor kind --
        and a second import must not trip the duplicate-name guard.
        """
        module = tmp_path / "rade_qnet_probe_repeat.py"
        module.write_text(
            "from src.rade_qnet.core.lifecycle.components import REPORTS\n"
            "class Probe:\n"
            "    pass\n"
            "REPORTS.register('repeat', Probe)\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        import_registrations(["rade_qnet_probe_repeat"])
        import_registrations(["rade_qnet_probe_repeat"])

        assert get_report("repeat").__name__ == "Probe"

    def test_a_missing_module_says_what_it_was_for(self):
        """
        The error names the module and explains the worker/parent split.

        An `ImportError` several frames below anything mentioning a job does
        not tell a user that the cause is where their model is defined.
        """
        with pytest.raises(ComponentError, match="registers a component"):
            import_registrations(["rade_qnet_module_that_does_not_exist"])
```

---

## 3. `tests/rade_qnet/core/lifecycle/test_lifecycle_context.py`

11010 bytes · SHA-256 `c187dac84db6bf6d`

```python
"""
Tests for the run context.

Three behaviours are the point of this module.

A **failing hook never fails a run**, and neither does a failing tracker. Both
are observers; the bundle on disk is the system of record. A broken tracking
credential must not destroy four hours of training.

A **job's seed is derived by hashing**, so re-running one failed member of a
job set alone reproduces exactly what it would have produced inside the set.

The **context is frozen**, so a stage cannot reconfigure the run it is part of.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from src.rade_qnet.core.lifecycle.context import Catalog, RunContext, Tracker
from src.rade_qnet.core.lifecycle.hooks import PipelineHook
from src.rade_qnet.core.provenance.logging import configure_logging, current_context
from src.rade_qnet.storage.runs.catalog import InMemoryCatalog
from src.rade_qnet.storage.runs.tracker import NullTracker
from src.rade_qnet.testkit.fixtures import RecordingHook


@pytest.fixture
def context(run_directory):
    """
    Provide a context with one recording hook attached.

    Parameters
    ----------
    run_directory
        Per-test working directory.

    Returns
    -------
    RunContext
        A context whose single hook records every event.
    """
    return RunContext(
        run_id="r-001",
        spec_digest="d" * 64,
        output_directory=run_directory,
        seed=100,
        hooks=(RecordingHook(),),
    )


class TestImmutability:
    """A stage cannot reconfigure its own run."""

    def test_the_context_is_frozen(self, context):
        """
        Assignment raises.

        A stage that could change the output directory or the seed would make
        the bundle a record of something that did not happen.
        """
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.seed = 999


class TestDirectories:
    """Derived paths are consistent, so two stages agree on where to write."""

    def test_reports_and_bundles_sit_under_the_output_directory(self, context):
        """Everything a run writes is beneath one root."""
        assert context.reports_directory.parent == context.output_directory
        assert context.bundles_directory.parent == context.output_directory

    def test_the_paths_are_stable(self, context):
        """
        Two calls return the same path.

        A derived path that varied -- by timestamp, say -- would mean the
        writer and the reader of a report disagree about where it is.
        """
        assert context.reports_directory == context.reports_directory


class TestJobDerivation:
    """Per-job contexts are derived without mutating the parent."""

    def test_two_jobs_get_different_seeds(self, context):
        """Members of a job set must be independent."""
        assert context.for_job("EURUSD").seed != context.for_job("USDJPY").seed

    def test_a_job_seed_is_reproducible(self, context):
        """
        Deriving twice gives the same seed.

        This is what lets a single failed job be re-run on its own and
        reproduce what it would have done in the full set.
        """
        assert context.for_job("EURUSD").seed == context.for_job("EURUSD").seed

    def test_job_seeds_do_not_depend_on_order(self, context):
        """
        Derivation is by hash, not by position.

        With ``base + index`` seeding, re-running a subset of jobs -- or
        reordering them -- would change every member's seed, so a job set
        would not be reproducible job by job.
        """
        in_order = [context.for_job(job).seed for job in ("a", "b", "c")]
        reversed_order = [context.for_job(job).seed for job in ("c", "b", "a")]
        assert in_order == list(reversed(reversed_order))

    def test_each_job_gets_its_own_directory(self, context):
        """Two members writing to one directory would overwrite each other."""
        assert context.for_job("EURUSD").output_directory != (
            context.for_job("USDJPY").output_directory
        )

    def test_the_directory_can_be_overridden(self, context, tmp_path):
        """A caller may place a job's output wherever it likes."""
        elsewhere = tmp_path / "elsewhere"
        assert context.for_job("EURUSD", output_directory=elsewhere).output_directory == elsewhere

    def test_observers_and_stores_are_inherited(self, context):
        """A job's events reach the same hooks as the run's."""
        derived = context.for_job("EURUSD")
        assert derived.hooks == context.hooks
        assert derived.run_id == context.run_id
        assert derived.spec_digest == context.spec_digest

    def test_the_parent_is_unchanged(self, context):
        """Derivation returns a new context rather than mutating this one."""
        context.for_job("EURUSD")
        assert context.job_id is None


class TestActivate:
    """Activation binds the identifiers used to attribute log lines."""

    def test_identifiers_bind_inside_the_block(self, context):
        """A log line emitted in a stage carries the run and the stage."""
        with context.activate(stage="fit"):
            assert current_context() == {"run_id": "r-001", "stage": "fit"}

    def test_the_job_identifier_binds_when_present(self, context):
        """A job set's logs must be attributable to the job."""
        with context.for_job("EURUSD").activate():
            assert current_context()["job_id"] == "EURUSD"

    def test_identifiers_are_released_on_exit(self, context):
        """Nothing leaks into the next stage."""
        with context.activate(stage="fit"):
            pass
        assert current_context() == {}


class TestNotify:
    """Hook fan-out is tolerant of a broken hook."""

    def test_every_hook_is_called(self, run_directory):
        """Observers see events in the order they were registered."""
        first, second = RecordingHook(), RecordingHook()
        context = RunContext(
            run_id="r",
            spec_digest="d",
            output_directory=run_directory,
            hooks=(first, second),
        )
        context.notify(lambda hook: hook.on_stage_start("fit"), description="fit start")
        assert first.names() == ("stage_start",)
        assert second.names() == ("stage_start",)

    def test_a_raising_hook_does_not_propagate(self, run_directory, caplog):
        """
        The central rule: an observer never fails a run.

        A broken dashboard or an expired tracking credential must not destroy
        a completed training run.
        """

        class Broken(PipelineHook):
            """Raises on every stage start."""

            def on_stage_start(self, stage: str) -> None:
                """Raise, to prove the failure is contained."""
                message = "hook is broken"
                raise RuntimeError(message)

        context = RunContext(
            run_id="r", spec_digest="d", output_directory=run_directory, hooks=(Broken(),)
        )
        with caplog.at_level(logging.WARNING):
            context.notify(lambda hook: hook.on_stage_start("fit"), description="fit start")
        assert "Broken" in caplog.text

    def test_a_raising_hook_does_not_silence_the_others(self, run_directory):
        """
        One broken observer must not suppress the working ones.

        Otherwise registration order would decide whether a dashboard
        receives events.
        """

        class Broken(PipelineHook):
            """Raises on every stage start."""

            def on_stage_start(self, stage: str) -> None:
                """Raise."""
                message = "hook is broken"
                raise RuntimeError(message)

        working = RecordingHook()
        context = RunContext(
            run_id="r",
            spec_digest="d",
            output_directory=run_directory,
            hooks=(Broken(), working),
        )
        context.notify(lambda hook: hook.on_stage_start("fit"), description="fit start")
        assert working.names() == ("stage_start",)


class TestTracking:
    """Tracking is optional and never load-bearing."""

    def test_metrics_without_a_tracker_are_a_no_op(self, context):
        """A run with no tracker configured still runs."""
        context.track_metrics({"mae": 0.1})

    def test_artifacts_without_a_tracker_are_a_no_op(self, context):
        """Same, for artifacts."""
        context.track_artifact(Path("/tmp/x"))

    def test_a_failing_tracker_is_tolerated(self, run_directory, caplog):
        """
        An unreachable tracking server does not end the run.

        The bundle on disk is the system of record; the tracker is a
        convenience.
        """

        class Offline:
            """A tracker that cannot reach its server."""

            def log_params(self, params):
                """Raise."""
                raise ConnectionError

            def log_metrics(self, metrics, *, step=None):
                """Raise."""
                raise ConnectionError

            def log_artifact(self, path, *, name=None):
                """Raise."""
                raise ConnectionError

            def finish(self, *, succeeded):
                """Raise."""
                raise ConnectionError

        context = RunContext(
            run_id="r", spec_digest="d", output_directory=run_directory, tracker=Offline()
        )
        with caplog.at_level(logging.WARNING):
            context.track_metrics({"mae": 0.1})
            context.track_artifact(Path("/tmp/x"))
        assert "tracker failed" in caplog.text


class TestProtocols:
    """Storage satisfies the protocols core declares, structurally."""

    def test_the_in_memory_catalog_satisfies_the_catalog_protocol(self):
        """
        ``core`` declares the interface; ``storage`` implements it.

        The dependency may not run the other way, which is why the protocol
        lives in ``core`` at all.
        """
        assert isinstance(InMemoryCatalog(), Catalog)

    def test_the_null_tracker_satisfies_the_tracker_protocol(self):
        """The default tracker is a real implementation of "do not track"."""
        assert isinstance(NullTracker(), Tracker)

    def test_an_unrelated_object_satisfies_neither(self):
        """The checks discriminate rather than accepting anything."""
        assert not isinstance(object(), Catalog)
        assert not isinstance(object(), Tracker)


class TestDescribe:
    """The opening log line states what the run was configured with."""

    def test_every_salient_field_is_described(self, context):
        """
        A run's own log says how it was set up.

        When a result is questioned months later, this is usually the first
        thing anyone reads.
        """
        configure_logging(level=logging.CRITICAL, force=True)
        rendered = "\n".join(context.describe())
        assert "r-001" in rendered
        assert "100" in rendered
        assert "RecordingHook" in rendered
```

---

## 4. `tests/rade_qnet/core/lifecycle/test_lifecycle_errors.py`

3905 bytes · SHA-256 `58c2af643b46fd7a`

```python
"""
Tests for the error hierarchy.

The property that matters most here is unusual enough to be worth naming: that
``SpecError`` is a ``ValueError``. That dual inheritance is what lets pydantic
collect a cross-field validator's error and attach a field path to it. Without
it, a field-level failure and a cross-field failure report through two
different mechanisms with two different shapes, and a caller cannot handle both.
"""

from __future__ import annotations

import pytest

from src.rade_qnet.core.lifecycle.errors import (
    BundleError,
    CapabilityError,
    ComponentError,
    ContractError,
    RadeQNetError,
    SpecError,
    StageError,
)


class TestHierarchy:
    """Every error is catchable as one framework base."""

    @pytest.mark.parametrize(
        "error_type",
        [SpecError, ContractError, CapabilityError, ComponentError, BundleError],
    )
    def test_every_error_derives_from_the_framework_base(self, error_type):
        """A caller can catch anything the framework raises with one except."""
        assert issubclass(error_type, RadeQNetError)

    def test_stage_error_derives_from_the_framework_base(self):
        """StageError is constructed differently but belongs to the hierarchy."""
        assert issubclass(StageError, RadeQNetError)

    def test_spec_error_is_also_a_value_error(self):
        """
        Assert the dual inheritance pydantic depends on.

        Pydantic only collects ``ValueError`` from a validator. If this
        inheritance is removed, a cross-field validator's error escapes raw
        and loses the field path that a field-level error keeps -- so the two
        failure modes stop reporting the same way.
        """
        assert issubclass(SpecError, ValueError)

    def test_contract_error_is_not_a_value_error(self):
        """
        A contract failure is a framework bug, not bad user input.

        Keeping it off ``ValueError`` means a pydantic validator cannot
        swallow it, which is correct: a contract violation should surface as
        itself rather than as a field-validation message.
        """
        assert not issubclass(ContractError, ValueError)


class TestStageError:
    """StageError attributes a failure to the stage that produced it."""

    def test_message_names_the_stage_and_the_cause(self):
        """The message identifies where to look, not just what broke."""
        error = StageError("fit", KeyError("features"))
        assert "stage 'fit' failed" in str(error)
        assert "KeyError" in str(error)
        assert "features" in str(error)

    def test_message_includes_the_run_when_one_is_known(self):
        """A job-set failure must say which run it came from."""
        error = StageError("build_data", ValueError("bad"), run_id="r-007")
        assert "r-007" in str(error)

    def test_message_omits_the_run_when_none_is_known(self):
        """A run-less failure does not render an empty prefix."""
        error = StageError("build_data", ValueError("bad"))
        assert "run" not in str(error).split("stage")[0]

    def test_the_cause_is_retained(self):
        """
        The original exception stays reachable.

        The wrapper adds attribution; it must not discard the detail needed to
        diagnose the underlying fault.
        """
        cause = KeyError("features")
        error = StageError("fit", cause)
        assert error.cause is cause
        assert error.stage == "fit"


class TestMessages:
    """Errors carry their message, since every message is a diagnosis."""

    @pytest.mark.parametrize(
        "error_type",
        [SpecError, ContractError, CapabilityError, ComponentError, BundleError],
    )
    def test_message_round_trips(self, error_type):
        """str() returns what was passed, with no decoration."""
        assert str(error_type("something specific went wrong")) == "something specific went wrong"
```

---

## 5. `tests/rade_qnet/core/lifecycle/test_lifecycle_hooks.py`

4449 bytes · SHA-256 `7732a3679d7e16f5`

```python
"""
Tests for the pipeline hook base.

The property being pinned down is that :class:`PipelineHook` is usable by
*partial* implementation. Every method is a no-op, so a subclass overrides only
what it cares about and keeps working when the framework adds a new hook point.
If any method were abstract, adding one would break every hook in existence --
including hooks written outside this repository.
"""

from __future__ import annotations

import inspect

import pytest

from src.rade_qnet.core.lifecycle.hooks import PipelineHook

#: Every hook point, with arguments that satisfy its signature. Kept as data so
#: a newly added hook point is a one-line change here rather than a new test.
HOOK_CALLS = (
    ("on_run_start", ("r-1", "digest"), {}),
    ("on_run_end", ("r-1",), {"succeeded": True}),
    ("on_stage_start", ("fit",), {}),
    ("on_stage_end", ("fit",), {"seconds": 1.5}),
    ("on_stage_error", ("fit", ValueError("boom")), {}),
    ("on_epoch_end", (0, {"train_loss": 0.5}), {}),
    ("on_metrics", ("evaluate", {"mae": 0.1}), {}),
    ("on_artifact", ("summary", "/tmp/summary.md"), {}),
)


class TestDefaultImplementation:
    """The base is a working "observe nothing" implementation."""

    @pytest.mark.parametrize(("method", "args", "kwargs"), HOOK_CALLS)
    def test_every_hook_point_is_callable_and_returns_none(self, method, args, kwargs):
        """
        The base can be used directly, and every hook point is a no-op.

        Returning ``None`` is part of the contract: a hook observes and must
        not be able to alter what the pipeline does, because then two runs
        with identical specs would differ according to who was watching.
        """
        assert getattr(PipelineHook(), method)(*args, **kwargs) is None

    @pytest.mark.parametrize(("method", "args", "kwargs"), HOOK_CALLS)
    def test_no_hook_point_is_abstract(self, method, args, kwargs):
        """
        A partial implementation is valid.

        If any method were abstract, adding a hook point would be a breaking
        change for every existing hook.
        """
        del args, kwargs
        assert not getattr(getattr(PipelineHook, method), "__isabstractmethod__", False)

    def test_the_base_is_instantiable(self):
        """Used as a null object wherever a hook is optional."""
        assert isinstance(PipelineHook(), PipelineHook)


class TestPartialSubclass:
    """A subclass overriding one method inherits the rest."""

    def test_overriding_one_method_leaves_the_others_working(self):
        """The property that keeps hooks cheap to write."""

        class OnlyStages(PipelineHook):
            """Observes stage starts and nothing else."""

            def __init__(self) -> None:
                self.stages: list[str] = []

            def on_stage_start(self, stage: str) -> None:
                """Record the stage."""
                self.stages.append(stage)

        hook = OnlyStages()
        hook.on_stage_start("fit")
        # The inherited methods still work, so a pipeline can call all of them
        # without checking which the subclass implemented.
        hook.on_run_start("r-1", "digest")
        hook.on_epoch_end(0, {"train_loss": 0.1})
        hook.on_artifact("summary", "/tmp/x")
        assert hook.stages == ["fit"]


class TestSignatures:
    """Keyword-only arguments guard against positional misreading."""

    @pytest.mark.parametrize(
        ("method", "parameter"),
        [("on_run_end", "succeeded"), ("on_stage_end", "seconds")],
    )
    def test_ambiguous_arguments_are_keyword_only(self, method, parameter):
        """
        A bare ``True`` or ``1.5`` at a call site says nothing.

        Making these keyword-only means a hook implementation cannot silently
        bind them in the wrong order, and a reader of the call site can tell
        what the value means.
        """
        signature = inspect.signature(getattr(PipelineHook, method))
        assert signature.parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY

    def test_the_error_hook_receives_the_exception_itself(self):
        """
        Not a formatted string.

        A hook that wants to classify failures needs the exception's type and
        its traceback, neither of which survives being rendered to text.
        """
        signature = inspect.signature(PipelineHook.on_stage_error)
        assert signature.parameters["error"].annotation == "BaseException"
```

---

## 6. `tests/rade_qnet/core/lifecycle/test_lifecycle_pipeline.py`

11275 bytes · SHA-256 `d07c5906200a2d10`

```python
"""
Tests for the pipeline base and its instrumented step runner.

``step()`` is the single choke point every stage passes through, and this module
pins down everything a user's stage override gets for free by going through it:
timing, lifecycle events, logging context, and a failure that always names its
stage. The point of centralising those is that an override written later cannot
forget any of them -- so these tests are really testing that the customisation
tiers hold.
"""

from __future__ import annotations

import logging

import pytest

from src.rade_qnet.core.lifecycle.context import RunContext
from src.rade_qnet.core.lifecycle.errors import StageError
from src.rade_qnet.core.lifecycle.hooks import PipelineHook
from src.rade_qnet.core.lifecycle.pipeline import Pipeline
from src.rade_qnet.core.provenance.logging import configure_logging, current_context
from src.rade_qnet.testkit.fixtures import RecordingHook


class DemoPipeline(Pipeline[str]):
    """
    A three-stage pipeline that can be made to fail at a chosen stage.

    Parameters
    ----------
    context
        Ambient state for the run.
    fail_at
        Stage that should raise, or ``None`` for a clean run.
    """

    stages = ("first", "second", "third")

    def __init__(self, context: RunContext, *, fail_at: str | None = None) -> None:
        """Store the failure point."""
        super().__init__(context)
        self.fail_at = fail_at
        self.context_during_stage: dict[str, str] | None = None

    def run(self) -> str:
        """Run three stages and return a marker."""
        for stage in self.stages:
            self.step(stage, lambda stage=stage: self._work(stage))
        return "done"

    def _work(self, stage: str) -> str:
        """Do a stage's work, recording the bound logging identifiers."""
        self.context_during_stage = current_context()
        if self.fail_at == stage:
            message = f"{stage} could not complete"
            raise KeyError(message)
        return stage


@pytest.fixture
def context(run_directory):
    """
    Provide a context with a recording hook.

    Parameters
    ----------
    run_directory
        Per-test working directory.

    Returns
    -------
    RunContext
        A context whose hook records every event.
    """
    configure_logging(level=logging.CRITICAL, force=True)
    return RunContext(
        run_id="r-001",
        spec_digest="d" * 64,
        output_directory=run_directory,
        hooks=(RecordingHook(),),
    )


class TestSuccessfulRun:
    """A clean run reports the full lifecycle in order."""

    def test_the_result_is_returned(self, context):
        """``execute`` returns what ``run`` produced."""
        assert DemoPipeline(context).execute() == "done"

    def test_every_stage_is_recorded_as_completed(self, context):
        """Completion order is recorded, not just a count."""
        pipeline = DemoPipeline(context)
        pipeline.execute()
        assert pipeline.completed_stages == ("first", "second", "third")

    def test_every_stage_is_timed(self, context):
        """
        Timing is automatic.

        A stage override gets it without doing anything, which is the point
        of routing everything through one runner.
        """
        pipeline = DemoPipeline(context)
        pipeline.execute()
        assert sorted(pipeline.timings) == ["first", "second", "third"]
        assert all(duration >= 0.0 for duration in pipeline.timings.values())

    def test_the_event_sequence_is_correct(self, context):
        """
        Run start, then paired stage events, then run end.

        The pairing is what a progress display depends on.
        """
        hook = context.hooks[0]
        DemoPipeline(context).execute()
        assert hook.names() == (
            "run_start",
            "stage_start",
            "stage_end",
            "stage_start",
            "stage_end",
            "stage_start",
            "stage_end",
            "run_end",
        )

    def test_the_run_is_reported_as_succeeded(self, context):
        """``on_run_end`` tells the truth about the outcome."""
        hook = context.hooks[0]
        DemoPipeline(context).execute()
        assert hook.events[-1] == ("run_end", "r-001", True)


class TestStageContext:
    """Each stage runs with its own logging identifiers bound."""

    def test_the_stage_name_is_bound_during_the_stage(self, context):
        """
        A log line from inside a stage is attributed to that stage.

        Including a line emitted by a user's model code, which knows nothing
        about the pipeline.
        """
        pipeline = DemoPipeline(context)
        pipeline.execute()
        assert pipeline.context_during_stage == {"run_id": "r-001", "stage": "third"}

    def test_identifiers_are_released_after_the_run(self, context):
        """Nothing leaks into whatever runs next in the process."""
        DemoPipeline(context).execute()
        assert current_context() == {}


class TestFailure:
    """A failure is attributed, reported, and not swallowed."""

    def test_the_failure_names_the_stage(self, context):
        """
        The whole reason for wrapping.

        A bare ``KeyError: 'features'`` six frames into a forward pass does
        not say which stage produced it.
        """
        with pytest.raises(StageError, match="stage 'second' failed"):
            DemoPipeline(context, fail_at="second").execute()

    def test_the_original_exception_is_chained(self, context):
        """Attribution adds information; it must not discard any."""
        with pytest.raises(StageError) as caught:
            DemoPipeline(context, fail_at="second").execute()
        assert isinstance(caught.value.__cause__, KeyError)

    def test_the_run_identifier_is_included(self, context):
        """A job set's failure must say which run it came from."""
        with pytest.raises(StageError, match="r-001"):
            DemoPipeline(context, fail_at="first").execute()

    def test_later_stages_do_not_run(self, context):
        """A pipeline stops at the first failure."""
        pipeline = DemoPipeline(context, fail_at="second")
        with pytest.raises(StageError):
            pipeline.execute()
        assert pipeline.completed_stages == ("first",)

    def test_the_failing_stage_is_still_timed(self, context):
        """
        How long a stage ran before failing is diagnostic.

        A stage that failed after two seconds and one that failed after two
        hours are different problems.
        """
        pipeline = DemoPipeline(context, fail_at="second")
        with pytest.raises(StageError):
            pipeline.execute()
        assert "second" in pipeline.timings

    def test_the_error_is_reported_to_hooks(self, context):
        """An observer learns which stage failed and how."""
        hook = context.hooks[0]
        with pytest.raises(StageError):
            DemoPipeline(context, fail_at="second").execute()
        assert ("stage_error", "second", "KeyError") in hook.events

    def test_run_end_fires_on_failure(self, context):
        """
        An observer is always told the run is over.

        A hook holding an open file or a tracker run must be able to close it
        whatever happened.
        """
        hook = context.hooks[0]
        with pytest.raises(StageError):
            DemoPipeline(context, fail_at="second").execute()
        assert hook.events[-1] == ("run_end", "r-001", False)

    def test_a_nested_stage_error_is_not_wrapped_twice(self, context):
        """
        Attribution happens once.

        A sub-pipeline's ``StageError`` already names its stage; re-wrapping
        would produce "stage fit failed [StageError] stage fit failed ...".
        """

        class Outer(Pipeline[str]):
            """Runs a stage that raises an already-attributed error."""

            stages = ("outer",)

            def run(self) -> str:
                """Raise a pre-attributed StageError from inside a step."""
                return self.step("outer", self._inner)

            def _inner(self) -> str:
                """Raise as though a nested pipeline had failed."""
                raise StageError("inner", ValueError("deep failure"), run_id="r-001")

        with pytest.raises(StageError) as caught:
            Outer(context).execute()
        assert str(caught.value).count("failed") == 1
        assert "inner" in str(caught.value)


class TestReporting:
    """Metrics and artifacts reach hooks and the tracker."""

    def test_metrics_are_published(self, context):
        """A dashboard sees metrics without the pipeline knowing about it."""

        class Reporting(Pipeline[None]):
            """Publishes one metric set."""

            stages = ()

            def run(self) -> None:
                """Publish metrics."""
                self.report_metrics("evaluate", {"mae": 0.25})

        hook = context.hooks[0]
        Reporting(context).execute()
        assert ("metrics", "evaluate", {"mae": 0.25}) in hook.events

    def test_artifacts_are_published(self, context):
        """A written file is announced as a path, not as contents."""

        class Writing(Pipeline[None]):
            """Publishes one artifact."""

            stages = ()

            def run(self) -> None:
                """Publish an artifact."""
                self.report_artifact("summary", self.context.reports_directory / "summary.md")

        hook = context.hooks[0]
        Writing(context).execute()
        assert ("artifact", "summary") in hook.events


class TestHookTolerance:
    """A broken observer cannot break a run, even at run level."""

    def test_a_hook_failing_at_run_start_does_not_fail_the_run(self, run_directory):
        """Consistent with the rule everywhere else that a hook may not."""

        class Broken(PipelineHook):
            """Raises on every hook point."""

            def on_run_start(self, run_id: str, spec_digest: str) -> None:
                """Raise."""
                message = "hook is broken"
                raise RuntimeError(message)

            def on_stage_start(self, stage: str) -> None:
                """Raise."""
                message = "hook is broken"
                raise RuntimeError(message)

        context = RunContext(
            run_id="r", spec_digest="d", output_directory=run_directory, hooks=(Broken(),)
        )
        assert DemoPipeline(context).execute() == "done"


class TestStageDeclaration:
    """The declared sequence is part of the pipeline's interface."""

    def test_stages_are_declared(self, context):
        """
        Declared rather than inferred.

        So the sequence can be rendered in a report and compared against what
        actually ran.
        """
        assert DemoPipeline(context).stages == ("first", "second", "third")

    def test_completed_stages_match_the_declaration_on_a_clean_run(self, context):
        """
        What ran is what was declared.

        A stage that runs without being declared would be invisible to any
        progress display driven by the declaration.
        """
        pipeline = DemoPipeline(context)
        pipeline.execute()
        assert pipeline.completed_stages == pipeline.stages
```

---

## 7. `tests/rade_qnet/core/lifecycle/test_lifecycle_registry.py`

4747 bytes · SHA-256 `801deef65d258f62`

```python
"""
Tests for the generic registry container.

What is under test here is the *mechanism*: a named store that holds classes,
refuses a duplicate with a message naming the kind, and reports what it has.
It knows nothing about models or engines, which is the point of it being a
separate module -- the four concrete registries and the decorators that fill
them are tested next door in ``test_lifecycle_components.py``.
"""

from __future__ import annotations

import pytest

from src.rade_qnet.core.lifecycle.components import (
    Registry,
)
from src.rade_qnet.core.lifecycle.errors import ComponentError
from src.rade_qnet.testkit.fixtures import isolated_registries


@pytest.fixture(autouse=True)
def _isolate():
    """
    Restore every registry after each test.

    The registries are module-level and shared by the whole process. Without
    this, a test that registers ``demo`` makes the *next* test's registration
    fail with a duplicate-name error -- a failure that depends on collection
    order and appears in the wrong test.
    """
    with isolated_registries():
        yield


class TestRegistry:
    """The generic registry behaves predictably."""

    def test_a_registered_component_is_retrievable(self):
        """The basic contract."""
        registry: Registry[type] = Registry("widgets")
        registry.register("alpha", int)
        assert registry.get("alpha") is int

    def test_an_unknown_name_lists_what_is_available(self):
        """
        The error message names the alternatives.

        A bare ``KeyError: 'ridgee'`` makes the user hunt for the real name;
        listing the registered names usually makes the typo self-evident.
        """
        registry: Registry[type] = Registry("widgets")
        registry.register("ridge", int)
        with pytest.raises(ComponentError, match="ridge"):
            registry.get("ridgee")

    def test_a_duplicate_name_is_rejected(self):
        """
        Registering twice raises rather than overwriting.

        A silent overwrite means one of two components is unreachable and
        nothing says which, so a run can quietly train the wrong model.
        """
        registry: Registry[type] = Registry("widgets")
        registry.register("alpha", int)
        with pytest.raises(ComponentError, match="alpha"):
            registry.register("alpha", float)

    def test_names_are_sorted(self):
        """Stable ordering, so an error message and a listing are stable."""
        registry: Registry[type] = Registry("widgets")
        for name in ("gamma", "alpha", "beta"):
            registry.register(name, int)
        assert registry.names() == ("alpha", "beta", "gamma")

    def test_membership_and_length(self):
        """Convenience access used throughout the test suite."""
        registry: Registry[type] = Registry("widgets")
        registry.register("alpha", int)
        assert "alpha" in registry
        assert "missing" not in registry
        assert len(registry) == 1

    def test_metadata_is_retained(self):
        """
        An entry can carry metadata beyond the component itself.

        Used to record which engine a model declares, which the lookups read
        back without instantiating anything.
        """
        registry: Registry[type] = Registry("widgets")
        registry.register("alpha", int, metadata={"engine": "torch"})
        assert registry.entry("alpha").metadata == {"engine": "torch"}


class TestSnapshotAndRestore:
    """Snapshot and restore is what makes test isolation possible."""

    def test_restore_undoes_a_registration(self):
        """The mechanism behind ``isolated_registries``."""
        registry: Registry[type] = Registry("widgets")
        snapshot = registry.snapshot()
        registry.register("temporary", int)
        registry.restore(snapshot)
        assert "temporary" not in registry

    def test_restore_reinstates_a_removed_registration(self):
        """Restoring is a full replacement, not a subtraction."""
        registry: Registry[type] = Registry("widgets")
        registry.register("permanent", int)
        snapshot = registry.snapshot()
        registry.restore({})
        registry.restore(snapshot)
        assert registry.get("permanent") is int

    def test_a_snapshot_is_not_a_live_view(self):
        """
        Taking a snapshot copies it.

        If the snapshot aliased the registry's own storage, later
        registrations would appear in it and restoring would be a no-op --
        which would make every test's isolation silently ineffective.
        """
        registry: Registry[type] = Registry("widgets")
        snapshot = registry.snapshot()
        registry.register("later", int)
        assert "later" not in snapshot
```

