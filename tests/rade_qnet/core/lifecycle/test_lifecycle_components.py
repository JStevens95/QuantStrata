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
