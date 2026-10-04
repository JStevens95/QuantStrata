# `src/rade_qnet/core/runtime`

9 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 62 | 3025 | `041af9b78a9927f4` |
| 2 | `components.py` | 548 | 15865 | `d4892d7b478201e0` |
| 3 | `context.py` | 403 | 14024 | `79547c2c0a1e60a0` |
| 4 | `errors.py` | 153 | 5378 | `f13eb08c3452f71f` |
| 5 | `hashing.py` | 231 | 6505 | `f6d9f4a484edae26` |
| 6 | `hooks.py` | 165 | 5574 | `a12ae9f374abea76` |
| 7 | `logging.py` | 262 | 9009 | `f107778d8a86e24c` |
| 8 | `pipeline.py` | 267 | 9822 | `49e5a0d89f731b58` |
| 9 | `seeding.py` | 240 | 7820 | `bf9dcd660de9c2cb` |

---

## 1. `src/rade_qnet/core/runtime/__init__.py`

3025 bytes · SHA-256 `041af9b78a9927f4`

```python
"""
The machinery that executes a pipeline.

Where ``spec`` and ``contract`` describe *what* a run is, ``runtime`` provides
the apparatus that makes a run happen and makes it observable: a context object
threaded through every stage, an instrumented step runner, lifecycle hooks,
component lookup by name, deterministic seeding and the framework's error
hierarchy.

This package is deliberately small and dependency-free.  It is imported by
every pipeline, every engine and every report writer, so anything heavy placed
here would be paid for by every process the framework starts.

Modules
-------
``errors.py``
    ``RadeQNetError`` and its specialisations: ``SpecError``, ``StageError``,
    ``ContractError``, ``BundleError``, ``CapabilityError``, ``ComponentError``.
    [Phase 1, delivered]
``hashing.py``
    Stable hashes for a spec, a payload, an array set and a file, used for
    cache keys and bundle provenance.  SHA-256 over canonical JSON, never
    Python's ``hash()``, which is salted per process.  [Phase 1, delivered]
``logging.py``
    Structured logging with contextual identifiers (run, job, stage) carried on
    ``contextvars`` so a worker's output is attributable without threading a
    logger through every call.  Configuration is explicit and never happens at
    import.  [Phase 1, delivered]
``seeding.py``
    ``seed_everything(seed, determinism)`` where determinism is ``off``,
    ``warn`` or ``strict``.  Unlike a blanket deterministic flag, this is
    explicit about the performance and kernel-support trade-off being made.
    Engines register their own seeding callbacks, because ``core`` may not
    import a training library.  [Phase 1, delivered]
``components.py``
    The name registry: ``@model``, ``@engine``, ``@learner``, ``@report`` and
    the lookups that resolve a string in a spec to a class.  Resolution by name
    (rather than by importable dotted path) keeps specs stable across
    refactors.  [Phase 1, delivered]
``hooks.py``
    ``PipelineHook`` -- the run, stage, epoch, metric and artifact observation
    points.  A hook may observe but never alter, and a failing hook never fails
    a run.  [Phase 1, delivered]
``context.py``
    ``RunContext`` -- run identifier, working directory, resolved seed, bound
    logging identifiers, hook fan-out and handles to the catalog and tracker.
    One object passed down instead of a dozen keyword arguments.  Also declares
    the ``Catalog`` and ``Tracker`` protocols that ``storage`` implements, since
    the dependency may not run the other way.  [Phase 1, delivered]
``pipeline.py``
    The ``Pipeline`` base and its ``step()`` runner.  Every stage goes through
    ``step()``, which times it, emits start and end events, and wraps failures
    in a ``StageError`` naming the stage.  [Phase 1, delivered]

Planned modules
---------------
``cache.py``
    The step cache, so tuning and evaluation can reuse an expensive data build
    across trials keyed by spec digest.  [Phase 5]
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/core/runtime/components.py`

15865 bytes · SHA-256 `d4892d7b478201e0`

```python
"""
The component registry: resolving a name in a specification to a class.

A specification says ``model: hybrid_gnn_rnn``, not
``model: rade_qnet.models.hybrid_gnn_rnn.register.HybridGnnRnnModel``. The
indirection buys two things.

**Specifications survive refactors.** A saved spec referencing an importable
dotted path becomes invalid the moment a class moves. A registered name is
stable, so a six-month-old bundle still describes how to rebuild its model.

**The framework does not import the model library.** A pipeline resolves a name
through this registry. If it imported a model instead, the framework would
depend on the library it exists to serve, and a user could not add a model
without editing framework code.

On module-global state
----------------------
These registries are process-global, which the coding standard otherwise
forbids. The exception is deliberate and narrow: there is exactly one set of
importable components per process, and a registry is append-only with
duplicate names rejected, so it cannot be mutated into meaning something
different mid-run. :meth:`Registry.snapshot` and :meth:`Registry.restore`
exist so a test, or a scoped plugin load, can contain its registrations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import TypeVar

from .errors import ComponentError

__all__ = [
    "ENGINES",
    "LEARNERS",
    "MODELS",
    "REPORTS",
    "Registry",
    "RegistryEntry",
    "engine",
    "get_engine",
    "get_learner",
    "get_model",
    "get_report",
    "import_registrations",
    "learner",
    "model",
    "registration_modules",
    "report",
]

#: Bound for the decorators below, which accept and return the same class.
ClassT = TypeVar("ClassT", bound=type)


@dataclass(frozen=True, slots=True)
class RegistryEntry[ComponentT]:
    """
    One registered component and the metadata registered alongside it.

    Parameters
    ----------
    name
        The name a specification uses.
    component
        The registered object, usually a class.
    metadata
        Free-form annotations, such as which engine a model requires. Kept
        beside the component rather than on it so that a registry can be
        queried without importing or instantiating anything.
    defining_module
        The dotted module whose import registered this component.

        Recorded so that a worker process can rebuild the registry its
        parent had. A spawned worker starts with a bare interpreter, and the
        only components it knows about are the ones its own imports brought
        in -- which, for a module whose entire purpose is a registration
        side effect, is none of them. It appears to work when the user's
        entry point happens to import the model, because spawn re-imports
        the main module; it then stops working the moment the entry point
        changes, which is a failure that depends on who is calling rather
        than on what is configured.

        Recorded rather than derived from a naming convention, so that a
        model living in a user's own package works exactly as a built-in
        one does.
    """

    name: str
    component: ComponentT
    metadata: Mapping[str, object] = field(default_factory=dict)
    defining_module: str | None = None


class Registry[ComponentT]:
    """
    A name-to-component mapping for one kind of component.

    Parameters
    ----------
    kind
        What the registry holds, used in error messages. Naming the kind is
        what turns "unknown name 'ridge'" into "no model named 'ridge'".
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, RegistryEntry[ComponentT]] = {}

    @property
    def kind(self) -> str:
        """What this registry holds."""
        return self._kind

    def register(
        self,
        name: str,
        component: ComponentT,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """
        Add a component under a name.

        Parameters
        ----------
        name
            The name a specification will use. Must be non-empty.
        component
            The object to register.
        metadata
            Optional annotations stored alongside the component.

        Raises
        ------
        ComponentError
            If the name is empty, or already registered. A duplicate is an
            error rather than a replacement because the alternative is a
            last-import-wins race: which model a name refers to would depend
            on import order, and a specification would silently train a
            different model than the one its author meant.
        """
        if not name:
            raise ComponentError(f"a {self._kind} name must be a non-empty string")
        if name in self._entries:
            existing = self._entries[name].component
            raise ComponentError(
                f"a {self._kind} named {name!r} is already registered "
                f"({existing!r}); names must be unique"
            )
        self._entries[name] = RegistryEntry(
            name=name,
            component=component,
            metadata=metadata or {},
            # The component's own module, not this one's. A decorator runs
            # inside the module being imported, so `__module__` on the
            # decorated class is exactly the module whose import caused the
            # registration -- which is the thing a worker needs to import.
            defining_module=getattr(component, "__module__", None),
        )

    def get(self, name: str) -> ComponentT:
        """
        Resolve a name to its component.

        Parameters
        ----------
        name
            The registered name.

        Returns
        -------
        ComponentT
            The registered component.

        Raises
        ------
        ComponentError
            If the name is not registered. The message lists what *is*
            registered, because the usual cause is a typo or a model package
            that was never imported, and both are obvious from the list.
        """
        try:
            return self._entries[name].component
        except KeyError:
            available = ", ".join(self.names()) or "<none registered>"
            raise ComponentError(
                f"no {self._kind} named {name!r}; available: {available}"
            ) from None

    def entry(self, name: str) -> RegistryEntry[ComponentT]:
        """
        Resolve a name to its full entry, including metadata.

        Parameters
        ----------
        name
            The registered name.

        Returns
        -------
        RegistryEntry
            The entry.

        Raises
        ------
        ComponentError
            If the name is not registered.
        """
        self.get(name)
        return self._entries[name]

    def names(self) -> tuple[str, ...]:
        """
        Return every registered name, sorted.

        Returns
        -------
        tuple of str
            Sorted names, so error messages and listings are deterministic.
        """
        return tuple(sorted(self._entries))

    def snapshot(self) -> dict[str, RegistryEntry[ComponentT]]:
        """
        Return a shallow copy of the registry's contents.

        Intended for scoped registration: take a snapshot, register, then
        restore. A test that registers a throwaway model must not leak it into
        every test that follows.

        Returns
        -------
        dict
            A copy of the internal mapping.
        """
        return dict(self._entries)

    def restore(self, snapshot: Mapping[str, RegistryEntry[ComponentT]]) -> None:
        """
        Replace the registry's contents with a snapshot.

        Parameters
        ----------
        snapshot
            A mapping previously returned by :meth:`snapshot`.
        """
        self._entries = dict(snapshot)

    def __contains__(self, name: object) -> bool:
        """Return whether a name is registered."""
        return name in self._entries

    def __len__(self) -> int:
        """Return the number of registered components."""
        return len(self._entries)

    def __repr__(self) -> str:
        """Return a representation naming the kind and the registered names."""
        return f"Registry(kind={self._kind!r}, names={list(self.names())!r})"


#: Models, keyed by the name a specification's ``model`` field uses.
MODELS: Registry[type] = Registry("model")

#: Engines, keyed by the name a training spec's ``engine`` discriminator uses.
ENGINES: Registry[type] = Registry("engine")

#: Learners -- update rules -- keyed by name.
LEARNERS: Registry[type] = Registry("learner")

#: Report writers, keyed by the names listed in ``ReportsSpec.enabled``.
REPORTS: Registry[type] = Registry("report")


def model(name: str, *, engine: str) -> Callable[[ClassT], ClassT]:
    """
    Register a model class under a name.

    Parameters
    ----------
    name
        The name a specification will use.
    engine
        Which engine the model requires. Recorded as metadata so a pipeline
        can reject a mismatched training spec before building anything, rather
        than failing partway through construction.

    Returns
    -------
    callable
        A decorator that registers the class and returns it unchanged.

    Examples
    --------
    ::

        @model("ridge", engine="sklearn")
        class Ridge(SupervisedModel):
            spec = RidgeSpec
    """

    def decorate(cls: ClassT) -> ClassT:
        MODELS.register(name, cls, metadata={"engine": engine})
        # Written onto the class as well as into the registry so that an
        # instance can report what it was registered as. Error messages about
        # a live object are far more useful with its registered name than with
        # its class name.
        cls.component_name = name  # type: ignore[attr-defined]
        cls.component_engine = engine  # type: ignore[attr-defined]
        return cls

    return decorate


def engine(name: str) -> Callable[[ClassT], ClassT]:
    """
    Register an engine class under a name.

    Parameters
    ----------
    name
        The name used by a training spec's ``engine`` discriminator.

    Returns
    -------
    callable
        A decorator that registers the class and returns it unchanged.
    """

    def decorate(cls: ClassT) -> ClassT:
        ENGINES.register(name, cls)
        cls.component_name = name  # type: ignore[attr-defined]
        return cls

    return decorate


def learner(name: str, *, engine: str) -> Callable[[ClassT], ClassT]:
    """
    Register a learner -- an update rule -- under a name.

    Parameters
    ----------
    name
        The name a training spec's ``learner`` field uses.
    engine
        Which engine the learner belongs to. A learner is inherently
        engine-specific, so the engine is required rather than optional.

    Returns
    -------
    callable
        A decorator that registers the class and returns it unchanged.
    """

    def decorate(cls: ClassT) -> ClassT:
        LEARNERS.register(name, cls, metadata={"engine": engine})
        cls.component_name = name  # type: ignore[attr-defined]
        cls.component_engine = engine  # type: ignore[attr-defined]
        return cls

    return decorate


def report(name: str) -> Callable[[ClassT], ClassT]:
    """
    Register a report writer under a name.

    Parameters
    ----------
    name
        The name listed in ``ReportsSpec.enabled``.

    Returns
    -------
    callable
        A decorator that registers the class and returns it unchanged.
    """

    def decorate(cls: ClassT) -> ClassT:
        REPORTS.register(name, cls)
        cls.component_name = name  # type: ignore[attr-defined]
        return cls

    return decorate


def get_model(name: str) -> type:
    """
    Resolve a model name.

    Parameters
    ----------
    name
        A registered model name.

    Returns
    -------
    type
        The model class.

    Raises
    ------
    ComponentError
        If the name is not registered.
    """
    return MODELS.get(name)


def registration_modules(*names: tuple[Registry[object], str]) -> tuple[str, ...]:
    """
    Return the modules whose import registered the named components.

    What a worker process needs in order to resolve the same names its
    parent could. A spawned worker starts with a bare interpreter, so a
    component registered purely as an import side effect is simply absent
    there -- and the symptom is "no model named ...", from a specification
    that is perfectly correct and that worked a moment ago in the parent.

    Resolving each name first is deliberate: it means an unregistered name
    fails here, in the parent, where the error can list what *is* available,
    rather than inside a worker where it would surface as a dead process.

    Parameters
    ----------
    *names
        Pairs of registry and name to look up.

    Returns
    -------
    tuple of str
        Dotted module names, deduplicated, in the order given. Components
        with no recorded module are omitted; nothing can be done for them,
        and a worker that cannot resolve one will say so clearly.

    Raises
    ------
    ComponentError
        If any name is not registered in the parent.
    """
    modules: list[str] = []
    for registry, name in names:
        module = registry.entry(name).defining_module
        if module is not None and module not in modules:
            modules.append(module)
    return tuple(modules)


def import_registrations(modules: Iterable[str]) -> None:
    """
    Import modules for their registration side effects.

    The worker-side counterpart of :func:`registration_modules`. Importing
    is idempotent, so this is safe to call unconditionally -- in a parent
    that already has everything, every import is a dictionary lookup.

    Parameters
    ----------
    modules
        Dotted module names.

    Raises
    ------
    ComponentError
        If a module cannot be imported, naming it. The alternative is an
        ``ImportError`` from inside a worker, several frames below anything
        that mentions a job or a model.
    """
    for module in modules:
        try:
            import_module(module)
        except ImportError as error:
            raise ComponentError(
                f"could not import {module!r}, which registers a component this "
                f"run needs. A worker process cannot see registrations made only "
                f"in the process that launched it, so the module has to be "
                f"importable by name here: {error}"
            ) from error


def get_engine(name: str) -> type:
    """
    Resolve an engine name.

    Parameters
    ----------
    name
        A registered engine name.

    Returns
    -------
    type
        The engine class.

    Raises
    ------
    ComponentError
        If the name is not registered.
    """
    return ENGINES.get(name)


def get_learner(name: str) -> type:
    """
    Resolve a learner name.

    Parameters
    ----------
    name
        A registered learner name.

    Returns
    -------
    type
        The learner class.

    Raises
    ------
    ComponentError
        If the name is not registered.
    """
    return LEARNERS.get(name)


def get_report(name: str) -> type:
    """
    Resolve a report name.

    Parameters
    ----------
    name
        A registered report name.

    Returns
    -------
    type
        The report class.

    Raises
    ------
    ComponentError
        If the name is not registered.
    """
    return REPORTS.get(name)
```

---

## 3. `src/rade_qnet/core/runtime/context.py`

14024 bytes · SHA-256 `79547c2c0a1e60a0`

```python
"""
The ambient state every pipeline stage can reach.

:class:`RunContext` is what a stage is given besides its typed input: where to
write, which run it belongs to, which seed it was given, who is observing it,
and where to record what it produced. Threading those through every signature
individually would make each stage's parameters mostly plumbing; hiding them
in module globals would make two concurrent runs in one process impossible.

Why the storage protocols are declared here
-------------------------------------------
:class:`Catalog` and :class:`Tracker` are *implemented* in ``rade_qnet.storage``
but *declared* in ``core``. That inversion is forced and is worth
understanding, because it recurs across the framework.

``core`` may not import ``storage`` -- the dependency runs the other way, and
the layering test enforces it. But a ``RunContext`` living in ``core`` has to
be able to hold a catalog. Declaring the protocol here resolves it: ``core``
owns the *interface*, ``storage`` owns the *implementation*, and the arrow
still points from ``storage`` to ``core``. This is dependency inversion, and
it is also what lets a test substitute an in-memory catalog with no filesystem
at all.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .hooks import PipelineHook
from .logging import bound_context, get_logger
from .seeding import derive_seed

if TYPE_CHECKING:
    from logging import Logger

    from ..contract.bundle import Manifest

__all__ = ["Catalog", "RunContext", "Tracker"]

_LOGGER = get_logger(__name__)


@runtime_checkable
class Catalog(Protocol):
    """
    The registry of what has been trained, and where it was put.

    Implemented by ``rade_qnet.storage.catalog``; declared here for the reason
    given in the module docstring.

    The implementation is single-writer by construction. Version assignment
    must be atomic, because the obvious implementation -- read the catalog,
    compute ``max + 1``, write it back -- loses a bundle whenever two runs
    finish together, and a parallel job set finishes runs together by design.

    Only the three methods a *pipeline* calls are declared here. The query
    side -- ``entries`` and ``records``, which
    :class:`~rade_qnet.storage.registry.RunRegistry` reads -- is deliberately
    absent, for the reason given at
    :class:`~rade_qnet.core.capability.supervised.RebuildableDataModule`: an
    ``isinstance`` check against a runtime-checkable protocol only tests that
    the methods exist, so widening this one would make every three-method
    stub in the suite stop satisfying it, failing training runs over methods
    the training path never calls. The registry constructs a concrete
    :class:`~rade_qnet.storage.catalog.JsonlCatalog` instead of accepting any
    ``Catalog``, which is what makes that safe.
    """

    def next_version(self, model_name: str, *, job_id: str | None = None) -> int:
        """
        Reserve the next version number for a model, atomically.

        Parameters
        ----------
        model_name
            Registered model name.
        job_id
            Job identifier within a job set, or ``None`` for a single run.
            Versions are numbered per job, so each member of a job set has its
            own sequence.

        Returns
        -------
        int
            A version number no other caller will be given.
        """
        ...

    def record(self, manifest: Manifest, *, location: Path | None = None) -> None:
        """
        Record a written bundle.

        Called after the bundle is on disk, never before: a catalog entry
        pointing at a directory that does not exist is worse than a bundle
        with no entry, because the first breaks every reader and the second
        is recoverable by rescanning.

        Parameters
        ----------
        manifest
            The manifest of the bundle that was written.
        location
            The bundle's directory. Recorded so a run selected from the
            catalog -- by tag, by alias, by best metric -- can be opened
            without the reader knowing the directory layout, which differs
            between a single run and a job set. Optional so an entry can
            still be recorded where no directory exists.
        """
        ...

    def latest(self, model_name: str, *, job_id: str | None = None) -> Manifest | None:
        """
        Return the most recent recorded manifest.

        Parameters
        ----------
        model_name
            Registered model name.
        job_id
            Job identifier, or ``None``.

        Returns
        -------
        Manifest or None
            The latest manifest, or ``None`` if nothing is recorded.
        """
        ...


@runtime_checkable
class Tracker(Protocol):
    """
    An experiment tracker.

    Declared in ``core`` for the same reason as :class:`Catalog`, and with the
    same consequence: the framework's default implementation does nothing, and
    a tracking failure degrades to a warning rather than ending the run. A
    model that trained successfully but could not reach a tracking server has
    still trained successfully.
    """

    def log_params(self, params: Mapping[str, object]) -> None:
        """
        Record the run's configuration.

        Parameters
        ----------
        params
            Flattened specification values.
        """
        ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """
        Record metrics, optionally against a step.

        Parameters
        ----------
        metrics
            Metric name to value.
        step
            Epoch or boosting round, or ``None`` for a final metric.
        """
        ...

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        """
        Record a written file.

        Parameters
        ----------
        path
            The file.
        name
            Logical name, defaulting to the file name.
        """
        ...

    def finish(self, *, succeeded: bool) -> None:
        """
        Close the tracking run.

        Parameters
        ----------
        succeeded
            Whether the run completed.
        """
        ...


@dataclass(frozen=True, slots=True)
class RunContext:
    """
    Ambient state for one run, passed to every stage.

    Frozen, so a stage cannot reconfigure the run it is part of. Derived
    contexts -- one per job in a job set -- are produced by :meth:`for_job`,
    which returns a new context rather than mutating this one.

    Parameters
    ----------
    run_id
        Identifier for this run, used in log lines, directory names and
        catalog entries.
    spec_digest
        Digest of the run specification.
    output_directory
        Root for everything this run writes.
    seed
        Base seed. A job's seed is derived from it rather than shared; see
        :meth:`for_job`.
    job_id
        Job identifier within a job set, or ``None`` for a single run.
    hooks
        Observers, notified in order.
    catalog
        Where bundles are recorded, or ``None`` to skip recording.
    tracker
        Where metrics are tracked, or ``None`` to skip tracking.
    """

    run_id: str
    spec_digest: str
    output_directory: Path
    seed: int = 0
    job_id: str | None = None
    hooks: tuple[PipelineHook, ...] = ()
    catalog: Catalog | None = None
    tracker: Tracker | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def logger(self) -> Logger:
        """
        A logger whose records carry this run's identifiers.

        The identifiers are attached by the contextvar-based filter rather
        than by this property, so a log line emitted deep inside a model's
        own code is tagged too, without that code knowing about the context.
        """
        return get_logger("rade_qnet.run")

    @property
    def reports_directory(self) -> Path:
        """Where reports and figures are written."""
        return self.output_directory / "reports"

    @property
    def bundles_directory(self) -> Path:
        """Where bundles are written."""
        return self.output_directory / "bundles"

    def activate(self, *, stage: str | None = None) -> AbstractContextManager[None]:
        """
        Return a context manager binding this run's identifiers for logging.

        Parameters
        ----------
        stage
            Stage name to bind alongside the run and job identifiers.

        Returns
        -------
        AbstractContextManager
            A manager that binds on entry and restores on exit, so a nested
            stage does not leak its name to its parent.
        """
        return bound_context(run_id=self.run_id, job_id=self.job_id, stage=stage)

    def for_job(self, job_id: str, *, output_directory: Path | None = None) -> RunContext:
        """
        Derive a context for one member of a job set.

        The derived seed is a hash of the base seed and the job identifier,
        not ``base + index``. Two properties follow, and both matter for a
        parallel job set:

        - **Order independence.** A job's seed depends on its identifier, so
          re-running a job set with the jobs reordered, or re-running one
          failed job alone, reproduces the same result.
        - **No accidental correlation.** Consecutive integer seeds produce
          correlated streams in some generators, which would make members of
          a job set less independent than they appear.

        Parameters
        ----------
        job_id
            Identifier for the job.
        output_directory
            Override the derived directory. Defaults to a subdirectory named
            after the job.

        Returns
        -------
        RunContext
            A new context; this one is unchanged.
        """
        return RunContext(
            run_id=self.run_id,
            spec_digest=self.spec_digest,
            output_directory=output_directory or (self.output_directory / "jobs" / job_id),
            seed=derive_seed(self.seed, job_id),
            job_id=job_id,
            hooks=self.hooks,
            catalog=self.catalog,
            tracker=self.tracker,
            metadata=self.metadata,
        )

    def notify(self, call: Callable[[PipelineHook], None], *, description: str) -> None:
        """
        Invoke a callable against every hook, tolerating failures.

        Takes a callable rather than a method name so the call is type
        checked: ``context.notify(lambda hook: hook.on_stage_start(stage))``
        is verified against :class:`PipelineHook`, where a string name would
        not be.

        Centralising the error handling here is what makes the "a hook may
        not break a run" rule true everywhere rather than in each of the
        dozen places a hook is called. A raising hook is logged with its class
        and the description below, and the remaining hooks still run -- one
        broken observer must not silence the others.

        Parameters
        ----------
        call
            What to invoke on each hook.
        description
            What was being reported, for the warning message.
        """
        for hook in self.hooks:
            try:
                call(hook)
            # Deliberately broad: see the rule above -- observers never fail runs.
            except Exception:
                _LOGGER.warning(
                    "hook %s raised while reporting %s; continuing",
                    type(hook).__name__,
                    description,
                    exc_info=True,
                )

    def track_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """
        Send metrics to the tracker, if there is one, tolerating failures.

        Parameters
        ----------
        metrics
            Metric name to value.
        step
            Epoch or boosting round, or ``None``.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(metrics, step=step)
        except Exception:  # Broad by design: tracking is never load-bearing.
            _LOGGER.warning("tracker failed to log metrics; continuing", exc_info=True)

    def track_artifact(self, path: Path, *, name: str | None = None) -> None:
        """
        Send an artifact to the tracker, if there is one, tolerating failures.

        Parameters
        ----------
        path
            The file that was written.
        name
            Logical name, defaulting to the file name.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_artifact(path, name=name)
        except Exception:  # Broad by design: tracking is never load-bearing.
            _LOGGER.warning("tracker failed to log artifact %s; continuing", path, exc_info=True)

    def describe(self) -> Sequence[str]:
        """
        Return a short human-readable description, for the run's opening log.

        Returns
        -------
        Sequence of str
            One line per salient field.
        """
        lines = [
            f"run_id         {self.run_id}",
            f"job_id         {self.job_id or '-'}",
            f"spec_digest    {self.spec_digest}",
            f"seed           {self.seed}",
            f"output         {self.output_directory}",
            f"hooks          {', '.join(type(h).__name__ for h in self.hooks) or '-'}",
            f"catalog        {type(self.catalog).__name__ if self.catalog else '-'}",
            f"tracker        {type(self.tracker).__name__ if self.tracker else '-'}",
        ]
        return tuple(lines)
```

---

## 4. `src/rade_qnet/core/runtime/errors.py`

5378 bytes · SHA-256 `f13eb08c3452f71f`

```python
"""
The framework's error hierarchy.

Every error the framework raises derives from :class:`RadeQNetError`, and the
subclass carries information a bare ``ValueError`` cannot: **whose fault it
is**.

That distinction is not pedantry. A :class:`SpecError` means the user wrote
something invalid and can fix it by editing their configuration. A
:class:`ContractError` means the framework or a model violated an internal
promise, and no amount of configuration will help. Collapsing both into
``ValueError`` throws that information away at exactly the moment someone is
trying to work out what to do next.
"""

from __future__ import annotations

__all__ = [
    "BundleError",
    "CapabilityError",
    "ComponentError",
    "ContractError",
    "EngineError",
    "RadeQNetError",
    "SpecError",
    "StageError",
]


class RadeQNetError(Exception):
    """
    Base class for every error the framework raises.

    Catching this catches everything the framework considers its own, and
    nothing else. A ``KeyError`` escaping from inside a stage is therefore
    distinguishable from a deliberate framework error -- it is a bug.
    """


class SpecError(RadeQNetError, ValueError):
    """
    A specification is invalid.

    Actionable by the user: a key is misspelled, a value is out of range, or
    two settings contradict each other. Raised during validation, before any
    expensive work begins.

    Notes
    -----
    This is the one error that also derives from ``ValueError``, and the reason
    is pydantic. A cross-field validator raising a plain ``Exception`` escapes
    pydantic untouched, so it arrives without the dotted path to the offending
    field -- while a field-level failure arrives with one. The result is two
    classes of configuration error reported in two different shapes.

    Deriving from ``ValueError`` makes pydantic collect these alongside
    field-level failures, so ``parse_run_spec`` can report every problem in one
    message with every path. The dual inheritance is therefore not convenience:
    it is what makes configuration errors uniform.
    """


class ContractError(RadeQNetError):
    """
    A stage contract was violated.

    Not actionable by the user -- this is a framework or model bug. Raised
    when a payload arrives with the wrong shape, a required field is absent,
    or a declared signature does not match the data produced.
    """


class CapabilityError(RadeQNetError):
    """
    A model was asked for a capability it does not implement.

    Raised in preference to returning a plausible-looking result. A model with
    no inductive capability asked to predict for an unseen entity must fail,
    not guess.
    """


class ComponentError(RadeQNetError):
    """
    A component could not be registered or resolved by name.

    Covers both directions: two components claiming the same name, and a
    specification naming a component that was never registered.
    """


class EngineError(RadeQNetError):
    """
    An engine could not do what it was asked.

    Covers the engine's own failures and the mismatches it is the first to
    notice: a training spec belonging to a different engine, a model whose
    shapes disagree with the signature it was built from, a checkpoint whose
    parameter names do not match the model loading it.

    Distinct from :class:`ContractError` because the two have different
    audiences. A contract violation is a framework or model bug; an engine
    error is frequently a *configuration* problem the user can fix -- asking
    for a graph network on the XGBoost engine, or for a precision the device
    cannot provide.

    Notes
    -----
    An *absent* accelerator is deliberately not one of these. A run that asked
    for CUDA on a machine without it warns and continues on the CPU, because
    failing would discard a run that would otherwise have succeeded. What is
    reported here is a request that can never be honoured, not one that cannot
    be honoured here.
    """


class BundleError(RadeQNetError):
    """
    A bundle could not be written, read or verified.

    Includes checksum mismatches, which is how silent corruption is turned
    into a loud failure at load time rather than into inexplicable
    predictions later.
    """


class StageError(RadeQNetError):
    """
    A pipeline stage failed.

    Always names the stage, so a traceback from a forty-job run identifies
    *where* in the lifecycle the failure happened without needing to be read.
    The original exception is retained on :attr:`cause` and chained with
    ``raise ... from``, so nothing is lost by wrapping it.

    Parameters
    ----------
    stage
        Name of the stage that failed, as passed to ``Pipeline.step``.
    cause
        The exception the stage raised.
    run_id
        Identifier of the run, included in the message when available.
    """

    def __init__(self, stage: str, cause: BaseException, *, run_id: str | None = None) -> None:
        self.stage = stage
        self.cause = cause
        self.run_id = run_id
        # The run identifier is the first thing needed to find the artifacts of
        # a failed job, so it leads the message when it is known.
        prefix = f"run '{run_id}': " if run_id is not None else ""
        super().__init__(f"{prefix}stage '{stage}' failed [{type(cause).__name__}] {cause}")
```

---

## 5. `src/rade_qnet/core/runtime/hashing.py`

6505 bytes · SHA-256 `f6d9f4a484edae26`

```python
"""
Stable content hashes for specifications, arrays and files.

These digests are used as cache keys and as bundle provenance, which imposes a
requirement stronger than it first appears: **the digest of equal inputs must
be equal in a different process, on a different day, on a different machine.**

Python's built-in ``hash()`` satisfies none of that. String hashing is salted
per interpreter, so ``hash("a")`` differs between two processes of the same
program. A cache keyed on it misses every time, and a hyper-parameter search
silently rebuilds its dataset once per trial instead of once in total. Every
digest here is therefore a SHA-256 over a canonical byte encoding.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic import BaseModel

__all__ = [
    "abbreviate_digest",
    "canonical_json",
    "digest_arrays",
    "digest_file",
    "digest_payload",
    "digest_spec",
]

#: Read size for file hashing. Large enough that the loop is not syscall-bound,
#: small enough that a multi-gigabyte checkpoint is not held in memory.
_FILE_CHUNK_BYTES = 1 << 20


def _json_default(value: object) -> Any:  # noqa: ANN401 - json hook signature
    """
    Encode values ``json`` cannot serialise on its own.

    Only types that genuinely appear in specifications and lineage are
    handled. Anything else raises, which is deliberate: silently encoding an
    unknown object via ``repr`` would produce a digest that changes when an
    unrelated ``__repr__`` changes.

    Parameters
    ----------
    value
        The object ``json.dumps`` could not encode.

    Returns
    -------
    Any
        A JSON-encodable replacement.

    Raises
    ------
    TypeError
        If the value has no canonical encoding.
    """
    if isinstance(value, Path):
        # Posix form so a digest computed on Windows matches one from Linux.
        return value.as_posix()
    if isinstance(value, (set, frozenset)):
        # Sets have no order, so one must be imposed or the digest is unstable.
        return sorted(str(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"no canonical JSON encoding for type {type(value).__name__!r}")


def canonical_json(payload: object) -> str:
    """
    Render a payload as JSON in a form that depends only on its content.

    Keys are sorted and separators are fixed, so two mappings that differ only
    in insertion order produce identical text -- and therefore identical
    digests.

    Parameters
    ----------
    payload
        Any JSON-encodable structure, plus the extra types handled by the
        module's encoder hook.

    Returns
    -------
    str
        Canonical JSON text.

    Raises
    ------
    TypeError
        If the payload contains a value with no canonical encoding.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def digest_payload(payload: object) -> str:
    """
    Return the SHA-256 digest of a payload's canonical JSON form.

    Parameters
    ----------
    payload
        Any structure acceptable to :func:`canonical_json`.

    Returns
    -------
    str
        Lowercase hexadecimal digest, 64 characters.
    """
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def digest_spec(spec: BaseModel) -> str:
    """
    Return the digest of a specification.

    The spec is dumped in JSON mode first, so that ``Path``, ``Enum`` and
    similar fields are reduced to their serialised form. This matters because
    the digest must match the one computed from the same spec after it has
    been written to disk and read back -- that round trip is what a step cache
    relies on.

    Parameters
    ----------
    spec
        Any pydantic model.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    return digest_payload(spec.model_dump(mode="json"))


def digest_file(path: Path, *, chunk_bytes: int = _FILE_CHUNK_BYTES) -> str:
    """
    Return the SHA-256 digest of a file's contents.

    Read in chunks so that hashing a large checkpoint does not require holding
    it in memory.

    Parameters
    ----------
    path
        File to hash.
    chunk_bytes
        Read size in bytes.

    Returns
    -------
    str
        Lowercase hexadecimal digest.

    Raises
    ------
    OSError
        If the file cannot be read.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    """
    Return a digest over a named collection of arrays.

    Shape and dtype are folded into the digest alongside the bytes, so two
    arrays holding the same values in different shapes or precisions hash
    differently. Without that, a ``float32`` and ``float64`` view of the same
    data would be treated as interchangeable by a cache, and they are not.

    Parameters
    ----------
    arrays
        Mapping of name to array. Names are hashed in sorted order, so the
        mapping's iteration order is irrelevant.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    hasher = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def abbreviate_digest(digest: str, *, length: int = 12) -> str:
    """
    Shorten a digest for use in a directory name or log line.

    Parameters
    ----------
    digest
        A full hexadecimal digest.
    length
        Number of leading characters to keep. Twelve hexadecimal characters is
        48 bits, which makes an accidental collision across the number of runs
        a person will ever inspect by hand effectively impossible.

    Returns
    -------
    str
        The leading ``length`` characters.
    """
    return digest[:length]
```

---

## 6. `src/rade_qnet/core/runtime/hooks.py`

5574 bytes · SHA-256 `a12ae9f374abea76`

```python
"""
Observation points a caller can attach to a pipeline without subclassing it.

Hooks exist to keep the customisation tiers separate. A user who wants to log
to an experiment tracker, stream progress to a dashboard or assert an
invariant between stages should not have to subclass a pipeline to do it --
subclassing is for changing *what a stage computes*, and conflating the two
produces overrides whose only purpose is to add a print statement and which
then silently drift from the base implementation.

The central design rule
-----------------------
**A hook may observe, but it may not alter.** Hooks receive payloads and return
``None``. A hook that could rewrite a payload would make the pipeline's
behaviour depend on observation, and two runs with identical specs would stop
being comparable because one had a dashboard attached.

**A failing hook does not fail the run.** A broken tracker credential must not
destroy four hours of training. Hook exceptions are caught, logged at warning
level with the hook and stage named, and execution continues. The one
exception is :meth:`PipelineHook.on_run_start`, where a hook that cannot
initialise should say so before any compute is spent -- see
:class:`~rade_qnet.core.runtime.pipeline.Pipeline` for where that line is drawn.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["PipelineHook"]


class PipelineHook:
    """
    Base class for pipeline observers.

    Every method is a no-op, so a subclass overrides only what it needs and
    gains new hook points without modification when the framework adds them.
    This is the opposite trade-off from a protocol -- here the default
    behaviour is genuinely useful, so inheritance costs nothing and spares
    every hook from stubbing out seven methods.

    Notes
    -----
    A hook instance is not shared across processes. Under a parallel job set
    each worker constructs its own, so a hook holding an open file handle or a
    network session is safe, but a hook accumulating state will see only its
    own worker's events.
    """

    def on_run_start(self, run_id: str, spec_digest: str) -> None:
        """
        Observe the start of the run, before any stage executes.

        The place to open a tracker run or create a directory. Of all the
        hook points, this is the one where failing loudly may be preferable,
        because nothing has been computed yet.

        Parameters
        ----------
        run_id
            Identifier for this run.
        spec_digest
            Digest of the run specification, for correlating with a bundle.
        """

    def on_run_end(self, run_id: str, *, succeeded: bool) -> None:
        """
        Observe the end of the run, whether or not it succeeded.

        Called after the last stage, including when a stage failed, so a hook can close a file or finalise a tracker run
        whatever happened. Check ``succeeded`` rather than assuming: marking a
        crashed run as complete is worse than not marking it at all.

        Parameters
        ----------
        run_id
            Identifier for this run.
        succeeded
            Whether every stage completed.
        """

    def on_stage_start(self, stage: str) -> None:
        """
        Observe the start of a pipeline stage.

        Parameters
        ----------
        stage
            Stage name, such as ``build_data`` or ``fit``.
        """

    def on_stage_end(self, stage: str, *, seconds: float) -> None:
        """
        Observe the successful completion of a stage.

        Not called for a stage that raised; see :meth:`on_stage_error`.

        Parameters
        ----------
        stage
            Stage name.
        seconds
            Wall time for the stage.
        """

    def on_stage_error(self, stage: str, error: BaseException) -> None:
        """
        Observe a stage failure.

        The exception is re-raised after every hook has been notified, so this
        is for recording, not for recovery. A hook cannot suppress the error.

        Parameters
        ----------
        stage
            Stage name.
        error
            The exception, before it is wrapped in a
            :class:`~rade_qnet.core.runtime.errors.StageError`.
        """

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """
        Observe the end of an epoch, or of a boosting round.

        Deliberately engine-agnostic: a gradient loop and a boosted-tree fit
        both report here, so a progress dashboard works for either without
        knowing which it is watching.

        Parameters
        ----------
        epoch
            Zero-based epoch index.
        metrics
            Metrics for the epoch, including losses.
        """

    def on_metrics(self, stage: str, metrics: Mapping[str, float]) -> None:
        """
        Observe metrics produced by a stage.

        Parameters
        ----------
        stage
            Stage that produced them.
        metrics
            Metric name to value, in original target units.
        """

    def on_artifact(self, name: str, path: str) -> None:
        """
        Observe a file a stage wrote.

        A figure, a report, a bundle. Reported as a path rather than as
        contents, because a hook that wanted to upload a two-gigabyte
        checkpoint should decide that for itself.

        Parameters
        ----------
        name
            Logical name of the artifact.
        path
            Where it was written.
        """
```

---

## 7. `src/rade_qnet/core/runtime/logging.py`

9009 bytes · SHA-256 `f107778d8a86e24c`

```python
"""
Structured logging with contextual run identifiers.

A forty-job run produces forty interleaved streams of log output. Without
identifiers attached to every record, that output is unreadable, and the usual
remedy -- threading a pre-configured logger through every function -- puts an
infrastructure parameter into the signature of code that has nothing to do
with logging.

Instead the identifiers live in :class:`~contextvars.ContextVar` slots. Any
module can call :func:`get_logger` and its records are automatically stamped
with the current run, job and stage.

Crossing a process boundary
---------------------------
Context variables do not survive ``fork`` or ``spawn``. A worker must re-bind
them, which is what :func:`context_payload` and :func:`apply_context_payload`
are for: the parent serialises the context into the job payload, and the
worker applies it before doing any work.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import IO, Final

__all__ = [
    "ContextFilter",
    "apply_context_payload",
    "bound_context",
    "configure_logging",
    "context_payload",
    "current_context",
    "get_logger",
]

#: Identifier of the current run. ``None`` outside a run.
RUN_ID: ContextVar[str | None] = ContextVar("rade_qnet_run_id", default=None)

#: Identifier of the current job within a job set. ``None`` for a single run.
JOB_ID: ContextVar[str | None] = ContextVar("rade_qnet_job_id", default=None)

#: Name of the pipeline stage currently executing. ``None`` between stages.
STAGE: ContextVar[str | None] = ContextVar("rade_qnet_stage", default=None)

_CONTEXT_VARIABLES: Final = {"run_id": RUN_ID, "job_id": JOB_ID, "stage": STAGE}

#: Root logger name. Every framework logger is a descendant, so a user can
#: raise or lower the framework's verbosity with a single call without
#: touching their own loggers.
ROOT_LOGGER_NAME: Final = "rade_qnet"

_LOG_FORMAT: Final = "%(asctime)s %(levelname)-7s [%(rade_qnet_context)s] %(name)s: %(message)s"
_TIME_FORMAT: Final = "%Y-%m-%d %H:%M:%S"


class ContextFilter(logging.Filter):
    """
    Attach the current run, job and stage identifiers to every record.

    Implemented as a filter rather than an adapter so that records emitted by
    code which knows nothing about this module -- including a third-party
    library logging under the framework's logger tree -- are stamped too.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add a ``rade_qnet_context`` attribute to the record.

        Parameters
        ----------
        record
            The record being emitted. Mutated in place.

        Returns
        -------
        bool
            Always ``True``: this filter annotates, it never discards.
        """
        identifiers = current_context()
        # A compact single field rather than three, so the format string stays
        # readable and a line with no context does not carry empty brackets.
        record.rade_qnet_context = (
            " ".join(f"{key}={value}" for key, value in identifiers.items()) or "-"
        )
        return True


def get_logger(name: str) -> logging.Logger:
    """
    Return the framework logger for a module.

    Parameters
    ----------
    name
        Usually ``__name__``. A leading ``rade_qnet`` is not duplicated, and any
        other name is placed under the framework's logger tree so that
        configuring one logger configures all of them.

    Returns
    -------
    logging.Logger
        A logger beneath :data:`ROOT_LOGGER_NAME`.
    """
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    # Strip the repository's import prefix so `src.rade_qnet.core.spec.run` and
    # `rade_qnet.core.spec.run` produce the same logger name. The package is
    # importable under both paths, and a log stream should not reveal which.
    trimmed = name.removeprefix("src.")
    if trimmed.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(trimmed)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{trimmed}")


def configure_logging(
    *,
    level: int = logging.INFO,
    stream: IO[str] | None = None,
    force: bool = False,
) -> logging.Logger:
    """
    Install a handler on the framework's root logger.

    Called explicitly, never at import. Importing ``rade_qnet`` must not change
    how an application's logging behaves -- a library that configures logging
    on import is a library that silently redirects somebody else's output.

    Parameters
    ----------
    level
        Threshold for the framework's logger.
    stream
        Destination. Defaults to standard error, so log output does not
        contaminate a program's results on standard output.
    force
        Replace any handler this function previously installed. Without it the
        call is idempotent, so repeated calls cannot produce duplicated lines.

    Returns
    -------
    logging.Logger
        The configured framework root logger.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    # Do not propagate to the root logger: an application that has configured
    # its own root handler would otherwise see every framework line twice.
    logger.propagate = False

    existing = [handler for handler in logger.handlers if getattr(handler, "_rade_qnet", False)]
    if existing and not force:
        return logger
    for handler in existing:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_TIME_FORMAT))
    handler.addFilter(ContextFilter())
    # Marked so a later call can recognise its own handler and leave any
    # handler the application added alone.
    handler._rade_qnet = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


def current_context() -> dict[str, str]:
    """
    Return the identifiers currently bound, omitting those that are unset.

    Returns
    -------
    dict of str to str
        Mapping with any of ``run_id``, ``job_id`` and ``stage`` that are set.
    """
    return {
        name: value
        for name, variable in _CONTEXT_VARIABLES.items()
        if (value := variable.get()) is not None
    }


@contextmanager
def bound_context(
    *,
    run_id: str | None = None,
    job_id: str | None = None,
    stage: str | None = None,
) -> Iterator[None]:
    """
    Bind identifiers for the duration of a block.

    Only the arguments given are changed; the rest keep their current values.
    That is what lets a stage bind ``stage`` without needing to know, or
    repeat, the run it belongs to.

    Every variable is restored on exit, including when the block raises, so a
    failed stage cannot leave its name attached to subsequent log lines.

    Parameters
    ----------
    run_id, job_id, stage
        Values to bind. ``None`` leaves the existing value in place.

    Yields
    ------
    None
    """
    updates = {"run_id": run_id, "job_id": job_id, "stage": stage}
    tokens = [
        _CONTEXT_VARIABLES[name].set(value) for name, value in updates.items() if value is not None
    ]
    try:
        yield
    finally:
        # Reset in reverse order so nested bindings unwind correctly.
        for token in reversed(tokens):
            token.var.reset(token)


def context_payload() -> dict[str, str]:
    """
    Serialise the current context for transport to a worker process.

    Returns
    -------
    dict of str to str
        A plain, picklable mapping suitable for inclusion in a job payload.
    """
    return current_context()


def apply_context_payload(payload: Mapping[str, str]) -> None:
    """
    Re-bind identifiers in a worker process.

    Unlike :func:`bound_context` this does not restore anything, because a
    worker's context should last for the whole of its task.

    This **replaces** the context rather than merging into it: an identifier
    absent from the payload is cleared. That matters because a process pool
    reuses its workers. If applying a payload merged, a worker that handled
    job ``EURUSD`` and then a run with no job identifier would keep logging
    ``EURUSD`` -- misattributing work to a job that had already finished,
    which is precisely the failure this mechanism exists to prevent.

    Unknown keys are ignored rather than raising: a newer parent sending an
    identifier an older worker does not recognise should degrade to slightly
    less informative logs, not to a crash.

    Parameters
    ----------
    payload
        A mapping produced by :func:`context_payload`. Pass an empty mapping
        to clear the context entirely.
    """
    for name, variable in _CONTEXT_VARIABLES.items():
        variable.set(payload.get(name))
```

---

## 8. `src/rade_qnet/core/runtime/pipeline.py`

9822 bytes · SHA-256 `49e5a0d89f731b58`

```python
"""
The template-method base every pipeline derives from.

A pipeline is a fixed sequence of small, typed, individually overridable
stages. :meth:`Pipeline.run` composes them; :meth:`Pipeline.step` wraps each
one. That split is what makes the four customisation tiers possible:

1. **Spec only.** Write no pipeline code at all.
2. **Add observation.** Attach hooks and reports.
3. **Override one stage.** Subclass, replace one method, inherit the rest --
   including its timing, logging, error wrapping and caching.
4. **Override** ``run``. Change the sequence itself. Rare, and the only tier
   that gives up the instrumentation.

Tier 3 is the one that justifies this design, and it only works because a
stage is small enough to be replaced in isolation. A monolithic ``train()``
offers tiers 1, 2 and 4 and nothing between them -- which in practice means
every non-standard model copies the whole function and diverges from it.

What the wrapper buys
---------------------
:meth:`step` is the single place that times a stage, logs its start and end,
notifies hooks, reports errors and wraps failures with the stage name. Putting
it in one place has a concrete payoff: an override written by a user gets all
of it for free and cannot forget any of it. The alternative -- each stage
instrumenting itself -- guarantees that the one stage someone adds later is
the one with no timing and an unhelpful traceback.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from .errors import StageError
from .logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from .context import RunContext

__all__ = ["Pipeline"]

_LOGGER = get_logger(__name__)


class Pipeline[ResultT](ABC):
    """
    Base class for the train, evaluate, infer and tune pipelines.

    Generic over its result type, so a subclass declares what it produces and
    a caller knows without inspecting the implementation.

    Parameters
    ----------
    context
        Ambient state for the run.

    Notes
    -----
    A pipeline instance is used once. It is not reset between runs, and it
    holds no state that two runs could share, which is what makes it safe to
    construct one per job in a parallel job set.
    """

    #: Stage names, in the order :meth:`run` invokes them. Declared rather than
    #: inferred so the sequence is documentation, can be rendered in a report,
    #: and can be compared against what actually ran.
    stages: tuple[str, ...] = ()

    def __init__(self, context: RunContext) -> None:
        """
        Store the run context.

        Parameters
        ----------
        context
            Ambient state for the run.
        """
        self.context = context
        self._completed: list[str] = []
        self._timings: dict[str, float] = {}

    @abstractmethod
    def run(self) -> ResultT:
        """
        Execute the pipeline.

        Implementations compose their stages through :meth:`step` so each one
        is timed, logged and reported. A subclass that overrides this method
        entirely takes on that responsibility itself.

        Returns
        -------
        ResultT
            Whatever this pipeline produces.

        Raises
        ------
        StageError
            Wrapping whatever a stage raised, naming the stage.
        """

    def execute(self) -> ResultT:
        """
        Run the pipeline with run-level instrumentation.

        The entry point a caller should use. :meth:`run` is the method a
        subclass writes; this is the method that surrounds it with the
        ``on_run_start`` and ``on_run_end`` notifications, the logging
        context, and the guarantee that ``on_run_end`` fires even on failure.

        Keeping the two apart means a subclass overriding ``run`` -- tier 4 --
        still gets the run-level bookkeeping, and that an overriding author
        cannot forget to report that the run ended.

        Returns
        -------
        ResultT
            The pipeline's result.

        Raises
        ------
        StageError
            Wrapping whatever a stage raised.
        """
        context = self.context
        succeeded = False
        with context.activate():
            _LOGGER.info("starting %s", type(self).__name__)
            for line in context.describe():
                _LOGGER.debug("  %s", line)
            context.notify(
                lambda hook: hook.on_run_start(context.run_id, context.spec_digest),
                description="run start",
            )
            try:
                result = self.run()
                succeeded = True
            finally:
                # Fires on the failure path too: a hook that opened a file or a
                # tracker run must be told the run is over, and told the truth
                # about whether it worked.
                context.notify(
                    lambda hook: hook.on_run_end(context.run_id, succeeded=succeeded),
                    description="run end",
                )
                if context.tracker is not None:
                    try:
                        context.tracker.finish(succeeded=succeeded)
                    except Exception:  # Broad by design: tracking is never load-bearing.
                        _LOGGER.warning("tracker failed to finish; continuing", exc_info=True)
            _LOGGER.info("completed %s in %.2fs", type(self).__name__, sum(self._timings.values()))
            return result

    def step[StepResultT](self, name: str, operation: Callable[[], StepResultT]) -> StepResultT:
        """
        Run one stage, instrumented.

        Takes a zero-argument callable rather than a method name so a stage's
        own arguments stay explicit and type checked at the call site:
        ``bundle = self.step("build_data", lambda: self.build_data(spec))``.

        Parameters
        ----------
        name
            Stage name. Should appear in :attr:`stages`.
        operation
            The work to perform.

        Returns
        -------
        StepResultT
            Whatever the operation returned.

        Raises
        ------
        StageError
            Wrapping whatever the operation raised. The original exception is
            chained, so no detail is lost, but the message names the stage --
            which turns "``KeyError: 'features'``" into something that
            identifies where to look.
        """
        context = self.context
        with context.activate(stage=name):
            context.notify(lambda hook: hook.on_stage_start(name), description=f"{name} start")
            _LOGGER.info("stage %s starting", name)
            started = time.perf_counter()
            try:
                result = operation()
            except Exception as error:
                # Rebound because Python unbinds an `except ... as` name at the
                # end of the block, which would leave the closure below empty.
                failure = error
                elapsed = time.perf_counter() - started
                self._timings[name] = elapsed
                _LOGGER.error("stage %s failed after %.2fs", name, elapsed)
                context.notify(
                    lambda hook: hook.on_stage_error(name, failure),
                    description=f"{name} error",
                )
                # A StageError is already attributed, so re-wrapping it would
                # produce "stage fit failed [StageError] stage fit failed ..."
                # for a nested pipeline. Let the inner attribution stand.
                if isinstance(failure, StageError):
                    raise
                raise StageError(name, failure, run_id=context.run_id) from failure
            elapsed = time.perf_counter() - started
            self._timings[name] = elapsed
            self._completed.append(name)
            _LOGGER.info("stage %s completed in %.2fs", name, elapsed)
            context.notify(
                lambda hook: hook.on_stage_end(name, seconds=elapsed),
                description=f"{name} end",
            )
            return result

    def report_metrics(self, stage: str, metrics: Mapping[str, float]) -> None:
        """
        Publish metrics to the hooks and the tracker.

        Parameters
        ----------
        stage
            Stage that produced them.
        metrics
            Metric name to value, in original target units. Metrics in a
            model's internal space must be inverted before reaching here --
            see :meth:`~rade_qnet.core.contract.state.FittedState.inverse_transform_targets`
            -- because everything downstream of this call treats them as
            comparable across runs.
        """
        self.context.notify(
            lambda hook: hook.on_metrics(stage, metrics),
            description=f"{stage} metrics",
        )
        self.context.track_metrics(metrics)

    def report_artifact(self, name: str, path: Path) -> None:
        """
        Publish a written file to the hooks and the tracker.

        Parameters
        ----------
        name
            Logical name of the artifact.
        path
            Where it was written.
        """
        self.context.notify(
            lambda hook: hook.on_artifact(name, str(path)),
            description=f"artifact {name}",
        )
        self.context.track_artifact(path, name=name)

    @property
    def completed_stages(self) -> tuple[str, ...]:
        """Stages that finished, in the order they finished."""
        return tuple(self._completed)

    @property
    def timings(self) -> Mapping[str, float]:
        """Wall time per stage, including a stage that failed."""
        return dict(self._timings)
```

---

## 9. `src/rade_qnet/core/runtime/seeding.py`

7820 bytes · SHA-256 `bf9dcd660de9c2cb`

```python
"""
Deterministic seeding, and derived seeds for job sets.

Two problems are solved here, and the second is easy to overlook until a
parallel run produces different numbers from a sequential one.

**Seeding what is installed, from a package that imports nothing.**
``core`` may not import a training library, yet seeding PyTorch is exactly the
kind of thing this module should do. The resolution is a registry: an engine
registers a seeding callback when it is imported, and :func:`seed_everything`
invokes whatever has registered. ``core`` therefore seeds the libraries that
happen to be present without ever naming them.

**Per-job seeds that do not depend on execution order.**
Forty jobs cannot share one seed, and they must not draw seeds from a counter
or from the process identifier -- either makes the result depend on scheduling.
:func:`derive_seed` hashes the run seed together with a stable label, so job
``EURUSD`` gets the same seed whether it runs first, last, or in a worker on
another machine.

Determinism
-----------
``determinism`` is an explicit three-way choice rather than a boolean, because
full determinism costs real performance and some operations have no
deterministic implementation at all:

``off``
    Fastest. Seeds are set, but non-deterministic kernels are permitted.
``warn``
    Request deterministic kernels; where none exists, warn and continue.
``strict``
    Require deterministic kernels; a missing one is an error.

This replaces a blanket ``use_deterministic_algorithms(True)`` buried in a
swallowed ``try``/``except``, where neither the cost nor the failure was
visible to anyone.
"""

from __future__ import annotations

import hashlib
import random
from typing import Literal, Protocol

import numpy as np

from .errors import SpecError
from .logging import get_logger

__all__ = [
    "Determinism",
    "Seeder",
    "derive_seed",
    "register_seeder",
    "registered_seeders",
    "seed_everything",
    "unregister_seeder",
]

#: How hard to work for bit-for-bit reproducibility.
Determinism = Literal["off", "warn", "strict"]

#: Upper bound for a derived seed. Chosen because the numerical libraries this
#: framework wraps accept a 32-bit unsigned seed.
_SEED_MODULUS = 2**32

_LOGGER = get_logger(__name__)


class Seeder(Protocol):
    """
    A callback that seeds one library.

    Registered by whichever package owns the library, so ``core`` never
    imports it. An engine registers at module import; a user could register
    one for a library the framework has never heard of.
    """

    def __call__(self, seed: int, determinism: Determinism) -> None:
        """
        Seed the library and apply the requested determinism level.

        Parameters
        ----------
        seed
            A non-negative integer below ``2 ** 32``.
        determinism
            The requested level. An implementation that cannot honour
            ``strict`` must raise rather than degrade silently.
        """


# Process-global and deliberately so: there is one set of installed libraries
# per process. Registration is keyed by name and replacing an entry is
# explicit, so this cannot be mutated into meaning something different.
_SEEDERS: dict[str, Seeder] = {}


def register_seeder(name: str, seeder: Seeder, *, replace: bool = False) -> None:
    """
    Register a callback to be invoked by :func:`seed_everything`.

    Parameters
    ----------
    name
        Library name, used for logging and for replacement.
    seeder
        The callback.
    replace
        Permit overwriting an existing registration. Required explicitly so
        that two packages accidentally claiming one name is an error rather
        than a silent last-one-wins.

    Raises
    ------
    SpecError
        If ``name`` is already registered and ``replace`` is false.
    """
    if name in _SEEDERS and not replace:
        raise SpecError(
            f"a seeder named {name!r} is already registered; "
            f"pass replace=True to override it deliberately"
        )
    _SEEDERS[name] = seeder


def unregister_seeder(name: str) -> None:
    """
    Remove a registered seeder, if present.

    Parameters
    ----------
    name
        Library name. Absent names are ignored, so teardown is idempotent.
    """
    _SEEDERS.pop(name, None)


def registered_seeders() -> tuple[str, ...]:
    """
    Return the names of registered seeders, sorted.

    Returns
    -------
    tuple of str
        Registered library names.
    """
    return tuple(sorted(_SEEDERS))


def seed_everything(seed: int, *, determinism: Determinism = "off") -> int:
    """
    Seed the standard library, NumPy and every registered library.

    Parameters
    ----------
    seed
        Base seed. Must be a non-negative integer below ``2 ** 32``.
    determinism
        Passed through to each registered seeder.

    Returns
    -------
    int
        The seed that was applied, so a caller can record it in a run's
        lineage without recomputing it.

    Raises
    ------
    SpecError
        If the seed is out of range, or if a seeder fails under ``strict``.
    """
    if not 0 <= seed < _SEED_MODULUS:
        raise SpecError(f"seed must satisfy 0 <= seed < {_SEED_MODULUS}, received {seed}")

    random.seed(seed)
    # The legacy global NumPy generator is seeded in addition to any explicit
    # Generator a caller holds, because third-party code frequently uses
    # `np.random.*` directly and would otherwise be left unseeded.
    np.random.seed(seed)  # noqa: NPY002

    for name, seeder in sorted(_SEEDERS.items()):
        try:
            seeder(seed, determinism)
        except Exception as exc:
            if determinism == "strict":
                # Under `strict` the user asked for reproducibility and must
                # be told it was not achieved, rather than receiving results
                # that quietly are not reproducible.
                raise SpecError(
                    f"seeder {name!r} failed under determinism='strict': {exc}"
                ) from exc
            _LOGGER.warning(
                "seeder %r failed; results from that library will not be reproducible",
                name,
                exc_info=True,
            )
    return seed


def derive_seed(base_seed: int, *labels: str) -> int:
    """
    Derive a stable child seed from a base seed and one or more labels.

    The derivation is a hash, not a counter, which gives three properties a
    counter cannot. It is independent of execution order, so a job set is
    reproducible regardless of scheduling. It is identical across processes and
    machines, because it never touches ``hash()``. And it is stable under
    insertion, so adding a forty-first job leaves the other forty unchanged --
    without which every job would be retrained on a different seed whenever the
    portfolio grew.

    Parameters
    ----------
    base_seed
        The run's seed.
    *labels
        Stable identifiers, such as a job id or a trial number. Order
        matters: ``derive_seed(0, "a", "b")`` differs from
        ``derive_seed(0, "b", "a")``.

    Returns
    -------
    int
        A seed in ``[0, 2 ** 32)``.

    Raises
    ------
    SpecError
        If no label is given. A derived seed with no label would just be a
        reformatting of the base seed, which is never what the caller wants.
    """
    if not labels:
        raise SpecError("derive_seed requires at least one label")

    # NUL is used as the separator because it cannot occur in the identifiers
    # this is called with, so ("a", "bc") and ("ab", "c") cannot collide.
    material = "\0".join((str(base_seed), *labels)).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big") % _SEED_MODULUS
```

