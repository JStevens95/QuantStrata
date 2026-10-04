"""
The four component registries: resolving a name in a specification to a class.

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

from collections.abc import Callable, Iterable
from importlib import import_module

from .errors import ComponentError
from .registry import ClassT, Registry

__all__ = [
    "ENGINES",
    "LEARNERS",
    "MODELS",
    "REPORTS",
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
