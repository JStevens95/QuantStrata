"""
The component registry: resolving a name in a specification to a class.

A specification says ``model: hybrid_gnn_rnn``, not
``model: rade_xl.models.hybrid_gnn_rnn.register.HybridGnnRnnModel``. The
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
        class Ridge(TabularModel):
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
