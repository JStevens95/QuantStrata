"""
The generic container a component registry is made of.

One class, deliberately: :class:`Registry` knows how to hold named classes,
refuse a duplicate and report what it has, and it knows nothing about what a
model or an engine is. The four concrete registries, and the decorators that
populate them, are next door in
:mod:`~rade_qnet.core.lifecycle.components`.

Why the split
-------------
The two halves change for different reasons and are read by different people.
This file changes when the *mechanism* changes -- how a collision is reported,
whether lookup is case-sensitive -- and it is read roughly never. The other
changes when the framework gains a new *kind* of pluggable thing, and it is
the file a contributor opens to find out what ``@model`` actually does.

Keeping them together meant five hundred lines in which the interesting part
was the last third.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from .errors import ComponentError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["ClassT", "Registry", "RegistryEntry"]

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
