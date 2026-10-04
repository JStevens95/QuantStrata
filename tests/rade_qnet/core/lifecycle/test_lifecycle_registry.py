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
