"""
The engine package convention, enforced.

``models/`` has had an enforced layout since Phase 3, and engines have not,
with the predictable result: ``xgboost`` is one module, ``sklearn`` was two
under a name (``adapters.py``) that described nothing, and ``torch`` had grown
to eleven flat files mixing five unrelated concerns.  Someone adding a fourth
backend had no template to copy and no vocabulary to obey.

This module closes that gap.  The sanctioned names are the four verbs an
engine performs and the three parts of *fit* big enough to need their own
package, and the rule is the same one ``test_model_layout.py`` applies to
models: ``engine.py`` is required, everything else is optional, and a name
outside the set is a failing test rather than a convention nobody mentioned.

Why a closed set is the load-bearing half
------------------------------------------
Requiring ``engine.py`` prevents nothing -- a backend obviously has one.  What
keeps the tree navigable is *forbidding the rest*, because drift happens one
plausible ``utils.py`` at a time and each addition is individually defensible.
Closing the set means a genuinely new concern has to be a reviewed extension
of the vocabulary here, argued once for every engine, rather than a judgement
call made quietly inside one of them.

What the progression across engines says
-----------------------------------------
The vocabulary is deliberately *optional* down to a single file, so the shape
of an engine package reports what its library actually owns.  ``xgboost`` is
one module because a boosted fit is a single call with no loop, no learner and
no device to choose.  ``sklearn`` is one module for the same reason.  Only
``torch`` carries ``training/``, ``learners/`` and ``hardware/``, because only
PyTorch makes the framework supply those.  A reader can therefore tell what a
backend does from ``ls`` alone, which is the whole return on the convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ..locations import PACKAGE_ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The one file an engine package must contain. It holds the class satisfying
#: :class:`~rade_qnet.engines.base.Engine`, and it is required because a
#: package here that does not implement that protocol is not an engine.
REQUIRED_FILES = frozenset({"__init__.py", "engine.py"})

#: Every other name an engine package may use, and nothing else. Each one is
#: a verb the engine performs, in the order a run performs them: build the
#: model, feed it, predict with it. Fitting is the fourth, and it is large
#: enough to be a package rather than a module -- see below.
OPTIONAL_FILES = frozenset(
    {
        "materialise.py",  # spec + signature -> an unfitted native model
        "loaders.py",  # a BatchSource -> whatever this library consumes
        "predictor.py",  # a fitted model -> predictions
    }
)

#: Sub-packages an engine may contain. These are the three parts of *fit* that
#: earn their own vocabulary, and they are separate from one another because
#: they change for different reasons: a new algorithm touches ``learners``
#: only, a new stopping rule touches ``training`` only, and a new accelerator
#: touches ``hardware`` only.
OPTIONAL_DIRECTORIES = frozenset(
    {
        "training",  # when a fit happens: drivers, callbacks, losses, checkpoints
        "learners",  # what one update means, one module per algorithm
        "hardware",  # where it runs, and whether it runs the same way twice
    }
)

#: Module names permitted inside ``training/``. Closed for the same reason the
#: top level is: this is where a fit's machinery accumulates fastest.
TRAINING_MODULES = frozenset(
    {
        "__init__.py",
        "loops.py",
        "callbacks.py",
        "losses.py",
        "checkpoint.py",
        "risk.py",
    }
)

#: Module names permitted inside ``hardware/``.
HARDWARE_MODULES = frozenset(
    {
        "__init__.py",
        "devices.py",
        "distributed.py",
        "determinism.py",
    }
)

#: Files directly under ``engines/`` that are not engine packages. ``base.py``
#: holds the protocols; ``loaders.py`` holds the drain shared by every
#: one-shot engine, hoisted here so that xgboost does not have to import from
#: sklearn to reach it.
FRAMEWORK_MODULES = frozenset({"__init__.py", "base.py", "loaders.py"})

ENGINES_ROOT = PACKAGE_ROOT / "engines"


def engine_packages() -> Iterator[Path]:
    """
    Yield every engine package directory.

    Yields
    ------
    pathlib.Path
        One directory per engine, in a stable alphabetical order so a
        failure names the same package on every machine.
    """
    for child in sorted(ENGINES_ROOT.iterdir()):
        if child.is_dir() and child.name != "__pycache__":
            yield child


def names_in(directory: Path) -> frozenset[str]:
    """
    Return the file and directory names in ``directory``, ignoring caches.

    Parameters
    ----------
    directory
        The directory to list.

    Returns
    -------
    frozenset of str
        Names, not paths, so a failure message is readable.
    """
    return frozenset(
        child.name
        for child in directory.iterdir()
        if child.name not in {"__pycache__", ".ruff_cache"}
    )


PACKAGES = list(engine_packages())


class TestTheEngineVocabulary:
    """Every engine package draws its filenames from one closed set."""

    @pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.name)
    def test_an_engine_package_has_an_engine_module(self, package: Path) -> None:
        """
        ``engine.py`` exists, because that is what makes it an engine.

        Parameters
        ----------
        package
            One engine package directory.
        """
        missing = REQUIRED_FILES - names_in(package)
        assert not missing, f"{package.name} is missing {sorted(missing)}"

    @pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.name)
    def test_an_engine_package_contains_nothing_unsanctioned(self, package: Path) -> None:
        """
        No file or directory is named anything outside the vocabulary.

        This is the check that keeps the rest honest. Without it the required
        file is a floor and the package drifts anyway.

        Parameters
        ----------
        package
            One engine package directory.
        """
        allowed = REQUIRED_FILES | OPTIONAL_FILES | OPTIONAL_DIRECTORIES
        unexpected = names_in(package) - allowed
        assert not unexpected, (
            f"{package.name} contains {sorted(unexpected)}, which is outside the engine "
            f"vocabulary {sorted(allowed)}. Either rename it to one of those, or extend "
            f"the vocabulary here and say in the charter what the new name means."
        )

    @pytest.mark.parametrize("package", PACKAGES, ids=lambda p: p.name)
    def test_a_sub_package_draws_from_its_own_closed_set(self, package: Path) -> None:
        """
        ``training/`` and ``hardware/`` are closed too.

        These are where a fit's machinery accumulates fastest, so leaving them
        open would move the drift one level down rather than prevent it.

        Parameters
        ----------
        package
            One engine package directory.
        """
        for directory, permitted in (
            ("training", TRAINING_MODULES),
            ("hardware", HARDWARE_MODULES),
        ):
            path = package / directory
            if not path.is_dir():
                continue
            unexpected = names_in(path) - permitted
            assert not unexpected, (
                f"{package.name}/{directory} contains {sorted(unexpected)}, outside "
                f"{sorted(permitted)}"
            )

    def test_engines_itself_holds_only_protocols_and_the_shared_drain(self) -> None:
        """
        Nothing accumulates directly under ``engines/``.

        The package root is the first place a "just this one helper" lands,
        and a helper there is reachable by every engine, which is how one
        backend's convenience becomes every backend's dependency.
        """
        loose = frozenset(
            child.name
            for child in ENGINES_ROOT.iterdir()
            if child.is_file() and child.suffix == ".py"
        )
        assert loose == FRAMEWORK_MODULES, (
            f"engines/ holds {sorted(loose)}; expected exactly {sorted(FRAMEWORK_MODULES)}"
        )


class TestWhatTheShapeReports:
    """The layout is informative, not merely tidy."""

    def test_only_torch_needs_the_fit_sub_packages(self) -> None:
        """
        A one-shot engine has no loop, no learner and no device policy.

        If this ever fails for ``sklearn`` or ``xgboost`` it is worth asking
        why: a boosted fit acquiring a ``training/`` package means the engine
        has started reimplementing a loop the framework already owns.
        """
        with_training = {p.name for p in PACKAGES if (p / "training").is_dir()}
        assert with_training == {"torch"}

    def test_no_engine_imports_another(self) -> None:
        """
        Backends are siblings, not a hierarchy.

        ``xgboost`` used to import ``sklearn.adapters`` for the drain that
        turns a batch stream into one matrix. The drain is pure NumPy and
        belongs to neither, so it was hoisted to ``engines/loaders.py``. The
        rule matters because a cross-engine import makes one backend's
        presence a condition of another's, which is exactly what installing
        only xgboost is supposed to avoid.
        """
        names = {p.name for p in PACKAGES}
        for package in PACKAGES:
            for module in package.rglob("*.py"):
                text = module.read_text()
                for other in names - {package.name}:
                    assert f"..{other}." not in text, (
                        f"{module} imports from the {other} engine; hoist the shared part "
                        f"to engines/loaders.py instead"
                    )
