"""
Where the package under test and its test tree live, worked out rather than assumed.

Why this module exists
----------------------
The suite runs in two places with two different layouts. In this repository
the package imports as ``src.rade_qnet`` and the tests sit at
``tests/rade_qnet``. Where it is deployed, the same files import as something
like ``tranql.models.rade.rade_qnet.rade_qnet``, with the tests in a sibling
package. A path such as ``Path("src/rade_qnet/models")`` is right in one and
wrong in the other, and it is also wrong here the moment pytest is started
from any directory but the repository root.

So nothing in the suite spells out a location. Every path and every dotted
name a test needs is derived here, from the one thing that is true in every
layout: the package imported successfully, so ``__name__`` and ``__file__``
say exactly where it is.

The import below is the only line in this module that names the package, and
the porting script rewrites it along with every other ``from src.rade_qnet``
import. Everything else follows from it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import src.rade_qnet as package

#: The package's dotted import name: ``src.rade_qnet`` here, the deployment's
#: own name elsewhere. Used wherever a test has to name a module as a string --
#: a ``monkeypatch`` target, an ``__import__``, a ``sys.modules`` key, or a
#: line of code handed to a fresh interpreter.
PACKAGE_NAME: Final = package.__name__

#: The package's directory, holding ``core``, ``engines``, ``docs`` and the rest.
PACKAGE_ROOT: Final = Path(package.__file__).resolve().parent

#: The directory that has to be on ``sys.path`` for :data:`PACKAGE_NAME` to
#: import. One level above the package per dotted component: the repository
#: root for ``src.rade_qnet``, the directory holding ``tranql`` for a deeper
#: name. A subprocess given this on its path can import the package by name.
IMPORT_ROOT: Final = PACKAGE_ROOT.parents[PACKAGE_NAME.count(".")]

#: The test tree's own directory.
TEST_ROOT: Final = Path(__file__).resolve().parent

#: Where the captured parity baseline lives. ``RADE_QNET_GOLDEN_ROOT`` wins
#: when set, which is the escape hatch for a deployment that keeps the arrays
#: somewhere of its own; otherwise the repository's convention, beneath
#: :data:`IMPORT_ROOT`. The testkit's ``load_golden`` honours the same
#: variable, so the tests and the loader can never disagree about where to
#: look.
GOLDEN_ROOT: Final = Path(
    os.environ.get(
        "RADE_QNET_GOLDEN_ROOT", IMPORT_ROOT / "tests" / "fixtures" / "rade_qnet" / "golden"
    )
)


def module_name(relative: str) -> str:
    """
    Return the full dotted name of a module inside the package.

    Parameters
    ----------
    relative
        The module's name relative to the package, such as
        ``"orchestration.pipelines.infer"``.

    Returns
    -------
    str
        The name it is importable under in this layout.
    """
    return f"{PACKAGE_NAME}.{relative}"
