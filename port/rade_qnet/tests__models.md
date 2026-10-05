# `tests/rade_qnet/models`

3 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 17 | 838 | `e11173949276a7e1` |
| 2 | `test_model_layout.py` | 525 | 22614 | `3a1a074e153fe65b` |
| 3 | `test_reference_models.py` | 395 | 15934 | `804c5e8b4c66f0c6` |

---

## 1. `tests/rade_qnet/models/__init__.py`

838 bytes · SHA-256 `e11173949276a7e1`

```python
"""
Placeholder for model test suites.

Model-specific tests are deliberately deferred: the framework is being proven
model-independently first, so that when the flagship is refactored into it, a
failure is attributable to the model rather than to foundations still in flux.

This package exists so the mirror is complete and the gap is visible. When a
model lands, its suite is added beneath here following the source layout --
``models/hybrid_gnn_rnn/layers/test_layers_gnn.py`` and so on -- and the
mirroring rule in ``test_scaffold.py`` is extended to cover it.

What a model suite must contain is not left to taste. Every model is run
through ``rade_qnet.testkit.conformance``, and a model refactored from an
existing implementation is additionally held to the parity levels in
``src/rade_qnet/docs/phases/PHASE_0_BASELINE.md``.
"""
```

---

## 2. `tests/rade_qnet/models/test_model_layout.py`

22614 bytes · SHA-256 `3a1a074e153fe65b`

```python
"""
The model package convention, enforced.

A convention that lives only in prose is a convention that holds until the
first person in a hurry, and the cost is not paid by them -- it is paid by
everyone who afterwards has to open a model package to find out what shape
it is. ``docs/MODEL_IMPLEMENTATION.md`` describes the layout; this module
is what makes the description true.

What is checked, and why each one
-----------------------------------
*Every model is a package.* So that ``rade_qnet.models.<name>`` means one
thing, and so that growing a model never means moving it.

*The four files always exist.* So that the procedure has no branches: a
contributor copies the same template whatever they are building.

*No file is named anything else.* This is the check that makes the rest
airtight. Without it the four required files are a floor and the layout
drifts anyway, one ``utils.py`` at a time.

*``model.py`` does not import the framework's wiring.* The split between
mathematics and plumbing is the reason for the convention. A ``model.py``
that imports the registry has quietly undone it while still passing every
other check here.

*``register.py`` declares exactly one model.* Two registrations in one
package make ``rade_qnet.models.<name>`` ambiguous as an address.

*Each package registers itself on import.* The framework has no plugin
scan: importing is the registration mechanism, so a package whose
``__init__`` does not reach its ``register`` module is invisible to a
specification despite looking complete.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.rade_qnet.core.lifecycle.components import LEARNERS, MODELS, REPORTS
from src.rade_qnet.testkit.fixtures import isolated_registries

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The five files every model package has, at every tier. Absent any one of
#: them the package is incomplete even if it imports and runs, because the
#: next reader cannot rely on the convention to find anything.
#: Present in every model package, whatever it learns. ``spec.py`` says what
#: can be configured and ``register.py`` says how it plugs in -- two questions
#: that have the same answer for a ridge regression and for a hedging agent.
UNIVERSAL_FILES = frozenset({"__init__.py", "spec.py", "register.py"})

#: The pair that names the paradigm. A supervised model learns from inputs
#: with known targets, so it has ``model.py`` and ``data.py``. An agent learns
#: by acting, so it has ``policy.py`` and ``environment.py``.
#:
#: Two shapes rather than one is a deliberate cost. Keeping the filenames
#: identical across every package would have been tidier, and would have meant
#: a file called ``data.py`` holding an environment -- a name that lies, in
#: the one directory a new joiner is told to copy. A reader can tell what a
#: package learns from ``ls``, which is worth more than the symmetry.
SUPERVISED_FILES = frozenset({"model.py", "data.py"})
POLICY_FILES = frozenset({"policy.py", "environment.py"})

#: Retained for the error messages below, which name what a supervised
#: package must contain -- still the overwhelmingly common case.
REQUIRED_FILES = UNIVERSAL_FILES | SUPERVISED_FILES

#: Files a model package may additionally contain, and nothing else. Each
#: name means a specific thing; see ``rade_qnet.models.__init__``. The point of
#: closing the set is that a new concern has to be either one of these or a
#: deliberate, reviewed extension of the vocabulary -- not a judgement call
#: made silently inside one model.
OPTIONAL_FILES = frozenset({"state.py", "reports.py", "visuals.py"})

#: Sub-directories a model package may contain.
OPTIONAL_DIRECTORIES = frozenset({"layers", "features", "pipelines"})

#: Stages a model may override. ``infer.py`` is absent deliberately:
#: inference is "load the bundle and run the model forward", and a model
#: that needs to change that has changed what its bundle means. See
#: ``ARCHITECTURE.md`` §11.
PIPELINE_FILES = frozenset({"__init__.py", "train.py", "eval.py", "tune.py"})

#: Modules ``model.py`` may not import, because importing them means the
#: mathematics has been mixed back into the wiring. Matched on the dotted
#: suffix so that a relative import reads the same as an absolute one.
WIRING_MODULES = ("core.lifecycle.components", "core.spec.run", "core.spec.training")

#: Statements of *ceremony* a tier 1 model may contain: everything in the
#: package except ``model.py``.
#:
#: ``model.py`` is excluded because it is the mathematics -- the thing the
#: user came to write. Capping it penalises a model for being a model, and
#: a budget that a complicated architecture can blow is a budget that stops
#: measuring what it was introduced to measure. What this number watches is
#: the cost the *framework* imposes: the spec, the registration, the data
#: declaration and the charter.
#:
#: Raised from 70 (whole package) when ``data.py`` became mandatory. That
#: was a real increase in the framework's floor, roughly eleven statements
#: per model, and it was accepted in exchange for the input contract being
#: declared and checked rather than discovered at runtime. Recorded here
#: rather than quietly absorbed, because a budget nobody has to justify
#: moving is not a budget.
TIER_1_BUDGET = 55

MODELS_ROOT = Path("src/rade_qnet/models")


def model_packages() -> list[Path]:
    """
    Return every model package directory.

    Returns
    -------
    list of Path
        One per model, sorted by name.
    """
    return sorted(
        path for path in MODELS_ROOT.iterdir() if path.is_dir() and path.name != "__pycache__"
    )


def statements(path: Path) -> int:
    """
    Count statements in a module, which is the tier 1 budget's unit.

    Statements rather than physical lines, so that the budget cannot be met
    by writing long lines and cannot be exceeded by documenting generously.
    The point is to constrain how much a model has to *do*, not how much it
    is allowed to explain.

    Parameters
    ----------
    path
        A module.

    Returns
    -------
    int
        Statement count, excluding bare docstring expressions.
    """
    return sum(
        1
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.stmt)
        and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    )


def imported_modules(path: Path) -> Iterator[str]:
    """
    Yield the dotted name of every module a file imports.

    Relative imports are yielded without their leading dots, so that
    ``from ...core.lifecycle.components import model`` and the absolute form
    produce the same string and a single rule covers both.

    Parameters
    ----------
    path
        A module.

    Yields
    ------
    str
        A dotted module name.
    """
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def registered_names(path: Path) -> list[str]:
    """
    Return the name in every ``@model(...)`` decorator in a file.

    Read from the syntax tree rather than by importing, so that the check
    reports a layout fault rather than an import error when the package is
    mid-edit.

    Parameters
    ----------
    path
        A module.

    Returns
    -------
    list of str
        One entry per decorated class.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "model"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
            ):
                found.append(str(decorator.args[0].value))
    return found


class TestEveryModelIsAPackage:
    """The addressing rule: one directory, one model, no exceptions."""

    def test_there_are_no_loose_modules_in_the_library(self) -> None:
        """
        Nothing but ``__init__.py`` sits directly under ``models/``.

        A loose ``ridge.py`` beside a ``hybrid_gnn_rnn/`` package is the
        state this convention replaced. It is not wrong so much as
        *unanswerable*: a contributor cannot tell which shape theirs should
        be, and whichever they pick is inconsistent with half the library.
        """
        loose = [path.name for path in MODELS_ROOT.glob("*.py") if path.name != "__init__.py"]
        assert not loose, (
            f"models are packages, not modules; found {loose} directly under "
            f"{MODELS_ROOT}. Move each into models/<name>/ with the four "
            f"required files -- see docs/MODEL_IMPLEMENTATION.md §3"
        )

    def test_the_library_is_not_empty(self) -> None:
        """
        Guard against the suite below passing by having nothing to check.

        Every parametrised test in this module iterates the packages, so a
        library that was accidentally emptied would turn the whole file
        green rather than red.
        """
        assert len(model_packages()) >= 2


class TestTheRequiredFilesExist:
    """The template, which is what makes the procedure branchless."""

    @pytest.mark.parametrize("package", model_packages(), ids=lambda p: p.name)
    def test_the_five_required_files_are_present(self, package: Path) -> None:
        """
        Every model has all five, including the smallest.

        Three of them are the same whatever the package learns; the other two
        name the paradigm. The ceremony is the price of there being one
        procedure rather than a judgement call -- see
        ``rade_qnet.models.__init__`` for the full argument.

        Parameters
        ----------
        package
            One model package directory.
        """
        present = {path.name for path in package.glob("*.py")}
        missing = sorted(UNIVERSAL_FILES - present)
        assert not missing, (
            f"{package.name} is missing {missing}. Every model package has "
            f"these, whatever it learns"
        )
        paradigm = [sorted(kind) for kind in (SUPERVISED_FILES, POLICY_FILES) if kind <= present]
        assert paradigm, (
            f"{package.name} has neither {sorted(SUPERVISED_FILES)} nor "
            f"{sorted(POLICY_FILES)}. A model package declares what it learns by "
            f"which pair it carries: a supervised model has model.py and data.py, "
            f"an agent has policy.py and environment.py. Copy the template from "
            f"src/rade_qnet/models/ridge/"
        )
        assert len(paradigm) == 1, (
            f"{package.name} carries both paradigms' files. A package learns one "
            f"way; two models are two packages"
        )

    @pytest.mark.parametrize("package", model_packages(), ids=lambda p: p.name)
    def test_data_py_declares_an_input_contract(self, package: Path) -> None:
        """
        ``data.py`` exports a ``REQUIRES``, and ``register.py`` binds it.

        This is the check that stops a mandatory ``data.py`` degenerating
        into a stub. Requiring the *file* only guarantees a file; the
        reason it is worth requiring is that it carries a declaration the
        framework then enforces, and without this check a model could
        satisfy the layout while declaring nothing.

        ``InputRequirement.unconstrained()`` is a perfectly good answer --
        a model that flattens everything genuinely has no constraints --
        but it has to be written down, because "no constraints" and "not
        thought about" are indistinguishable from the outside.
        """
        exported = {
            target.id
            for node in ast.walk(ast.parse((package / "data.py").read_text()))
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert "REQUIRES" in exported, (
            f"{package.name}/data.py must define REQUIRES, the input "
            f"contract the pipeline checks the data build against. Use "
            f"InputRequirement.unconstrained() if the model truly accepts "
            f"anything"
        )
        assert "REQUIRES" in (package / "register.py").read_text(), (
            f"{package.name}/register.py must bind `requires = REQUIRES`; a "
            f"contract nothing reads is not a contract"
        )

    @pytest.mark.parametrize("package", model_packages(), ids=lambda p: p.name)
    def test_no_file_is_named_outside_the_vocabulary(self, package: Path) -> None:
        """
        A model package contains only names the convention defines.

        This is the check that makes the layout airtight rather than merely
        a floor. A ``helpers.py`` or a ``utils.py`` is always defensible in
        the moment and always means the next reader has to open the package
        to find out what is in it.

        If a model genuinely needs a concept the vocabulary lacks, the fix
        is to add it here and to ``rade_qnet.models.__init__`` in the same
        commit, so the decision is made once for the library rather than
        privately inside one model.
        """
        allowed = UNIVERSAL_FILES | SUPERVISED_FILES | POLICY_FILES | OPTIONAL_FILES
        strays = sorted(path.name for path in package.glob("*.py") if path.name not in allowed)
        assert not strays, (
            f"{package.name} contains {strays}, which the model layout does "
            f"not define. Permitted files: {sorted(allowed)}. Permitted "
            f"directories: {sorted(OPTIONAL_DIRECTORIES)}"
        )

    @pytest.mark.parametrize("package", model_packages(), ids=lambda p: p.name)
    def test_no_directory_is_named_outside_the_vocabulary(self, package: Path) -> None:
        """Sub-directories are limited the same way, and for the same reason."""
        strays = sorted(
            path.name
            for path in package.iterdir()
            if path.is_dir()
            and path.name != "__pycache__"
            and path.name not in OPTIONAL_DIRECTORIES
        )
        assert not strays, (
            f"{package.name} contains directories {strays}; permitted: "
            f"{sorted(OPTIONAL_DIRECTORIES)}"
        )

    @pytest.mark.parametrize("package", model_packages(), ids=lambda p: p.name)
    def test_pipeline_overrides_are_named_for_their_stage(self, package: Path) -> None:
        """
        A ``pipelines/`` directory holds only stage overrides.

        ``infer.py`` is excluded deliberately. Inference is "load the
        bundle and run the model forward"; a model that needs to change
        that has changed what its bundle means, and the place to fix it is
        the bundle rather than a fourth override.
        """
        pipelines = package / "pipelines"
        if not pipelines.is_dir():
            pytest.skip(f"{package.name} overrides no pipeline stage")
        strays = sorted(
            path.name for path in pipelines.glob("*.py") if path.name not in PIPELINE_FILES
        )
        assert not strays, (
            f"{package.name}/pipelines contains {strays}; a stage override "
            f"is named for its stage: {sorted(PIPELINE_FILES)}"
        )


class TestTheSplitIsReal:
    """
    The convention is only worth its ceremony if the halves stay separate.

    These are the checks that stop the layout becoming four files that each
    contain a bit of everything, which would be strictly worse than the one
    file it replaced.
    """

    @pytest.mark.parametrize("package", model_packages(), ids=lambda p: p.name)
    def test_model_py_does_not_import_the_frameworks_wiring(self, package: Path) -> None:
        """
        ``model.py`` is mathematics and knows nothing about running.

        The payoff is concrete: the architecture can be imported, read,
        reviewed and unit-tested with no registry, no run specification and
        no engine in the picture. The moment it imports the decorator, the
        file is wiring again and every one of those properties is gone.

        Note that importing *contracts* -- a signature, an error type -- is
        fine. The line is at things that only exist because the model is
        being run by this framework.
        """
        offenders = sorted(
            name
            for name in imported_modules(package / "model.py")
            if name.lstrip(".").endswith(WIRING_MODULES)
        )
        assert not offenders, (
            f"{package.name}/model.py imports {offenders}; registration and "
            f"run-spec parsing belong in register.py so that model.py stays "
            f"readable as mathematics"
        )

    @pytest.mark.parametrize("package", model_packages(), ids=lambda p: p.name)
    def test_registration_happens_in_register_py_and_nowhere_else(self, package: Path) -> None:
        """
        Exactly one ``@model(...)`` per package, in ``register.py``.

        One, because ``rade_qnet.models.<name>`` is an address and two
        registrations make it ambiguous. In ``register.py``, because that
        is where a reader looks for it -- a decorator hidden in ``spec.py``
        works perfectly and is findable only by grep.
        """
        for path in sorted(package.rglob("*.py")):
            names = registered_names(path)
            if path.name == "register.py":
                assert len(names) == 1, (
                    f"{package.name}/register.py declares {len(names)} models "
                    f"({names}); it must declare exactly one"
                )
            else:
                assert not names, (
                    f"{path} registers {names}; registration belongs in {package.name}/register.py"
                )

    @pytest.mark.parametrize("package", model_packages(), ids=lambda p: p.name)
    def test_importing_the_package_registers_the_model(self, package: Path) -> None:
        """
        The ``__init__`` reaches ``register``, so importing is enough.

        There is no plugin scan and no entry-point discovery in this
        framework: import *is* the registration mechanism. A package whose
        ``__init__`` imports only its spec looks complete, passes every
        other check in this file, and is invisible to a specification that
        names it.

        Checked against an emptied registry *and* an evicted module cache,
        because either one alone makes the test lie. With the registry
        intact, a previously imported model satisfies the assertion for a
        package that registers nothing. With the cache intact, importing
        the package is a no-op and the assertion passes for a package whose
        ``__init__`` was emptied this morning -- a reload will not help,
        because reloading ``__init__`` does not re-execute the ``register``
        submodule that is still cached beneath it.

        So the test evicts the package's whole sub-tree and imports it
        cold, which is the only arrangement that actually reproduces what
        a user's first import does.
        """
        declared = registered_names(package / "register.py")[0]
        prefix = f"src.rade_qnet.models.{package.name}"
        cached = {
            name: module
            for name, module in sys.modules.items()
            if name == prefix or name.startswith(f"{prefix}.")
        }
        try:
            for name in cached:
                del sys.modules[name]
            with isolated_registries(empty=True):
                # `empty=True` clears models and engines only, by design:
                # those are names a specification claims, while reports and
                # learners are catalogues the pipeline renders from. A cold
                # import is the one case that needs all four cleared, since
                # a model that also registers reports -- as the flagship
                # does -- would otherwise collide with its own earlier
                # registration. Cleared here rather than by widening the
                # fixture, because the context manager restores every
                # registry on exit regardless.
                for registry in (LEARNERS, REPORTS):
                    registry.restore({})
                importlib.import_module(prefix)
                assert declared in MODELS.names(), (
                    f"importing rade_qnet.models.{package.name} did not "
                    f"register {declared!r}; its __init__ must import from "
                    f".register"
                )
        finally:
            # Put the originals back. The cold import above created a second
            # set of class objects, and leaving those in the cache would mean
            # a later test's `get_model(...) is SomeModel` compares a class
            # from one import against a class from another and fails for a
            # reason that has nothing to do with what it is testing.
            sys.modules.update(cached)


class TestSimpleModelsStayCheap:
    """
    The budget that makes the framework's cost visible.

    If a tier 1 model stops fitting, the finding is about the framework
    rather than about the model: something above it now demands boilerplate.
    Raising the budget to make this pass would delete the only signal that
    said so.
    """

    @pytest.mark.parametrize("name", ["ridge", "xgb_tabular", "lstm_tabular"])
    def test_a_tier_1_model_fits_in_the_budget(self, name: str) -> None:
        """
        Everything except ``model.py``, summed across the package.

        Summed rather than per file, because budgeting each file
        separately would mean the five-file template silently quintupled
        the allowance. ``model.py`` excluded, because the mathematics is
        not the framework's tax to charge.
        """
        package = MODELS_ROOT / name
        total = sum(
            statements(path) for path in sorted(package.glob("*.py")) if path.name != "model.py"
        )
        assert total <= TIER_1_BUDGET, (
            f"{name} spends {total} statements on ceremony against a budget "
            f"of {TIER_1_BUDGET}; either it is no longer tier 1, or the "
            f"framework is asking a simple model for too much"
        )
```

---

## 3. `tests/rade_qnet/models/test_reference_models.py`

15934 bytes · SHA-256 `804c5e8b4c66f0c6`

```python
"""
Tests for the baseline models.

Two different things are checked here, and the second is the reason this
phase exists.

The first is ordinary: each baseline builds, trains, saves and scores
through the unmodified lifecycle. That is worth asserting but not
surprising.

The second is structural, and it is a test of the *framework* rather than of
the models. Three engines with nothing in common -- an iterative one, a
closed-form one, and one that owns its own loop -- now run through the same
pipeline. If adding them required the pipeline to learn their names, the
abstraction failed and every subsequent engine will cost the same. If adding
a model required fifty lines of ceremony, the framework is not cheap enough
to be worth using for a simple model, and users will reach for a notebook
instead. Both are asserted rather than hoped for:
``test_no_pipeline_branches_on_an_engine_name`` and
``test_each_baseline_is_under_the_line_budget``.

Those two are the ones to read if this file ever fails after unrelated work.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.rade_qnet.api import evaluate, infer, train, train_jobs, tune
from src.rade_qnet.core.lifecycle.errors import StageError
from src.rade_qnet.core.spec.jobs import parse_job_set_spec
from src.rade_qnet.models.lstm_tabular.register import LstmTabularModel

if TYPE_CHECKING:
    pass

#: Marks a case that needs XGBoost, which is an optional dependency. A
#: contributor on a minimal install gets a skip rather than a failure, and
#: the structural tests above still run -- which are the ones that matter
#: most, because they are about the framework rather than about a library.
needs_xgboost = pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None, reason="xgboost is not installed"
)

#: Engine names the pipeline must never mention. A pipeline that branches on
#: one of these has special-cased a backend, which is the failure this phase
#: is designed to detect.
ENGINE_NAMES = ("torch", "sklearn", "xgboost", "lightgbm")

ORCHESTRATION = Path("src/rade_qnet/orchestration")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """
    Write a small linear problem to a CSV file.

    Linear on purpose: every baseline here can fit it, so a failure is never
    about the problem being too hard for one of them.

    Parameters
    ----------
    tmp_path
        Pytest's per-test directory.

    Returns
    -------
    Path
        The CSV file.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 4))
    targets = features @ np.array([1.5, -2.0, 0.5, 3.0]) + 0.25
    path = tmp_path / "linear.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"f{index}" for index in range(4)] + ["target"])
        writer.writerows([*row, target] for row, target in zip(features, targets, strict=True))
    return path


def specification(path: Path, name: str, engine: str, **training: object) -> dict:
    """
    Build a run specification for one baseline.

    Parameters
    ----------
    path
        The dataset.
    name
        Registered model name.
    engine
        Registered engine name.
    **training
        Extra training settings.

    Returns
    -------
    dict
        A specification ready for :func:`~rade_qnet.api.train`.
    """
    return {
        "task": "supervised",
        "model": {"name": name, "params": {}},
        "source": {"kind": "tabular", "path": str(path)},
        "training": {"engine": engine, **training},
        "reports": {"enabled": []},
        "hardware": {"device": "cpu"},
    }


class TestTheFrameworkStaysGeneral:
    """The claim the reference models exist to keep honest."""

    def test_no_pipeline_branches_on_an_engine_name(self) -> None:
        """
        No module under ``orchestration`` compares anything to an engine name.

        This is the phase's central claim, stated as a test. A pipeline that
        reads ``if engine == "xgboost"`` still works -- that is what makes it
        dangerous. It works for the engines somebody thought of, and the
        cost of the fourth engine is the same as the cost of the third,
        forever.

        The check is on the syntax tree rather than on the text, so that a
        name in a docstring or a comment is not a failure. Explaining why
        XGBoost owns its own loop is exactly the sort of comment that should
        be there.
        """
        offenders: list[str] = []
        for path in sorted(ORCHESTRATION.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and node.value in ENGINE_NAMES:
                    offenders.append(f"{path}:{node.lineno} -> {node.value!r}")
        assert not offenders, "orchestration names an engine:\n" + "\n".join(offenders)


class TestWhatTheFixtureCanAndCannotShow:
    """
    A finding about the Phase 0 fixture, pinned so it is not forgotten.

    The charter planned ``test_flagship_beats_baselines_on_the_fixture`` as
    a marked, informational check. Running the comparison showed the test
    could not mean anything, and why is more useful than the test would
    have been: the fixture's targets are near-exact linear combinations of
    its inputs, so a ridge regression reaches the noise floor and no model
    can beat it by more than rounding.

    That is not a defect in the fixture. Phase 0 built it to pin
    determinism and it does that well. It *is* a limit on what it can be
    used to argue, and the limit is invisible unless someone checks --
    which is what this does.
    """

    def test_the_targets_are_almost_exactly_linear_in_the_inputs(self) -> None:
        """
        Ordinary least squares explains essentially all of the variance.

        If this ever fails because the coefficient fell, the fixture has
        gained structure a linear model cannot capture, and a comparison
        between the flagship and the baselines has become worth running.
        Treat a failure here as good news and go read §8.6.
        """
        directory = Path("tests/fixtures/rade_qnet/golden/hybrid_gnn_rnn/input")
        if not directory.exists():  # pragma: no cover - depends on the checkout
            pytest.skip("the Phase 0 golden fixture is not present")

        features = np.load(directory / "elementary_pnl.npy")
        targets = np.load(directory / "target_pnl.npy")

        for index in range(targets.shape[1]):
            column = targets[:, index]
            fitted = features @ np.linalg.lstsq(features, column, rcond=None)[0]
            explained = 1 - ((column - fitted) ** 2).sum() / ((column - column.mean()) ** 2).sum()
            assert explained > 0.99, (
                f"target {index} is no longer linear in the inputs "
                f"(R^2 {explained:.4f}); the fixture may now be able to "
                f"discriminate between models -- see PHASE_6 §8.6"
            )


class TestEachBaselineRuns:
    """The whole lifecycle, on every engine, through unmodified pipelines."""

    @pytest.mark.parametrize(
        ("name", "engine", "settings"),
        [
            ("ridge", "sklearn", {}),
            pytest.param("xgb_tabular", "xgboost", {"n_estimators": 20}, marks=needs_xgboost),
            ("lstm_tabular", "torch", {"epochs": 3}),
        ],
    )
    def test_training_produces_a_bundle_that_rescores_identically(
        self, dataset: Path, tmp_path: Path, name: str, engine: str, settings: dict
    ) -> None:
        """
        The headline: train, save, reload, score, predict.

        Re-scoring is checked for *equality*, not closeness. A bundle that
        scores differently from the run that produced it means the saved
        model and the reported model are different models, and every
        comparison made from the report is then about something that was
        never saved.
        """
        __import__(f"src.rade_qnet.models.{name}")

        result = train(
            specification(dataset, name, engine, **settings),
            output_root=tmp_path / name,
        )
        assert result.bundle_directory is not None

        bundle = Path(result.bundle_directory)
        assert evaluate(bundle).metric("test", "mae") == result.metric("test", "mae")
        assert infer(bundle).n_predictions > 0

    @pytest.mark.parametrize(
        ("name", "engine", "settings"),
        [
            ("ridge", "sklearn", {}),
            pytest.param("xgb_tabular", "xgboost", {"n_estimators": 20}, marks=needs_xgboost),
        ],
    )
    def test_every_report_renders_for_a_one_shot_engine(
        self, dataset: Path, tmp_path: Path, name: str, engine: str, settings: dict
    ) -> None:
        """
        All four reports, on an engine with at most one epoch.

        The curves report is the one at risk: it draws a training history,
        and a closed-form fit has a history of one point. Rendering a
        one-point line is odd but honest; *failing* to render would make
        reporting a thing only gradient engines get, which would quietly
        make the one-shot engines second-class.
        """
        __import__(f"src.rade_qnet.models.{name}")

        run = specification(dataset, name, engine, **settings)
        run["reports"] = {"enabled": ["baselines", "curves", "quality", "summary"]}
        result = train(run, output_root=tmp_path / name)

        assert result.bundle_directory is not None
        reports = Path(result.bundle_directory).parents[2] / "reports"
        written = {path.name for path in reports.iterdir()}
        assert {"summary.md", "baselines.md", "data_quality.md"} <= written

    def test_a_one_shot_baseline_runs_in_a_job_set(self, dataset: Path, tmp_path: Path) -> None:
        """
        Phase 4's fan-out does not care that there is no training loop.

        Worth its own test because a job set runs each job in a separate
        process, and a one-shot engine's handle, history and bundle all
        have to survive that boundary the same way a network's do.
        """
        __import__("src.rade_qnet.models.ridge")

        manifest = train_jobs(
            parse_job_set_spec(
                {
                    "name": "baselines",
                    "output_root": str(tmp_path),
                    "defaults": {
                        "model": {"name": "ridge", "params": {"alpha": 0.1}},
                        "source": {"kind": "tabular", "path": str(dataset)},
                        "training": {"engine": "sklearn"},
                        "reports": {"enabled": []},
                        "hardware": {"device": "cpu", "threads_per_worker": 1},
                    },
                    "jobs": [{"id": "a"}, {"id": "b"}],
                    "placement": {"executor": "local"},
                }
            )
        )
        assert [job.succeeded for job in manifest.jobs] == [True, True]

    def test_a_one_shot_baseline_is_tunable(self, dataset: Path, tmp_path: Path) -> None:
        """
        A search over ``alpha``, end to end.

        Tuning is the stage most likely to assume a training loop, because
        pruning and intermediate reporting both need epochs. A search over
        an engine that has none must still run -- it just cannot prune.
        """
        __import__("src.rade_qnet.models.ridge")

        base = specification(dataset, "ridge", "sklearn")
        result = tune(
            {
                "model": "ridge",
                "base": base,
                "space": {"model.params.alpha": [0.01, 0.1, 1.0]},
                "trials": 3,
                "sampler": "grid",
                "objective": "mae",
                "direction": "minimise",
                "seed": 7,
                "name": "ridge-search",
            },
            output_root=tmp_path,
        )
        assert result.best is not None
        assert [record.succeeded for record in result.trials] == [True] * 3

    def test_the_closed_form_baseline_fits_a_linear_problem(
        self, dataset: Path, tmp_path: Path
    ) -> None:
        """
        Ridge on linear data should be nearly exact.

        Not a quality threshold so much as a wiring check: a column order
        scrambled anywhere between the CSV and the estimator leaves this
        number an order of magnitude larger while everything still runs.

        The penalty is set rather than left at its default, because a
        default of one shrinks the coefficients enough to put the error
        within the range a wiring fault would also produce -- which would
        make a passing test mean nothing.
        """
        __import__("src.rade_qnet.models.ridge")

        run = specification(dataset, "ridge", "sklearn")
        run["model"]["params"] = {"alpha": 0.01}
        assert train(run, output_root=tmp_path).metric("test", "mae") < 0.01


class TestTheInputContractIsEnforcedByThePipeline:
    """
    The declaration is only worth having because a run checks it.

    The unit tests in ``tests/rade_qnet/core/contract`` prove the contract
    type rejects what it should. These prove the pipeline asks it, which
    is the half that makes the difference between a declaration and a
    guarantee.
    """

    def test_a_model_that_accepts_anything_still_runs(self, dataset: Path, tmp_path: Path) -> None:
        """
        An unconstrained requirement is not a disabled pipeline stage.

        Worth asserting because the cheapest way to break this feature is
        to have the check silently skipped for every model that declares
        no constraints -- which is most of them.
        """
        __import__("src.rade_qnet.models.ridge")

        result = train(specification(dataset, "ridge", "sklearn"), output_root=tmp_path)
        assert result.bundle_directory is not None

    def test_a_signature_the_model_cannot_consume_is_refused(
        self, dataset: Path, tmp_path: Path
    ) -> None:
        """
        A mismatched pairing fails at ``declare_signature``, before building.

        The recurrent baseline consumes exactly one dynamic input. Handed
        a signature with two it must stop with a message naming both --
        not train on whichever one the dictionary yielded first, which is
        what it did before the contract existed.

        The second block is injected by patching the *definition's*
        signature rather than the framework's data module, and that is the
        case worth testing rather than a convenience. ``DatasetSource``
        already refuses a multi-block signature, so a user on
        ``TabularDataModule`` is covered without any contract at all. The
        user who is not covered is the one who wrote their own data module
        -- which, once ``data.py`` is mandatory, is most of them. This
        patch stands in for that data module.
        """
        original = LstmTabularModel.signature

        def two_blocks(self: LstmTabularModel, bundle: object):
            built = original(self, bundle)  # type: ignore[arg-type]
            extra = dict(built.dynamic)
            extra["an_unexpected_second_block"] = next(iter(built.dynamic.values()))
            return built.model_copy(update={"dynamic": extra})

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(LstmTabularModel, "signature", two_blocks)
            with pytest.raises(StageError) as caught:
                train(
                    specification(dataset, "lstm_tabular", "torch", epochs=1),
                    output_root=tmp_path,
                )

        assert caught.value.stage == "declare_signature"
        message = str(caught.value.__cause__)
        assert "exactly one unnamed" in message
        assert "an_unexpected_second_block" in message
```

