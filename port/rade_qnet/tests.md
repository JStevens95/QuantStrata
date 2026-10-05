# `tranql/models/rade/rade_qnet/tests`

8 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 32 | 1216 | `01e950fac0ad5abe` |
| 2 | `conftest.py` | 198 | 7384 | `81d71ab5fafa422c` |
| 3 | `locations.py` | 78 | 3231 | `59c4e2876f572acb` |
| 4 | `ruff.toml` | 50 | 2453 | `81fd794fd26156fb` |
| 5 | `test_api.py` | 628 | 22881 | `bdd9c331304761db` |
| 6 | `test_documentation.py` | 121 | 4376 | `a4784275d63c97e5` |
| 7 | `test_extensibility.py` | 346 | 12606 | `09b30ee0351cac76` |
| 8 | `test_scaffold.py` | 713 | 28364 | `6f2109cbb0ec4c83` |

---

## 1. `tranql/models/rade/rade_qnet/tests/__init__.py`

1216 bytes · SHA-256 `01e950fac0ad5abe`

```python
"""
Unit tests for the rade_qnet framework.

The tree below mirrors ``tranql/models/rade/rade_qnet/rade_qnet`` one directory at a time.  Mirroring is
not tidiness for its own sake: it means the question "is this component
tested?" is answered by looking at one predictable path, and an untested
sub-package shows up as an empty directory rather than as an absence nobody
notices.

Tests are built in the same order as the framework, so that each phase is
verified against components already proven in the phase before it.  See
``tranql/models/rade/rade_qnet/rade_qnet/docs/IMPLEMENTATION.md`` for the order and for each phase's
definition of done.

Scope
-----
These tests cover the **model-independent** framework.  Model-specific suites
arrive with the phase that builds the model, which is why ``models`` holds
placeholder packages rather than a full mirror until each model is delivered.

Naming
------
``test_<package>_<module>.py`` -- matching the convention used by the sibling
test suites in this repository, so a file name states which module it covers
without reference to its directory.

Running
-------
From the repository root::

    .venv/bin/python -m pytest tranql/models/rade/rade_qnet/tests -q
"""
```

---

## 2. `tranql/models/rade/rade_qnet/tests/conftest.py`

7384 bytes · SHA-256 `81d71ab5fafa422c`

```python
"""
Shared fixtures for the rade_qnet test suite.

Fixtures here are intentionally limited to *locating* things -- the source
tree, the test tree, the module list.  Behavioural fixtures (synthetic specs,
sources and bundles) belong in ``rade_qnet.testkit.fixtures`` so that model
authors outside this repository can use them too, and are re-exported from the
relevant sub-package ``conftest.py`` as each phase lands.

The suite is run from the repository root::

    .venv/bin/python -m pytest tranql/models/rade/rade_qnet/tests -q

Running it that way puts the repository root on ``sys.path``, which is what
makes the ``tranql.models.rade.rade_qnet.rade_qnet`` import path resolve.  The package itself uses
relative imports internally, so it is equally importable under any name.

The fixtures below do not assume that layout.  Every location comes from
:mod:`.locations`, which derives it from the imported package -- so the same
suite runs unchanged where the package imports under a deeper name, with the
tests in a sibling directory.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.provenance.logging import ROOT_LOGGER_NAME

from .locations import GOLDEN_ROOT, IMPORT_ROOT, PACKAGE_ROOT, TEST_ROOT

#: Artifacts whose presence stands for the whole capture.
#:
#: One from each half of it, chosen to catch the two ways the fixture goes
#: missing.  ``manifest.json`` is absent when nothing was ever captured.
#: ``elementary_pnl.npy`` is absent when only the text files arrived -- which
#: happens when the tree crossed a markdown-only proxy, since the arrays
#: cannot travel that way and the JSON beside them can.
#:
#: A presence check, not an integrity check: the manifest's digests are what
#: prove a fixture is the one it claims to be.
_GOLDEN_ARTIFACTS = (
    GOLDEN_ROOT / "hybrid_gnn_rnn" / "manifest.json",
    GOLDEN_ROOT / "hybrid_gnn_rnn" / "input" / "elementary_pnl.npy",
)


@pytest.fixture(autouse=True)
def _restore_framework_logging() -> Iterator[None]:
    """
    Undo any logging configuration a test installs.

    ``configure_logging`` is deliberately global and sticky: it sets a level,
    attaches a handler, and sets ``propagate = False`` so an application with
    its own root handler does not see every framework line twice.  All of that
    is right for an application and wrong for a test suite, because the state
    outlives the test that created it.

    The failure it causes is nastier than it sounds.  ``propagate = False``
    stops ``caplog`` from seeing framework log records at all, so a test
    asserting that something was logged passes in isolation and fails only
    when it runs after a test that configured logging.  Restoring the logger
    here keeps that ordering dependency out of the suite entirely.

    Yields
    ------
    None
        The test runs with the framework logger restored afterwards.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    level, propagate = logger.level, logger.propagate
    handlers = list(logger.handlers)
    try:
        yield
    finally:
        logger.setLevel(level)
        logger.propagate = propagate
        logger.handlers[:] = handlers


@pytest.fixture(scope="session")
def requires_golden() -> None:
    """
    Skip the requesting tests when the captured baseline is not present.

    Requested with ``@pytest.mark.usefixtures("requires_golden")`` on a test
    class, rather than by a module-level marker. The distinction matters:
    the classes that replay the baseline sit in files alongside classes that
    do not, and in one case the ratio is two tests to nineteen. Skipping a
    whole module to protect two tests would quietly stop running seventeen
    that work perfectly well without any fixture at all.

    Why skip rather than fail. The capture is a one-off step that a fresh
    clone has not run, and the arrays cannot cross a markdown-only proxy, so
    their absence is an ordinary condition rather than a defect. A suite that
    reports red for a missing optional input teaches people to ignore red.

    What is lost when this skips is worth stating plainly: these are the
    tests that compare this implementation's numbers against the original's.
    Everything else still runs, so the suite proves the framework behaves --
    it just stops proving it produces the same figures as the baseline.

    This also gates the tests that check the capture is *complete*, which
    reads like a contradiction and is not. The two answer different
    questions, and the order matters. This fixture asks whether a capture is
    present at all; the completeness tests then ask whether the one that is
    present has every artifact it should. Gating them means a half-copied
    tree -- the text files without the arrays -- is reported as "no fixture"
    rather than as a dozen failures, while a genuine capture missing one
    file still fails loudly, which is what those tests are for.

    Raises
    ------
    Skipped
        If any artifact in :data:`_GOLDEN_ARTIFACTS` is missing, naming the
        first one and how to produce it.
    """
    for artifact in _GOLDEN_ARTIFACTS:
        if not artifact.is_file():
            pytest.skip(
                f"golden fixture not captured: {artifact} is missing. "
                f"Capture it with examples/rade_qnet/phase0_capture_baseline.py, "
                f"or point RADE_QNET_GOLDEN_ROOT at an existing capture, "
                f"and note that a partial copy is worse than none -- the text "
                f"files alone stop these tests skipping without letting them pass"
            )


@pytest.fixture(scope="session")
def import_root() -> Path:
    """
    Return the directory that must be on ``sys.path`` to import the package.

    The repository root here; the directory holding the top-level package
    wherever the package imports under a deeper name. What a subprocess needs
    on its path to import the package by name.

    Returns
    -------
    Path
        See :data:`.locations.IMPORT_ROOT`.
    """
    return IMPORT_ROOT


@pytest.fixture(scope="session")
def package_root() -> Path:
    """
    Return the absolute path to the ``rade_qnet`` source package.

    Returns
    -------
    Path
        The directory holding ``core``, ``engines`` and the rest.
    """
    return PACKAGE_ROOT


@pytest.fixture(scope="session")
def test_root() -> Path:
    """
    Return the absolute path to the ``rade_qnet`` test package.

    Returns
    -------
    Path
        The directory holding this ``conftest.py``.
    """
    return TEST_ROOT


@pytest.fixture
def run_directory(tmp_path: Path) -> Iterator[Path]:
    """
    Provide an empty directory standing in for a run's working directory.

    Pipelines write bundles, reports and manifests beneath a run directory.
    Tests are given a fresh temporary one per test so that no test can observe
    another's artifacts, and so a failing test leaves its output behind for
    inspection rather than overwriting a shared location.

    Parameters
    ----------
    tmp_path
        Pytest's per-test temporary directory.

    Yields
    ------
    Path
        An existing, empty directory.
    """
    directory = tmp_path / "run"
    directory.mkdir()
    yield directory
```

---

## 3. `tranql/models/rade/rade_qnet/tests/locations.py`

3231 bytes · SHA-256 `59c4e2876f572acb`

```python
"""
Where the package under test and its test tree live, worked out rather than assumed.

Why this module exists
----------------------
The suite runs in two places with two different layouts. In this repository
the package imports as ``tranql.models.rade.rade_qnet.rade_qnet`` and the tests sit at
``tranql/models/rade/rade_qnet/tests``. Where it is deployed, the same files import as something
like ``tranql.models.rade.rade_qnet.rade_qnet``, with the tests in a sibling
package. A path such as ``Path("tranql/models/rade/rade_qnet/rade_qnet/models")`` is right in one and
wrong in the other, and it is also wrong here the moment pytest is started
from any directory but the repository root.

So nothing in the suite spells out a location. Every path and every dotted
name a test needs is derived here, from the one thing that is true in every
layout: the package imported successfully, so ``__name__`` and ``__file__``
say exactly where it is.

The import below is the only line in this module that names the package, and
the porting script rewrites it along with every other ``from tranql.models.rade.rade_qnet.rade_qnet``
import. Everything else follows from it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import tranql.models.rade.rade_qnet.rade_qnet as package

#: The package's dotted import name: ``tranql.models.rade.rade_qnet.rade_qnet`` here, the deployment's
#: own name elsewhere. Used wherever a test has to name a module as a string --
#: a ``monkeypatch`` target, an ``__import__``, a ``sys.modules`` key, or a
#: line of code handed to a fresh interpreter.
PACKAGE_NAME: Final = package.__name__

#: The package's directory, holding ``core``, ``engines``, ``docs`` and the rest.
PACKAGE_ROOT: Final = Path(package.__file__).resolve().parent

#: The directory that has to be on ``sys.path`` for :data:`PACKAGE_NAME` to
#: import. One level above the package per dotted component: the repository
#: root for ``tranql.models.rade.rade_qnet.rade_qnet``, the directory holding ``tranql`` for a deeper
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
```

---

## 4. `tranql/models/rade/rade_qnet/tests/ruff.toml`

2453 bytes · SHA-256 `81fd794fd26156fb`

```toml
# Lint configuration for the rade_qnet test tree.
#
# The test tree is held to the same standard as the source tree, so this file
# extends the package's configuration rather than restating it.  Only the
# handful of rules that are genuinely wrong for tests are relaxed below.
#
# Ruff resolves configuration by walking up from each linted file, and the test
# tree sits outside `tranql/models/rade/rade_qnet/rade_qnet`; without this file, tests would silently fall
# back to Ruff's defaults and the standard would apply to half the code.
extend = "../rade_qnet/ruff.toml"

[lint]
extend-ignore = [
    # A test's arguments are fixtures whose types are fixed by the fixture
    # definition, and every test returns None.  Annotating both adds noise
    # without adding safety.  The remaining annotation rules are relaxed for
    # the same reason and no other: a test module's helpers and fakes are
    # read beside their single call site, and their docstrings already carry
    # a Returns section.  Annotations stay mandatory in `tranql/models/rade/rade_qnet/rade_qnet`, which
    # is the code anyone outside this repository actually calls.
    "ANN001",
    "ANN002",
    "ANN003",
    "ANN201",
    "ANN202",
    "ANN206",
    # Asserting against a literal value is what a test does.
    "PLR2004",
    # A fixture is sometimes requested purely for its side effect (a temporary
    # directory, a patched environment) and never referenced in the body.
    "ARG001",
    # Much of this tree exists to supply fakes: stub models implementing a
    # capability protocol, callbacks handed to a writer, hooks recording that
    # they fired.  Their signatures are dictated by the interface they stand
    # in for, so ignoring an argument is the normal case rather than an
    # oversight.  In `tranql/models/rade/rade_qnet/rade_qnet` these stay enabled, and the one deliberate
    # case there is written as an explicit `del` with its reason.
    "ARG002",
    "ARG003",
    "ARG005",
    # A throwaway fake declaring `static: dict = {}` is clearer than the same
    # fake declaring `static: ClassVar[dict] = {}`.  The rule guards against
    # shared mutable state outliving an instance, which a per-test stub
    # cannot do.
    "RUF012",
]

# Docstrings stay mandatory.  A test name plus its docstring is how a failure
# is diagnosed from a CI log by someone who did not write the test, so this is
# the last place to relax documentation rules.
```

---

## 5. `tranql/models/rade/rade_qnet/tests/test_api.py`

22881 bytes · SHA-256 `bdd9c331304761db`

```python
"""
Tests for the front door.

``api`` adds no behaviour. Every function here assembles the same pieces a
caller could assemble by hand, which is the property that keeps it from
becoming a second, divergent way to run a model -- so these tests are mostly
about *equivalence*: that the one-line path produces what the long path
produces.

The one thing ``api`` genuinely contributes is reaching across layers that
are not allowed to see each other. ``orchestration`` may not import
``models``; :func:`~rade_qnet.api.train_groups` reads a group manifest,
expands it into jobs, and hands them to a runner that has no idea a group
exists.
"""

from __future__ import annotations

import json

import pytest
import yaml

from tranql.models.rade.rade_qnet.rade_qnet import api
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import (
    engine as register_engine,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import model as register_model
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import SpecError
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.compute.local import LocalExecutor
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.jobs.groups import MANIFEST_FILENAME
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import (
    SyntheticEngine,
    isolated_registries,
)

from .orchestration.jobs.support import (
    DIRECTORY_MODEL_NAME,
    ENGINE_TAG,
    GROUP_FILENAME,
    MODEL_NAME,
    SyntheticDirectoryModel,
    SyntheticSupervisedModel,
    job_set_payload,
    write_linear_dataset,
)


@pytest.fixture(autouse=True)
def _registries():
    """
    Isolate the component registries for every test here.

    Yields
    ------
    None
        For the duration of the test.
    """
    # ``empty=True`` because the stand-in engine below claims a shipped
    # engine's name. Without it, whether this fixture succeeds depends on
    # whether an earlier test imported the real one -- which made the suite
    # pass or fail on collection order.
    with isolated_registries(empty=True):
        register_engine(ENGINE_TAG)(SyntheticEngine)
        register_model(MODEL_NAME, engine=ENGINE_TAG)(SyntheticSupervisedModel)
        register_model(DIRECTORY_MODEL_NAME, engine=ENGINE_TAG)(SyntheticDirectoryModel)
        yield


@pytest.fixture
def dataset(tmp_path):
    """
    Write the exactly-linear dataset every run reads.

    Returns
    -------
    pathlib.Path
        The CSV file.
    """
    return write_linear_dataset(tmp_path / "linear.csv")


def run_payload(dataset, output_root):
    """
    Build a single-run configuration mapping.

    Parameters
    ----------
    dataset
        The CSV file.
    output_root
        Where the run writes.

    Returns
    -------
    dict
        The configuration.
    """
    return {
        "model": MODEL_NAME,
        "source": {"kind": "tabular", "path": str(dataset)},
        "training": {"engine": ENGINE_TAG},
        "reports": {"enabled": []},
        "output_root": str(output_root),
    }


class TestTrainingOneModel:
    """`api.train`, in the three forms a configuration arrives in."""

    def test_a_mapping_trains(self, dataset, tmp_path):
        """
        The notebook path.

        The fit is closed-form over exactly linear data, so an r-squared of
        one is the only correct answer and a tolerance would not be needed
        even if one were offered.
        """
        result = api.train(run_payload(dataset, tmp_path / "out"))

        assert result.evaluations["test"].metrics["r2"] == pytest.approx(1.0)

    def test_the_output_root_may_be_a_string(self, dataset, tmp_path):
        """
        A string directory is accepted, like every other path argument.

        `evaluate` and `infer` already took a bundle as either, and a
        specification can be a string, so an output root that took only
        `Path` was an inconsistency rather than a rule. It also failed
        badly: the string travelled several frames inwards before
        `root / run_id` raised `TypeError: unsupported operand type(s)
        for /`, naming neither the argument nor the mistake.
        """
        result = api.train(
            run_payload(dataset, tmp_path / "out"), output_root=str(tmp_path / "strings")
        )

        assert result.bundle_directory is not None
        assert str(tmp_path / "strings") in result.bundle_directory

    def test_a_yaml_file_trains(self, dataset, tmp_path):
        """The production path."""
        path = tmp_path / "run.yaml"
        path.write_text(yaml.safe_dump(run_payload(dataset, tmp_path / "out")), encoding="utf-8")

        assert api.train(path).evaluations["test"].metrics["r2"] == pytest.approx(1.0)

    def test_an_already_validated_specification_trains(self, dataset, tmp_path):
        """
        The test path, and the one that nearly broke silently.

        `RunSpec` is an annotated discriminated union, so checking against
        it in an instance test raises rather than returning `False` -- a
        mistake no passing test would reveal by accident.
        """
        from tranql.models.rade.rade_qnet.rade_qnet.core.spec.run import (  # noqa: PLC0415
            parse_run_spec,
        )

        spec = parse_run_spec(run_payload(dataset, tmp_path / "out"))

        assert api.train(spec).evaluations["test"].metrics["r2"] == pytest.approx(1.0)

    def test_the_run_directory_is_derived_from_the_specification(self, dataset, tmp_path):
        """
        Two runs of one configuration share a directory.

        The same reasoning as a job set's run identifier: a timestamped
        name leaves a trail of near-identical directories nobody can tell
        apart, and a re-run after a crash should complete the first rather
        than start a second.
        """
        output_root = tmp_path / "out"
        api.train(run_payload(dataset, output_root))

        before = {path.name for path in output_root.iterdir()}
        api.train(run_payload(dataset, output_root))

        assert {path.name for path in output_root.iterdir()} == before

    def test_a_changed_configuration_lands_somewhere_else(self, dataset, tmp_path):
        """
        Derived from the digest, so a different run is a different place.

        Otherwise a changed configuration would overwrite the results of
        the one it replaced.
        """
        output_root = tmp_path / "out"
        api.train(run_payload(dataset, output_root))
        api.train({**run_payload(dataset, output_root), "seed": 99})

        # Directories only: the catalog writes its index into the same root.
        assert len([path for path in output_root.iterdir() if path.is_dir()]) == 2

    def test_output_root_can_be_overridden(self, dataset, tmp_path):
        """
        A caller writing somewhere other than the file says.

        Common when one configuration is run against several destinations
        and editing the file for each would be the wrong way round.
        """
        elsewhere = tmp_path / "elsewhere"

        api.train(run_payload(dataset, tmp_path / "out"), output_root=elsewhere)

        assert elsewhere.exists()

    def test_a_supervised_model_in_an_interactive_run_is_refused_clearly(self, tmp_path):
        """
        Said here, not several stages later.

        ``api.train`` routes on the task, so an interactive configuration is
        no longer refused outright -- it goes to the interactive pipeline.
        What is refused is the *mismatch*: a model that learns from a fixed
        dataset cannot be trained by interaction, and the message names the
        model rather than reporting a missing environment from four stages
        in.

        The interactive path's own tests live in
        ``orchestration/pipelines/test_pipelines_reinforce.py``.
        """
        with pytest.raises(SpecError, match="learns from a fixed dataset"):
            api.train(
                {
                    "task": "reinforcement",
                    "model": MODEL_NAME,
                    "environment": {"name": "hedging"},
                    "training": {"engine": "torch"},
                    "output_root": str(tmp_path),
                }
            )


class TestTrainingAJobSet:
    """`api.train_jobs`."""

    def test_a_mapping_runs_every_job(self, dataset, tmp_path):
        """Two jobs in, two records out."""
        manifest = api.train_jobs(
            job_set_payload(dataset, tmp_path / "out"), executor=LocalExecutor()
        )

        assert len(manifest.succeeded) == 2

    def test_a_yaml_file_runs_every_job(self, dataset, tmp_path):
        """The documented path from `ARCHITECTURE.md` §12."""
        path = tmp_path / "set.yaml"
        path.write_text(
            yaml.safe_dump(job_set_payload(dataset, tmp_path / "out")), encoding="utf-8"
        )

        assert len(api.train_jobs(path, executor=LocalExecutor()).succeeded) == 2

    def test_a_failing_job_is_recorded_rather_than_raised(self, dataset, tmp_path):
        """
        The partial-failure policy, surfaced through the front door.

        Thirty-nine models that trained are thirty-nine results, and
        discarding them over one typo in the fortieth is the wrong trade.
        """
        manifest = api.train_jobs(
            job_set_payload(
                dataset,
                tmp_path / "out",
                jobs=[
                    {"id": "good"},
                    {"id": "bad", "overrides": {"source": {"path": str(tmp_path / "gone.csv")}}},
                ],
            ),
            executor=LocalExecutor(),
        )

        assert [record.job_id for record in manifest.succeeded] == ["good"]
        assert [record.job_id for record in manifest.failed] == ["bad"]


class TestTrainingAcrossGroups:
    """``api.train_groups``, which is why this module sits above the layers."""

    @pytest.fixture
    def data_root(self, tmp_path):
        """
        Lay out a two-group set, one data file per group.

        Each group gets its own file, which is the layout ``group_overrides``
        assumes: a directory per group, named in the source parameters.

        Returns
        -------
        pathlib.Path
            The group set directory.
        """
        root = tmp_path / "data"
        groups = [{"name": "alpha"}, {"name": "beta"}]
        root.mkdir()
        for entry in groups:
            directory = root / entry["name"]
            directory.mkdir()
            write_linear_dataset(directory / GROUP_FILENAME)
        (root / MANIFEST_FILENAME).write_text(json.dumps({"groups": groups}), encoding="utf-8")
        return root

    @pytest.fixture
    def defaults(self):
        """
        Provide the shared run-specification fragment.

        A model source, because that is the shape a group override produces:
        it names a directory, and reading a directory is the model's
        business rather than the framework's.

        Returns
        -------
        dict
            The defaults.
        """
        return {
            "model": {"name": DIRECTORY_MODEL_NAME},
            "source": {"kind": "model"},
            "training": {"engine": ENGINE_TAG},
            "reports": {"enabled": []},
        }

    def test_one_job_runs_per_group(self, data_root, defaults, tmp_path):
        """
        Read, expand, dispatch -- the three steps, as one call.

        The middle one is the step that is easy to get subtly wrong, which is
        why it is worth a function rather than a docstring.
        """
        manifest = api.train_groups(
            data_root,
            defaults=defaults,
            output_root=tmp_path / "out",
            executor=LocalExecutor(),
        )

        assert [record.job_id for record in manifest.jobs] == ["alpha", "beta"]

    def test_a_group_filter_restricts_the_set(self, data_root, defaults, tmp_path):
        """Naming some groups trains only those groups."""
        manifest = api.train_groups(
            data_root,
            defaults=defaults,
            output_root=tmp_path / "out",
            groups=["beta"],
            executor=LocalExecutor(),
        )

        assert [record.job_id for record in manifest.jobs] == ["beta"]

    def test_per_group_overrides_reach_the_jobs(self, data_root, defaults, tmp_path):
        """
        The hook that makes a group set a job set rather than a loop.

        Asserted through a setting that survives into the manifest, since the
        exact fit makes both groups score identically.
        """
        manifest = api.train_groups(
            data_root,
            defaults=defaults,
            output_root=tmp_path / "out",
            overrides_for=lambda group: {"seed": 7 if group.name == "alpha" else 9},
            executor=LocalExecutor(),
        )

        assert manifest.record("alpha").seed != manifest.record("beta").seed

    def test_a_missing_group_set_fails_before_anything_runs(self, defaults, tmp_path):
        """
        Read first, dispatch second.

        A directory that is not there is knowable without training anything.
        """
        with pytest.raises(SpecError):
            api.train_groups(tmp_path / "absent", defaults=defaults, output_root=tmp_path / "out")

    def test_two_variants_of_a_set_share_one_catalog(self, data_root, defaults, tmp_path):
        """
        So a registry can compare them.

        Each variant of a set has its own directory -- its name carries the
        specification digest -- and the catalog once defaulted to that
        directory, so no query could see both variants at once.
        """
        for seed in (1, 2):
            api.train_groups(
                data_root,
                defaults={**defaults, "seed": seed},
                output_root=tmp_path / "out",
                groups=["alpha"],
                executor=LocalExecutor(),
            )

        versions = [run.version for run in api.registry(tmp_path / "out").runs(job="alpha")]
        assert versions == [1, 2]


class TestSelectingTrainedRuns:
    """``api.registry``: from many trained runs back to the one wanted."""

    def test_runs_trained_with_a_tag_are_selected_by_it(self, dataset, tmp_path):
        """The tag set at training time is the one queried by."""
        root = tmp_path / "out"
        api.train({**run_payload(dataset, root), "tags": ["sweep"]})
        api.train({**run_payload(dataset, root), "seed": 3})

        selected = api.registry(root).runs(tags=["sweep"])

        assert [run.version for run in selected] == [1]

    def test_a_selected_run_can_be_promoted_and_evaluated(self, dataset, tmp_path):
        """
        The whole round trip, through public functions only.

        Train, pick the best, promote it, find it again by alias, and score
        the directory the registry hands back.
        """
        root = tmp_path / "out"
        api.train({**run_payload(dataset, root), "tags": ["sweep"]})
        runs = api.registry(root)
        runs.promote(runs.best("r2", direction="maximise", tags=["sweep"]), "production")

        production = runs.get(MODEL_NAME, alias="production")
        result = api.evaluate(production.directory)

        assert result.evaluations["test"].metrics["r2"] == pytest.approx(1.0)


def tune_payload(dataset, output_root):
    """
    Build a small search configuration mapping.

    Parameters
    ----------
    dataset
        The CSV file.
    output_root
        Where the search writes.

    Returns
    -------
    dict
        The configuration.
    """
    del output_root
    return {
        "model": MODEL_NAME,
        "base": {
            "source": {"kind": "tabular", "path": str(dataset)},
            "training": {"engine": ENGINE_TAG},
            "reports": {"enabled": []},
        },
        "space": {"seed": [1, 2, 3]},
        "trials": 3,
        "sampler": "grid",
        "name": "smoke",
    }


def trained_bundle(tmp_path, dataset):
    """
    Train one model through the front door and return its bundle.

    Returns
    -------
    pathlib.Path
        The bundle directory.
    """
    root = tmp_path / "runs"
    api.train(run_payload(dataset, root), output_root=root)
    bundles = sorted(root.rglob("manifest.json"))
    assert bundles, "training wrote no bundle"
    return bundles[0].parent


class TestEvaluatingASavedModel:
    """`api.evaluate`, which is the whole phase in one call."""

    def test_a_bundle_can_be_rescored(self, tmp_path, dataset):
        """The headline case, in one line as advertised."""
        result = api.evaluate(trained_bundle(tmp_path, dataset))

        assert "test" in result.evaluations
        assert result.evaluations["test"].in_original_units is True

    def test_rescoring_reproduces_the_training_metrics(self, tmp_path, dataset):
        """
        Exactly, which is the gate the whole phase is built around.

        Compared against the metrics recorded in the bundle rather than
        against the training call's return value, because the bundle is
        what anybody actually has weeks later.
        """
        bundle = trained_bundle(tmp_path, dataset)
        recorded = json.loads((bundle / "result.json").read_text(encoding="utf-8"))
        rescored = api.evaluate(bundle)

        for split, evaluation in recorded["evaluations"].items():
            assert rescored.evaluations[split].metrics == evaluation["metrics"]

    def test_a_reproduction_says_the_source_did_not_change(self, tmp_path, dataset):
        """Which is the half of provenance a reader acts on."""
        result = api.evaluate(trained_bundle(tmp_path, dataset))

        assert result.source_changed is False

    def test_selecting_splits_scores_only_those(self, tmp_path, dataset):
        """A re-score against new observations usually wants only test."""
        result = api.evaluate(trained_bundle(tmp_path, dataset), splits=("test",))

        assert set(result.evaluations) == {"test"}

    def test_the_run_is_rooted_beside_the_bundle(self, tmp_path, dataset):
        """
        Rather than in whatever directory the caller happened to be in.

        Asserted on the context rather than on the filesystem, because
        evaluation deliberately writes nothing: it returns a result, and a
        caller who wants it kept keeps it. Checking for a directory would
        be asserting a side effect this phase chose not to have.
        """
        bundle = trained_bundle(tmp_path, dataset)
        context = api._bundle_context(bundle, action="evaluate", output_root=None)

        assert context.output_directory.parent == bundle.parent
        assert context.metadata["bundle"] == str(bundle)

    def test_the_evaluation_run_has_no_catalog(self, tmp_path, dataset):
        """
        Because it writes no bundle, so there is nothing to record.

        Handing it one would invite a future version to register an
        evaluation as though it were a model.
        """
        bundle = trained_bundle(tmp_path, dataset)
        context = api._bundle_context(bundle, action="evaluate", output_root=None)

        assert context.catalog is None


class TestPredictingWithASavedModel:
    """`api.infer`, and the provenance it insists on."""

    def test_a_bundle_produces_predictions(self, tmp_path, dataset):
        """In the target's original units, as the contract says."""
        predictions = api.infer(trained_bundle(tmp_path, dataset))

        assert predictions.n_predictions > 0
        assert predictions.in_original_units is True

    def test_predictions_carry_their_provenance(self, tmp_path, dataset):
        """Reconciling predictions later is the ordinary case, not the exception."""
        bundle = trained_bundle(tmp_path, dataset)
        predictions = api.infer(bundle)

        assert predictions.bundle_version is not None
        assert predictions.provenance["model_name"] == MODEL_NAME
        assert predictions.provenance["predicted_at"]

    def test_predictions_match_the_evaluation_pass(self, tmp_path, dataset):
        """
        The two share their reload path.

        This is the test that the sharing is real rather than coincidental:
        if either grows its own version of a stage, the counts diverge.
        """
        import numpy as np  # noqa: PLC0415

        bundle = trained_bundle(tmp_path, dataset)
        predictions = api.infer(bundle)
        scored = api.evaluate(bundle, splits=("test",))

        assert predictions.n_predictions == scored.evaluations["test"].n_samples
        assert np.all(np.isfinite(predictions.values))


class TestSearchingASpace:
    """`api.tune`, in the three forms a configuration arrives in."""

    def test_a_search_runs_from_a_mapping(self, tmp_path, dataset):
        """The notebook case."""
        result = api.tune(tune_payload(dataset, tmp_path), output_root=tmp_path / "searches")

        assert len(result.trials) == 3

    def test_a_search_runs_from_a_file(self, tmp_path, dataset):
        """The production case."""
        path = tmp_path / "search.yaml"
        path.write_text(yaml.safe_dump(tune_payload(dataset, tmp_path)), encoding="utf-8")

        result = api.tune(path, output_root=tmp_path / "searches")
        assert len(result.trials) == 3

    def test_the_winner_is_identified(self, tmp_path, dataset):
        """With the direction and split recorded alongside it."""
        result = api.tune(tune_payload(dataset, tmp_path), output_root=tmp_path / "searches")

        assert result.best.succeeded
        assert result.direction == "minimise"
        assert result.objective_split == "validation"

    def test_each_trial_keeps_its_bundle(self, tmp_path, dataset):
        """Which is the difference between acting on a search and repeating it."""
        result = api.tune(tune_payload(dataset, tmp_path), output_root=tmp_path / "searches")

        assert result.best.bundle_directory is not None
        assert (tmp_path / "searches").exists()

    def test_the_winning_bundle_can_be_evaluated(self, tmp_path, dataset):
        """
        The two halves of the phase meeting.

        A search produces a bundle, and that bundle can be re-scored
        without retraining anything.
        """
        from pathlib import Path  # noqa: PLC0415

        result = api.tune(tune_payload(dataset, tmp_path), output_root=tmp_path / "searches")
        rescored = api.evaluate(Path(result.best.bundle_directory))

        assert rescored.metric("validation", "mae") == pytest.approx(result.best.objective)

    def test_a_search_is_named_in_its_directory(self, tmp_path, dataset):
        """So two searches in one root are told apart."""
        api.tune(tune_payload(dataset, tmp_path), output_root=tmp_path / "searches")

        assert any(path.name.startswith("smoke-") for path in (tmp_path / "searches").iterdir())
```

---

## 6. `tranql/models/rade/rade_qnet/tests/test_documentation.py`

4376 bytes · SHA-256 `a4784275d63c97e5`

````python
"""
Tests that the configuration examples in the documentation actually work.

``ARCHITECTURE.md`` §12 is the first thing anyone reads and the place they
copy their first configuration from. It had drifted: the job-set example was
written before Phase 1 settled the schema and used field names that no
longer existed, so a reader following it got a validation error on their
first run -- from the document that was supposed to be teaching them the
format.

Documentation drifts because nothing checks it. This checks it: the YAML is
extracted from the file and put through the real loader, so a schema change
that invalidates the example fails here rather than on a new user's machine.

Only the configuration blocks are checked, not the prose. A test that tried
to verify explanatory text would be a test nobody could keep passing.
"""

from __future__ import annotations

import re

import pytest
import yaml

from tranql.models.rade.rade_qnet.rade_qnet.core.spec.jobs import parse_job_set_spec

from .locations import PACKAGE_ROOT

#: The document under test.
ARCHITECTURE = PACKAGE_ROOT / "docs" / "ARCHITECTURE.md"

#: The example's filename, used to find its block rather than relying on
#: the block's position -- which changes whenever a section is added above.
JOB_SET_EXAMPLE = "configs/hybrid_portfolio.yaml"


@pytest.fixture(scope="module")
def job_set():
    """
    Parse the job-set example out of the architecture document.

    Returns
    -------
    JobSetSpec
        The validated job set.
    """
    # Imported for its registration side effect: the example names the
    # flagship, and validating it means resolving that name.
    import tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.register  # noqa: F401, PLC0415

    text = ARCHITECTURE.read_text(encoding="utf-8")
    match = re.search(rf"```yaml\n# {re.escape(JOB_SET_EXAMPLE)}\n(.*?)```", text, re.DOTALL)
    assert match is not None, f"no yaml block for {JOB_SET_EXAMPLE} in {ARCHITECTURE}"

    return parse_job_set_spec(yaml.safe_load(match.group(1)))


class TestTheDocumentedJobSet:
    """The example a reader copies first."""

    def test_it_parses(self, job_set):
        """
        The minimum bar, and the one the example previously failed.

        A document teaching a file format should produce a file that
        loads.
        """
        assert job_set.job_ids == ("FX__G10", "FX__EM", "RATES__USD")

    def test_every_job_validates(self, job_set):
        """
        Parsing is not enough: the jobs have to be real run specifications.

        A job set validates lazily enough that a malformed job survives
        loading and fails at dispatch, which is exactly the drift this
        guards against.
        """
        assert set(job_set.validate_jobs()) == {"FX__G10", "FX__EM", "RATES__USD"}

    def test_the_prose_about_merging_is_true(self, job_set):
        """
        The document claims naming one field leaves its siblings alone.

        `FX__EM` changes two model parameters. If the claim were false it
        would have lost the set's epochs, its thread budget and its
        sequence length -- so this asserts the explanation, not just the
        syntax.
        """
        run = job_set.validate_jobs()["FX__EM"]

        assert run.model.params["units"] == 64
        assert run.training.epochs == 200
        assert run.hardware.threads_per_worker == 1
        assert run.source.transforms.sequence.length == 20

    def test_the_thread_budget_is_pinned_as_the_text_says(self, job_set):
        """
        The example is also an instruction about reproducibility.

        It pins the budget and explains why; a later edit that dropped the
        field would leave the explanation attached to a file that no
        longer does it.
        """
        assert all(
            job_set.run_spec_for(job_id).hardware.threads_per_worker is not None
            for job_id in job_set.job_ids
        )

    def test_the_per_cluster_complexity_differs(self, job_set):
        """
        The point the example is making.

        An example where every job had the same model would illustrate
        nothing about why a portfolio is a job set.
        """
        widths = {
            job_id: job_set.run_spec_for(job_id).model.params["units"] for job_id in job_set.job_ids
        }

        assert len(set(widths.values())) > 1
````

---

## 7. `tranql/models/rade/rade_qnet/tests/test_extensibility.py`

12606 bytes · SHA-256 `09b30ee0351cac76`

```python
"""
A model defined outside ``rade_qnet`` trains through the whole lifecycle.

This is the framework's central promise, stated as a test. Everything else
is in service of it: a user brings a model, the framework brings the rest.

Why it needs its own file
-------------------------
Every model that ships lives under ``rade_qnet.models``, and every synthetic
model in the rest of the suite lives under ``tests``. Both are inside the
repository, and both are imported by code that already imports the
framework. Neither answers the question a user actually has, which is
whether a model in *their* distribution, importing ``rade_qnet`` as a
third-party dependency, works the same way.

It does, and the mechanism is that there is no mechanism: importing the
module registers it. But "it should work, there is nothing special about
it" is exactly the kind of claim that stops being true without anyone
noticing — the first import-time registry scan or packaging assumption
added for convenience would break it, and nothing else in the suite would
fail.

What is deliberately not done here
----------------------------------
The model below does not import anything private. It uses only the five
public names a user would reach for, and if any of them moves, this test
fails -- which is the point. A test that reached into internals to make
itself work would be testing that the internals exist rather than that the
public surface is sufficient.

The other half: where extensibility stops
-----------------------------------------
A model plugs in from outside. An *engine* does not, and the last class here
pins that boundary: a new engine name registers perfectly happily, and then
a specification naming it is refused, because ``TrainingSpec`` is a union
discriminated on a literal engine name. That trade is argued in the
``engines`` package charter -- each engine's settings get their own validated
type -- and it is asserted here so the limit is covered rather than inferred.

Nothing else in the suite would notice if it changed: every synthetic engine
registers under the name ``sklearn``, reusing a discriminator that already
exists, so none of them exercises a genuinely new engine name.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from pydantic import Field
from sklearn.ensemble import RandomForestRegressor

# The five public names a third-party model needs, and nothing else.
from tranql.models.rade.rade_qnet.rade_qnet.api import evaluate, infer, train
from tranql.models.rade.rade_qnet.rade_qnet.core.authoring.supervised import SupervisedModel
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import (
    engine as register_engine,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import get_engine, model
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import SpecError
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.base import Spec
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.run import parse_run_spec
from tranql.models.rade.rade_qnet.rade_qnet.engines import sklearn as _sklearn_engine  # noqa: F401
from tranql.models.rade.rade_qnet.rade_qnet.sources.dataset.tabular import TabularDataModule
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import (
    SyntheticEngine,
    isolated_registries,
)

#: The name the out-of-tree model claims. Registered inside a fixture rather
#: than at import, so that collecting this module does not leave a component
#: behind for every other test in the session to trip over.
MODEL_NAME = "third_party_forest"


class ThirdPartySpec(Spec):
    """
    Settings for a model the framework has never heard of.

    Parameters
    ----------
    n_estimators
        Trees in the forest.
    max_depth
        Depth of each tree.
    """

    n_estimators: int = Field(default=20, ge=1)
    max_depth: int = Field(default=4, ge=1)


class ThirdPartyForest(SupervisedModel):
    """
    A model as a user would write it, in a package of their own.

    Not decorated at class scope: the decorator runs inside the fixture, so
    the registration is undone when the test finishes.

    Attributes
    ----------
    spec
        Validates the ``model.params`` block.
    """

    spec = ThirdPartySpec

    def data_module(self, spec: object) -> TabularDataModule:
        """
        Use the framework's tabular reader.

        Parameters
        ----------
        spec
            The validated run specification, unused.

        Returns
        -------
        TabularDataModule
            The framework's own.
        """
        del spec
        return TabularDataModule()

    def build_model(self, spec: object, signature: object) -> RandomForestRegressor:
        """
        Return an unfitted forest.

        Parameters
        ----------
        spec
            The run specification, for the model's parameters.
        signature
            The declared interface, unused by a forest.

        Returns
        -------
        RandomForestRegressor
            Unfitted, seeded so the test is deterministic.
        """
        del signature
        settings = ThirdPartySpec.model_validate(dict(spec.model.params))  # type: ignore[attr-defined]
        return RandomForestRegressor(
            n_estimators=settings.n_estimators,
            max_depth=settings.max_depth,
            random_state=0,
        )


@pytest.fixture
def registered():
    """
    Register the out-of-tree model for one test.

    Yields
    ------
    None
        For the duration of the test.
    """
    with isolated_registries():
        model(MODEL_NAME, engine="sklearn")(ThirdPartyForest)
        yield


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """
    Write a small regression problem.

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
    features = rng.normal(size=(300, 4))
    targets = features @ np.array([1.5, -2.0, 0.5, 3.0])
    path = tmp_path / "book.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"f{index}" for index in range(4)] + ["target"])
        writer.writerows([*row, target] for row, target in zip(features, targets, strict=True))
    return path


def specification(dataset: Path) -> dict:
    """
    Build a run specification naming the out-of-tree model.

    Parameters
    ----------
    dataset
        The CSV to read.

    Returns
    -------
    dict
        An unvalidated specification.
    """
    return {
        "task": "supervised",
        "model": {"name": MODEL_NAME, "params": {"n_estimators": 20}},
        "source": {"kind": "tabular", "path": str(dataset)},
        "training": {"engine": "sklearn"},
        "reports": {"enabled": ["summary", "baselines"]},
        "hardware": {"device": "cpu"},
    }


class TestAModelFromOutsideTheFramework:
    """The framework's central promise, exercised end to end."""

    def test_naming_it_in_a_specification_resolves_it(
        self, registered, dataset: Path, tmp_path: Path
    ) -> None:
        """
        Importing is the whole registration mechanism.

        No entry point, no plugin manifest, no scan. If this ever needs
        one, a user's model has stopped being an ordinary Python class.
        """
        del registered
        result = train(specification(dataset), output_root=tmp_path / "runs")

        assert result.bundle_directory is not None

    def test_it_gets_the_full_lifecycle(self, registered, dataset: Path, tmp_path: Path) -> None:
        """
        Train, re-score and predict, with no model-specific support.

        Re-scoring is checked for equality rather than closeness: a bundle
        that scores differently from the run that produced it is a
        different model, whoever wrote it.
        """
        del registered
        result = train(specification(dataset), output_root=tmp_path / "runs")
        bundle = Path(str(result.bundle_directory))

        assert evaluate(bundle).metric("test", "mae") == result.metric("test", "mae")
        assert infer(bundle).values.shape[0] > 0

    def test_it_gets_the_reports_too(self, registered, dataset: Path, tmp_path: Path) -> None:
        """
        Reports are not reserved for models that ship with the framework.

        Worth asserting separately because reporting is the part most
        likely to acquire a quiet assumption about where a model lives --
        a lookup keyed on the model name, say.
        """
        del registered
        result = train(specification(dataset), output_root=tmp_path / "runs")

        reports = Path(str(result.bundle_directory)).parents[2] / "reports"
        assert {"summary.md", "baselines.md"} <= {p.name for p in reports.iterdir()}

    def test_its_settings_are_validated_like_any_other(
        self, registered, dataset: Path, tmp_path: Path
    ) -> None:
        """
        A user's own spec class gets the framework's validation.

        The constraint is declared on `ThirdPartySpec`, not anywhere in
        the framework, and it is still enforced -- which is what makes a
        bad configuration a parse error rather than an obscure failure
        inside scikit-learn some minutes later.
        """
        del registered
        invalid = specification(dataset)
        invalid["model"]["params"] = {"n_estimators": 0}

        with pytest.raises(Exception, match=r"n_estimators|greater than or equal"):
            train(invalid, output_root=tmp_path / "runs")


#: An engine name the framework has never heard of, used to locate the edge of
#: what can be added from outside.
UNKNOWN_ENGINE = "jax"


class TestTheBoundaryOfWhatPlugsIn:
    """Models arrive from outside; engines do not. Pinned, rather than assumed."""

    def test_a_new_engine_name_registers_without_complaint(self) -> None:
        """
        Half the boundary: the component registry is open to any name.

        Worth asserting separately from the refusal below, because together
        the two locate exactly where the limit falls. An engine is
        registrable from anywhere; what it cannot do is appear in a
        specification.
        """
        with isolated_registries():
            register_engine(UNKNOWN_ENGINE)(SyntheticEngine)

            assert get_engine(UNKNOWN_ENGINE) is SyntheticEngine

    def test_but_a_specification_naming_it_is_refused(self, dataset: Path) -> None:
        """
        The other half: ``TrainingSpec`` is a closed union.

        A fourth engine needs its own training-spec type declared in
        ``core.spec.training``. That is a deliberate trade -- each engine's
        settings get a validated type of their own, which is what stops a
        one-shot fit being configured with gradient-descent options -- but it
        does mean a backend cannot arrive from outside the distribution.

        Asserted so that opening the union later has to come past this test,
        rather than the limit being discovered by whoever first tries.

        The type is the assertion, with no ``match`` on the wording: that
        sentence is pydantic's, and the rest of the spec suite declines to
        pin it for the same reason. What matters here is that an unknown
        engine is a clean specification error rather than an obscure failure
        somewhere inside a pipeline. The *content* of the message is covered
        by the next test, against our own engine names.
        """
        with isolated_registries():
            register_engine(UNKNOWN_ENGINE)(SyntheticEngine)
            unknown = specification(dataset)
            unknown["training"] = {"engine": UNKNOWN_ENGINE}

            with pytest.raises(SpecError):
                parse_run_spec(unknown)

    def test_the_refusal_names_the_engines_that_would_work(self, dataset: Path) -> None:
        """
        Because the fix is to pick one of them, or to declare a fourth.

        A bare "validation error" would leave a reader guessing between a
        misspelt name, an uninstalled package and an engine that was never
        written -- three different problems with three different answers.
        """
        unknown = specification(dataset)
        unknown["training"] = {"engine": UNKNOWN_ENGINE}

        with pytest.raises(SpecError) as caught:
            parse_run_spec(unknown)

        message = str(caught.value)
        assert all(name in message for name in ("torch", "xgboost", "sklearn"))
```

---

## 8. `tranql/models/rade/rade_qnet/tests/test_scaffold.py`

28364 bytes · SHA-256 `6f2109cbb0ec4c83`

```python
"""
Structural tests for the rade_qnet scaffold.

These tests assert nothing about behaviour -- at this stage there is none.  They
assert the properties that make the rest of the build safe: that the package
tree is importable, that every package documents its own charter, that the test
tree mirrors the source tree, and that the dependency layering described in
``docs/ARCHITECTURE.md`` actually holds in the code.

The layering test is the load-bearing one, and it is written now, against an
almost empty tree, deliberately.  Architectural boundaries are not broken by a
decision to break them; they are broken by one convenient import in a hurry.
Encoding the boundary as a test means the first such import fails CI instead of
quietly becoming precedent.
"""

from __future__ import annotations

import ast
import importlib
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from tranql.models.rade.rade_qnet.rade_qnet.core.spec.base import Spec

from .locations import PACKAGE_NAME, PACKAGE_ROOT

# The package under test, under whatever name it was imported as.
rade_qnet = importlib.import_module(PACKAGE_NAME)

TEST_ROOT = Path(__file__).resolve().parent

# The top-level packages that make up the framework, in dependency order.
FRAMEWORK_LAYERS = (
    "core",
    "sources",
    "storage",
    "analysis",
    "engines",
    "orchestration",
    "models",
    "testkit",
)

# Which layers each layer is permitted to import from.  This table *is* the
# architecture: every entry encodes a decision recorded in the charter
# docstring of the corresponding package.
#
# Reading it top to bottom, the framework forms a one-way stack.  Two entries
# deserve their reasoning restated here, because they are the ones most likely
# to be questioned:
#
#   * ``orchestration`` may not import ``models``.  Pipelines resolve a model
#     by name through the component registry.  If a pipeline imported a model,
#     the framework would depend on the library it exists to serve, and no
#     user could add a model without editing the framework.
#   * No framework layer carries business vocabulary.  Knowing that a number
#     is a P&L in a particular currency is a model's business, expressed in
#     its own ``data.py``; the moment a training loop knows it, the framework
#     stops being reusable on the next problem.  There was once a ``domains``
#     layer for such knowledge; everything in it turned out to be either
#     generic (now in ``orchestration.jobs``) or one model's vocabulary (now
#     in that model), so the layer was removed.
ALLOWED_DEPENDENCIES: dict[str, frozenset[str]] = {
    # The vocabulary. Depends on nothing, which is what lets a spec be parsed
    # and hashed in a process that has never imported a training library.
    "core": frozenset(),
    # Where data comes from.
    "sources": frozenset({"core"}),
    # The system of record. Stores bytes an engine hands it; does not serialise
    # them itself, hence no dependency on engines.
    "storage": frozenset({"core"}),
    # Metrics, figures and reports. Receives plain arrays, never a live model.
    "analysis": frozenset({"core"}),
    # One adapter per training library.
    "engines": frozenset({"core", "sources"}),
    # The conductor. Sees everything below it, and the model library only
    # through the registry.
    "orchestration": frozenset({"core", "sources", "storage", "analysis", "engines"}),
    # The model library sits on top of the whole framework.
    "models": frozenset({"core", "sources", "storage", "analysis", "engines", "orchestration"}),
    # Conformance tooling exercises the framework, but never a specific model.
    "testkit": frozenset({"core", "sources", "storage", "analysis", "engines", "orchestration"}),
}

# Third-party distributions ``core`` is allowed to import.  Everything else it
# imports must come from the standard library.  An allowlist is used rather
# than a list of banned libraries because the property being protected is
# "core stays cheap to import", and that property is broken by the next heavy
# dependency nobody thought to ban.
#
# ``yaml`` was added in Phase 1 for ``core.spec.run.load_run_spec``.  A
# specification file is the user's interface to the framework, so parsing and
# validating it belongs with the specs rather than being reimplemented by each
# entry point.  PyYAML is a pure-parser dependency that imports in single-digit
# milliseconds, so it does not compromise the property above.  Extending this
# set is a deliberate, reviewable change -- which is the point of the
# allowlist.
CORE_PERMITTED_THIRD_PARTY = frozenset({"numpy", "pydantic", "typing_extensions", "yaml"})

# Sub-trees whose tests are deliberately deferred.  Model-specific tests
# arrive with the phase that builds the model, so
# requiring a mirrored test package for every layer beneath them now would
# create empty directories that assert nothing.
DEFERRED_TEST_SUBTREES = frozenset({"models"})

# Sub-trees inside a deferred one that have since been delivered, and are
# therefore held to the mirroring rule again.  Listed explicitly rather than
# inferred from whether a test package happens to exist, because inferring it
# would make the rule self-fulfilling: deleting a test package would remove
# the requirement to have one.  Adding a model here is the last step of the
# phase that builds it.
DELIVERED_TEST_SUBTREES = frozenset(
    {
        "models/hybrid_gnn_rnn",
        "models/ridge",
        "models/xgb_tabular",
        "models/lstm_tabular",
    }
)

# Directories that hold no importable code and are therefore exempt from the
# mirroring and packaging rules.
NON_CODE_DIRECTORIES = frozenset({"docs", "__pycache__", ".ruff_cache"})


def _iter_source_files(root: Path) -> Iterator[Path]:
    """
    Yield every Python file beneath ``root``, skipping non-code directories.

    Parameters
    ----------
    root
        Directory to walk.

    Yields
    ------
    Path
        Absolute path to a ``.py`` file.
    """
    for path in sorted(root.rglob("*.py")):
        # ``parts`` is checked rather than the immediate parent so that a file
        # nested several levels inside an excluded directory is also skipped.
        if NON_CODE_DIRECTORIES.isdisjoint(path.parts):
            yield path


def _iter_package_directories(root: Path) -> Iterator[Path]:
    """
    Yield every directory beneath ``root`` that is part of the package tree.

    Parameters
    ----------
    root
        Directory to walk.

    Yields
    ------
    Path
        Absolute path to a directory, excluding non-code directories.
    """
    for path in sorted(root.rglob("*")):
        if path.is_dir() and NON_CODE_DIRECTORIES.isdisjoint(path.parts):
            yield path


def _is_delivered(relative: Path) -> bool:
    """
    Report whether a package sits inside a delivered sub-tree.

    Parameters
    ----------
    relative
        A package path relative to the package root.

    Returns
    -------
    bool
        True when the package is, or lies beneath, a delivered sub-tree.
    """
    posix = relative.as_posix()
    return any(
        posix == delivered or posix.startswith(f"{delivered}/")
        for delivered in DELIVERED_TEST_SUBTREES
    )


def _dotted_package_of(source_file: Path) -> tuple[str, ...]:
    """
    Return the dotted package containing ``source_file``, as path components.

    The result is relative to and includes the package root, so a module at
    ``rade_qnet/engines/torch/engine.py`` reports ``("rade_qnet", "engines",
    "torch")``.  For an ``__init__.py`` the containing package is the directory
    itself, which is what Python's relative-import resolution uses.

    Parameters
    ----------
    source_file
        A ``.py`` file inside the package.

    Returns
    -------
    tuple of str
        Components of the containing package's dotted name.
    """
    relative = source_file.relative_to(PACKAGE_ROOT)
    # ``parts[:-1]`` drops the filename; for ``__init__.py`` that leaves the
    # directory, which is exactly the package it defines.
    return ("rade_qnet", *relative.parts[:-1])


def _resolve_relative_import(containing_package: tuple[str, ...], node: ast.ImportFrom) -> str:
    """
    Resolve an explicit relative import to an absolute dotted module name.

    One level (``.``) refers to the containing package, two (``..``) to its
    parent, and so on -- so ``level - 1`` components are stripped from the
    containing package before the imported suffix is appended.

    Parameters
    ----------
    containing_package
        Components of the package containing the importing module.
    node
        The ``from ... import ...`` node, with ``node.level`` greater than zero.

    Returns
    -------
    str
        The absolute dotted name the import refers to.
    """
    retained = len(containing_package) - (node.level - 1)
    base = containing_package[:retained]
    suffix = tuple(node.module.split(".")) if node.module else ()
    return ".".join((*base, *suffix))


def _imported_modules(source_file: Path) -> Iterator[str]:
    """
    Yield the absolute dotted name of every module imported by ``source_file``.

    Relative imports are resolved against the file's location, so a caller can
    treat every yielded name uniformly without knowing which import form
    produced it.

    Parameters
    ----------
    source_file
        A ``.py`` file inside the package.

    Yields
    ------
    str
        An absolute dotted module name.
    """
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    containing_package = _dotted_package_of(source_file)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                yield _resolve_relative_import(containing_package, node)
            elif node.module:
                yield node.module


def _layer_of(dotted_name: str) -> str | None:
    """
    Return the framework layer a dotted name belongs to, if any.

    Both ``rade_qnet.core.spec`` and the same module under the name the
    package was actually imported as -- ``tranql.models.rade.rade_qnet.rade_qnet.core.spec`` here, or
    something deeper where it is vendored -- resolve to ``"core"``, so the
    answer does not depend on how the package was imported.  A name that is not inside
    ``rade_qnet``, or that is a top-level module such as ``rade_qnet.api``, has no
    layer.

    Parameters
    ----------
    dotted_name
        An absolute dotted module name.

    Returns
    -------
    str or None
        The layer name, or ``None`` if the name sits outside the layered tree.
    """
    parts = dotted_name.split(".")
    prefix = PACKAGE_NAME.split(".")
    if parts[: len(prefix)] == prefix:
        parts = ["rade_qnet", *parts[len(prefix) :]]
    if parts[0] != "rade_qnet" or len(parts) < 2:
        return None
    return parts[1] if parts[1] in ALLOWED_DEPENDENCIES else None


def _distribution_of(dotted_name: str) -> str:
    """
    Return the top-level distribution or module name of a dotted import.

    Parameters
    ----------
    dotted_name
        An absolute dotted module name.

    Returns
    -------
    str
        The first component, which is what gets installed or shipped.
    """
    return dotted_name.split(".", maxsplit=1)[0]


def _import_every_spec_module() -> None:
    """
    Import every module under ``core.spec``.

    Spec classes are only discoverable once their defining module has been
    imported, and nothing else in this file imports them.  Importing the
    package by walking the directory means a spec module added later is picked
    up without anybody editing this test.
    """
    for source_file in _iter_source_files(PACKAGE_ROOT / "core" / "spec"):
        relative = source_file.relative_to(PACKAGE_ROOT)
        # Unlike `_dotted_package_of`, the module's own name is needed here:
        # importing the containing package would not necessarily import the
        # module that defines the specs.
        parts = (*relative.parts[:-1], relative.stem)
        importlib.import_module(".".join((PACKAGE_NAME, *parts)))


def _all_spec_types() -> list[type[Spec]]:
    """
    Return every concrete spec class, found through the class hierarchy.

    Derived from ``Spec.__subclasses__`` transitively rather than from a list,
    so a spec cannot be added without these invariants applying to it.

    Returns
    -------
    list
        Public spec classes, in a deterministic order.
    """
    _import_every_spec_module()

    found: dict[str, type[Spec]] = {}
    pending = [Spec]
    while pending:
        for subclass in pending.pop().__subclasses__():
            pending.append(subclass)
            # Private bases (`_RunSpecBase`, `_SourceSpecBase`) exist to share
            # fields between the real specs and are never instantiated by a
            # user, so holding them to the user-facing invariants would be
            # testing an implementation detail.
            if not subclass.__name__.startswith("_"):
                found[subclass.__name__] = subclass
    return [found[name] for name in sorted(found)]


def _bare_instance(spec_type: type[Spec]) -> Spec | None:
    """
    Construct a spec with no arguments, or report that it cannot be.

    Parameters
    ----------
    spec_type
        The spec class to try.

    Returns
    -------
    Spec or None
        The instance, or ``None`` if the spec requires fields.
    """
    try:
        return spec_type()
    except ValidationError:
        # The only expected failure: a spec with a required field. Anything
        # else is a real defect and is deliberately left to propagate.
        return None


def _defaulted_spec_types(spec_types: Sequence[type[Spec]]) -> frozenset[type[Spec]]:
    """
    Return the specs that something else constructs on the user's behalf.

    A spec reached through a ``default_factory`` must be constructible with no
    arguments, because that is exactly what the factory does. A spec the user
    always names explicitly carries no such obligation -- see the note in
    ``PHASE_1_CORE.md`` §7.1 on why the original, stronger rule was narrowed.

    Parameters
    ----------
    spec_types
        Every discovered spec class.

    Returns
    -------
    frozenset
        The subset reached through a default factory.
    """
    defaulted: set[type[Spec]] = set()
    for spec_type in spec_types:
        for field in spec_type.model_fields.values():
            factory = field.default_factory
            # A `default_factory` is usually the spec class itself, which is
            # the case worth following; anything else (`dict`, `list`) is not
            # a spec and is ignored.
            if isinstance(factory, type) and issubclass(factory, Spec):
                defaulted.add(factory)
    return frozenset(defaulted)


# Collected at import time so each module appears as its own test case, making
# a failure name the offending file directly in the pytest report.
SOURCE_FILES = list(_iter_source_files(PACKAGE_ROOT))
SOURCE_FILE_IDS = [str(path.relative_to(PACKAGE_ROOT)) for path in SOURCE_FILES]

SPEC_TYPES = _all_spec_types()
SPEC_TYPE_IDS = [spec_type.__name__ for spec_type in SPEC_TYPES]
DEFAULTED_SPEC_TYPES = _defaulted_spec_types(SPEC_TYPES)


class TestPackageIntegrity:
    """The package tree is importable and self-describing."""

    def test_package_reports_a_version(self):
        """``rade_qnet`` exposes a version, which every bundle records."""
        assert isinstance(rade_qnet.__version__, str)
        assert rade_qnet.__version__

    def test_every_layer_exists_as_a_package(self):
        """Each layer named in the architecture is present on disk."""
        missing = [
            layer
            for layer in FRAMEWORK_LAYERS
            if not (PACKAGE_ROOT / layer / "__init__.py").is_file()
        ]
        assert not missing, f"layers declared in the architecture but absent: {missing}"

    def test_every_layer_is_covered_by_the_dependency_table(self):
        """No layer can be added without stating what it may depend on."""
        assert set(FRAMEWORK_LAYERS) == set(ALLOWED_DEPENDENCIES)

    def test_dependency_table_references_only_real_layers(self):
        """The table cannot permit a dependency on a package that does not exist."""
        for layer, permitted in ALLOWED_DEPENDENCIES.items():
            unknown = permitted - set(FRAMEWORK_LAYERS)
            assert not unknown, f"{layer} is permitted to import unknown layers: {unknown}"

    def test_dependencies_are_acyclic(self):
        """
        The layers form a one-way stack.

        If two layers could import each other, neither could be understood,
        tested or replaced on its own, which is the whole return on having
        layers.
        """
        for layer, permitted in ALLOWED_DEPENDENCIES.items():
            for dependency in permitted:
                assert layer not in ALLOWED_DEPENDENCIES[dependency], (
                    f"circular dependency declared between {layer} and {dependency}"
                )

    @pytest.mark.parametrize("directory", _iter_package_directories(PACKAGE_ROOT), ids=str)
    def test_every_directory_is_a_package(self, directory: Path):
        """
        Every code directory declares itself with an ``__init__.py``.

        Implicit namespace packages would work at run time, but they break
        editor navigation and let a directory exist with no charter explaining
        why it is there.
        """
        assert (directory / "__init__.py").is_file(), f"{directory} has no __init__.py"

    @pytest.mark.parametrize("source_file", SOURCE_FILES, ids=SOURCE_FILE_IDS)
    def test_every_module_has_a_docstring(self, source_file: Path):
        """
        Every module opens with a docstring.

        For a package ``__init__.py`` this is its charter: what belongs in the
        package, what does not, and which modules it will hold.  That charter
        is the first thing a contributor reads, and it is how a sub-package
        resists becoming a dumping ground.
        """
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        docstring = ast.get_docstring(tree)
        assert docstring is not None, f"{source_file} has no module docstring"
        assert docstring.strip(), f"{source_file} has an empty module docstring"


class TestDependencyLayering:
    """Imports respect the architecture's one-way dependency stack."""

    @pytest.mark.parametrize("source_file", SOURCE_FILES, ids=SOURCE_FILE_IDS)
    def test_module_imports_only_permitted_layers(self, source_file: Path):
        """
        A module imports only from its own layer and the layers below it.

        Top-level modules such as ``api.py`` and ``cli.py`` have no layer of
        their own and are free to compose the whole framework -- that is their
        purpose.
        """
        importing_layer = _layer_of(".".join(_dotted_package_of(source_file)))
        if importing_layer is None:
            return

        permitted = ALLOWED_DEPENDENCIES[importing_layer] | {importing_layer}
        violations = [
            imported
            for imported in _imported_modules(source_file)
            if (target := _layer_of(imported)) is not None and target not in permitted
        ]
        assert not violations, (
            f"{source_file.relative_to(PACKAGE_ROOT)} is in layer '{importing_layer}', "
            f"which may import {sorted(permitted)}, but imports: {sorted(violations)}"
        )

    @pytest.mark.parametrize("source_file", SOURCE_FILES, ids=SOURCE_FILE_IDS)
    def test_core_stays_free_of_training_libraries(self, source_file: Path):
        """
        ``core`` imports only the standard library, pydantic and numpy.

        This is what allows a specification to be loaded, validated, hashed and
        shipped to a worker in a process that has never imported PyTorch.  A
        job set that spawns sixteen workers pays the import cost sixteen times,
        so the restriction is a throughput concern as much as a design one.
        """
        if _layer_of(".".join(_dotted_package_of(source_file))) != "core":
            return

        forbidden = sorted(
            {
                distribution
                for imported in _imported_modules(source_file)
                if (distribution := _distribution_of(imported)) not in sys.stdlib_module_names
                and distribution not in CORE_PERMITTED_THIRD_PARTY
                and distribution not in {"rade_qnet", PACKAGE_NAME.split(".")[0]}
            }
        )
        assert not forbidden, (
            f"{source_file.relative_to(PACKAGE_ROOT)} is in 'core', which may import only the "
            f"standard library and {sorted(CORE_PERMITTED_THIRD_PARTY)}, but imports: {forbidden}"
        )


class TestEverySpecObeysTheSpecInvariants:
    """
    The four spec invariants hold for every spec, discovered rather than listed.

    Each invariant is also tested directly in ``core/spec/test_spec_*.py``,
    which is where a *failure* is diagnosed.  The tests here exist for a
    different reason: they are the ones that catch a spec added in a later
    phase that nobody remembered to test.  A hand-written list would have to
    be updated by the same person who forgot, so the list is derived from the
    class hierarchy instead.
    """

    def test_specs_were_actually_discovered(self):
        """
        The discovery itself works.

        Without this, a broken walk would collect nothing and every test below
        would pass vacuously -- the failure mode that makes a structural test
        worse than no test, because it reports a guarantee it never checked.
        """
        assert len(SPEC_TYPES) > 15

    @pytest.mark.parametrize("spec_type", SPEC_TYPES, ids=SPEC_TYPE_IDS)
    def test_a_misspelled_key_is_refused(self, spec_type: type[Spec]):
        """
        ``extra="forbid"``, so a typo cannot be silently ignored.

        Defect 1 in the architecture's table: a key that is dropped rather
        than rejected produces a run configured with one value and trained
        with another, reporting plausible numbers the whole way.
        """
        assert spec_type.model_config.get("extra") == "forbid"

    @pytest.mark.parametrize("spec_type", SPEC_TYPES, ids=SPEC_TYPE_IDS)
    def test_the_spec_cannot_be_mutated_after_validation(self, spec_type: type[Spec]):
        """
        ``frozen=True``, so a recorded spec describes what actually ran.

        A pipeline that received a spec and a bundle that recorded it must
        hold the same values, or the bundle documents a run that never
        happened.
        """
        assert spec_type.model_config.get("frozen") is True

    @pytest.mark.parametrize("spec_type", SPEC_TYPES, ids=SPEC_TYPE_IDS)
    def test_a_defaulted_spec_is_bare_constructible(self, spec_type: type[Spec]):
        """
        A spec reached through a ``default_factory`` constructs bare.

        Defect 2: a required field hidden inside a defaulted sub-spec means the
        enclosing spec cannot be built at all, and the error names the inner
        field rather than the outer one the user was editing.

        Scoped to defaulted specs on purpose.  A spec that a user must name
        explicitly -- a source or a split strategy -- is entitled to require
        fields, because nothing constructs it on the user's behalf.
        """
        if spec_type not in DEFAULTED_SPEC_TYPES:
            pytest.skip("not reached through a default_factory; may require fields")
        spec_type()

    @pytest.mark.parametrize("spec_type", SPEC_TYPES, ids=SPEC_TYPE_IDS)
    def test_a_spec_round_trips_exactly(self, spec_type: type[Spec]):
        """
        Dumping and reloading returns an equal spec.

        Defect 1 again, from the other direction: the round trip is what makes
        a spec saved in a bundle a faithful record, and what makes the spec
        digest stable enough for a step cache to key on.

        Applied to anything constructible with no arguments, not only to
        defaulted specs.  ``SklearnTrainingSpec`` is a case in point: nothing
        defaults into it, because it is a sibling of ``TorchTrainingSpec`` in
        the training union, but a user selects it with one line of YAML and it
        has to round-trip exactly like any other.
        """
        original = _bare_instance(spec_type)
        if original is None:
            pytest.skip("requires fields, so there is no canonical instance to round-trip")
        assert type(original).model_validate_json(original.model_dump_json()) == original


class TestTestTreeMirrorsSourceTree:
    """The test tree shadows the source tree, so coverage gaps are visible."""

    def test_every_framework_package_has_a_test_package(self):
        """
        Each framework package has a matching test package.

        Mirroring means an untested sub-package is an empty directory rather
        than an absence nobody notices.  Model and domain sub-trees are
        excluded until the phase that builds them delivers them, at which
        point they are listed in ``DELIVERED_TEST_SUBTREES`` and mirrored
        like anything else.
        """
        missing: list[str] = []
        for directory in _iter_package_directories(PACKAGE_ROOT):
            relative = directory.relative_to(PACKAGE_ROOT)
            # A deferred sub-tree still needs its own top-level test package as
            # a placeholder, but not a mirror of everything beneath it.
            if (
                relative.parts[0] in DEFERRED_TEST_SUBTREES
                and len(relative.parts) > 1
                and not _is_delivered(relative)
            ):
                continue
            if not (TEST_ROOT / relative).is_dir():
                missing.append(str(relative))
        assert not missing, f"source packages with no mirrored test package: {missing}"

    def test_no_orphaned_test_packages(self):
        """
        Every test package corresponds to a real source package.

        Catches the opposite failure: a test package left behind after its
        source package was renamed or removed, which would otherwise sit there
        passing and testing nothing.
        """
        orphaned: list[str] = []
        for directory in _iter_package_directories(TEST_ROOT):
            relative = directory.relative_to(TEST_ROOT)
            if not (PACKAGE_ROOT / relative).is_dir():
                orphaned.append(str(relative))
        assert not orphaned, f"test packages with no matching source package: {orphaned}"


class TestDocumentation:
    """The documentation a contributor is pointed at actually exists."""

    @pytest.mark.parametrize(
        "document",
        [
            "ARCHITECTURE.md",
            "IMPLEMENTATION.md",
            "CODING_STANDARDS.md",
            "README.md",
        ],
    )
    def test_top_level_document_exists(self, document: str):
        """A document referenced from the package charter is present."""
        path = PACKAGE_ROOT / "docs" / document
        assert path.is_file(), f"{path} is referenced by the package but missing"
        assert path.read_text(encoding="utf-8").strip(), f"{path} is empty"

    def test_every_phase_referenced_by_the_plan_exists(self):
        """
        Every phase document linked from ``IMPLEMENTATION.md`` resolves.

        The implementation plan is the entry point for anyone -- human or
        agent -- picking up the build.  A dead link there sends them looking
        for instructions that do not exist.
        """
        plan = (PACKAGE_ROOT / "docs" / "IMPLEMENTATION.md").read_text(encoding="utf-8")
        phase_documents = sorted((PACKAGE_ROOT / "docs" / "phases").glob("PHASE_*.md"))
        assert phase_documents, "no phase documents found"

        unreferenced = [path.name for path in phase_documents if path.name not in plan]
        assert not unreferenced, (
            f"phase documents not referenced by IMPLEMENTATION.md: {unreferenced}"
        )
```

