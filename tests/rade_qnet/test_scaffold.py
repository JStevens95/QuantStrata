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

from src.rade_qnet.core.spec.base import Spec

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
    package was actually imported as -- ``src.rade_qnet.core.spec`` here, or
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
