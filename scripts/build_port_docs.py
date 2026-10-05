"""
Emit the ``rade_qnet`` source and test trees as markdown, for transport
through a proxy that passes nothing but ``.md``.

The problem this solves
-----------------------
A locked-down machine can fetch documentation and nothing else. ``git clone``,
a release zip and PyPI are all unavailable, so the only way the code can cross
is as prose. That constraint is what this script answers: one markdown
document per directory, each file inside it fenced verbatim, plus an
index carrying a SHA-256 for every file so the far side can prove the paste
landed intact rather than discovering it three days later in a stack trace.

Why one document per directory
------------------------------
Per *top-level package* would put 304 KB of ``core`` into a single page, which
GitHub truncates in the rendered view and which is miserable to scroll. Per
*file* would mean several hundred documents to navigate. Per directory keeps
the largest around 140 KB, and it matches how someone rebuilding the tree
actually works: make a directory, fill it, move on.

Why the tests travel too
------------------------
Source alone crosses as something nobody can trust. The suite is what turns a
pasted tree into a verified one: it is the difference between "the files
appear to be there" and "this behaves the way it did on the machine it left".
The far side runs it once and knows, rather than finding out during the first
real run.

The two trees share every mechanism and differ only in a document-name prefix,
so ``tests/rade_qnet/core/spec`` becomes ``tests__core__spec.md`` beside the
``core__spec.md`` it exercises. Source document names are unchanged by the
addition.

Why the fence is measured rather than fixed
--------------------------------------------
A file containing a code fence of its own would close its own block early and
silently truncate. The source tree happens to contain none; the test tree
contains one, in ``test_documentation.py``, which greps ``ARCHITECTURE.md``
for a fenced YAML block and so necessarily spells one out.

Rather than pick a longer fence globally and hope, each file is given the
shortest fence that cannot occur inside it -- one backtick longer than its
longest run, never fewer than three. Files with no backticks keep the ordinary
three, so the common case is unchanged, and a future file that embeds a fence
needs no intervention at all.

Why the output is verified rather than trusted
----------------------------------------------
A generator that writes a code fence is easy; a generator that writes a code
fence which round-trips to the original byte-for-byte is the only kind worth
having, because the entire point is fidelity. :func:`verify` re-parses every
document it just wrote and compares the recovered bytes against the source.
If that check ever fails the script exits non-zero and writes nothing further,
because a port document that is subtly wrong is worse than no port document.

Why the layout is rewritten on the way out
------------------------------------------
Here the package imports as ``src.rade_qnet`` and its suite lives in
``tests/rade_qnet``. On the far side it is vendored into a larger repository
as ``tranql.models.rade.rade_qnet.rade_qnet``, with the suite beside it as
``tranql.models.rade.rade_qnet.tests``. Exporting the repository's spelling
would leave five hundred imports for someone to correct by hand inside a paste,
which is exactly where a typo goes unnoticed.

So the export takes a :class:`Layout` and writes every file as it must read in
that layout: headings name the target paths, imports name the target package,
the test tree's ruff configuration extends the package's from wherever it now
sits, and isort treats the host's top-level package as first-party.

The package itself imports only relatively, so the rewrite touches the tests
and the prose rather than the code; that is what makes it mechanical rather
than clever. A longer package name does lengthen import lines, though, and an
import that no longer fits must be re-wrapped or ``ruff format --check`` fails
on arrival. Rather than imitate the formatter, the rewritten tree is written to
a scratch directory and ruff is run over it there -- the same configuration,
discovered the same way it will be on the far side -- and the export refuses to
proceed if anything is left for ruff to report.

``pyproject.toml`` travels only in the repository layout. In a vendored layout
the host repository already has one, and pasting ours over it would be far
worse than useless; the dependencies are listed in the index instead.

Usage
-----
::

    python scripts/build_port_docs.py            # write and verify, work layout
    python scripts/build_port_docs.py --verify   # verify an existing set
    python scripts/build_port_docs.py --package src.rade_qnet --tests tests.rade_qnet
"""

from __future__ import annotations

import argparse
import hashlib
import posixpath
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

#: Repository root, derived from this file's location so the script works from
#: any working directory.
ROOT = Path(__file__).resolve().parent.parent

#: Where the documents land.
OUTPUT = ROOT / "port" / "rade_qnet"


@dataclass(frozen=True, slots=True)
class Tree:
    """
    One directory tree exported as a set of documents.

    Parameters
    ----------
    source
        The directory to walk.
    prefix
        Path components prepended to every document name, keeping the two
        trees' documents distinguishable in one flat output directory.
        Empty for the source tree, so its document names are unchanged by
        the test tree's arrival.
    label
        How the tree is described in the index.
    blurb
        A sentence for the index explaining what the tree is for and what
        the reader should do with it.
    """

    source: Path
    prefix: tuple[str, ...]
    label: str
    blurb: str
    suffixes: frozenset[str] = frozenset({".py"})


#: The package itself.
SOURCE_TREE = Tree(
    source=ROOT / "src" / "rade_qnet",
    prefix=(),
    label="Source",
    blurb=(
        "The package. Work down the list in order: a parent directory "
        "always appears before its children, so the tree is importable at "
        "every step."
    ),
)

#: The prose. Carried rather than fetched separately -- see the module
#: docstring for why being inside the manifest is the point.
DOCS_TREE = Tree(
    source=ROOT / "src" / "rade_qnet" / "docs",
    prefix=("docs",),
    label="Documentation",
    blurb=(
        "The prose, including `ARCHITECTURE.md`. These are already markdown "
        "and could be fetched directly, but they are carried here so they "
        "land in the manifest: a truncated paste then shows up as a digest "
        "mismatch rather than as a puzzling test failure. Four tests read "
        "these files and check the examples in them still parse, so the "
        "suite needs them present at these exact paths."
    ),
    suffixes=frozenset({".md"}),
)

#: The suite that proves the package arrived intact.
TESTS_TREE = Tree(
    source=ROOT / "tests" / "rade_qnet",
    prefix=("tests",),
    label="Tests",
    blurb=(
        "The suite. Rebuild it after the source and run it as shown "
        "above -- that run is what turns a pasted tree "
        "into a verified one. Each document's name mirrors the source "
        "document it exercises: `tests__core__spec.md` tests `core__spec.md`."
    ),
)

#: Every tree, in the order the far side should rebuild them.
TREES = (SOURCE_TREE, DOCS_TREE, TESTS_TREE)

#: The package and test-tree locations this repository uses, as paths. Every
#: target path and every rewrite is expressed relative to these two.
REPOSITORY_PACKAGE = "src/rade_qnet"
REPOSITORY_TESTS = "tests/rade_qnet"

#: The test tree's ruff configuration, and the line in it that points at the
#: package's. Matched exactly so that a change to either fails the export
#: loudly rather than leaving the far side with a dangling ``extend``.
_TESTS_RUFF = f"{REPOSITORY_TESTS}/ruff.toml"
_EXTEND_LINE = f'extend = "../../{REPOSITORY_PACKAGE}/ruff.toml"'

#: The package's ruff configuration, and its isort first-party declaration.
_PACKAGE_RUFF = f"{REPOSITORY_PACKAGE}/ruff.toml"
_FIRST_PARTY_LINE = 'known-first-party = ["rade_qnet", "src"]'


@dataclass(frozen=True, slots=True)
class Layout:
    """
    Where the package and its tests are imported from on the far side.

    Parameters
    ----------
    package
        The package's dotted import name, e.g. ``src.rade_qnet``.
    tests
        The test tree's dotted import name, e.g. ``tests.rade_qnet``.

    Both are taken as import names rather than paths because that is how the
    far side describes its own layout, and because every directory on the way
    down must then be a package -- a constraint a path would hide.
    """

    package: str
    tests: str

    def __post_init__(self) -> None:
        """Refuse a name that could not be imported, before anything is written."""
        for name in (self.package, self.tests):
            if not all(part.isidentifier() for part in name.split(".")):
                raise SystemExit(f"{name!r} is not a dotted import name")
        if self.package == self.tests:
            raise SystemExit("the package and its tests cannot share a location")

    @property
    def package_dir(self) -> str:
        """The package's directory, relative to the far side's root."""
        return self.package.replace(".", "/")

    @property
    def tests_dir(self) -> str:
        """The test tree's directory, relative to the far side's root."""
        return self.tests.replace(".", "/")

    @property
    def is_repository(self) -> bool:
        """Whether this is the layout the files already have, needing no rewrite."""
        return (self.package_dir, self.tests_dir) == (
            REPOSITORY_PACKAGE,
            REPOSITORY_TESTS,
        )

    def where(self, path: Path) -> str:
        """
        Return the path a repository file takes on the far side.

        Parameters
        ----------
        path
            Any path inside the repository.

        Returns
        -------
        str
            A POSIX-style path relative to the far side's root.
        """
        return self._moved(relative(path))

    def _moved(self, here: str) -> str:
        """
        Return where a repository-relative path lands in this layout.

        Parameters
        ----------
        here
            A POSIX-style path relative to the repository root.

        Returns
        -------
        str
            The same path relative to the far side's root.
        """
        for old, new in (
            (REPOSITORY_PACKAGE, self.package_dir),
            (REPOSITORY_TESTS, self.tests_dir),
        ):
            if here == old or here.startswith(f"{old}/"):
                return new + here[len(old) :]
        return here

    def _relink(self, here: str, text: str) -> str:
        """
        Recompute every relative markdown link in a document for this layout.

        A link is a path from the document to its target, and the two can
        move by different amounts -- the docs move with the package, the
        test tree's configuration with the tests -- so a link cannot be
        rewritten as text. It is resolved where it stands, both ends are
        moved, and the path between them is measured again.

        Parameters
        ----------
        here
            The document's repository-relative path.
        text
            Its contents.

        Returns
        -------
        str
            The contents with every relative link pointing where it did.
        """
        origin = posixpath.dirname(here)
        moved_origin = posixpath.dirname(self._moved(here))

        def relink(match: re.Match[str]) -> str:
            target = posixpath.normpath(posixpath.join(origin, match["path"]))
            moved = posixpath.relpath(self._moved(target), moved_origin)
            return f"]({moved}{match['anchor'] or ''})"

        return _RELATIVE_LINK.sub(relink, text)

    def rewrite(self, path: Path, data: bytes) -> bytes:
        """
        Return a file's contents as they must read in this layout.

        Structured settings are replaced exactly and must be found; free text
        -- imports, docstrings, prose -- is replaced wherever a location is
        spelled out. The two are kept apart because a setting that silently
        failed to match would ship a broken configuration, whereas prose that
        mentions no location simply needs no change.

        Parameters
        ----------
        path
            The file's location in the repository.
        data
            Its contents there.

        Returns
        -------
        bytes
            Its contents in this layout. Unchanged for the repository layout.

        Raises
        ------
        SystemExit
            If a setting that must be rewritten is no longer where expected.
        """
        if self.is_repository:
            return data
        text = data.decode()
        here = relative(path)
        if here == _TESTS_RUFF:
            # Exact, and before the prose pass, which would otherwise rewrite
            # the `src/rade_qnet` inside it into a path that does not exist.
            target = posixpath.relpath(self.package_dir, self.tests_dir)
            text = _replace_setting(
                here, text, _EXTEND_LINE, f'extend = "{target}/ruff.toml"'
            )
        if path.suffix == ".md":
            text = self._relink(here, text)
        if here == _PACKAGE_RUFF:
            host = self.package.split(".")[0]
            text = _replace_setting(
                here,
                text,
                _FIRST_PARTY_LINE,
                f'known-first-party = ["rade_qnet", "{host}"]',
            )
        replacements = (
            (_DOTTED_PACKAGE, self.package),
            (_DOTTED_TESTS, self.tests),
            (_SLASHED_PACKAGE, self.package_dir),
            (_SLASHED_TESTS, self.tests_dir),
        )
        for pattern, replacement in replacements:
            text = pattern.sub(replacement, text)
        if path.suffix == ".py" and (leak := _REPOSITORY_IMPORT.search(text)):
            raise SystemExit(
                f"{here} still imports through the repository's `src` after "
                f"rewriting ({leak.group().strip()!r}), which cannot work in "
                f"{self.package}. Spell it `src.rade_qnet...` so the rewrite "
                f"finds it, or resolve it through the test tree's locations."
            )
        return text.encode()


#: Free-text spellings of the two locations. The lookbehinds stop a match in
#: the middle of a longer name or path -- ``mysrc.rade_qnet``, or the
#: ``../../src/rade_qnet`` that :meth:`Layout.rewrite` handles exactly.
_DOTTED_PACKAGE = re.compile(r"(?<![\w.])src\.rade_qnet\b")
_DOTTED_TESTS = re.compile(r"(?<![\w.])tests\.rade_qnet\b")
_SLASHED_PACKAGE = re.compile(r"(?<![\w./])src/rade_qnet\b")
_SLASHED_TESTS = re.compile(r"(?<![\w./])tests/rade_qnet\b")

#: A markdown link to a relative path, split into the path and any anchor.
#: Only links starting ``./`` or ``../`` are relative in the sense that can
#: break; a bare name points beside its document, and both move together.
_RELATIVE_LINK = re.compile(r"\]\((?P<path>\.{1,2}/[^)#\s]*)(?P<anchor>#[^)\s]*)?\)")

#: An import statement naming the repository's ``src`` package, in any of
#: the forms the free-text patterns cannot reach -- ``from src import
#: rade_qnet`` above all. Checked after rewriting, as the last line of defence.
_REPOSITORY_IMPORT = re.compile(r"^\s*(?:from|import)\s+src\b.*$", re.MULTILINE)

#: The layout the files already have.
REPOSITORY_LAYOUT = Layout(package="src.rade_qnet", tests="tests.rade_qnet")

#: The layout the port is written for by default: the package vendored into
#: ``tranql``, with its suite as a sibling directory.
WORK_LAYOUT = Layout(
    package="tranql.models.rade.rade_qnet.rade_qnet",
    tests="tranql.models.rade.rade_qnet.tests",
)

#: The suite's result on the far side, where the binary parity fixtures are
#: absent and every test needing them skips. Measured, not derived: re-run the
#: suite without fixtures and update this whenever tests are added.
EXPECTED_RESULT = "3070 passed, 101 skipped"


def _replace_setting(where: str, text: str, old: str, new: str) -> str:
    """
    Replace one configuration line that must be present exactly once.

    Parameters
    ----------
    where
        The file's repository path, for the error message.
    text
        Its contents.
    old
        The line as it reads in the repository.
    new
        The line as it must read in the target layout.

    Returns
    -------
    str
        The contents with the line replaced.

    Raises
    ------
    SystemExit
        If the line is absent or ambiguous.
    """
    if text.count(old) != 1:
        raise SystemExit(
            f"{where} no longer contains {old!r} exactly once; update the "
            f"rewrite in {Path(__file__).name} to match before exporting."
        )
    return text.replace(old, new)


#: Directories never exported: caches, and the documentation tree itself.
#: ``docs/`` is already markdown, so it crosses the proxy unchanged and
#: duplicating its 7,498 lines here would only create a second copy to drift.
SKIP_DIRECTORIES = frozenset({"__pycache__", ".ruff_cache", "docs"})

#: Fence language for each exported suffix. ``markdown`` inside a markdown
#: fence is legal and renders as plain text, which is what a reader copying
#: a document back out actually wants.
LANGUAGES = {".py": "python", ".md": "markdown"}

#: Non-Python files that are exported anyway, mapped to their fence language.
EXTRA_FILES = {"ruff.toml": "toml"}

#: Files from the repository root that travel too, mapped to their fence
#: language. ``pyproject.toml`` is what makes the far side's tree installable
#: with ``pip install -e .``; without it the package only imports under the
#: ``src.`` prefix, from the repository root.
ROOT_FILES = {"pyproject.toml": "toml"}

#: The document holding :data:`ROOT_FILES`, listed first in the index because
#: it describes the tree every other document fills in.
ROOT_DOCUMENT = "_repository.md"

#: The shortest fence used for any code block. Three backticks give GitHub a
#: copy button and syntax highlighting; :func:`fence_for` lengthens it for the
#: rare file that spells out a fence of its own.
FENCE = "```"

#: A line that is nothing but a run of three or more backticks, optionally
#: followed by a language tag. Used by :func:`parse_document` to find block
#: boundaries without knowing in advance how long a given file's fence is.
_FENCE_LINE = re.compile(r"^(`{3,})([^`]*)$")


def fence_for(data: bytes) -> str:
    """
    Return the shortest fence that cannot appear inside some bytes.

    One backtick longer than the file's longest run, floored at three. That
    makes the closing delimiter unambiguous by construction rather than by
    inspection: no line of the file can match it, so the block cannot close
    early no matter what the file contains.

    Parameters
    ----------
    data
        The file's bytes.

    Returns
    -------
    str
        A run of backticks.
    """
    runs = (len(run) for run in re.findall(r"`+", data.decode()))
    return "`" * max(len(FENCE), max(runs, default=0) + 1)


def relative(path: Path) -> str:
    """
    Return a path as it should appear on the far side.

    Paths are written relative to the repository root rather than to
    ``SOURCE``, so a reader can copy the heading straight into a shell without
    working out what it is relative to.

    Parameters
    ----------
    path
        Any path inside the repository.

    Returns
    -------
    str
        A POSIX-style path relative to the repository root.
    """
    return path.relative_to(ROOT).as_posix()


def digest(data: bytes) -> str:
    """
    Return the SHA-256 of some bytes, abbreviated to 16 hex characters.

    Abbreviated because the digest is here to catch a mangled paste, not to
    resist an adversary, and sixteen characters is already far beyond the
    collision probability of a few hundred files. A full digest would wrap in
    the index table and make it unreadable.

    Parameters
    ----------
    data
        The file's bytes.

    Returns
    -------
    str
        The first 16 characters of the hex digest.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def exported_directories(tree: Tree) -> list[Path]:
    """
    Return every directory in a tree that directly contains an exported file.

    Sorted so that a parent sorts before its children, which is the order
    someone rebuilding the tree wants to work in.

    Parameters
    ----------
    tree
        The tree to walk.

    Returns
    -------
    list of Path
        Directories, each holding at least one file to export.
    """
    found: set[Path] = set()
    for path in tree.source.rglob("*"):
        if not path.is_file():
            continue
        if any(
            part in SKIP_DIRECTORIES for part in path.relative_to(tree.source).parts
        ):
            continue
        if path.suffix in tree.suffixes or path.name in EXTRA_FILES:
            found.add(path.parent)
    return sorted(found)


def collect_root_files() -> list[tuple[Path, bytes, str]]:
    """
    Return the repository-root files to export, in :data:`ROOT_FILES` order.

    Separate from :func:`collect` because the repository root holds plenty
    that must not travel -- other packages' scripts among them -- so only the
    named files are taken, never everything with a matching suffix.

    Returns
    -------
    list of tuple
        ``(path, contents, fence_language)`` for each file that exists.
    """
    files: list[tuple[Path, bytes, str]] = []
    for name, language in ROOT_FILES.items():
        path = ROOT / name
        if path.is_file():
            files.append((path, _checked_bytes(path), language))
    return files


def _checked_bytes(path: Path) -> bytes:
    """
    Read a file, refusing one that a fenced block cannot carry exactly.

    Parameters
    ----------
    path
        The file.

    Returns
    -------
    bytes
        Its contents.

    Raises
    ------
    SystemExit
        If the file lacks a final newline, which a fenced block always adds.
        A file containing a fence of its own is *not* refused: it is given a
        longer fence by :func:`fence_for`.
    """
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        # A fenced block always restores a final newline, so a file
        # without one cannot round-trip. Refusing is better than
        # exporting something that differs by a byte nobody will see.
        raise SystemExit(
            f"{relative(path)} does not end with a newline and so cannot "
            f"be represented exactly in a fenced block. Add one."
        )
    return data


def collect(directory: Path) -> list[tuple[Path, bytes, str]]:
    """
    Return the files to export from one directory.

    ``__init__.py`` is forced to the front because it is the file that makes
    the directory a package, and a reader who creates it first gets an
    importable tree at every intermediate step.

    Parameters
    ----------
    directory
        A directory from :func:`exported_directories`.

    Returns
    -------
    list of tuple
        ``(path, contents, fence_language)``, ``__init__.py`` first and the
        remainder alphabetical.

    Raises
    ------
    SystemExit
        If any file lacks a final newline.
    """
    files: list[tuple[Path, bytes, str]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        language = LANGUAGES.get(path.suffix) or EXTRA_FILES.get(path.name)
        if language is None:
            continue
        files.append((path, _checked_bytes(path), language))
    return sorted(files, key=lambda item: (item[0].name != "__init__.py", item[0].name))


def apply_layout(
    documents: list[tuple[Tree, Path, list[tuple[Path, bytes, str]]]], layout: Layout
) -> list[tuple[Tree, Path, list[tuple[Path, bytes, str]]]]:
    """
    Rewrite every collected file for a layout, then let ruff tidy the result.

    Parameters
    ----------
    documents
        Every tree, directory and its files, as collected.
    layout
        The far side's layout.

    Returns
    -------
    list of tuple
        The same documents, each file's contents as it must read there.
    """
    if layout.is_repository:
        return documents
    rewritten = [
        (
            tree,
            directory,
            [(path, layout.rewrite(path, data), lang) for path, data, lang in files],
        )
        for tree, directory, files in documents
    ]
    tidied = tidy(
        {
            layout.where(path): data
            for _, _, files in rewritten
            for path, data, _ in files
        }
    )
    return [
        (
            tree,
            directory,
            [(path, tidied[layout.where(path)], lang) for path, _, lang in files],
        )
        for tree, directory, files in rewritten
    ]


def tidy(files: dict[str, bytes]) -> dict[str, bytes]:
    """
    Sort imports and format a rewritten tree exactly as ruff would there.

    The tree is materialised in a scratch directory at its target paths, so
    ruff discovers the rewritten configuration files the way it will on the
    far side rather than this repository's. Only import sorting is fixed --
    the one rule a longer package name can break -- and then the whole tree
    is checked, so anything else wrong stops the export here instead of
    surfacing as a lint failure after a three-hour paste.

    Parameters
    ----------
    files
        Target path to rewritten contents, for every exported file.

    Returns
    -------
    dict
        The same mapping, Python files sorted and formatted.

    Raises
    ------
    SystemExit
        If ruff is unavailable, or reports anything it cannot fix.
    """
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        for name, data in files.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_bytes(data)
        ruff = [sys.executable, "-m", "ruff"]
        steps = (
            [*ruff, "check", "--select", "I", "--fix", "--exit-zero", "--quiet", "."],
            [*ruff, "format", "--quiet", "."],
            [*ruff, "check", "--quiet", "--output-format=concise", "."],
            [*ruff, "format", "--check", "--quiet", "."],
        )
        for step in steps:
            try:
                result = subprocess.run(
                    step, cwd=root, capture_output=True, text=True, check=False
                )
            except OSError as error:
                raise SystemExit(f"ruff could not be run: {error}") from error
            if result.returncode:
                raise SystemExit(
                    f"ruff rejected the rewritten tree:\n{result.stdout}{result.stderr}"
                )
        return {name: (root / name).read_bytes() for name in files}


def document_name(directory: Path, tree: Tree) -> str:
    """
    Return the markdown filename for a directory.

    ``core/spec`` becomes ``core__spec.md``: flat, so every document sits in
    one place and the index is a single list, but with the hierarchy still
    legible in the name. The tree's prefix joins on the front by the same
    rule, so the matching test document is ``tests__core__spec.md`` -- which
    means the two sort together in a directory listing.

    Parameters
    ----------
    directory
        A directory from :func:`exported_directories`.
    tree
        The tree it came from.

    Returns
    -------
    str
        The document filename.
    """
    if directory == ROOT:
        return ROOT_DOCUMENT
    parts = (*tree.prefix, *directory.relative_to(tree.source).parts)
    return ("__".join(parts) if parts else "_root") + ".md"


def render_document(
    directory: Path, files: list[tuple[Path, bytes, str]], layout: Layout
) -> str:
    """
    Render one directory as a markdown document.

    Each file gets a heading carrying its full path, a line of metadata, and a
    fenced block holding it verbatim. The metadata line is what makes a bad
    paste detectable without running anything: line count, byte count and
    digest are all visible on both sides.

    Parameters
    ----------
    directory
        The directory being documented.
    files
        Its files, from :func:`collect`.
    layout
        The far side's layout, which decides the paths the headings name.

    Returns
    -------
    str
        The document.
    """
    where = layout.where(directory)
    lines = [
        f"# `{where}`",
        "",
        (
            f"{len(files)} file(s). Create the directory, then create "
            f"each file below with the exact contents of its block."
        ),
        "",
        "| # | File | Lines | Bytes | SHA-256 |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for index, (path, data, _) in enumerate(files, start=1):
        count = data.decode().count("\n")
        lines.append(
            f"| {index} | `{path.name}` | {count} | {len(data)} | `{digest(data)}` |"
        )
    lines.append("")

    for index, (path, data, language) in enumerate(files, start=1):
        text = data.decode()
        lines += [
            "---",
            "",
            f"## {index}. `{layout.where(path)}`",
            "",
        ]
        if not data:
            # An empty __init__.py is still load-bearing: without it the
            # directory is not a package. Saying so explicitly stops a reader
            # skipping a block that looks like a rendering fault.
            lines += [
                (
                    "**This file is empty.** Create it with no contents "
                    "at all — it exists to make the directory a package."
                ),
                "",
            ]
            continue

        # The closing fence sits on its own line, which *is* the file's final
        # newline. So the body is every line of the file with that implied
        # newline's empty trailing element removed -- not `rstrip`, which
        # would also swallow a deliberate blank line at the end of a file and
        # make the export silently lossy. Two files in this tree end that
        # way, which is how the round-trip check found this.
        body = text.split("\n")
        if body[-1] == "":
            body.pop()
        fence = fence_for(data)
        lines += [
            f"{len(data)} bytes · SHA-256 `{digest(data)}`",
            "",
            f"{fence}{language}",
            *body,
            fence,
            "",
        ]

    return "\n".join(lines) + "\n"


def render_index(
    documents: list[tuple[Tree, Path, list[tuple[Path, bytes, str]]]], layout: Layout
) -> str:
    """
    Render the index: what to fetch, in what order, and how to check it.

    Parameters
    ----------
    documents
        Every tree, directory and its files, in creation order.
    layout
        The far side's layout, which decides every path and command shown.

    Returns
    -------
    str
        The index document.
    """
    total_files = sum(len(files) for _, _, files in documents)
    total_bytes = sum(len(data) for _, _, files in documents for _, data, _ in files)
    total_lines = sum(
        data.decode().count("\n") for _, _, files in documents for _, data, _ in files
    )

    lines = [
        "# Porting `rade_qnet` through a markdown-only proxy",
        "",
        (
            f"{len(documents)} documents, {total_files} files, "
            f"{total_lines:,} lines, {total_bytes:,} bytes."
        ),
        "",
        (
            "Each document below covers one directory: create the "
            "directory, then create each file in it from the block that "
            "carries it. Rebuild the source tree first, then the tests."
        ),
        "",
        *_layout_section(layout),
        "## What is not here, and what to expect because of it",
        "",
        *_installation_paragraphs(layout),
        (
            "**The golden parity fixtures do not travel.** "
            "`tests/fixtures/rade_qnet/` holds `.npy` and `.npz` arrays — "
            "binary, and so impossible to carry as text. They guard "
            "numerical parity against a captured reference, so if that "
            "matters on the far side the arrays have to cross by some "
            "other route. The suite looks for them under "
            "`tests/fixtures/rade_qnet/golden` at the root the package is "
            "imported from; to keep them anywhere else, point the "
            "`RADE_QNET_GOLDEN_ROOT` environment variable at the directory "
            "holding `hybrid_gnn_rnn/`."
        ),
        "",
        (
            "Every test that needs them skips cleanly, so **a correct "
            "paste is all-green** and any red at all means something did "
            "not land. From the repository root, run the suite and compare:"
        ),
        "",
        f"{FENCE}",
        f"pytest {layout.tests_dir}",
        f"  -> {EXPECTED_RESULT}",
        FENCE,
        "",
        (
            "The 66 extra skips relative to a full checkout are the parity "
            "tests: the ones that compare this implementation's numbers "
            "against the original's. Everything else runs, so the suite "
            "still proves the framework behaves — it just stops proving it "
            "reproduces the baseline's figures."
        ),
        "",
        (
            "Nine of the fixtures are `.json` and could in principle travel "
            "as text. Copying only those gains nothing: the arrays beside "
            "them are what the tests read, so the same tests skip either "
            "way. It is all of them or none, and none is a perfectly good "
            "answer."
        ),
        "",
        "## A note on fence lengths",
        "",
        (
            "Almost every block below is fenced with three backticks. A "
            "file that spells out a fence of its own gets four, so that it "
            "cannot close its own block early. Copy whatever sits *between* "
            "the fence lines and the length never matters."
        ),
        "",
    ]

    position = 0
    for tree in TREES:
        group = [(d, files) for t, d, files in documents if t is tree]
        if not group:
            continue
        count = sum(len(files) for _, files in group)
        size = sum(len(data) for _, files in group for _, data, _ in files)
        lines += [
            f"## {tree.label}: {len(group)} documents, {count} files, {size:,} bytes",
            "",
            tree.blurb,
            "",
            "| # | Document | Directory | Files | Bytes |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        for directory, files in group:
            position += 1
            name = document_name(directory, tree)
            lines.append(
                f"| {position} | [`{name}`]({name}) | `{layout.where(directory)}` | "
                f"{len(files)} | {sum(len(d) for _, d, _ in files):,} |"
            )
        lines.append("")

    lines += [
        "## Verifying the result",
        "",
        (
            "Every file above carries the first 16 hex characters of its "
            "SHA-256. Save the manifest at the end of this page as "
            "`MANIFEST.txt` in the repository root, then run the script "
            "below from the same place. It names every file that is "
            "missing or whose contents differ, which is the one check "
            "that turns a silent bad paste into a reported one."
        ),
        "",
        f"{FENCE}python",
        _VERIFIER.strip(),
        FENCE,
        "",
        "## Manifest",
        "",
        f"{FENCE}",
        "\n".join(
            f"{digest(data)}  {layout.where(path)}"
            for _, _, files in documents
            for path, data, _ in files
        ),
        FENCE,
        "",
    ]
    return "\n".join(lines) + "\n"


def _layout_section(layout: Layout) -> list[str]:
    """
    Return the index paragraphs describing where the tree lands.

    Parameters
    ----------
    layout
        The far side's layout.

    Returns
    -------
    list of str
        Markdown lines; empty for the repository layout, which needs no
        explanation.
    """
    if layout.is_repository:
        return []
    parents = layout.package_dir.split("/")[:-1]
    packages = [
        "/".join(parents[: depth + 1]) + "/__init__.py" for depth in range(len(parents))
    ]
    return [
        "## Where it lands",
        "",
        (
            f"This export is written for a vendored layout: the package "
            f"imports as `{layout.package}` and its suite as "
            f"`{layout.tests}`. Every heading below already names its file's "
            f"path in that layout, and every import already uses that name, "
            f"so nothing needs editing after the paste."
        ),
        "",
        (
            "The directories above the package must already be packages, "
            "which in an existing repository they normally are. If any of "
            "these is missing, create it empty:"
        ),
        "",
        *(f"- `{name}`" for name in packages),
        "",
        (
            "Run every command from the repository root -- the directory "
            f"holding `{parents[0]}/` -- since the suite imports the package "
            "by its full name."
        ),
        "",
    ]


def _installation_paragraphs(layout: Layout) -> list[str]:
    """
    Return the index paragraphs on installation and dependencies.

    Parameters
    ----------
    layout
        The far side's layout.

    Returns
    -------
    list of str
        Markdown lines.
    """
    if layout.is_repository:
        return [
            (
                "Everything needed to install and run is carried, including "
                "`pyproject.toml` (the first document, so the rebuilt tree "
                "installs with `pip install -e .`) and the documentation."
            ),
            "",
        ]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    lines = [
        (
            "**`pyproject.toml` does not travel**, because the host "
            "repository has its own and pasting this one over it would "
            "break it. What it would have contributed is the interpreter "
            "and dependency floor, so make sure the host environment has:"
        ),
        "",
        f"- Python `{project['requires-python']}`",
        *(f"- `{requirement}`" for requirement in project["dependencies"]),
        "",
    ]
    optional = project.get("optional-dependencies", {})
    if optional:
        lines += [
            (
                "And, for the engines and sources that need them -- tests "
                "of anything absent skip rather than fail:"
            ),
            "",
            *(
                f"- `{requirement}` ({group})"
                for group, requirements in optional.items()
                for requirement in requirements
                # An extra naming the project itself only aggregates the
                # others, which are already listed.
                if not requirement.startswith(project["name"])
            ),
            "",
        ]
    return lines


#: The verification script, embedded in the index so the far side can run it
#: without needing a second file to cross the proxy.
_VERIFIER = '''
"""Check a ported tree against MANIFEST.txt. Run from the repository root."""

import hashlib
import pathlib
import sys

missing: list[str] = []
differs: list[str] = []
checked = 0

for line in pathlib.Path("MANIFEST.txt").read_text().splitlines():
    if not line.strip():
        continue
    expected, _, name = line.partition("  ")
    path = pathlib.Path(name)
    if not path.exists():
        missing.append(name)
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    if actual != expected:
        differs.append(name)
    checked += 1

for name in missing:
    print(f"MISSING  {name}")
for name in differs:
    print(f"DIFFERS  {name}")
print(f"\\nchecked {checked}, missing {len(missing)}, differs {len(differs)}")
sys.exit(1 if missing or differs else 0)
'''


def parse_document(text: str) -> dict[str, bytes]:
    """
    Recover the files a document encodes.

    The inverse of :func:`render_document`, used only by :func:`verify`. It
    reads the headings for paths and the fenced blocks for contents, which is
    exactly what a human copying out of the rendered page does — so if this
    round-trips, so will they.

    Parameters
    ----------
    text
        A rendered document.

    Returns
    -------
    dict
        Repository-relative path to file contents.
    """
    recovered: dict[str, bytes] = {}
    current: str | None = None
    buffer: list[str] | None = None
    opened: str = FENCE

    for line in text.splitlines():
        delimiter = _FENCE_LINE.match(line)
        # Inside a block every line is content, including one that looks
        # like a heading. Carrying the documentation made that distinction
        # load-bearing: ARCHITECTURE.md has `## ` headings of its own, and
        # reading them as file paths silently truncated it.
        if buffer is None and line.startswith("## ") and "`" in line:
            current = line.split("`")[1]
            # An empty file never opens a block, so record it now; a later
            # fence for the same heading simply overwrites this.
            recovered[current] = b""
        elif delimiter and buffer is None and current is not None:
            # Remember the exact opening run. Each file is fenced with one
            # backtick more than its longest internal run, so only a line
            # this long or longer can be the close -- and a shorter run
            # inside the file is content, to be copied through untouched.
            opened = delimiter.group(1)
            buffer = []
        elif (
            delimiter and buffer is not None and len(delimiter.group(1)) >= len(opened)
        ):
            recovered[current] = ("\n".join(buffer) + "\n").encode()
            buffer = None
        elif buffer is not None:
            buffer.append(line)

    return recovered


def verify(
    documents: list[tuple[Tree, Path, list[tuple[Path, bytes, str]]]], layout: Layout
) -> int:
    """
    Re-read every written document and compare it against what was exported.

    Parameters
    ----------
    documents
        Every tree, directory and its files, already rewritten for ``layout``.
    layout
        The layout the documents were written for.

    Returns
    -------
    int
        The number of files that failed to round-trip.
    """
    failures = 0
    for tree, directory, files in documents:
        path = OUTPUT / document_name(directory, tree)
        recovered = parse_document(path.read_text())
        for source_path, data, _ in files:
            key = layout.where(source_path)
            if recovered.get(key) != data:
                print(f"ROUND-TRIP FAILED  {key}")
                failures += 1
    return failures


def main() -> int:
    """
    Write every document, then prove each one round-trips.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check the existing documents without rewriting them",
    )
    parser.add_argument(
        "--package",
        default=WORK_LAYOUT.package,
        help=f"the package's dotted import name on the far side (default: {WORK_LAYOUT.package})",
    )
    parser.add_argument(
        "--tests",
        default=WORK_LAYOUT.tests,
        help=f"the suite's dotted import name on the far side (default: {WORK_LAYOUT.tests})",
    )
    arguments = parser.parse_args()
    layout = Layout(package=arguments.package, tests=arguments.tests)

    # The repository's own pyproject.toml only makes sense in the
    # repository's own layout; anywhere else it would replace the host's.
    documents = (
        [(SOURCE_TREE, ROOT, collect_root_files())] if layout.is_repository else []
    )
    for tree in TREES:
        documents += [(tree, d, collect(d)) for d in exported_directories(tree)]
    documents = apply_layout(
        [(tree, d, files) for tree, d, files in documents if files], layout
    )

    if not arguments.verify:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        # A directory deleted from the source must disappear from the port
        # too; otherwise the far side faithfully rebuilds a package that no
        # longer exists, and nothing reports it.
        expected = {document_name(d, tree) for tree, d, _ in documents} | {"INDEX.md"}
        for stale in sorted(OUTPUT.glob("*.md")):
            if stale.name not in expected:
                print(f"removing stale document {stale.name}")
                stale.unlink()
        for tree, directory, files in documents:
            (OUTPUT / document_name(directory, tree)).write_text(
                render_document(directory, files, layout)
            )
        (OUTPUT / "INDEX.md").write_text(render_index(documents, layout))

    failures = verify(documents, layout)
    total = sum(len(files) for _, _, files in documents)
    if failures:
        print(f"\n{failures} of {total} files did not round-trip")
        return 1

    print(
        f"{len(documents)} documents, {total} files for {layout.package}, all round-trip exactly"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
