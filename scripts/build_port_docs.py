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

Usage
-----
::

    python scripts/build_port_docs.py            # write and verify
    python scripts/build_port_docs.py --verify   # verify an existing set
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
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
        "The suite. Rebuild it after the source and run "
        "`pytest tests/rade_qnet` -- that run is what turns a pasted tree "
        "into a verified one. Each document's name mirrors the source "
        "document it exercises: `tests__core__spec.md` tests `core__spec.md`."
    ),
)

#: Every tree, in the order the far side should rebuild them.
TREES = (SOURCE_TREE, DOCS_TREE, TESTS_TREE)

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
        if any(part in SKIP_DIRECTORIES for part in path.relative_to(tree.source).parts):
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


def render_document(directory: Path, files: list[tuple[Path, bytes, str]]) -> str:
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

    Returns
    -------
    str
        The document.
    """
    where = relative(directory)
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
            f"| {index} | `{path.name}` | {count} | {len(data)} | "
            f"`{digest(data)}` |"
        )
    lines.append("")

    for index, (path, data, language) in enumerate(files, start=1):
        text = data.decode()
        lines += [
            "---",
            "",
            f"## {index}. `{relative(path)}`",
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


def render_index(documents: list[tuple[Tree, Path, list[tuple[Path, bytes, str]]]]) -> str:
    """
    Render the index: what to fetch, in what order, and how to check it.

    Parameters
    ----------
    documents
        Every tree, directory and its files, in creation order.

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
        "## What is not here, and what to expect because of it",
        "",
        (
            "Everything needed to install and run is carried, including "
            "`pyproject.toml` (the first document, so the rebuilt tree "
            "installs with `pip install -e .`) and the documentation."
        ),
        "",
        (
            "**The golden parity fixtures do not travel.** "
            "`tests/fixtures/rade_qnet/` holds `.npy` and `.npz` arrays — "
            "binary, and so impossible to carry as text. They guard "
            "numerical parity against a captured reference, so if that "
            "matters on the far side the arrays have to cross by some "
            "other route."
        ),
        "",
        (
            "Most tests that need them skip cleanly. **Twenty-six do not** "
            "— they fail or error on the missing file instead. That is a "
            "gap in those tests rather than in this port, but it means a "
            "correct paste is *not* all-green. Run the suite and compare "
            "against the expected result below; anything else means "
            "something did not land."
        ),
        "",
        f"{FENCE}",
        "pytest tests/rade_qnet",
        "  -> 7 failed, 3072 passed, 70 skipped, 19 errors",
        FENCE,
        "",
        (
            "**Do not copy across a subset of the fixtures.** The nine "
            "`.json` files among them are text and look portable, but "
            "supplying those without the arrays is worse than supplying "
            "none: the loader then finds the directory, the tests stop "
            "skipping, and the failure count rises to 22. It is all of "
            "them or none."
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
                f"| {position} | [`{name}`]({name}) | `{relative(directory)}` | "
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
            f"{digest(data)}  {relative(path)}"
            for _, _, files in documents
            for path, data, _ in files
        ),
        FENCE,
        "",
    ]
    return "\n".join(lines) + "\n"


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
        elif delimiter and buffer is not None and len(delimiter.group(1)) >= len(opened):
            recovered[current] = ("\n".join(buffer) + "\n").encode()
            buffer = None
        elif buffer is not None:
            buffer.append(line)

    return recovered


def verify(documents: list[tuple[Tree, Path, list[tuple[Path, bytes, str]]]]) -> int:
    """
    Re-read every written document and compare it against the source.

    Parameters
    ----------
    documents
        Every tree, directory and its files.

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
            key = relative(source_path)
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
    arguments = parser.parse_args()

    documents = [(SOURCE_TREE, ROOT, collect_root_files())]
    for tree in TREES:
        documents += [(tree, d, collect(d)) for d in exported_directories(tree)]
    documents = [(tree, d, files) for tree, d, files in documents if files]

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
                render_document(directory, files)
            )
        (OUTPUT / "INDEX.md").write_text(render_index(documents))

    failures = verify(documents)
    total = sum(len(files) for _, _, files in documents)
    if failures:
        print(f"\n{failures} of {total} files did not round-trip")
        return 1

    print(f"{len(documents)} documents, {total} files, all round-trip exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
