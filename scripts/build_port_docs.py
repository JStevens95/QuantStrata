"""
Emit the ``rade_qnet`` source tree as markdown, for transport through a proxy
that passes nothing but ``.md``.

The problem this solves
-----------------------
A locked-down machine can fetch documentation and nothing else. ``git clone``,
a release zip and PyPI are all unavailable, so the only way the code can cross
is as prose. That constraint is what this script answers: one markdown
document per source directory, each file inside it fenced verbatim, plus an
index carrying a SHA-256 for every file so the far side can prove the paste
landed intact rather than discovering it three days later in a stack trace.

Why one document per directory
------------------------------
Per *top-level package* would put 304 KB of ``core`` into a single page, which
GitHub truncates in the rendered view and which is miserable to scroll. Per
*file* would mean 164 documents to navigate. Per directory lands at 37
documents, the largest around 140 KB, and it matches how someone rebuilding
the tree actually works: make a directory, fill it, move on.

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
import sys
from pathlib import Path

#: Repository root, derived from this file's location so the script works from
#: any working directory.
ROOT = Path(__file__).resolve().parent.parent

#: The tree being exported.
SOURCE = ROOT / "src" / "rade_qnet"

#: Where the documents land.
OUTPUT = ROOT / "port" / "rade_qnet"

#: Directories never exported: caches, and the documentation tree itself.
#: ``docs/`` is already markdown, so it crosses the proxy unchanged and
#: duplicating its 7,498 lines here would only create a second copy to drift.
SKIP_DIRECTORIES = frozenset({"__pycache__", ".ruff_cache", "docs"})

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

#: The fence used for every code block. Three backticks give GitHub a copy
#: button and syntax highlighting. Safe only while no exported file contains a
#: fence of its own, which :func:`collect` asserts rather than assumes.
FENCE = "```"


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


def exported_directories() -> list[Path]:
    """
    Return every directory that directly contains an exported file.

    Sorted so that a parent sorts before its children, which is the order
    someone rebuilding the tree wants to work in.

    Returns
    -------
    list of Path
        Directories, each holding at least one file to export.
    """
    found: set[Path] = set()
    for path in SOURCE.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRECTORIES for part in path.relative_to(SOURCE).parts):
            continue
        if path.suffix == ".py" or path.name in EXTRA_FILES:
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
        If the file contains a code fence, which would close its own block
        early, or lacks a final newline, which a fenced block always adds.
    """
    data = path.read_bytes()
    if FENCE.encode() in data:
        raise SystemExit(
            f"{relative(path)} contains a {FENCE!r} fence, which would "
            f"close its own code block. Switch FENCE to a longer run of "
            f"backticks before exporting."
        )
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
        If any file contains a code fence, which would terminate its own
        block early and silently truncate the exported source.
    """
    files: list[tuple[Path, bytes, str]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        language = "python" if path.suffix == ".py" else EXTRA_FILES.get(path.name)
        if language is None:
            continue
        files.append((path, _checked_bytes(path), language))
    return sorted(files, key=lambda item: (item[0].name != "__init__.py", item[0].name))


def document_name(directory: Path) -> str:
    """
    Return the markdown filename for a directory.

    ``core/spec`` becomes ``core__spec.md``: flat, so every document sits in
    one place and the index is a single list, but with the hierarchy still
    legible in the name.

    Parameters
    ----------
    directory
        A directory from :func:`exported_directories`.

    Returns
    -------
    str
        The document filename.
    """
    if directory == ROOT:
        return ROOT_DOCUMENT
    parts = directory.relative_to(SOURCE).parts
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
        lines += [
            f"{len(data)} bytes · SHA-256 `{digest(data)}`",
            "",
            f"{FENCE}{language}",
            *body,
            FENCE,
            "",
        ]

    return "\n".join(lines) + "\n"


def render_index(documents: list[tuple[Path, list[tuple[Path, bytes, str]]]]) -> str:
    """
    Render the index: what to fetch, in what order, and how to check it.

    Parameters
    ----------
    documents
        Every directory and its files, in creation order.

    Returns
    -------
    str
        The index document.
    """
    total_files = sum(len(files) for _, files in documents)
    total_bytes = sum(len(data) for _, files in documents for _, data, _ in files)
    total_lines = sum(
        data.decode().count("\n") for _, files in documents for _, data, _ in files
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
            "Each document below covers one directory. Work down the "
            "list in order: a parent directory always appears before its "
            "children, so the tree is importable at every step."
        ),
        "",
        "## What is not here",
        "",
        (
            "`src/rade_qnet/docs/` is already markdown, so it crosses the "
            "proxy unchanged — fetch those files directly rather than "
            "through this set. `pyproject.toml` is included, as the first "
            "document, so the rebuilt tree installs with "
            "`pip install -e .`. The test suite, fixtures and examples are "
            "excluded by scope; the golden parity fixtures under "
            "`tests/fixtures/rade_qnet/` are binary `.npy` files and "
            "cannot travel as text at all."
        ),
        "",
        "## Directories, in creation order",
        "",
        "| # | Document | Directory | Files | Bytes |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for index, (directory, files) in enumerate(documents, start=1):
        name = document_name(directory)
        size = sum(len(data) for _, data, _ in files)
        lines.append(
            f"| {index} | [`{name}`]({name}) | `{relative(directory)}` | "
            f"{len(files)} | {size:,} |"
        )

    lines += [
        "",
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
            for _, files in documents
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

    for line in text.splitlines():
        if line.startswith("## ") and "`" in line:
            current = line.split("`")[1]
            # An empty file never opens a block, so record it now; a later
            # fence for the same heading simply overwrites this.
            recovered[current] = b""
        elif line.startswith(FENCE) and buffer is None and current is not None:
            buffer = []
        elif line.startswith(FENCE) and buffer is not None:
            recovered[current] = ("\n".join(buffer) + "\n").encode()
            buffer = None
        elif buffer is not None:
            buffer.append(line)

    return recovered


def verify(documents: list[tuple[Path, list[tuple[Path, bytes, str]]]]) -> int:
    """
    Re-read every written document and compare it against the source.

    Parameters
    ----------
    documents
        Every directory and its files.

    Returns
    -------
    int
        The number of files that failed to round-trip.
    """
    failures = 0
    for directory, files in documents:
        path = OUTPUT / document_name(directory)
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

    documents = [(ROOT, collect_root_files())]
    documents += [(d, collect(d)) for d in exported_directories()]
    documents = [(d, files) for d, files in documents if files]

    if not arguments.verify:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        # A directory deleted from the source must disappear from the port
        # too; otherwise the far side faithfully rebuilds a package that no
        # longer exists, and nothing reports it.
        expected = {document_name(directory) for directory, _ in documents} | {"INDEX.md"}
        for stale in sorted(OUTPUT.glob("*.md")):
            if stale.name not in expected:
                print(f"removing stale document {stale.name}")
                stale.unlink()
        for directory, files in documents:
            (OUTPUT / document_name(directory)).write_text(
                render_document(directory, files)
            )
        (OUTPUT / "INDEX.md").write_text(render_index(documents))

    failures = verify(documents)
    total = sum(len(files) for _, files in documents)
    if failures:
        print(f"\n{failures} of {total} files did not round-trip")
        return 1

    print(f"{len(documents)} documents, {total} files, all round-trip exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
