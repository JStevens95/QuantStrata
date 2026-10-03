"""
Building and verifying a bundle's file manifest.

The manifest is what turns a directory of files into a bundle that can be
trusted. Every file is listed with its SHA-256 digest and its size, so loading
a bundle can establish that it is the same bundle that was written -- rather
than one that was truncated by a full disk, partially synchronised by a file
share, or edited by someone's debugging session six weeks ago.

Why verification is a separate, explicit step
---------------------------------------------
:func:`verify_manifest` is not called automatically on every read. Hashing a
multi-gigabyte checkpoint takes real time, and paying it to list bundles in a
UI would make the UI unusable. The caller chooses: cheap metadata reads skip
it, and anything that is about to produce predictions does not.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..core.contract.bundle import BUNDLE_SCHEMA_VERSION, Manifest, ManifestEntry
from ..core.runtime.errors import BundleError
from ..core.runtime.hashing import digest_file
from ..core.runtime.logging import get_logger

__all__ = ["build_manifest", "collect_entries", "verify_manifest"]

_LOGGER = get_logger(__name__)

#: Files excluded from a manifest. The manifest cannot list itself -- its own
#: digest would have to be known before it was written -- and editor or
#: filesystem debris must not invalidate an otherwise sound bundle.
EXCLUDED_NAMES = frozenset({"manifest.json", ".DS_Store", "Thumbs.db"})

#: Directories skipped entirely while walking a bundle.
EXCLUDED_DIRECTORIES = frozenset({"__pycache__", ".ipynb_checkpoints"})


def collect_entries(directory: Path) -> tuple[ManifestEntry, ...]:
    """
    Hash every file beneath a directory.

    Entries are sorted by path, which matters for more than tidiness: a
    manifest's own digest is computed over its JSON, so a stable file order is
    what lets two bundles with identical contents produce identical
    manifests.

    Parameters
    ----------
    directory
        The bundle directory.

    Returns
    -------
    tuple of ManifestEntry
        One entry per file, sorted by relative path.

    Raises
    ------
    BundleError
        If the directory does not exist.
    """
    if not directory.is_dir():
        raise BundleError(f"cannot build a manifest: {directory} is not a directory")

    entries: list[ManifestEntry] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        if EXCLUDED_DIRECTORIES & set(path.relative_to(directory).parts):
            continue
        entries.append(
            ManifestEntry(
                # POSIX form so a bundle written on Windows verifies on Linux.
                relative_path=path.relative_to(directory).as_posix(),
                sha256=digest_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(entries)


def build_manifest(
    directory: Path,
    *,
    model_name: str,
    engine: str,
    version: int,
    framework_version: str,
    spec_digest: str,
    job_id: str | None = None,
    metrics: dict[str, float] | None = None,
    tags: tuple[str, ...] = (),
) -> Manifest:
    """
    Build a manifest describing a bundle directory as it currently stands.

    Called after every other file has been written, because it hashes what it
    finds. Writing a file into a bundle afterwards produces a manifest that no
    longer describes it, and :func:`verify_manifest` will say so.

    Parameters
    ----------
    directory
        The bundle directory, fully written.
    model_name
        Registered model name.
    engine
        Registered engine name.
    version
        Version assigned by the catalog.
    framework_version
        Version of ``rade_xl`` writing this bundle.
    spec_digest
        Digest of the run specification.
    job_id
        Job identifier, or ``None`` for a single run.
    metrics
        Headline metrics, duplicated into the manifest so a catalog listing
        need not open every result file.
    tags
        Free-form labels.

    Returns
    -------
    Manifest
        A manifest listing every file with its digest.
    """
    entries = collect_entries(directory)
    _LOGGER.debug("manifest for %s lists %d file(s)", directory, len(entries))
    return Manifest(
        schema_version=BUNDLE_SCHEMA_VERSION,
        model_name=model_name,
        engine=engine,
        version=version,
        created_at=datetime.now(UTC),
        framework_version=framework_version,
        spec_digest=spec_digest,
        job_id=job_id,
        files=entries,
        metrics=metrics or {},
        tags=tags,
    )


def verify_manifest(directory: Path, manifest: Manifest) -> None:
    """
    Check that a directory still matches its manifest.

    All three failure modes are checked, and they are genuinely different
    problems:

    - A **missing** file means an incomplete or partially copied bundle.
    - A **changed** digest means corruption, or an edit.
    - An **extra** file means something wrote into the bundle after it was
      sealed, which makes the bundle no longer the thing the manifest
      describes -- even though every listed file is intact.

    Every discrepancy is collected before raising, rather than failing on the
    first. Being told that one file of two hundred is wrong, when in fact
    eighty are, sends the reader after the wrong explanation.

    Parameters
    ----------
    directory
        The bundle directory.
    manifest
        The manifest it should match.

    Raises
    ------
    BundleError
        If the directory and the manifest disagree, with every discrepancy
        listed.
    """
    if manifest.schema_version != BUNDLE_SCHEMA_VERSION:
        raise BundleError(
            f"bundle at {directory} has schema version {manifest.schema_version}, "
            f"but this version of rade_xl reads version {BUNDLE_SCHEMA_VERSION}; "
            f"refusing to load it partially"
        )

    problems: list[str] = []
    for entry in manifest.files:
        path = directory / entry.relative_path
        if not path.is_file():
            problems.append(f"{entry.relative_path}: listed in the manifest but not present")
            continue
        actual_size = path.stat().st_size
        if actual_size != entry.size_bytes:
            problems.append(
                f"{entry.relative_path}: size is {actual_size} bytes, "
                f"manifest records {entry.size_bytes}"
            )
            continue
        actual_digest = digest_file(path)
        if actual_digest != entry.sha256:
            problems.append(
                f"{entry.relative_path}: contents have changed "
                f"(digest {actual_digest[:12]}, manifest records {entry.sha256[:12]})"
            )

    present = {entry.relative_path for entry in collect_entries(directory)}
    for extra in sorted(present - set(manifest.file_paths)):
        problems.append(f"{extra}: present but not listed in the manifest")

    if problems:
        detail = "\n  ".join(problems)
        raise BundleError(f"bundle at {directory} does not match its manifest:\n  {detail}")
