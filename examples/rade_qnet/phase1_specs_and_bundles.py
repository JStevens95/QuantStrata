"""
Phase 1 end to end: a specification in, a verified bundle out.

Everything Phase 1 delivers, exercised in one script and with no engine
anywhere. That absence is the point. The framework can read a configuration,
version it, write a bundle atomically, reopen it, verify every byte, restore a
fitted transform and render a report before a single line of PyTorch exists --
which is what makes the engines of Phase 2 replaceable rather than load
bearing.

What this demonstrates, in order
--------------------------------
1. A YAML specification is parsed, validated and digested, and the digest is
   stable under key reordering.
2. A catalog reserves versions under a lock, so concurrent jobs cannot collide.
3. A bundle is written atomically, with a SHA-256 recorded per file.
4. The bundle is reopened, verified, and its fitted state round-trips, so
   predictions can be returned to original units months later.
5. A deliberately corrupted bundle is refused rather than silently loaded.
6. The summary report renders from the bundle alone.

Run it with::

    .venv/bin/python examples/rade_qnet/phase1_specs_and_bundles.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

# Put the repository root on the import path so `src.rade_qnet.*` resolves when
# this file is run directly, following the convention of the other example
# scripts here. An installed distribution imports `rade_qnet` and needs none of
# this; the package uses relative imports internally so both spellings work.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rade_qnet.analysis.reports.base import ReportContext
from src.rade_qnet.analysis.reports.summary import SummaryReport
from src.rade_qnet.core.contract.bundle import ModelBundle, SavedBundle
from src.rade_qnet.core.runtime.errors import BundleError
from src.rade_qnet.core.runtime.hashing import digest_spec
from src.rade_qnet.core.runtime.logging import configure_logging
from src.rade_qnet.core.spec.run import RunSpec, load_run_spec, parse_run_spec
from src.rade_qnet.storage.bundle import load_fitted_state, open_bundle, write_bundle
from src.rade_qnet.storage.runs.catalog import JsonlCatalog
from src.rade_qnet.testkit.fixtures import StandardisingState, make_lineage, make_model_bundle

#: A complete run specification. Only `model` is strictly required; the rest is
#: spelled out to make the defaults visible rather than because it is needed.
EXAMPLE_YAML = """
model:
  name: demo_ridge
  alpha: 0.5
seed: 7
tags: [example, phase1]

hardware:
  device: cpu
  determinism: warn

training:
  engine: torch
  epochs: 5
  learning_rate: 0.001

reports:
  enabled: [summary]
"""

#: Identifies the run throughout. A single place to change, so the catalog
#: reservation, the bundle path and the manifest cannot drift apart.
MODEL_NAME = "demo_ridge"
JOB_ID = "EURUSD"


def show(label: str, detail: str) -> None:
    """
    Print one labelled line of output.

    Parameters
    ----------
    label
        Short name for what is being shown.
    detail
        The value or outcome.
    """
    print(f"  {label:<24} {detail}")


def heading(number: int, title: str) -> None:
    """
    Print a numbered section heading.

    Parameters
    ----------
    number
        Section number, matching the list in this module's docstring.
    title
        What the section demonstrates.
    """
    print(f"\n{number}. {title}")


def parse_specification(root: Path) -> RunSpec:
    """
    Read the example specification from a file and digest it.

    Deliberately written to disk first rather than parsed from the string in
    memory: reading a file is what a user actually does, and it is the path
    that can fail on encoding, on YAML syntax, and on a top level that is a
    list rather than a mapping.

    Parameters
    ----------
    root
        Directory to write the configuration file into.

    Returns
    -------
    RunSpec
        The validated specification.
    """
    heading(1, "Parsing and digesting the specification")
    config_path = root / "run.yaml"
    config_path.write_text(EXAMPLE_YAML, encoding="utf-8")

    spec = load_run_spec(config_path)
    show("model", spec.model.describe())
    # `task` was never written in the file. It defaults during parsing, which
    # is what selects the supervised branch of the RunSpec union.
    show("task (defaulted)", spec.task)
    show("engine", spec.training.engine)
    # False by default: a run should train the epoch budget it was configured
    # with, and the default monitor is a validation metric.
    show("early stopping", str(spec.training.early_stopping.enabled))

    digest = digest_spec(spec)
    show("spec digest", f"{digest[:16]}...")

    # The same configuration with its top-level keys reversed. The digest is
    # taken over a canonical form, so this is recognised as the same run --
    # which is what makes a step cache trustworthy rather than merely fast.
    payload = yaml.safe_load(EXAMPLE_YAML)
    reordered = dict(reversed(list(payload.items())))
    show("stable under reorder", str(digest_spec(parse_run_spec(reordered)) == digest))
    return spec


def reserve_version(catalog: JsonlCatalog) -> int:
    """
    Take a version number from the catalog.

    Parameters
    ----------
    catalog
        The model store's catalog.

    Returns
    -------
    int
        A version number nobody else holds.
    """
    heading(2, "Reserving a version from the catalog")
    version = catalog.next_version(MODEL_NAME, job_id=JOB_ID)
    show("reserved", f"v{version}")
    # The next caller gets a different number even though nothing has been
    # recorded yet. That gap between reserving and recording is exactly the
    # race two concurrent jobs would otherwise lose, and closing it is why
    # reservation is a lock-guarded catalog operation rather than a directory
    # listing.
    show("next caller gets", f"v{catalog.next_version(MODEL_NAME, job_id=JOB_ID)}")
    return version


def write(bundle: ModelBundle, catalog: JsonlCatalog, root: Path, version: int) -> SavedBundle:
    """
    Write a bundle to the store and record it in the catalog.

    Parameters
    ----------
    bundle
        The finished run, in memory.
    catalog
        The catalog to record the manifest in.
    root
        The repository-free temporary root, used to shorten the printed path.
    version
        The reserved version number.

    Returns
    -------
    SavedBundle
        Where it landed, and the manifest describing it.
    """
    heading(3, "Writing a bundle atomically")
    saved = write_bundle(
        bundle,
        root=root / "store",
        version=version,
        framework_version="0.1.0.dev0",
        engine="sklearn",
        model_name=MODEL_NAME,
        job_id=JOB_ID,
        # The engine owns weight serialisation. `storage` writes the bytes it
        # is handed and never imports a training library, which is what lets a
        # bundle be opened on a host with no engine installed.
        write_weights=lambda path: path.write_bytes(b"<the engine writes real weights here>"),
        tags=("example",),
    )
    # Recorded only after the write succeeded. A catalog entry pointing at a
    # bundle that does not exist is worse than no entry at all.
    catalog.record(saved.manifest, location=saved.directory)

    show("written to", str(saved.directory.relative_to(root)))
    show("files hashed", str(len(saved.manifest.files)))
    show("identifier", saved.manifest.identifier)
    return saved


def reopen(saved: SavedBundle, digest: str) -> None:
    """
    Reopen the bundle, verify it, and restore the fitted state.

    Parameters
    ----------
    saved
        The bundle just written.
    digest
        Digest of the specification it was built from, to check against the
        manifest.
    """
    heading(4, "Reopening and verifying")
    reopened = open_bundle(saved.directory)
    show("verified", "every file matches its recorded digest")
    # The manifest's digest comes from the lineage, not from re-digesting the
    # saved spec: the data build is the authority on which configuration
    # produced these rows.
    show("spec digest matches", str(reopened.manifest.spec_digest == digest))

    state = load_fitted_state(reopened, StandardisingState)
    standardised = np.array([-1.0, 0.0, 1.0])
    # The reason fitted state is part of a bundle at all. A model predicts in
    # standardised space, and a prediction nobody can convert back into basis
    # points is not an answer to anything.
    original = state.inverse_transform_targets(standardised)
    show("inverse of [-1, 0, 1]", str(np.round(original, 3).tolist()))


def corrupt(saved: SavedBundle) -> None:
    """
    Truncate a file in the bundle and confirm it is then refused.

    Parameters
    ----------
    saved
        The bundle to damage. Done last, because it leaves it unusable.
    """
    heading(5, "Refusing a corrupted bundle")
    saved.spec_path.write_text("{}", encoding="utf-8")
    try:
        open_bundle(saved.directory)
    except BundleError as error:
        # The last line names the file that failed and how; the first is just
        # the bundle's path, which is already known here.
        show("refused", str(error).strip().splitlines()[-1].strip())
    else:
        show("refused", "NOT REFUSED -- verification is not working")


def report(bundle: ModelBundle, root: Path) -> None:
    """
    Render the summary report from the in-memory bundle.

    Parameters
    ----------
    bundle
        The finished run.
    root
        Directory to create the report directory beneath.
    """
    heading(6, "Rendering the summary report")
    # A report is handed a directory that already exists. Creating it is the
    # pipeline's job, so a report cannot decide where output goes.
    directory = root / "reports"
    directory.mkdir()

    # `render_safely`, not `render`: a report must never be load-bearing, so a
    # failure here would come back as a skipped outcome carrying its reason
    # rather than as an exception that loses a completed training run.
    outcome = SummaryReport().render_safely(ReportContext(bundle=bundle, directory=directory))
    show("succeeded", str(outcome.succeeded))
    for path in outcome.paths:
        show("wrote", path.name)


def main() -> None:
    """Run the whole demonstration in a temporary directory."""
    # Called explicitly. Importing rade_qnet never configures logging, so a host
    # application's own logging setup is left alone.
    configure_logging()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)

        spec = parse_specification(root)
        digest = digest_spec(spec)

        catalog = JsonlCatalog(root / "store")
        version = reserve_version(catalog)

        # A real run gets its bundle from the train pipeline. Phase 1 has no
        # engine to fit one, so the testkit supplies a bundle of the same
        # shape -- the spec and lineage are the real ones parsed above, so the
        # digest recorded in the manifest is genuine.
        bundle = make_model_bundle(
            spec=spec,
            lineage=make_lineage(spec_digest=digest),
            state=StandardisingState(mean=1.5, scale=2.5),
        )

        saved = write(bundle, catalog, root, version)
        reopen(saved, digest)
        corrupt(saved)
        # Safe after the corruption above, because a report reads the bundle
        # it is given in memory and never goes back to disk for it.
        report(bundle, root)

        heading(7, "What is deliberately still missing")
        show("fit", "needs an engine -- Phase 2")
        show("evaluate", "needs an engine -- Phase 2")
        show("weight serialisation", "supplied by the engine -- Phase 2")
        print()


if __name__ == "__main__":
    main()
