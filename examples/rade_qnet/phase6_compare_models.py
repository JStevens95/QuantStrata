"""
Run three engines against the flagship, on one problem, and show the numbers.

This is the Phase 6 deliverable in executable form, and it is the first
script here whose output is a *finding* rather than a demonstration. The
earlier examples show that the framework does what it claims. This one asks
whether the flagship is worth what it costs, which is a different question
and a less comfortable one.

What it demonstrates
--------------------
1. **Three engines, one lifecycle.** A closed-form ridge regression, a
   gradient-boosted forest and a recurrent network each train, save, and
   re-score through pipelines that contain no mention of any of them. The
   same four reports render for all three. Nothing in ``orchestration``
   knows which engine it is driving, and
   ``test_no_pipeline_branches_on_an_engine_name`` keeps it that way.

2. **What the flagship's extra machinery buys.** The baselines see a
   flattened matrix of the same P&L the flagship sees. They do not see the
   graph, because there is nothing in a design matrix for it to be: the
   adapters refuse static inputs rather than silently dropping them. That
   asymmetry is the experiment, not a flaw in it -- the question is
   precisely whether the structure the flagship is given pays for itself.

3. **A comparison, not a ranking.** The table below prints metrics side by
   side and declares no winner. Choosing one needs a metric, a split and a
   tolerance, and all three are the reader's judgement rather than this
   script's. A hundred and sixty timesteps is not enough evidence to retire
   anything.

How the comparison is made fair
-------------------------------
The fixture carries twenty elementary instruments and three targets. The
flagship predicts all three at once; the baselines predict one. Comparing
the flagship's pooled error against a baseline's single-target error would
flatter whichever happened to have the easier mix.

So the flagship is scored *per target* -- which its Phase 5 evaluation
override already reports -- and only the target the baselines were trained
on is compared. Both numbers are then a mean absolute error on the same
held-out rows of the same series, in the same units.

What remains unequal, and is stated in the output rather than hidden: the
flagship sees a sequence window and a graph, the recurrent baseline sees the
window alone, and the two tabular baselines see a single flattened row.

Running it
----------
::

    python examples/rade_qnet/phase6_compare_models.py

It reads the golden fixture captured in Phase 0, so it needs no external
data. Artifacts land under ``artifacts/rade_qnet/phase6``.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

# Importing a model package is what registers it. The framework does not
# import models on its own behalf -- a host that only wants a ridge
# regression should not pay for Torch to load -- so a script that goes
# through `api` asks for what it needs. Defect 12 in the architecture notes.
#
# The order here is not arbitrary. See `PHASE_6_ADDITIONAL_ENGINES.md` §8.2:
# on macOS the XGBoost and PyTorch wheels each bring their own `libomp`, and
# whichever loads first owns the process. The XGBoost engine package imports
# Torch first for exactly this reason, so any order works here -- but it is
# worth knowing why, because the symptom of getting it wrong is a hang with
# no error at all.
import rade_qnet.models.lstm_tabular
import rade_qnet.models.ridge
import rade_qnet.models.xgb_tabular
from rade_qnet.api import evaluate, train

import rade_qnet.models.hybrid_gnn_rnn  # noqa: F401  isort:skip

#: The Phase 0 fixture, as every example in this folder uses.
FIXTURE = Path(__file__).resolve().parents[2] / (
    "tests/fixtures/rade_qnet/golden/hybrid_gnn_rnn/input"
)

#: Where this script writes.
OUTPUT_ROOT = Path("artifacts/rade_qnet/phase6")

#: Which of the fixture's three targets the baselines predict, and therefore
#: which of the flagship's per-target errors is comparable. The first, with
#: no particular reason beyond needing to pick one and say which.
TARGET_INDEX = 0

#: Window the recurrent models read. Short, because the fixture is short:
#: a hundred and sixty timesteps does not support a long memory.
SEQUENCE_LENGTH = 4

#: Epochs for the gradient-trained models. Enough to converge on a problem
#: this size and small enough that the script finishes in under a minute.
N_EPOCHS = 30

#: Boosting rounds, with early stopping allowed to end it sooner.
N_ROUNDS = 200

#: The metric every row of the table reports.
METRIC = "mae"


def write_tabular_view(destination: Path) -> Path:
    """
    Flatten the fixture into the matrix the tabular baselines can read.

    One row per timestep: the twenty elementary P&L series as features, and
    one of the three target series as the label. This is the whole of what a
    model without a graph can be given, which is why it is written out
    explicitly rather than assembled somewhere less visible.

    Parameters
    ----------
    destination
        Directory to write into.

    Returns
    -------
    Path
        The CSV file.
    """
    elementary = np.load(FIXTURE / "elementary_pnl.npy")
    targets = np.load(FIXTURE / "target_pnl.npy")
    universe = json.loads((FIXTURE / "universe.json").read_text())

    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "tabular.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*universe["elementary_ids"], "target"])
        writer.writerows(
            [*row, target]
            for row, target in zip(elementary, targets[:, TARGET_INDEX], strict=True)
        )
    return path


def baseline_specification(
    dataset: Path, name: str, engine: str, **training: object
) -> dict[str, object]:
    """
    Build a run specification for one baseline.

    Every baseline gets the same source, the same split and the same seed.
    Only the model and the engine differ, which is what makes the resulting
    numbers comparable at all.

    Parameters
    ----------
    dataset
        The flattened CSV.
    name
        Registered model name.
    engine
        Registered engine name.
    **training
        Engine-specific training settings.

    Returns
    -------
    dict
        An unvalidated specification.
    """
    return {
        "task": "supervised",
        "name": f"phase6-{name}",
        "seed": 0,
        "model": {"name": name, "params": {}},
        "source": {
            "kind": "tabular",
            "path": str(dataset),
            "transforms": {"sequence": {"length": SEQUENCE_LENGTH}},
            "split": {
                "kind": "chronological",
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
            },
        },
        "training": {"engine": engine, **training},
        "reports": {"enabled": ()},
        "hardware": {"device": "cpu"},
    }


def flagship_specification() -> dict[str, object]:
    """
    Build the flagship's specification, on the full fixture.

    Returns
    -------
    dict
        An unvalidated specification.
    """
    return {
        "task": "supervised",
        "name": "phase6-flagship",
        "seed": 0,
        "model": {"name": "hybrid_gnn_rnn", "params": {"units": 16}},
        "source": {
            "kind": "model",
            "transforms": {"sequence": {"length": SEQUENCE_LENGTH}},
            "split": {
                "kind": "chronological",
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
            },
            "loader": {"batch_size": 16, "shuffle": True},
            "params": {
                "directory": str(FIXTURE),
                "encoder": {},
                "graph": {"n_neighbours": 5},
            },
        },
        "training": {"engine": "torch", "epochs": N_EPOCHS, "loss": "mse"},
        "reports": {"enabled": ()},
        "hardware": {"device": "cpu"},
    }


def run_baselines(dataset: Path) -> list[tuple[str, str, int, float]]:
    """
    Train every baseline and collect its held-out error.

    Parameters
    ----------
    dataset
        The flattened CSV every baseline reads.

    Returns
    -------
    list of tuple
        Model name, engine name, epochs recorded, and test metric.
    """
    configured = [
        ("ridge", "sklearn", {}),
        ("xgb_tabular", "xgboost", {"n_estimators": N_ROUNDS, "early_stopping_rounds": 10}),
        ("lstm_tabular", "torch", {"epochs": N_EPOCHS}),
    ]

    rows: list[tuple[str, str, int, float]] = []
    for name, engine, settings in configured:
        result = train(
            baseline_specification(dataset, name, engine, **settings),
            output_root=OUTPUT_ROOT / "runs",
        )

        # Re-scored from disk rather than read off the training result, so
        # the number in the table is one that survived a save and a reload.
        # A comparison made from numbers that only ever existed in memory
        # would not tell a reader anything about the model they can deploy.
        bundle = Path(str(result.bundle_directory))
        rows.append(
            (name, engine, len(result.fit.history), evaluate(bundle).metric("test", METRIC))
        )
    return rows


def run_flagship() -> tuple[int, float, dict[str, object]]:
    """
    Train the flagship and return its error on the compared target.

    Returns
    -------
    tuple
        Epochs recorded, pooled test metric, and the per-target notes its
        evaluation override wrote.
    """
    result = train(flagship_specification(), output_root=OUTPUT_ROOT / "runs")
    scored = evaluate(Path(str(result.bundle_directory)))
    return len(result.fit.history), scored.metric("test", METRIC), dict(scored.notes)


def main() -> None:
    """Run every model and print the comparison."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    dataset = write_tabular_view(OUTPUT_ROOT / "data")
    print(f"flattened view: {dataset}")
    print(f"comparing on target index {TARGET_INDEX}\n")

    rows = run_baselines(dataset)
    epochs, pooled, notes = run_flagship()

    print(f"\n{'model':<16}{'engine':<10}{'epochs':>8}{'  test ' + METRIC:>14}")
    print("-" * 48)
    for name, engine, recorded, metric in rows:
        print(f"{name:<16}{engine:<10}{recorded:>8}{metric:>14.6f}")
    print(f"{'hybrid_gnn_rnn':<16}{'torch':<10}{epochs:>8}{pooled:>14.6f}  (pooled)")

    # Printed rather than folded into the table, because it is the one
    # number in this script that needed a model-specific pipeline to exist
    # at all -- and because a reader should see that it is a different
    # quantity from the pooled error above it.
    print("\nthe flagship, per target (its Phase 5 evaluation override):")
    for key in sorted(notes):
        print(f"  {key}: {notes[key]}")

    # No verdict, deliberately. See `PHASE_6_ADDITIONAL_ENGINES.md` §3.8:
    # picking a winner needs a metric, a split and a tolerance, and all
    # three belong to whoever is reading this rather than to the script.
    print(
        "\nNo winner is declared. The baselines saw a flattened matrix; the\n"
        "recurrent baseline saw a window; the flagship saw a window and a\n"
        "graph. One hundred and sixty timesteps is not enough evidence to\n"
        "retire anything, and the comparison is here to be argued with."
    )


if __name__ == "__main__":
    main()
