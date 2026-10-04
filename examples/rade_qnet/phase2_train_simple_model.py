"""
Phase 2 end to end: a CSV file in, a trained bundle and a report set out.

What Phase 1 could not do. Phase 1 proved the framework could read a
specification, version a bundle, write it atomically, verify it and report from
it -- with no engine anywhere. This script closes the loop: a real PyTorch
model is built, materialised, placed, trained against a real objective,
scored, persisted and reported on, and every one of those steps happens
through the same abstractions a user's own model would use.

The user-written part is deliberately tiny. :class:`TabularMlp` below is
twenty lines: a data module, a model, and nothing else. Everything else --
the stage order, the split, the scaler fitted on training rows only, the
alignment of predictions to targets, the versioning, the bundle, the reports
-- is the framework's.

What this demonstrates, in order
--------------------------------
1. A model is declared by subclassing ``TabularModel``: one line of data
   wiring and one of model construction.
2. The pipeline resolves its components from the specification before it
   touches any data, so an unsupported combination fails in milliseconds.
3. The data is built once: read, split chronologically, scaled on the
   training rows only, and fingerprinted.
4. Lazy parameters are materialised from the input signature before an
   optimiser exists -- defect 6.
5. The model trains, with early stopping and best-weight restoration driven
   by a real validation split.
6. Every split is scored through an *ordered* view of its source, so a
   reshuffling training loader cannot misalign the two passes scoring needs.
7. The bundle is written atomically and versioned, and the report set is
   rendered from it.

Run it with::

    .venv/bin/python examples/rade_qnet/phase2_train_simple_model.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

# Put the repository root on the import path so `src.rade_qnet.*` resolves when
# this file is run directly, following the convention of the other example
# scripts here. An installed distribution imports `rade_qnet` and needs none of
# this; the package uses relative imports internally so both spellings work.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn

from src.rade_qnet.analysis.reports import baselines, curves, quality, summary  # noqa: F401
from src.rade_qnet.core.capability.simple import TabularModel
from src.rade_qnet.core.contract.bundle import ModelBundle
from src.rade_qnet.core.contract.data import DataBundle
from src.rade_qnet.core.contract.result import TrainingResult
from src.rade_qnet.core.contract.signature import InputSignature
from src.rade_qnet.core.runtime.components import model as register_model
from src.rade_qnet.core.runtime.context import RunContext
from src.rade_qnet.core.runtime.hashing import digest_spec
from src.rade_qnet.core.runtime.logging import configure_logging
from src.rade_qnet.core.spec.run import SupervisedRunSpec
from src.rade_qnet.engines.base import ModelHandle
from src.rade_qnet.engines.torch import TorchEngine  # noqa: F401  (registers 'torch')
from src.rade_qnet.orchestration.pipelines.train import TrainPipeline
from src.rade_qnet.sources.dataset.module import TabularDataModule
from src.rade_qnet.storage.catalog import JsonlCatalog

#: Identifies the model in the registry, the specification and the catalog. A
#: single place to change, so the three cannot drift apart.
MODEL_NAME = "phase2_mlp"

#: Rows of synthetic data. Four hundred scenarios is enough for a 70/15/15
#: chronological split to leave a validation period early stopping can act on.
N_SCENARIOS = 400

#: The relationship the model has to find. Linear plus a mild interaction, so a
#: two-layer network beats a linear baseline but the problem stays learnable in
#: a handful of epochs on a CPU.
TRUE_WEIGHTS = np.array([1.5, -2.0, 0.75, 0.0], dtype=np.float64)


class Mlp(nn.Module):
    """
    A two-layer network whose input width is inferred, not declared.

    ``LazyLinear`` is used on purpose. It is the common case in real models --
    anything whose feature count depends on the data build -- and it is the
    case that makes the materialise stage necessary rather than optional. Built
    this way, the module has *no parameters at all* until it has seen one
    batch's shapes, so an optimiser constructed over it would track an empty
    parameter group and update nothing.

    Parameters
    ----------
    hidden
        Width of the hidden layer.
    """

    def __init__(self, hidden: int = 32) -> None:
        """Build the stack."""
        super().__init__()
        self.hidden = nn.LazyLinear(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Run the forward pass.

        The parameter is named ``features`` because that is the key the data
        build puts in each batch. The engine calls ``model(**inputs)``, so a
        model's parameter names are how it declares what it consumes -- which
        is what lets a graph model take a static adjacency alongside its
        features without the engine knowing anything about graphs.

        Parameters
        ----------
        features
            Batch of input rows.

        Returns
        -------
        torch.Tensor
            One prediction per row.
        """
        return self.head(torch.relu(self.hidden(features)))


@register_model(MODEL_NAME, engine="torch")
class TabularMlp(TabularModel):
    """
    The whole of a user's model definition.

    Two methods. ``TabularModel`` supplies ``build_data`` and ``signature``,
    which between them do the reading, splitting, scaling, windowing,
    fingerprinting and source construction -- around twenty lines that would
    otherwise be copied into every simple model, each copy a separate
    opportunity to fit a scaler on the test period.

    ``data_module`` is abstract rather than defaulted to the built-in tabular
    module because ``core`` cannot import ``sources``: a default would invert
    the one-way dependency stack the architecture rests on. The cost is this
    one line.
    """

    def data_module(self, spec: SupervisedRunSpec) -> TabularDataModule:
        """
        Return the data module that builds this model's dataset.

        Parameters
        ----------
        spec
            The validated run specification.

        Returns
        -------
        TabularDataModule
            The built-in tabular module, which is sufficient for a CSV input.
        """
        del spec
        return TabularDataModule()

    def build_model(self, spec: SupervisedRunSpec, signature: object) -> nn.Module:
        """
        Construct the untrained model.

        The signature is available here and deliberately unused: this model
        infers its own input width, which is the case the materialise stage
        exists to serve. A model that wanted to size its layers explicitly
        would read ``signature.dynamic["features"].shape`` instead.

        Parameters
        ----------
        spec
            The validated run specification.
        signature
            The data build's declared interface.

        Returns
        -------
        torch.nn.Module
            The untrained model.
        """
        del signature
        return Mlp(hidden=int(spec.model.params.get("hidden", 32)))


def show(label: str, detail: object) -> None:
    """
    Print one labelled line of output.

    Parameters
    ----------
    label
        Short name for what is being shown.
    detail
        The value or outcome.
    """
    print(f"  {label:<28} {detail}")


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


def write_dataset(path: Path) -> Path:
    """
    Write a synthetic CSV with a learnable target and one deliberate flaw.

    The flaw is a block in the middle of one column where the feed stopped
    updating -- every value repeating the last one it saw. It is there so the
    data-quality report has something real to find, and a frozen feed is the
    right flaw to demonstrate because it is the one with *no other symptom*:
    every value is present, finite and plausible, and only the consecutive
    repetition says anything is wrong.

    A block of *missing* values would be a different lesson. A NaN in a
    training feature makes the loss NaN on the first batch, which makes every
    parameter NaN, and no later epoch recovers -- so the loop refuses at the
    epoch it diverged in rather than letting the failure surface two stages
    later as a message about non-finite predictions.

    Parameters
    ----------
    path
        Destination file.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    rng = np.random.default_rng(20_260_102)
    features = rng.normal(loc=0.0, scale=1.0, size=(N_SCENARIOS, 4))
    # A mild interaction on top of the linear part, so a two-layer network has
    # something a linear model cannot reach.
    target = features @ TRUE_WEIGHTS + 0.4 * features[:, 0] * features[:, 1]
    target += rng.normal(scale=0.1, size=N_SCENARIOS)

    recorded = features.copy()
    recorded[150:170, 3] = recorded[149, 3]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["spot", "carry", "vol", "skew", "target"])
        writer.writerows(np.column_stack([recorded, target]).tolist())
    return path


def build_specification(dataset: Path) -> SupervisedRunSpec:
    """
    Build the run specification this example trains from.

    Written as a dictionary and validated rather than constructed field by
    field, so it goes through the same discriminated-union resolution and the
    same validators a YAML file would.

    Parameters
    ----------
    dataset
        Path to the CSV input.

    Returns
    -------
    SupervisedRunSpec
        The validated specification.
    """
    heading(1, "Declaring the run")
    spec = SupervisedRunSpec.model_validate(
        {
            "model": {"name": MODEL_NAME, "hidden": 32},
            "seed": 11,
            "source": {
                "kind": "tabular",
                "path": dataset,
                "target_column": "target",
                "split": {"kind": "chronological", "validation_fraction": 0.15},
                "transforms": {"scaling": {"method": "standard", "scale_target": True}},
                "loader": {"batch_size": 32, "shuffle": True},
            },
            "training": {
                "engine": "torch",
                "epochs": 40,
                "learning_rate": 0.01,
                "early_stopping": {"enabled": True, "monitor": "val_loss", "patience": 8},
                "checkpoint": {"enabled": True, "monitor": "val_loss"},
                "scheduler": {"kind": "cosine"},
            },
            "hardware": {"device": "cpu"},
            "reports": {"enabled": ["summary", "curves", "quality", "baselines"]},
        }
    )
    show("model", spec.model.describe())
    show("engine", spec.training.engine)
    show("epoch budget", spec.training.epochs)
    show("batch order shuffled", spec.source.loader.shuffle)
    # The same flag used to drive both the split and the batch order -- defect
    # 3. Shuffling batch order is standard; shuffling a time-series split is
    # leakage, and they are now separate decisions.
    show("split strategy", spec.source.split.kind)
    return spec


def build_pipeline(spec: SupervisedRunSpec, root: Path) -> TrainPipeline:
    """
    Wire the pipeline and resolve its components without running anything.

    Parameters
    ----------
    spec
        The validated specification.
    root
        Directory for the catalog and the run output.

    Returns
    -------
    TrainPipeline
        The resolved pipeline.
    """
    heading(2, "Resolving components before touching any data")
    context = RunContext(
        run_id="phase2-example",
        spec_digest=digest_spec(spec),
        output_directory=root / "run",
        seed=spec.seed,
        catalog=JsonlCatalog(root / "store"),
    )
    pipeline = TrainPipeline(context=context, spec=spec, definition=TabularMlp())

    # Resolution instantiates the engine and every report and executes
    # nothing. An unsupported combination -- a distributed request on an
    # engine that cannot distribute, a report that is not registered -- fails
    # here, in milliseconds, rather than after the expensive data build.
    pipeline.resolve()
    show("engine", pipeline.engine.capabilities().describe())
    show("stages", " -> ".join(pipeline.stages[:4]) + " -> ...")
    show("stage count", len(pipeline.stages))
    return pipeline


def train(pipeline: TrainPipeline) -> TrainingResult:
    """
    Run the pipeline and report what the fit did.

    Parameters
    ----------
    pipeline
        The resolved pipeline.

    Returns
    -------
    TrainingResult
        The pipeline's result.
    """
    heading(3, "Training")
    result = pipeline.execute()

    fit = result.fit
    show("epochs run", f"{fit.n_epochs} of {pipeline.spec.training.epochs}")
    show("stopped early", fit.stopped_early)
    show("monitor", fit.monitor)
    # Zero-based in the data, one-based where it is shown to a person.
    show("best epoch", "n/a" if fit.best_epoch is None else fit.best_epoch + 1)
    show("best monitor value", f"{fit.best_monitor_value:.6g}")
    # When this is false, the reported metrics and the saved weights describe
    # different models -- which is why the summary report states it.
    show("best weights restored", fit.restored_best)
    show("first epoch loss", f"{fit.history[0].train_loss:.6g}")
    show("final epoch loss", f"{fit.history[-1].train_loss:.6g}")
    return result


def show_scores(result: TrainingResult) -> None:
    """
    Print each split's score beside a naive baseline's.

    Parameters
    ----------
    result
        The pipeline's training result.
    """
    heading(4, "Scoring every split")
    for name in ("train", "validation", "test"):
        evaluation = result.evaluations.get(name)
        if evaluation is None:
            continue
        r_squared = evaluation.metrics.get("r2")
        baseline = evaluation.baseline_metrics.get("r2")
        detail = f"r2={r_squared:.4f} over {evaluation.n_samples} samples"
        if baseline is not None:
            detail += f" (baseline r2={baseline:.4f})"
        show(name, detail)

    # The training split is scored through an *ordered* view of its source.
    # Without that, the two passes scoring needs -- one for predictions, one
    # for targets -- walk a reshuffling loader in different orders. Every
    # metric still computes, and a correctly trained model reports a negative
    # r-squared on its own training data.
    train_r2 = result.evaluations["train"].metrics["r2"]
    show("train r2 is positive", train_r2 > 0.0)
    show("metrics in original units", result.evaluations["test"].in_original_units)


def show_artifacts(pipeline: TrainPipeline) -> None:
    """
    Print what was written, and where.

    Parameters
    ----------
    pipeline
        The pipeline that ran.
    """
    heading(5, "What was written")
    context = pipeline.context
    for label, directory in (
        ("bundle", context.bundles_directory),
        ("reports", context.reports_directory),
    ):
        names = sorted(path.name for path in directory.rglob("*") if path.is_file())
        show(f"{label} files", len(names))
        show("", ", ".join(names))


def show_quality(bundle: ModelBundle) -> None:
    """
    Print the quality metrics the data build recorded.

    Parameters
    ----------
    bundle
        The persisted model bundle.
    """
    heading(6, "What the data build noticed")
    for name in sorted(bundle.lineage.quality):
        show(name, f"{bundle.lineage.quality[name]:.4f}")
    # The twenty missing rows written into one column, found without anyone
    # having looked for them. These are the numbers that distinguish "the
    # model got worse" from "the data got worse".
    show("scenarios", bundle.lineage.n_scenarios)
    show("split sizes", bundle.lineage.split_sizes)


def main() -> None:
    """Run the whole example in a temporary directory."""
    configure_logging(level="WARNING")
    print(__doc__.strip().splitlines()[0])

    with tempfile.TemporaryDirectory(prefix="rade_qnet_phase2_") as temporary:
        root = Path(temporary)
        dataset = write_dataset(root / "fx.csv")

        spec = build_specification(dataset)
        pipeline = build_pipeline(spec, root)

        # The bundle is captured on its way through the persist stage. A
        # caller who only needs the metrics uses the result; this example
        # wants to show what went into the bundle as well.
        captured: list[ModelBundle] = []
        persist = pipeline.persist

        def capture(
            handle: ModelHandle,
            data: DataBundle[object],
            signature: InputSignature,
            result: TrainingResult,
        ) -> ModelBundle:
            """Record the bundle on its way past the persist stage."""
            bundle = persist(handle, data, signature, result)
            captured.append(bundle)
            return bundle

        pipeline.persist = capture

        result = train(pipeline)
        show_scores(result)
        show_artifacts(pipeline)
        show_quality(captured[0])

        heading(7, "What is deliberately still missing")
        show("multi-cluster job sets", "Phase 4")
        show("evaluate / infer / tune", "Phase 5")
        show("XGBoost and sklearn engines", "Phase 6")
        print()


if __name__ == "__main__":
    main()
