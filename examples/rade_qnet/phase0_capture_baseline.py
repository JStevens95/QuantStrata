"""
Capture the golden fixture from the *unmodified* ``rade_ml_pt`` flagship.

The refactor's entire correctness claim is that the flagship model in
``rade_qnet`` produces the same results as the flagship in ``rade_ml_pt``. That
claim is backed either by an artifact captured before anything changed, or it
is an assertion. This script produces the artifact.

Two rules govern it, and both are about discipline rather than code.

**It imports nothing from ``rade_qnet``.** Not even the parity helpers. If new
code could influence what is captured, the fixture records what the refactor
*intends* rather than what the original *does*, and the comparison becomes
circular. The only dependency is on ``rade_ml_pt`` and the scientific stack.

**It records defects.** Reading the original closely enough to capture it
means noticing its bugs -- that is how all ten in ``ARCHITECTURE.md`` §13 were
found -- and "fix it while capturing it" is then irresistible. Two of those
defects change numerical output and are deliberately reproduced here:

- *Defect 9, basis-selection leakage.* ``dimension_reduction`` runs over the
  full scaled history including validation and test rows, so the basis is
  chosen with knowledge of held-out data. The fixture captures that. The
  refactor reproduces it only with ``basis.fit_on="all"``; its default is
  ``train``.
- *Defect 3, the shared shuffle flag.* One flag drives both the scenario split
  and the batch order. The fixture captures the **exact split indices** so the
  refactor can replay them rather than having to reproduce the coupling.

Fixing a bug and proving a refactor are two different tasks. Done in one
change neither is verified, because a parity failure could be the fix working
or the refactor broken and there is no way to tell.

What the fixture contains, and why the input is stored the way it is
--------------------------------------------------------------------
The input is checked in as plain ``.npy`` and ``.json``, never as a pickle,
and this script materialises the four pickles ``rade_ml_pt``'s loader expects
into a temporary directory at capture time. A checked-in pickle is code that
executes on load, which is the wrong thing to put in a fixture whose purpose
is to be trusted. The conversion is lossless and explicit, and it keeps the
committed artifact reviewable in a diff.

Run with::

    .venv/bin/python examples/rade_qnet/phase0_capture_baseline.py

Re-running reproduces the fixture bit for bit. That is asserted by
``--verify``, which captures into a temporary directory and diffs against the
committed one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

# The repository root, so this script runs from anywhere.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.rade_ml_pt.data.hybrid_gnn_rnn.build import (  # noqa: E402
    HybridGnnRnnResult,
    build_dataset,
)
from src.rade_ml_pt.data.hybrid_gnn_rnn.config import (  # noqa: E402
    AttributeEncoderConfig,
    BasisSelectionConfig,
    DimensionalityConfig,
    FolderEnvironmentConfig,
    GraphBuilderConfig,
    HybridGnnRnnDataConfig,
)
from src.rade_ml_pt.data.io import CacheLoader  # noqa: E402
from src.rade_ml_pt.models.hybrid_gnn_rnn.config import HybridGnnRnnModelConfig  # noqa: E402
from src.rade_ml_pt.models.hybrid_gnn_rnn.model import HybridGnnRnn  # noqa: E402

#: Where the committed fixture lives.
FIXTURE_ROOT = _ROOT / "tests" / "fixtures" / "rade_qnet" / "golden"
FIXTURE_NAME = "hybrid_gnn_rnn"

#: The one seed that governs input generation, the split, weight
#: initialisation and the training loop. A single value rather than several,
#: so "re-run with the same seed" is unambiguous.
SEED = 42

#: Deliberately tiny. Large enough to exercise a sparse graph, a recurrence
#: and a non-trivial basis selection; small enough that the fixture commits
#: and the parity suite runs in seconds. A parity suite taking twenty minutes
#: gets run once.
N_SCENARIOS = 160
N_TARGETS = 3

#: The elementary book is laid out as (underlying, product) groups with
#: several instruments each, and the shape is load-bearing rather than
#: cosmetic. ``dimension_reduction`` runs basis selection *per group*, so a
#: book with two instruments per group has nothing to reduce: every
#: instrument survives, the post-reduction index arrays come out equal to the
#: pre-reduction ones, and the trap in ``build_metadata`` goes uncaptured --
#: the fixture would then pass against a refactor that gets it wrong. Five
#: per group driven by two factors reduces to roughly two, which is what
#: makes level 1 able to fail.
UNDERLYINGS = ("EURUSD", "GBPUSD")
PRODUCT_TYPES = ("vanilla_option", "forward")
INSTRUMENTS_PER_GROUP = 5
GROUP_FACTORS = 2
N_ELEMENTARY = len(UNDERLYINGS) * len(PRODUCT_TYPES) * INSTRUMENTS_PER_GROUP

#: Window length. Greater than one so the sequence-aware split is exercised:
#: at a length of one every boundary question disappears, and the boundary is
#: where the interesting bugs are.
SEQ_LENGTH = 4

#: How many epochs level 4 captures. Five is enough for the curve to have a
#: shape and short enough that accumulated non-determinism stays inside the
#: 1e-3 relative tolerance.
N_EPOCHS = 5

#: Narrow layers, because parity is about agreement rather than accuracy. A
#: wider model would make the fixture larger and the capture slower without
#: testing anything the narrow one does not.
LAYER_UNITS = 16

_LOGGER = logging.getLogger("phase0.capture")


def build_input(directory: Path) -> dict[str, Any]:
    """
    Generate the fixed input and write it in a reviewable, pickle-free form.

    The P&L is built so the targets are a genuine linear combination of a few
    elementary instruments plus noise. That matters for the basis selection:
    against pure noise, the selected basis would be arbitrary and reordering
    it would not be detectable, which is exactly the trap level 1 exists to
    catch.

    Parameters
    ----------
    directory
        The fixture's ``input/`` directory, created if absent.

    Returns
    -------
    dict
        The generated frames and attribute dictionaries.
    """
    directory.mkdir(parents=True, exist_ok=True)

    # `RandomState` rather than `default_rng`: its stream is guaranteed stable
    # across NumPy versions, and a fixture whose input changes when NumPy is
    # upgraded is not a baseline.
    rng = np.random.RandomState(SEED)

    # Built group by group, because basis selection runs per group. Each
    # group of five instruments is driven by two factors plus a small
    # idiosyncratic term, so the group has genuine internal rank and
    # selection has something to find -- roughly two survivors per group.
    elementary_ids: list[str] = []
    columns: list[np.ndarray] = []
    for group_index, (underlying, product) in enumerate(
        (u, p) for u in UNDERLYINGS for p in PRODUCT_TYPES
    ):
        factors = rng.randn(N_SCENARIOS, GROUP_FACTORS)
        loadings = rng.randn(GROUP_FACTORS, INSTRUMENTS_PER_GROUP)
        block = factors @ loadings + 0.05 * rng.randn(N_SCENARIOS, INSTRUMENTS_PER_GROUP)
        for offset in range(INSTRUMENTS_PER_GROUP):
            index = group_index * INSTRUMENTS_PER_GROUP + offset
            elementary_ids.append(f"{underlying}|{product}|{index:03d}")
            columns.append(block[:, offset])

    elementary = np.column_stack(columns) * 0.01
    target_ids = [f"EURUSD|vanilla_option|tgt_{index}" for index in range(N_TARGETS)]

    # Targets are a sparse combination of elementary instruments, which is
    # what a replicating portfolio actually looks like.
    weights = np.zeros((N_ELEMENTARY, N_TARGETS))
    for target in range(N_TARGETS):
        chosen = rng.choice(N_ELEMENTARY, size=5, replace=False)
        weights[chosen, target] = rng.uniform(0.5, 1.5, size=5)
    targets = elementary @ weights + 0.001 * rng.randn(N_SCENARIOS, N_TARGETS)

    def attributes(trade_ids: list[str], trade_type: str) -> dict[str, Any]:
        """Build the attribute dictionary the encoder expects."""
        count = len(trade_ids)
        return {
            "trade_id": trade_ids,
            "moneyness": rng.uniform(0.8, 1.2, count).tolist(),
            "yrs_to_maturity": rng.uniform(0.1, 2.0, count).tolist(),
            "delta": rng.uniform(-1.0, 1.0, count).tolist(),
            "vega": rng.uniform(0.0, 0.5, count).tolist(),
            "product_type": [tid.split("|")[1] for tid in trade_ids],
            "product_subtype": ["european"] * count,
            "trade_type": [trade_type] * count,
            "underlying_risk_factors": [["FX"]] * count,
        }

    payload = {
        "elementary_pnl": pd.DataFrame(elementary, columns=elementary_ids),
        "target_pnl": pd.DataFrame(targets, columns=target_ids),
        "elementary_attribs": attributes(elementary_ids, "elementary"),
        "target_attribs": attributes(target_ids, "target"),
    }

    np.save(directory / "elementary_pnl.npy", elementary)
    np.save(directory / "target_pnl.npy", targets)
    _write_json(
        directory / "universe.json",
        {
            "elementary_ids": elementary_ids,
            "target_ids": target_ids,
        },
    )
    _write_json(directory / "elementary_attributes.json", payload["elementary_attribs"])
    _write_json(directory / "target_attributes.json", payload["target_attribs"])
    return payload


def load_input(directory: Path) -> dict[str, Any]:
    """
    Read a previously captured input instead of regenerating it.

    Preferred over regeneration whenever the committed input exists, so the
    baseline cannot drift underneath the fixture if this script's generation
    code is ever edited. A changed input and a changed result are
    indistinguishable in a parity failure.

    Parameters
    ----------
    directory
        The fixture's ``input/`` directory.

    Returns
    -------
    dict
        The frames and attribute dictionaries.
    """
    universe = json.loads((directory / "universe.json").read_text(encoding="utf-8"))
    return {
        "elementary_pnl": pd.DataFrame(
            np.load(directory / "elementary_pnl.npy"), columns=universe["elementary_ids"]
        ),
        "target_pnl": pd.DataFrame(
            np.load(directory / "target_pnl.npy"), columns=universe["target_ids"]
        ),
        "elementary_attribs": json.loads(
            (directory / "elementary_attributes.json").read_text(encoding="utf-8")
        ),
        "target_attribs": json.loads(
            (directory / "target_attributes.json").read_text(encoding="utf-8")
        ),
    }


def materialise_job(payload: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """
    Write the four pickles ``rade_ml_pt``'s loader reads, into a scratch dir.

    The loader takes four file paths and reads them through ``CacheLoader``.
    Rather than check those pickles in, the capture writes them fresh each
    run from the reviewable input. The fixture therefore contains no
    executable payload, and what ``rade_ml_pt`` reads is provably the same
    data a reader can inspect.

    Parameters
    ----------
    payload
        The frames and attribute dictionaries.
    workdir
        A temporary directory.

    Returns
    -------
    dict
        The job dictionary, with its ``cluster_info`` paths.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "elementary_pnl_path": str(workdir / "elementary_pnl.pkl"),
        "target_pnl_path": str(workdir / "target_pnl.pkl"),
        "elementary_attribs_path": str(workdir / "elementary_attributes.pkl"),
        "target_attribs_path": str(workdir / "target_attributes.pkl"),
    }
    CacheLoader.save_data(payload["elementary_pnl"], paths["elementary_pnl_path"])
    CacheLoader.save_data(payload["target_pnl"], paths["target_pnl_path"])
    CacheLoader.save_data(payload["elementary_attribs"], paths["elementary_attribs_path"])
    CacheLoader.save_data(payload["target_attribs"], paths["target_attribs_path"])
    return {"cluster_info": paths}


def data_config(workdir: Path) -> HybridGnnRnnDataConfig:
    """
    Build the data configuration the baseline is captured under.

    ``shuffle=False`` is the one choice worth explaining. Defect 3 is that a
    single flag drives both the scenario split and the batch order, so
    ``shuffle=True`` produces a *random split* -- a leak. The fixture is
    captured with it off, which makes the split chronological and
    reproducible, and the split indices are then recorded so the refactor can
    replay them exactly rather than having to reproduce the coupling.

    ``reduction_mode="basis_selection"`` is on deliberately. With reduction
    disabled the post-reduction index arrays would equal the pre-reduction
    ones and the trap in ``build_metadata`` would go uncaptured -- the fixture
    would pass against a refactor that gets it wrong.

    Parameters
    ----------
    workdir
        Scratch directory for the original's own intermediate files.

    Returns
    -------
    HybridGnnRnnDataConfig
        The configuration.
    """
    return HybridGnnRnnDataConfig(
        folders=FolderEnvironmentConfig(root_folder=str(workdir)),
        validation_split=0.2,
        test_split=0.2,
        seq_length=SEQ_LENGTH,
        batch_size=16,
        shuffle=False,
        cache=False,
        drop_remainder=False,
        transform_type="standard",
        dimensionality=DimensionalityConfig(
            reduction_mode="basis_selection",
            basis_selection=BasisSelectionConfig(
                var_threshold=0.99, weight_tail=1.0, method="pca", max_components=200
            ),
        ),
        graph_builder=GraphBuilderConfig(
            k=5,
            distance_metric="euclidean",
            include_quota=False,
            alpha_moneyness=1.0,
            alpha_maturity=1.0,
            alpha_delta=1.0,
            alpha_vega=1.0,
            alpha_prod_type=1.0,
            alpha_prod_subtype=0.5,
            alpha_underlying=1.0,
            alpha_underlying_rf=0.5,
        ),
        attribute_encoder=AttributeEncoderConfig(
            numeric_keys=["moneyness", "yrs_to_maturity", "delta", "vega"],
            categorical_keys=["product_type", "product_subtype", "trade_type"],
            multi_label_keys=["underlying_risk_factors"],
            num_decay_terms=3,
        ),
        plot_trade_graph=False,
        plot_pnl_distribution=False,
        save_intermediate_files=False,
        seed=SEED,
    )


def model_config() -> dict[str, Any]:
    """
    Build the model configuration the baseline is captured under.

    Returns
    -------
    dict
        The nested dict the model expects.
    """
    return HybridGnnRnnModelConfig.from_dict(
        {
            "gnn_layer": {"parameters": {"units": LAYER_UNITS}},
            "rnn_layer": {
                "general": {"layer_type": "lstm"},
                "parameters": {"units": LAYER_UNITS},
            },
            "fusion_layer": {"parameters": {"units": LAYER_UNITS}},
            "attention_layer": {"parameters": {"units": LAYER_UNITS}},
            "projection_layer": {"parameters": {"units": LAYER_UNITS}},
        }
    ).to_dict()


def capture_level1(result: HybridGnnRnnResult, directory: Path) -> None:
    """
    Capture the fitted state, which level 1 compares exactly.

    Two artifacts here are the ones a refactor gets subtly wrong.

    ``selected_basis.json`` records **order**, not just membership. Basis
    selection returns an ordered list and that order fixes column positions
    in every array downstream, so a refactor selecting the same instruments
    in a different order passes a set comparison and fails everything after
    it.

    ``elementary_idx`` and ``target_idx`` are captured **post-reduction**.
    The original's ``build_metadata`` recomputes them as ``0..n_e`` and
    ``n_e..n_e+n_t`` *after* the basis is selected. Capturing the
    pre-reduction indices would make level 1 pass against the wrong thing,
    and the error would surface as a quietly worse model rather than as a
    failure.

    Parameters
    ----------
    result
        The original's build result.
    directory
        The fixture's ``level1_state/`` directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    metadata = result.metadata

    elementary_transformer = metadata["elementary_pnl_transformer"]
    target_transformer = metadata["target_pnl_transformer"]
    np.save(directory / "scaler_mean.npy", np.asarray(elementary_transformer.mean_))
    np.save(directory / "scaler_scale.npy", np.asarray(elementary_transformer.scale_))
    np.save(directory / "target_scaler_mean.npy", np.asarray(target_transformer.mean_))
    np.save(directory / "target_scaler_scale.npy", np.asarray(target_transformer.scale_))

    _write_json(directory / "selected_basis.json", list(metadata["selected_trades"]))

    encoder_results = result.encoder_results
    np.save(directory / "combined_features.npy", np.asarray(encoder_results["combined_features"]))

    graph = result.graph_results
    np.save(directory / "adjacency_indices.npy", np.asarray(graph["sparse_indices"]))
    np.save(directory / "adjacency_values.npy", np.asarray(graph["sparse_values"]))
    np.save(directory / "adjacency_shape.npy", np.asarray(graph["sparse_shape"], dtype=np.int64))

    # Post-reduction. See this function's docstring -- this is the trap.
    np.save(
        directory / "elementary_idx.npy", np.asarray(metadata["elementary_idx"], dtype=np.int64)
    )
    np.save(directory / "target_idx.npy", np.asarray(metadata["target_idx"], dtype=np.int64))

    _write_json(
        directory / "universe.json",
        {
            "elementary_ids": list(metadata["elementary_ids"]),
            "target_ids": list(metadata["target_ids"]),
        },
    )


def capture_level2(result: HybridGnnRnnResult, directory: Path) -> None:
    """
    Capture the split indices and the first batch of each split.

    The split indices are the compatibility mechanism for defect 3: rather
    than making the refactor reproduce a shuffle flag that drives two
    unrelated things, it replays these indices through an explicit split.

    Only the first batch per split is stored. Every later batch exercises the
    same code on different rows, so storing them all would multiply the
    fixture's size without adding a distinct failure it could catch.

    Parameters
    ----------
    result
        The original's build result.
    directory
        The fixture's ``level2_tensors/`` directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    metadata = result.metadata

    # Two different things, and conflating them costs a day of debugging.
    #
    # `*_indices` is which scenario rows belong to the split. It is what the
    # P&L scalers were fitted on, so the refactor has to replay exactly this
    # set or every fitted statistic comes out different.
    #
    # `*_starts` is where each rolling window begins, which is a strict
    # subset: a window needs `sequence_length` rows, so the last few rows of
    # a split can never start one. The refactor *derives* these from the row
    # membership rather than replaying them, which is why both are captured
    # -- the derivation is itself something worth checking.
    _write_json(
        directory / "split_indices.json",
        {
            "train": [int(index) for index in metadata["train_indices"]],
            "validation": [int(index) for index in metadata["val_indices"]],
            "test": [int(index) for index in metadata["test_indices"]],
        },
    )
    _write_json(
        directory / "split_window_starts.json",
        {
            "train": [int(index) for index in metadata["train_starts"]],
            "validation": [int(index) for index in metadata["val_starts"]],
            "test": [int(index) for index in metadata["test_starts"]],
        },
    )

    for split, loader in (
        ("train", result.train_ds),
        ("validation", result.val_ds),
        ("test", result.test_ds),
    ):
        if loader is None:
            continue
        inputs, target = next(iter(loader))
        arrays = {key: value.detach().cpu().numpy() for key, value in inputs.items()}
        arrays["target"] = target.detach().cpu().numpy()
        np.savez(directory / f"{split}_batch_000.npz", **arrays)


def capture_level3(result: HybridGnnRnnResult, config: dict[str, Any], directory: Path) -> None:
    """
    Capture initial weights and the forward output for one fixed batch.

    Captured in ``eval`` mode and under ``no_grad``. Dropout is active in
    training mode, so a forward pass captured there would depend on the
    global RNG state at the moment it ran -- the fixture would then record a
    sample rather than a function, and no refactor could reproduce it.

    Parameters
    ----------
    result
        The original's build result.
    config
        The model configuration.
    directory
        The fixture's ``level3_forward/`` directory.
    """
    directory.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    model = HybridGnnRnn(config=config)

    inputs, _ = next(iter(result.train_ds))
    model.eval()
    with torch.no_grad():
        # The first call also materialises the lazy parameters, which is why
        # the state dict is saved afterwards rather than before.
        outputs = model(inputs)

    torch.save(model.state_dict(), directory / "state_dict.pt")
    np.save(directory / "outputs.npy", outputs.detach().cpu().numpy())


def _without_dropout(config: dict[str, Any]) -> dict[str, Any]:
    """
    Return the model configuration with every dropout rate set to zero.

    Walks the nested configuration rather than naming each layer, because
    the original spreads ``dropout_rate`` across five layer blocks and
    missing one would leave the curve partly stochastic -- which would
    show up as a parity failure nobody could explain.

    Parameters
    ----------
    config
        The model configuration.

    Returns
    -------
    dict
        A deep copy with the rates zeroed.
    """
    if isinstance(config, dict):
        return {
            key: 0.0 if key == "dropout_rate" else _without_dropout(value)
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [_without_dropout(item) for item in config]
    return config


def capture_level4(result: HybridGnnRnnResult, config: dict[str, Any], directory: Path) -> None:
    """
    Capture a short training curve under an explicit, minimal loop.

    The loop is written out here rather than delegated to ``rade_ml_pt``'s
    trainer, and that is a deliberate narrowing of what level 4 claims. The
    trainer applies early stopping, learning-rate reduction and best-weight
    restoration; a curve captured through it would be a record of those
    callbacks as much as of the model, and the refactored framework
    implements its own callbacks by design. Comparing the two would test
    whether two callback implementations agree, which is not the question.

    What this captures instead is narrower and actually meaningful: the same
    architecture, on the same batches, in the same order, under the same
    optimiser and loss, produces the same losses. That isolates the model and
    the data -- which is what the refactor moved.

    Dropout is disabled and the initial weights are reloaded from the level 3
    capture, for the same reason level 3 is captured under ``eval``. With
    dropout active, each epoch's loss depends on the global random state at
    the moment it ran, so the curve would record a sample of a stochastic
    process rather than a function -- and no refactor could reproduce it,
    because reproducing it would mean reproducing the exact order in which
    the original consumed random numbers. Lazy parameters make that worse
    still: they draw their initial values at a different point in the
    sequence than eager ones do, so even the starting weights would differ.
    Pinning both leaves a curve determined by the architecture, the data and
    the optimiser, which is what level 4 is for.

    Parameters
    ----------
    result
        The original's build result.
    config
        The model configuration.
    directory
        The fixture's ``level4_training/`` directory.
    """
    directory.mkdir(parents=True, exist_ok=True)

    # Dropout off: see the docstring. The layer holds no parameters, so the
    # state dict loaded below is unaffected by the change.
    config = _without_dropout(config)

    torch.manual_seed(SEED)
    model = HybridGnnRnn(config=config)

    # Materialise the lazy parameters before the optimiser is constructed.
    # This is defect 6 in the original: an optimiser built over uninitialised
    # parameters holds references to placeholders and silently updates
    # nothing. Doing it correctly here is not a fix to `rade_ml_pt` -- it is
    # the capture script avoiding a trap, which is explicitly permitted where
    # modifying the original is not.
    model.train()
    inputs, _ = next(iter(result.train_ds))
    model(inputs)

    # The level 3 weights, reloaded rather than re-seeded. Identical in
    # principle -- same seed, same construction -- but reloading states the
    # starting point as a file a refactor can read, instead of as a
    # consequence of matching the original's random-number consumption.
    model.load_state_dict(
        torch.load(directory.parent / "level3_forward" / "state_dict.pt", weights_only=True)
    )

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.L1Loss()

    curve: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for epoch in range(N_EPOCHS):
        model.train()
        batch_losses = []
        for batch_inputs, batch_target in result.train_ds:
            optimiser.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_inputs), batch_target)
            loss.backward()
            optimiser.step()
            batch_losses.append(float(loss.detach()))
        curve["train_loss"].append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            validation_losses = [
                float(loss_function(model(batch_inputs), batch_target))
                for batch_inputs, batch_target in result.val_ds
            ]
        curve["val_loss"].append(float(np.mean(validation_losses)))
        _LOGGER.info(
            "epoch %d/%d  train %.6f  val %.6f",
            epoch + 1,
            N_EPOCHS,
            curve["train_loss"][-1],
            curve["val_loss"][-1],
        )

    _write_json(directory / "curve.json", curve)


def write_manifest(directory: Path, config: HybridGnnRnnDataConfig) -> None:
    """
    Record what was captured, from where, and under which versions.

    When a long-passing parity test starts failing, the first question is
    whether the baseline moved rather than the code. Without the source
    commit there is no way to answer it.

    Parameters
    ----------
    directory
        The fixture's root.
    config
        The data configuration used, recorded in full.
    """
    _write_json(
        directory / "manifest.json",
        {
            "fixture": FIXTURE_NAME,
            "captured_at": datetime.now(UTC).isoformat(),
            "source_commit": _git_commit(),
            "source_module": "src.rade_ml_pt.data.hybrid_gnn_rnn.build",
            "seed": SEED,
            "shape": {
                "n_scenarios": N_SCENARIOS,
                "n_elementary": N_ELEMENTARY,
                "n_targets": N_TARGETS,
                "seq_length": SEQ_LENGTH,
            },
            "epochs": N_EPOCHS,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
            },
            "data_config": _plain(config.to_dict()),
            "model_config": _plain(model_config()),
            "preserved_defects": {
                "defect_9_basis_selection_leakage": (
                    "dimension_reduction runs over the full scaled history including "
                    "held-out rows. Reproduce with basis.fit_on='all'; the default is "
                    "'train'."
                ),
                "defect_3_shared_shuffle_flag": (
                    "one flag drives both the split and the batch order. Captured with "
                    "shuffle=False and the exact split indices recorded, so the "
                    "refactor replays them via an explicit split."
                ),
            },
        },
    )


def capture(root: Path) -> Path:
    """
    Run the original end to end and write every level.

    Parameters
    ----------
    root
        Directory to write the fixture into.

    Returns
    -------
    pathlib.Path
        The fixture directory.
    """
    directory = root / FIXTURE_NAME
    directory.mkdir(parents=True, exist_ok=True)

    input_directory = directory / "input"
    if (input_directory / "elementary_pnl.npy").is_file():
        _LOGGER.info("reusing the committed input, so the baseline cannot drift")
        payload = load_input(input_directory)
    else:
        _LOGGER.info("generating the input for the first time")
        payload = build_input(input_directory)

    with tempfile.TemporaryDirectory() as scratch:
        workdir = Path(scratch)
        job = materialise_job(payload, workdir / "job")
        config = data_config(workdir / "work")

        _LOGGER.info("running rade_ml_pt build_dataset (unmodified)")
        result = build_dataset(job=job, config=config)

        capture_level1(result, directory / "level1_state")
        capture_level2(result, directory / "level2_tensors")

        model = model_config()
        capture_level3(result, model, directory / "level3_forward")
        capture_level4(result, model, directory / "level4_training")
        write_manifest(directory, config)

    return directory


def verify(root: Path) -> int:
    """
    Re-capture into a temporary directory and diff against the committed one.

    The definition of done requires the capture to be reproducible bit for
    bit. If it is not, the fixture records one run rather than the original's
    behaviour, and a parity failure could mean nothing at all.

    ``manifest.json`` is excluded from the comparison because it records the
    capture timestamp, which differs by construction.

    Parameters
    ----------
    root
        Where the committed fixture lives.

    Returns
    -------
    int
        Process exit code: zero when every artifact matched.
    """
    committed = root / FIXTURE_NAME
    if not committed.is_dir():
        _LOGGER.error("nothing to verify: no fixture at %s", committed)
        return 1

    with tempfile.TemporaryDirectory() as scratch:
        # Seed the temporary capture with the committed input, so this
        # verifies the *capture* is reproducible rather than that the input
        # generator is.
        fresh_root = Path(scratch)
        fresh_input = fresh_root / FIXTURE_NAME / "input"
        fresh_input.mkdir(parents=True)
        for path in (committed / "input").iterdir():
            fresh_input.joinpath(path.name).write_bytes(path.read_bytes())

        capture(fresh_root)
        fresh = fresh_root / FIXTURE_NAME

        differences = []
        for path in sorted(committed.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            relative = path.relative_to(committed)
            other = fresh / relative
            if not other.is_file():
                differences.append(f"{relative}: missing from the re-capture")
            elif _digest(path) != _digest(other):
                differences.append(f"{relative}: differs between captures")

    for difference in differences:
        _LOGGER.error("%s", difference)
    if differences:
        _LOGGER.error(
            "the capture is NOT reproducible, so the fixture records one run rather "
            "than the original's behaviour"
        )
        return 1

    _LOGGER.info("the capture is reproducible bit for bit")
    return 0


def _write_json(path: Path, payload: object) -> None:
    """
    Write JSON deterministically.

    Sorted keys and a fixed indent, with a trailing newline. Without this the
    bit-for-bit check would fail on dictionary ordering rather than on
    content, and the fixture would produce noisy diffs on every re-capture.

    Parameters
    ----------
    path
        Destination.
    payload
        Anything JSON-serialisable.
    """
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_plain) + "\n",
        encoding="utf-8",
    )


def _plain(value: object) -> object:
    """
    Reduce a value to something JSON can hold.

    Parameters
    ----------
    value
        Any value appearing in a configuration.

    Returns
    -------
    object
        A JSON-serialisable equivalent.
    """
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        # Absolute scratch paths would change between runs and break the
        # bit-for-bit check, so only the final component is recorded.
        return value.name
    # Anything else -- a fitted transformer, an enum -- is recorded as its
    # repr. The manifest documents the configuration; it is not a format the
    # configuration is rebuilt from.
    return value if isinstance(value, str | int | float | bool) or value is None else repr(value)


def _digest(path: Path) -> str:
    """
    Hash a file's contents.

    Parameters
    ----------
    path
        The file.

    Returns
    -------
    str
        Hex SHA-256.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    """
    Return the current commit, or a marker when it cannot be determined.

    Returns
    -------
    str
        The commit hash, or ``"unknown"``.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def main() -> int:
    """
    Parse arguments and run.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-capture into a temporary directory and diff against the committed fixture",
    )
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"
    )
    # The original logs heavily at INFO, which buries the capture's own
    # progress without adding anything a reader of this script needs.
    logging.getLogger("src.rade_ml_pt").setLevel(logging.WARNING)

    if arguments.verify:
        return verify(FIXTURE_ROOT)

    directory = capture(FIXTURE_ROOT)
    total_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    _LOGGER.info("wrote the fixture to %s (%.1f KB)", directory, total_bytes / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
