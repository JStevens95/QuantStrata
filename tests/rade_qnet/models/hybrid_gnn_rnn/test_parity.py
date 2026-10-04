"""
Parity levels 2 and 4: the batches, and the training curve.

Level 1 compares the fitted state and level 3 compares one forward pass;
both are checked where they are produced. These two are checked here
because they need the framework around them -- level 2 needs the batch
sources the loader builds, and level 4 needs an optimiser stepping over
several epochs.

Why the tolerances widen
------------------------
Level 2 is exact: a batch is an arrangement of numbers already computed at
level 1, so an inexact comparison here would mean the arranging changed
something, which it must not.

Level 4 is the loosest at ``rtol=1e-3``, and deliberately so. Five epochs
of Adam accumulate the last-bit differences level 3 tolerates, and the
accumulation is genuine rather than a measurement artefact: two
mathematically identical implementations that differ in operation order
really do reach slightly different weights. Demanding exactness here would
either fail on a BLAS upgrade or force the refactor to preserve operation
order, which would block every improvement the refactor exists to make.

What level 4 does *not* claim
------------------------------
The captured curve came from an explicit minimal loop, not from the
original's trainer. The trainer applies early stopping, learning-rate
reduction and best-weight restoration, and the framework implements its
own callbacks by design -- so a comparison through both would be testing
whether two callback implementations agree. This compares the narrower and
more meaningful thing: same architecture, same batches, same order, same
optimiser and loss, same losses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.rade_qnet.core.spec.data import (
    ExplicitSplitSpec,
    LoaderSpec,
    ModelSourceSpec,
    ReductionSpec,
    SequenceSpec,
    TransformsSpec,
)
from src.rade_qnet.models.hybrid_gnn_rnn.data import HybridDataModule
from src.rade_qnet.models.hybrid_gnn_rnn.model import HybridGnnRnn
from src.rade_qnet.models.hybrid_gnn_rnn.spec import HybridModelSpec
from src.rade_qnet.sources.batching.dataset import sources_for
from src.rade_qnet.testkit.parity import compare_curve, compare_tensors, load_golden

#: The golden fixture captured from the original implementation.
FIXTURE_NAME = "hybrid_gnn_rnn"
FIXTURE = Path("tests/fixtures/rade_qnet/golden") / FIXTURE_NAME

#: The capture's settings, which the replay must match exactly. Named
#: rather than inlined so that a mismatch reads as a changed constant
#: rather than as a mysterious number.
SEQUENCE_LENGTH = 4
BATCH_SIZE = 16
N_EPOCHS = 5
LEARNING_RATE = 1e-3
SEED = 0
UNITS = 16


def replay_spec() -> ModelSourceSpec:
    """
    Build the source spec that reproduces the capture.

    Every compatibility flag the baseline needs is set here and only here.
    Each defaults to the *correct* behaviour, so a production
    configuration that omits one gets the fixed behaviour rather than the
    bug -- forgetting is the safe failure.

    Returns
    -------
    ModelSourceSpec
        The replay specification.
    """
    rows = json.loads((FIXTURE / "level2_tensors" / "split_indices.json").read_text())
    return ModelSourceSpec(
        params={
            "directory": str(FIXTURE / "input"),
            # Compatibility flag: the baseline's encoder fitted and
            # transformed in float32.
            "encoder": {"numeric_precision": "float32"},
            # Its twin, for the graph's row normalisation.
            "graph": {"n_neighbours": 5, "precision": "float32"},
        },
        # The baseline's split, replayed rather than reproduced, so a
        # difference in splitting cannot masquerade as a difference in
        # anything downstream of it.
        split=ExplicitSplitSpec(
            train=tuple(rows["train"]),
            validation=tuple(rows["validation"]),
            test=tuple(rows["test"]),
        ),
        # Shuffling off, because the capture walked the windows in order
        # and batch zero must be the same sixteen windows on both sides.
        loader=LoaderSpec(batch_size=BATCH_SIZE, shuffle=False),
        transforms=TransformsSpec(
            # Compatibility flag: defect 9. The baseline selected the
            # basis over the full scaled history rather than over the
            # training rows alone.
            reduction=ReductionSpec(method="basis_selection", fit_on="all"),
            sequence=SequenceSpec(
                length=SEQUENCE_LENGTH,
                # Compatibility flag: the baseline required each window to
                # lie entirely inside its own split, dropping the first
                # three labels of each. The framework's default instead
                # keeps them and relies on a boundary gap -- the two cost
                # the same scenarios, but the fixture's splits are
                # adjacent, so the replay has to confine.
                confine_to_split=True,
            ),
        ),
    )


@pytest.fixture(scope="module")
def replay():
    """Build the dataset and its batch sources once, through the framework."""
    spec = replay_spec()
    module = HybridDataModule()
    prepared = module.build(spec, seed=SEED)
    return prepared, sources_for(
        prepared,
        loader=spec.loader,
        sequence=spec.transforms.sequence,
        seed=SEED,
    )


def first_batches(sources) -> dict[str, dict[str, np.ndarray]]:
    """
    Collect each split's first batch, merging in the static inputs.

    The capture wrote one dictionary per split holding everything a
    forward pass receives. The framework keeps the per-sample tensors and
    the batch-invariant ones apart -- which is the point of the static
    path -- so they are recombined here for the comparison.

    Parameters
    ----------
    sources
        Split name to batch source.

    Returns
    -------
    dict
        Split name to tensor name to array.
    """
    collected: dict[str, dict[str, np.ndarray]] = {}
    for split, source in sources.items():
        batch = next(iter(source.batches()))
        merged = {name: np.asarray(value) for name, value in batch.items()}
        merged.update({name: np.asarray(value) for name, value in source.static.items()})
        # The capture used the original's key for the dense shape.
        merged["adjacency_dense_shape"] = merged.pop("adjacency_shape")
        collected[split] = merged
    return collected


class TestLevel2Tensors:
    """The batches the loader produces, compared exactly."""

    def test_the_batch_contents_match(self, replay) -> None:
        """
        Every tensor of every captured batch, bit for bit.

        Exact rather than approximate, with one exception: a batch
        rearranges numbers that level 1 already verified, so any
        difference here is a difference in the arranging -- a window
        offset, a split boundary, a transposed index -- and none of those
        has a tolerance. The exception is the graph's edge weights, which
        the batch carries verbatim from the fitted state and which
        therefore carry the identical one-ULP tolerance they do at level
        1.

        ``elementary_indices`` is allowed to be missing. The original
        passed it into every batch and never read it; the refactored
        signature does not declare it, and every other tensor matching
        is what proves the omission changes nothing.
        """
        _, sources = replay
        report = compare_tensors(
            first_batches(sources),
            load_golden(FIXTURE_NAME),
            allow_missing=("elementary_indices",),
        )
        assert report.passed, report.summary()

    def test_the_static_inputs_are_not_collated_per_sample(self, replay) -> None:
        """
        The graph is carried once, not once per row.

        This is defect 4. The adjacency does not vary by scenario, so
        collating it into every sample multiplies its memory by the batch
        size and its host-to-device transfer by the number of batches.
        The numbers would be identical either way, which is exactly why
        this needs its own assertion.
        """
        _, sources = replay
        source = sources["train"]
        batch = next(iter(source.batches()))
        assert set(source.static) == {
            "trade_features",
            "adjacency_indices",
            "adjacency_values",
            "adjacency_shape",
            "target_indices",
        }
        assert set(batch) == {"pnl_history", "target"}

    def test_the_window_count_matches_the_capture(self, replay) -> None:
        """
        The same windows, derived rather than replayed.

        Only the split's row membership is taken from the fixture; which
        windows that implies is the framework's own calculation, and a
        window length off by one would show up here before it showed up
        as a mysterious tensor mismatch.
        """
        _, sources = replay
        expected = json.loads((FIXTURE / "level2_tensors" / "split_window_starts.json").read_text())
        for split, source in sources.items():
            assert source.n_samples == len(expected[split]), split


class TestLevel4TrainingCurve:
    """Several epochs of training, compared at a widened tolerance."""

    def test_the_loss_curve_matches(self, replay) -> None:
        """
        Five epochs of Adam over the same batches reach the same losses.

        Written as an explicit loop rather than run through the
        framework's trainer, matching how the fixture was captured. The
        framework's callbacks -- early stopping, learning-rate reduction,
        best-weight restoration -- are its own design and comparing them
        against the original's would be testing the wrong thing. What is
        compared here is the model and the data, which is what moved.
        """
        prepared, sources = replay

        # Dropout off and the starting weights reloaded, matching how the
        # capture was taken. With dropout active the curve would record
        # the random-number sequence as much as the model, and no refactor
        # could reproduce it without reproducing that sequence exactly.
        model = HybridGnnRnn(HybridModelSpec(units=UNITS, dropout=0.0), prepared.signature)
        model.load_state_dict(
            torch.load(FIXTURE / "level3_forward" / "state_dict.pt", weights_only=True)
        )
        # No materialise step: every parameter already exists, which is
        # the fix for defect 6. The capture script had to materialise by
        # hand before building its optimiser, and an optimiser built one
        # line earlier would have tracked nothing.
        optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_function = torch.nn.L1Loss()

        curve: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        for _ in range(N_EPOCHS):
            model.train()
            curve["train_loss"].append(
                float(
                    np.mean(
                        [
                            _step(model, optimiser, loss_function, inputs, target)
                            for inputs, target in _batches(sources["train"])
                        ]
                    )
                )
            )
            model.eval()
            with torch.no_grad():
                curve["val_loss"].append(
                    float(
                        np.mean(
                            [
                                float(loss_function(model(**inputs), target))
                                for inputs, target in _batches(sources["validation"])
                            ]
                        )
                    )
                )

        report = compare_curve(curve, load_golden(FIXTURE_NAME))
        assert report.passed, report.summary()


def _batches(source):
    """
    Yield one pass over a source as forward-ready inputs and targets.

    Parameters
    ----------
    source
        The batch source.

    Yields
    ------
    tuple
        The keyword arguments for the forward pass, and the target.
    """
    static = {name: torch.as_tensor(np.asarray(value)) for name, value in source.static.items()}
    for batch in source.batches():
        inputs = {
            name: torch.as_tensor(np.asarray(value))
            for name, value in batch.items()
            if name != "target"
        }
        yield inputs | static, torch.as_tensor(np.asarray(batch["target"]))


def _step(model, optimiser, loss_function, inputs, target) -> float:
    """
    Take one optimiser step and return the loss.

    Parameters
    ----------
    model
        The network.
    optimiser
        The update rule.
    loss_function
        What to minimise.
    inputs
        Forward-pass keyword arguments.
    target
        The batch's target.

    Returns
    -------
    float
        The loss before the step.
    """
    optimiser.zero_grad(set_to_none=True)
    loss = loss_function(model(**inputs), target)
    loss.backward()
    optimiser.step()
    return float(loss.detach())
