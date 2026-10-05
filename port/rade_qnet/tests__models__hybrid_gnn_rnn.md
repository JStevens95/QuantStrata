# `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn`

9 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 1 | 51 | `5383ca55a0c3650b` |
| 2 | `test_data.py` | 491 | 19791 | `e630fd59a0014469` |
| 3 | `test_model.py` | 300 | 11886 | `1a0289dc02f4c4e4` |
| 4 | `test_parity.py` | 355 | 13452 | `0937aea9a32f8f33` |
| 5 | `test_register.py` | 159 | 5978 | `b193105cb59734a7` |
| 6 | `test_reports.py` | 193 | 7129 | `0a0031459e0895d1` |
| 7 | `test_state.py` | 252 | 10469 | `501cd328cd45e06b` |
| 8 | `test_universe.py` | 137 | 4883 | `9cb3857df194e430` |
| 9 | `test_visuals.py` | 150 | 4792 | `358c1bc5be3d1ee1` |

---

## 1. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/__init__.py`

51 bytes · SHA-256 `5383ca55a0c3650b`

```python
"""Mirrors ``rade_qnet.models.hybrid_gnn_rnn``."""
```

---

## 2. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_data.py`

19791 bytes · SHA-256 `e630fd59a0014469`

```python
"""
Tests for the flagship's data build.

Two groups. The first checks the build's own invariants against a small
synthetic cluster -- which axis each stage fits on, what the signature
declares, what the index arrays point at. The second replays the golden
fixture and asserts that the fitted state matches the implementation this
one replaces, element for element.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.data import (
    ChronologicalSplitSpec,
    ExplicitSplitSpec,
    ModelSourceSpec,
    ReductionSpec,
    SequenceSpec,
    TransformsSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.data import (
    HybridDataModule,
    _merge_attributes,
)
from tranql.models.rade.rade_qnet.rade_qnet.testkit.parity import compare_state, load_golden

from ...locations import GOLDEN_ROOT

FIXTURE = GOLDEN_ROOT
HYBRID_FIXTURE = FIXTURE / "hybrid_gnn_rnn"

SEQUENCE_LENGTH = 4
N_SCENARIOS = 120


@pytest.fixture
def cluster(tmp_path: Path) -> Path:
    """
    Write a small synthetic cluster to disk.

    Two underlyings and two product types, five instruments each driven by
    two factors, so basis selection has something real to reduce -- a book
    whose instruments were already independent would make every reduction
    test vacuous.

    Parameters
    ----------
    tmp_path
        Pytest's temporary directory.

    Returns
    -------
    pathlib.Path
        The cluster directory.
    """
    generator = np.random.default_rng(0)
    directory = tmp_path / "cluster"
    directory.mkdir()

    elementary_ids, blocks = [], []
    for underlying in ("EURUSD", "GBPUSD"):
        for product in ("forward", "vanilla_option"):
            drivers = generator.normal(size=(N_SCENARIOS, 2))
            blocks.append(drivers @ generator.normal(size=(2, 5)))
            elementary_ids += [f"{underlying}|{product}|{number:03d}" for number in range(5)]
    elementary = np.hstack(blocks)
    target_ids = ["EURUSD|target|900", "GBPUSD|target|901"]
    target = generator.normal(size=(N_SCENARIOS, 2))

    np.save(directory / "elementary_pnl.npy", elementary)
    np.save(directory / "target_pnl.npy", target)
    (directory / "universe.json").write_text(
        json.dumps({"elementary_ids": elementary_ids, "target_ids": target_ids})
    )

    def attributes(identifiers: list[str]) -> dict[str, list]:
        count = len(identifiers)
        return {
            "trade_id": identifiers,
            "moneyness": list(generator.uniform(0.9, 1.1, count)),
            "yrs_to_maturity": list(generator.uniform(0.1, 2.0, count)),
            "delta": list(generator.uniform(0.2, 0.8, count)),
            "vega": list(generator.uniform(5.0, 20.0, count)),
            "product_type": [name.split("|")[1] for name in identifiers],
            "product_subtype": ["standard"] * count,
            "trade_type": ["target" if "target" in name else "elementary" for name in identifiers],
            "underlying_risk_factors": [[name.split("|")[0]] for name in identifiers],
        }

    (directory / "elementary_attributes.json").write_text(json.dumps(attributes(elementary_ids)))
    (directory / "target_attributes.json").write_text(json.dumps(attributes(target_ids)))
    return directory


def make_spec(directory: Path, **overrides) -> ModelSourceSpec:
    """
    Build a source spec pointing at a cluster.

    Parameters
    ----------
    directory
        The cluster directory.
    **overrides
        Fields to replace on the spec.

    Returns
    -------
    ModelSourceSpec
        The specification.
    """
    defaults = {
        "params": {"directory": str(directory)},
        "split": ChronologicalSplitSpec(validation_fraction=0.2, test_fraction=0.2),
        "transforms": TransformsSpec(
            reduction=ReductionSpec(method="basis_selection"),
            sequence=SequenceSpec(length=SEQUENCE_LENGTH),
        ),
    }
    return ModelSourceSpec(**(defaults | overrides))


@pytest.fixture
def built(cluster: Path):
    """Run every stage and return the pieces, so tests can assert on any of them."""
    module, spec = HybridDataModule(), make_spec(cluster)
    raw = module.load(spec)
    splits = module.split(raw, spec)
    state = module.fit_state(raw, spec, train_indices=splits.train)
    features, target = module.transform(raw, state)
    signature = module.signature(spec, features=features, state=state)
    return module, spec, raw, splits, state, features, target, signature


class TestLoading:
    """Reading a cluster off disk."""

    def test_the_shapes_line_up(self, built) -> None:
        """P&L, universe and attributes all describe the same instruments."""
        _, _, raw, *_ = built
        assert raw.elementary_pnl.shape == (N_SCENARIOS, 20)
        assert raw.target_pnl.shape == (N_SCENARIOS, 2)
        assert len(raw.elementary_ids) == 20

    def test_a_missing_directory_is_named(self, tmp_path: Path) -> None:
        """
        The error has to say which folder and which file.

        A bare `FileNotFoundError` from inside a loader is the least useful
        message a user can get when a job set of forty clusters has one bad
        path.
        """
        with pytest.raises(ContractError, match="not a hybrid cluster directory"):
            HybridDataModule().load(make_spec(tmp_path / "absent"))

    def test_an_absent_directory_setting_is_named(self) -> None:
        """A spec with no directory should say so, not fail on `None / 'x'`."""
        with pytest.raises(ContractError, match="needs a 'directory'"):
            HybridDataModule().load(ModelSourceSpec(params={}))

    def test_misaligned_scenario_counts_are_refused(self, cluster: Path) -> None:
        """
        Elementary and target P&L are aligned row by row.

        A mismatch means every target is attributed to the wrong day, which
        trains to a plausible loss and is undetectable downstream.
        """
        np.save(cluster / "target_pnl.npy", np.zeros((N_SCENARIOS - 5, 2)))
        with pytest.raises(ContractError, match="scenario"):
            HybridDataModule().load(make_spec(cluster))

    def test_a_universe_that_disagrees_with_the_pnl_is_refused(self, cluster: Path) -> None:
        """A column count that does not match the identifier count misnames every column."""
        universe = json.loads((cluster / "universe.json").read_text())
        universe["elementary_ids"] = universe["elementary_ids"][:-1]
        (cluster / "universe.json").write_text(json.dumps(universe))
        with pytest.raises(ContractError, match="column"):
            HybridDataModule().load(make_spec(cluster))


class TestWhichAxisIsFittedOnWhat:
    """
    The distinction the whole module turns on.

    The scenario axis must see training rows only; the entity axis must see
    the whole universe. Getting either backwards is a bug that trains fine.
    """

    def test_the_scalers_see_training_rows_only(self, built) -> None:
        """
        Fitted on the training split, not on the full history.

        Checked by refitting on the training rows directly and comparing:
        a scaler fitted on everything would differ, and the difference is
        exactly the amount by which the backtest would flatter itself.
        """
        _, _, raw, splits, state, *_ = built
        expected = raw.elementary_pnl[splits.train, :].mean(axis=0)
        assert state.feature_scaler.centre == pytest.approx(expected)
        assert state.feature_scaler.centre != pytest.approx(raw.elementary_pnl.mean(axis=0))

    def test_the_encoder_sees_every_instrument(self, built) -> None:
        """
        The entity axis is not restricted, and that is correct.

        Which instruments exist and what their attributes are is known
        before any P&L is observed. The graph spans the selected basis plus
        every target, whichever split their P&L happens to land in.
        """
        _, _, _, _, state, *_ = built
        assert state.graph.n_nodes == len(state.selected_basis) + state.universe.n_targets

    def test_the_basis_defaults_to_training_rows(self, built) -> None:
        """
        Defect 9 is fixed by default, not preserved by default.

        A compatibility flag whose default reproduces the defect would mean
        every run that forgot to set it leaked.
        """
        _, spec, *_ = built
        assert spec.transforms.reduction.fit_on == "train"

    def test_fitting_the_basis_on_all_rows_changes_the_answer(self, cluster: Path) -> None:
        """
        The flag must actually do something, or the parity tests prove nothing.

        It is also the measure of the leak: the two bases differ, so a
        backtest run with `all` selected its features partly from the data
        it was scored on.
        """
        module = HybridDataModule()
        bases = {}
        for fit_on in ("train", "all"):
            spec = make_spec(
                cluster,
                transforms=TransformsSpec(
                    reduction=ReductionSpec(method="basis_selection", fit_on=fit_on),
                    sequence=SequenceSpec(length=SEQUENCE_LENGTH),
                ),
            )
            raw = module.load(spec)
            splits = module.split(raw, spec)
            bases[fit_on] = module.fit_state(raw, spec, train_indices=splits.train).selected_basis
        assert bases["train"] != bases["all"]

    def test_the_leak_is_recorded_in_the_lineage(self, cluster: Path) -> None:
        """
        A run that leaked must say so where someone will read it.

        Six months later, the only thing distinguishing a leaked backtest
        from a sound one is this annotation.
        """
        spec = make_spec(
            cluster,
            transforms=TransformsSpec(
                reduction=ReductionSpec(method="basis_selection", fit_on="all")
            ),
        )
        assert "leakage_warning" in HybridDataModule().lineage_notes(spec)


class TestTheIndexArrays:
    """The subtlest trap in the build."""

    def test_they_are_computed_after_reduction(self, built) -> None:
        """
        ``0..n_e`` and ``n_e..n_e + n_t`` over the *selected* basis.

        Carrying the pre-reduction indices forward produces arrays of
        plausible length holding plausible values that address the wrong
        rows of the encoding. The model trains, the loss falls, and the
        predictions belong to different instruments.
        """
        _, _, raw, _, state, *_ = built
        n_selected = len(state.selected_basis)
        assert n_selected < len(raw.elementary_ids)
        assert np.array_equal(state.elementary_indices, np.arange(n_selected))
        assert np.array_equal(state.target_indices, np.arange(n_selected, n_selected + 2))

    def test_they_address_the_encoding_they_describe(self, built) -> None:
        """
        The indices must cover the attribute matrix exactly.

        An off-by-one here silently drops the last target from every
        prediction.
        """
        _, _, raw, _, state, *_ = built
        attributes = _merge_attributes(
            raw,
            selected_basis=state.selected_basis,
            elementary_ids=raw.elementary_ids,
        )
        rows = state.encoder.transform(attributes).features.shape[0]
        assert rows == len(state.elementary_indices) + len(state.target_indices)
        assert int(state.target_indices[-1]) == rows - 1


class TestTransformAndSignature:
    """What the model is handed."""

    def test_the_features_are_narrowed_to_the_basis(self, built) -> None:
        """One column per selected instrument, over the full scenario axis."""
        _, _, _, _, state, features, _, _ = built
        assert features.shape == (N_SCENARIOS, len(state.selected_basis))

    def test_held_out_rows_are_transformed_too(self, built) -> None:
        """
        Every row is scaled, not only the training ones.

        Not a leak: the scaler's parameters came from training rows alone,
        and a held-out row scaled differently would reach the model on a
        scale it never trained on.
        """
        _, _, raw, *_, features, _, _ = built
        assert features.shape[0] == raw.elementary_pnl.shape[0]

    def test_the_static_set_holds_the_graph(self, built) -> None:
        """
        Declaring the graph static is what retires the baseline's collation.

        There it was merged into every sample and compared across the batch
        to recover the one copy -- every batch, every epoch, to establish
        something true by construction.
        """
        *_, signature = built
        assert set(signature.static) == {
            "trade_features",
            "adjacency_indices",
            "adjacency_values",
            "adjacency_shape",
            "target_indices",
        }

    def test_elementary_indices_are_absent_from_the_signature(self, built) -> None:
        """
        A redundant index array is an invitation for the two to disagree.

        The elementary block always occupies rows ``0..n_e``, so the array
        is an `arange` of a length the model already has.
        """
        *_, signature = built
        assert "elementary_indices" not in signature.static

    def test_the_dynamic_shape_matches_the_sequence_length(self, built) -> None:
        """The window the recurrence consumes, with a free batch dimension."""
        _, _, _, _, state, *_, signature = built
        assert signature.dynamic["pnl_history"].shape == (
            None,
            SEQUENCE_LENGTH,
            len(state.selected_basis),
        )

    def test_the_feature_names_are_post_reduction(self, built) -> None:
        """
        The names must describe the columns that survived reduction.

        Reporting pre-reduction names against post-reduction columns
        mislabels every attribution figure in the run report.
        """
        module, _, raw, _, state, features, _, _ = built
        names = module.feature_names(raw, state)
        assert names == state.selected_basis
        assert len(names) == features.shape[1]


class TestStatelessness:
    """The module must not remember a previous build."""

    def test_building_twice_gives_the_same_answer(self, cluster: Path) -> None:
        """
        No stage may stash anything on ``self``.

        If one did, the second build on an instance would silently reuse the
        first build's state -- and a cached dataset would carry a signature
        fitted to data it was not built from.
        """
        module, spec = HybridDataModule(), make_spec(cluster)
        results = []
        for _ in range(2):
            raw = module.load(spec)
            splits = module.split(raw, spec)
            state = module.fit_state(raw, spec, train_indices=splits.train)
            results.append((state.selected_basis, module.transform(raw, state)[0]))
        assert results[0][0] == results[1][0]
        assert np.array_equal(results[0][1], results[1][1])


@pytest.mark.usefixtures("requires_golden")
class TestParityAgainstTheBaseline:
    """
    Level 1: the fitted state, compared against the recorded baseline.

    The compatibility flags are set *here* and only here. Each reproduces a
    behaviour of the implementation being replaced, and each defaults to the
    correct one, so a production configuration that omits them gets the
    fixed behaviour.
    """

    @pytest.fixture
    def replay(self):
        """Rebuild the fixture's cluster through the refactored module."""
        rows = json.loads((HYBRID_FIXTURE / "level2_tensors" / "split_indices.json").read_text())
        spec = ModelSourceSpec(
            params={
                "directory": str(HYBRID_FIXTURE / "input"),
                # Compatibility flag: the baseline's encoder fitted and
                # transformed in float32. See AttributeEncoderSpec.
                "encoder": {"numeric_precision": "float32"},
                # Its twin, for the graph's row normalisation.
                "graph": {"n_neighbours": 5, "precision": "float32"},
            },
            # The baseline's split, replayed rather than reproduced, so that
            # a difference in splitting cannot masquerade as a difference in
            # fitting.
            split=ExplicitSplitSpec(
                train=tuple(rows["train"]),
                validation=tuple(rows["validation"]),
                test=tuple(rows["test"]),
            ),
            transforms=TransformsSpec(
                # Compatibility flag: defect 9. The baseline selected the
                # basis over the full scaled history.
                reduction=ReductionSpec(method="basis_selection", fit_on="all"),
                sequence=SequenceSpec(length=SEQUENCE_LENGTH),
            ),
        )
        module = HybridDataModule()
        raw = module.load(spec)
        splits = module.split(raw, spec)
        state = module.fit_state(raw, spec, train_indices=splits.train)
        return raw, state

    def test_the_fitted_state_matches(self, replay) -> None:
        """
        Every artifact the baseline produced, reproduced.

        Twelve of the thirteen comparisons are exact. The thirteenth,
        ``adjacency_values``, carries a documented one-ULP tolerance -- see
        ``ADJACENCY_VALUE_ATOL`` for why, and
        ``TestTheGraphIsRobust`` for the evidence that it is safe.
        """
        raw, state = replay
        attributes = _merge_attributes(
            raw, selected_basis=state.selected_basis, elementary_ids=raw.elementary_ids
        )
        report = compare_state(
            {
                "scaler_mean": state.feature_scaler.centre,
                "scaler_scale": state.feature_scaler.scale,
                "target_scaler_mean": state.target_scaler.centre,
                "target_scaler_scale": state.target_scaler.scale,
                "selected_basis": list(state.selected_basis),
                "combined_features": state.encoder.transform(attributes).features,
                "adjacency_indices": state.graph.indices,
                "adjacency_values": state.graph.values,
                "adjacency_shape": np.array(state.graph.dense_shape, dtype=np.int64),
                "elementary_idx": state.elementary_indices,
                "target_idx": state.target_indices,
                "universe": {
                    "elementary_ids": list(state.universe.elementary_ids),
                    "target_ids": list(state.universe.target_ids),
                },
            },
            load_golden("hybrid_gnn_rnn", root=GOLDEN_ROOT),
        )
        assert report.passed, str(report)

    def test_the_window_starts_are_derived_correctly(self, replay) -> None:
        """
        Window starts must follow from row membership.

        The framework derives them; the baseline computed them separately.
        Worth checking rather than assuming, because the two sets differ --
        a split's last few rows cannot begin a window -- and a refactor
        that confused them would fit every scaler on the wrong rows. It did,
        once.
        """
        expected = json.loads(
            (HYBRID_FIXTURE / "level2_tensors" / "split_window_starts.json").read_text()
        )
        rows = json.loads((HYBRID_FIXTURE / "level2_tensors" / "split_indices.json").read_text())
        for split, starts in expected.items():
            # A window of length L starting at row r needs rows r..r+L-1, so
            # the valid starts are the split's rows minus its last L-1.
            assert starts == rows[split][: len(rows[split]) - SEQUENCE_LENGTH + 1]
```

---

## 3. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_model.py`

11886 bytes · SHA-256 `1a0289dc02f4c4e4`

```python
"""
Tests for the assembled network, including parity level 3.

The layer tests check each block in isolation. These check that they are
wired together in the right order with the right widths, that the model
holds no state beyond its parameters, and -- the one that matters most --
that the whole network reproduces the original bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tranql.models.rade.rade_qnet.rade_qnet.core.contract.signature import (
    InputSignature,
    TensorSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.model import HybridGnnRnn
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.spec import HybridModelSpec
from tranql.models.rade.rade_qnet.rade_qnet.testkit.parity import compare_forward, load_golden

from ...locations import GOLDEN_ROOT

#: The golden fixture captured from the original implementation.
FIXTURE = "hybrid_gnn_rnn"

#: Shapes of the captured batch, so a signature can be built without the
#: data module. The parity test asserts these against the fixture rather
#: than trusting them.
N_NODES, N_ATTRIBUTES = 11, 13
N_ELEMENTARY, SEQUENCE, N_TARGETS = 8, 4, 3
N_EDGES, BATCH = 66, 16


def signature(
    *,
    n_nodes: int = N_NODES,
    n_attributes: int = N_ATTRIBUTES,
    n_elementary: int = N_ELEMENTARY,
    n_targets: int = N_TARGETS,
) -> InputSignature:
    """Build a signature matching the fixture's batch."""
    return InputSignature(
        dynamic={"pnl_history": TensorSpec(shape=(None, SEQUENCE, n_elementary), dtype="float32")},
        static={
            "trade_features": TensorSpec(shape=(n_nodes, n_attributes), dtype="float32"),
            "adjacency_indices": TensorSpec(shape=(N_EDGES, 2), dtype="int64"),
            "adjacency_values": TensorSpec(shape=(N_EDGES,), dtype="float32"),
            "adjacency_shape": TensorSpec(shape=(2,), dtype="int64"),
            "target_indices": TensorSpec(shape=(n_targets,), dtype="int64"),
        },
        target=TensorSpec(shape=(None, n_targets), dtype="float32"),
    )


@pytest.fixture
def batch() -> dict[str, torch.Tensor]:
    """Return the captured training batch, keyed for the forward pass."""
    golden = load_golden(FIXTURE, root=GOLDEN_ROOT)
    arrays = golden.arrays("level2_tensors/train_batch_000.npz")
    renamed = {"adjacency_dense_shape": "adjacency_shape"}
    # `elementary_indices` is dropped, not renamed. The original passed it
    # into every batch and never read it; the refactored signature does not
    # declare it, and the parity test below is what proves the omission
    # changes no number. Keeping it here would hide that.
    discarded = {"target", "elementary_indices"}
    return {
        renamed.get(name, name): torch.tensor(value)
        for name, value in arrays.items()
        if name not in discarded
    }


@pytest.fixture
def model() -> HybridGnnRnn:
    """Return the network at the original's production width."""
    return HybridGnnRnn(HybridModelSpec(units=16), signature())


class TestConstruction:
    """What the signature determines."""

    def test_every_width_comes_from_the_signature(self) -> None:
        """
        Nothing is assumed about the data's shape.

        A width hard-coded here would make the model unusable on a
        cluster of a different size, which is the single most common
        thing to vary across a job set.
        """
        network = HybridGnnRnn(
            HybridModelSpec(units=16),
            signature(n_nodes=20, n_attributes=9, n_elementary=5, n_targets=4),
        )
        assert network.gnn_block.gnn_layers[0].fusion_dense.in_features == 3 * 9
        assert network.rnn_block.rnn.input_size == 5
        assert network.projection_layer._baseline_kernels.shape[0] == 4

    def test_the_model_is_fully_parameterised_before_any_forward_pass(
        self, model: HybridGnnRnn
    ) -> None:
        """
        Defect 6, at the level of the whole network.

        An optimiser constructed over a model with lazy parameters tracks
        only the eager ones. The lazy ones then materialise on the first
        forward call and are never updated -- and the loss still falls,
        through the layers that were tracked, so nothing reports a fault.
        """
        assert all(
            not isinstance(parameter, torch.nn.UninitializedParameter)
            for parameter in model.parameters()
        )

    def test_a_missing_static_input_is_refused(self) -> None:
        """
        With a message naming what is missing and what was declared.

        The alternative is a ``KeyError`` deep in a forward pass, which
        names the key but not which side of the contract was wrong.
        """
        declared = signature()
        partial = InputSignature(
            dynamic=declared.dynamic,
            static={
                name: spec for name, spec in declared.static.items() if name != "trade_features"
            },
            target=declared.target,
        )
        with pytest.raises(ContractError, match="trade_features"):
            HybridGnnRnn(HybridModelSpec(units=16), partial)

    def test_a_variable_width_is_refused(self) -> None:
        """
        Only the batch axis may vary.

        A variable feature axis is not something to default around: it
        means the data module and the network disagree about what is
        fixed, and guessing would build a layer of some arbitrary size.
        """
        declared = signature()
        vague = InputSignature(
            dynamic={"pnl_history": TensorSpec(shape=(None, SEQUENCE, None), dtype="float32")},
            static=declared.static,
            target=declared.target,
        )
        with pytest.raises(ContractError, match="variable"):
            HybridGnnRnn(HybridModelSpec(units=16), vague)


@pytest.mark.usefixtures("requires_golden")
class TestForward:
    """What the assembled network computes."""

    def test_one_prediction_per_target_per_sample(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """The output is the shape the target declares."""
        model.eval()
        with torch.no_grad():
            assert model(**batch).shape == (BATCH, N_TARGETS)

    def test_every_parameter_receives_a_gradient(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        All 62 tensors are connected to the output.

        Across five blocks this is the check that a whole block has not
        been constructed, counted in the parameter total, and then left
        out of the forward pass.
        """
        model(**batch).sum().backward()
        unused = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        assert not unused

    def test_both_streams_reach_the_prediction(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        Perturbing the attributes or the history both move the output.

        A hybrid model in which one stream is disconnected still trains,
        still scores reasonably, and is not the model anyone signed off.
        """
        model.eval()
        with torch.no_grad():
            base = model(**batch)
            attributes = dict(batch)
            attributes["trade_features"] = attributes["trade_features"] + 1.0
            history = dict(batch)
            history["pnl_history"] = history["pnl_history"] + 1.0
            assert not torch.allclose(base, model(**attributes))
            assert not torch.allclose(base, model(**history))


@pytest.mark.usefixtures("requires_golden")
class TestNoHiddenState:
    """The model holds parameters and nothing else."""

    def test_two_forward_passes_agree(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        Calling twice gives the same answer.

        The original cached its graph embedding on the module and
        invalidated the cache on a mode change but not on a parameter
        change. A model that remembers anything between calls can return
        a stale number that looks entirely plausible.
        """
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(**batch), model(**batch))

    def test_nothing_is_stashed_on_the_module(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        The attribute set is unchanged by a forward pass.

        Model-held mutable state is how two jobs sharing a process see
        each other's results, and it is invisible until the day two jobs
        share a process.
        """
        model.eval()
        before = set(vars(model))
        with torch.no_grad():
            model(**batch)
        assert set(vars(model)) == before

    def test_the_precomputed_path_matches_the_ordinary_one(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        The capability replaces the cache without changing the answer.

        Reusing the graph embedding across an evaluation pass is a large
        saving -- the graph stream is recomputed identically for every
        batch otherwise -- but only if it is the same computation.
        """
        static = {name: value for name, value in batch.items() if name != "pnl_history"}
        model.eval()
        with torch.no_grad():
            direct = model(**batch)
            reused = model.forward_with_precomputed(batch, model.precompute(static))
        torch.testing.assert_close(direct, reused)

    def test_the_model_declares_that_it_handles_unseen_entities(self, model: HybridGnnRnn) -> None:
        """
        The capability is declared, and the output head backs it up.

        Declaring it without the head's borrowing machinery would let an
        inference pipeline hand the model a trade it cannot price.
        """
        assert model.supports_unseen_entities


@pytest.mark.usefixtures("requires_golden")
class TestParityAgainstTheBaseline:
    """Level 3: the forward pass, against the captured original."""

    def test_the_parameter_tensors_match_by_name_and_shape(self, model: HybridGnnRnn) -> None:
        """
        All 62 of them.

        Names as well as shapes, because the saved weights are restored
        by name: a port whose shapes matched but whose names differed
        would build a model that cannot load its own predecessor's
        checkpoint, and the failure would surface only on reload.
        """
        golden = load_golden(FIXTURE, root=GOLDEN_ROOT)
        expected = torch.load(golden.directory / "level3_forward/state_dict.pt", weights_only=True)
        produced = {name: tuple(t.shape) for name, t in model.state_dict().items()}
        assert produced == {name: tuple(t.shape) for name, t in expected.items()}

    def test_the_forward_output_matches(
        self, model: HybridGnnRnn, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        The whole network, on the captured batch, under the original's weights.

        This is the test the entire port exists to pass. It is checked
        under ``eval`` and ``no_grad``, because the fixture was captured
        that way -- dropout active would make the comparison a record of
        the global random state rather than of the model.
        """
        golden = load_golden(FIXTURE, root=GOLDEN_ROOT)
        model.load_state_dict(
            torch.load(golden.directory / "level3_forward/state_dict.pt", weights_only=True)
        )
        model.eval()
        with torch.no_grad():
            output = model(**batch).numpy()

        report = compare_forward(np.asarray(output, dtype=np.float64), golden)
        assert report.passed, report.describe()
```

---

## 4. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_parity.py`

13452 bytes · SHA-256 `0937aea9a32f8f33`

```python
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

import numpy as np
import pytest
import torch

from tranql.models.rade.rade_qnet.rade_qnet.core.spec.data import (
    ExplicitSplitSpec,
    LoaderSpec,
    ModelSourceSpec,
    ReductionSpec,
    SequenceSpec,
    TransformsSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.data import HybridDataModule
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.model import HybridGnnRnn
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.spec import HybridModelSpec
from tranql.models.rade.rade_qnet.rade_qnet.sources.batching.dataset import sources_for
from tranql.models.rade.rade_qnet.rade_qnet.testkit.parity import (
    compare_curve,
    compare_tensors,
    load_golden,
)

from ...locations import GOLDEN_ROOT

#: The golden fixture captured from the original implementation.
FIXTURE_NAME = "hybrid_gnn_rnn"
FIXTURE = GOLDEN_ROOT / FIXTURE_NAME

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


@pytest.mark.usefixtures("requires_golden")
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
            load_golden(FIXTURE_NAME, root=GOLDEN_ROOT),
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


@pytest.mark.usefixtures("requires_golden")
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

        report = compare_curve(curve, load_golden(FIXTURE_NAME, root=GOLDEN_ROOT))
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
```

---

## 5. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_register.py`

5978 bytes · SHA-256 `b193105cb59734a7`

```python
"""
Tests for the model's framework declaration.

These check the wiring rather than the mathematics: that the name in a
specification resolves to this model, that the engine and the reports the
declaration names are actually registered by the time anything asks for
them, and that the definition builds its parts from the spec alone.

Registration by accident is the failure worth naming. A declaration whose
subject happens to be registered -- because some other import pulled it in
-- passes every test run from a session that imported everything, and
fails the first real run that imports only what it needs.
"""

from __future__ import annotations

import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import (
    ENGINES,
    MODELS,
    REPORTS,
    get_model,
)
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.run import parse_run_spec
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.data import HybridDataModule
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.model import HybridGnnRnn
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.pipelines.train import (
    HYBRID_REPORTS,
    HybridTrainPipeline,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.register import HybridGnnRnnModel
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.spec import (
    HybridDataSpec,
    HybridModelSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.state import HybridState
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.train import TrainPipeline

from ...locations import GOLDEN_ROOT
from .test_model import signature

FIXTURE = GOLDEN_ROOT / "hybrid_gnn_rnn" / "input"


def run_spec(**report_overrides: object):
    """
    Build a minimal validated run specification naming the model.

    Parameters
    ----------
    **report_overrides
        Fields to set on the reports block.

    Returns
    -------
    SupervisedRunSpec
        The validated specification.
    """
    return parse_run_spec(
        {
            "task": "supervised",
            "model": {"name": "hybrid_gnn_rnn", "params": {"units": 8}},
            "source": {"kind": "model", "params": {"directory": str(FIXTURE)}},
            "training": {"engine": "torch"},
            "reports": dict(report_overrides) or {},
        }
    )


class TestRegistration:
    """What importing the model package makes resolvable."""

    def test_the_model_resolves_by_the_name_a_spec_uses(self) -> None:
        """
        A specification names a string; this is what turns it into a class.

        Resolved through the registry rather than imported, because that
        is the path a real run takes.
        """
        assert "hybrid_gnn_rnn" in MODELS
        assert get_model("hybrid_gnn_rnn") is HybridGnnRnnModel

    def test_the_declared_engine_is_registered(self) -> None:
        """
        The decorator declares ``torch``; importing the model provides it.

        A declaration whose subject may or may not be registered,
        depending on what else the process happened to import, is the
        classic source of "no engine named 'torch'" from a configuration
        that is perfectly correct.
        """
        assert HybridGnnRnnModel.component_engine == "torch"
        assert "torch" in ENGINES

    def test_the_model_report_is_registered(self) -> None:
        """
        For the same reason, and by the same mechanism.

        The train pipeline adds ``hybrid_graph`` to every run, so a name
        that resolved only when something else had imported the report
        module would fail the run at its first stage.
        """
        assert "hybrid_graph" in REPORTS
        assert all(name in REPORTS for name in HYBRID_REPORTS)

    def test_the_declared_types_are_the_real_ones(self) -> None:
        """
        The specs, state and pipelines a reader would look for.

        These are the only place the wiring is written down, so a wrong
        one here is a wrong one everywhere.
        """
        assert HybridGnnRnnModel.spec is HybridModelSpec
        assert HybridGnnRnnModel.data_spec is HybridDataSpec
        assert HybridGnnRnnModel.state_cls is HybridState
        assert HybridGnnRnnModel.pipelines["train"] is HybridTrainPipeline

    def test_the_pipeline_mapping_cannot_be_mutated(self) -> None:
        """
        It is class-level, so a write would affect every run in the process.

        Not a hypothetical: a test or a notebook rebinding a stage for
        one run would silently rebind it for all of them.
        """
        with pytest.raises(TypeError):
            HybridGnnRnnModel.pipelines["train"] = TrainPipeline  # type: ignore[index]


class TestDefinition:
    """What the definition builds."""

    def test_the_data_module_is_fresh_each_time(self) -> None:
        """
        Not held on the definition, so two jobs cannot reach the same object.

        The module carries no state -- every setting comes from the spec
        it is handed -- so a fresh one costs nothing and a shared one is
        a thing that could accumulate something.
        """
        definition = HybridGnnRnnModel()
        first = definition.data_module(run_spec())
        second = definition.data_module(run_spec())
        assert isinstance(first, HybridDataModule)
        assert first is not second

    def test_the_model_is_built_from_the_spec_and_signature_alone(self) -> None:
        """
        No data reaches the constructor.

        This is what makes a six-month-old bundle reloadable: the saved
        signature plus the saved weights are sufficient to rebuild the
        identical object, with no need to reproduce the dataset that
        originally shaped it.
        """
        spec = run_spec()
        built = HybridGnnRnnModel().build_model(spec, signature())
        assert isinstance(built, HybridGnnRnn)
        assert built.spec.units == 8
```

---

## 6. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_reports.py`

7129 bytes · SHA-256 `0a0031459e0895d1`

```python
"""Tests for the graph diagnostics report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.analysis.reports.base import ReportContext
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.data import (
    ModelSourceSpec,
    ReductionSpec,
    SequenceSpec,
    TransformsSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.data import HybridDataModule
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.features.graph import (
    SparseGraphState,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.reports import (
    GRAPH_FILENAME,
    HybridGraphReport,
)
from tranql.models.rade.rade_qnet.rade_qnet.testkit.fixtures import StandardisingState

from ...locations import GOLDEN_ROOT

FIXTURE = GOLDEN_ROOT / "hybrid_gnn_rnn" / "input"

#: Window length. Short, because this report never looks at a window.
SEQUENCE_LENGTH = 4


@pytest.fixture(scope="module")
def state():
    """Fit the fixture's cluster once and return the resulting state."""
    spec = ModelSourceSpec(
        params={"directory": str(FIXTURE), "graph": {"n_neighbours": 5}},
        transforms=TransformsSpec(
            reduction=ReductionSpec(method="basis_selection"),
            sequence=SequenceSpec(length=SEQUENCE_LENGTH),
        ),
    )
    module = HybridDataModule()
    raw = module.load(spec)
    splits = module.split(raw, spec)
    return module.fit_state(raw, spec, train_indices=splits.train)


def context(tmp_path: Path, bundle_state) -> ReportContext:
    """
    Build a report context around a state.

    Only the bundle's state is read, so the rest of the bundle is a
    placeholder -- which is itself worth noting: the report re-renders
    from a saved run without the data and without the weights.

    Parameters
    ----------
    tmp_path
        Where to write.
    bundle_state
        The fitted state to report on.

    Returns
    -------
    ReportContext
        The context.
    """

    class _Bundle:
        state = bundle_state

    return ReportContext(bundle=_Bundle(), directory=tmp_path)  # type: ignore[arg-type]


@pytest.mark.usefixtures("requires_golden")
class TestRendering:
    """What the report writes."""

    def test_a_page_and_three_figures(self, tmp_path: Path, state) -> None:
        """The Markdown page first, so a reader opening the list finds it."""
        paths = HybridGraphReport().render(context(tmp_path, state))
        assert paths[0].name == GRAPH_FILENAME
        assert len(paths) == 4
        assert all(path.is_file() for path in paths)

    def test_the_page_reports_the_graph_dimensions(self, tmp_path: Path, state) -> None:
        """
        Counted from the state rather than restated from the spec.

        A report that echoed the configuration would agree with it even
        when the graph did not.
        """
        HybridGraphReport().render(context(tmp_path, state))
        page = (tmp_path / GRAPH_FILENAME).read_text()
        assert f"| instruments | {state.graph.n_nodes} |" in page
        assert f"| edges (self-loops included) | {state.graph.n_edges} |" in page

    def test_the_quoted_neighbour_count_excludes_the_self_loop(self, tmp_path: Path, state) -> None:
        """
        The self-loop is excluded from the quoted count.

        "Five neighbours" meaning "four and itself" survives into a
        conversation with a desk, and is wrong there.
        """
        HybridGraphReport().render(context(tmp_path, state))
        page = (tmp_path / GRAPH_FILENAME).read_text()
        expected = int(np.median(state.graph.degrees() - 1))
        assert f"| neighbours per instrument (median) | {expected} |" in page

    def test_the_worst_covered_instruments_are_named(self, tmp_path: Path, state) -> None:
        """
        Named, not counted.

        The action this prompts is per instrument -- look at it, decide
        whether its attributes are wrong or whether it is genuinely an
        outlier -- and a count supports none of that.
        """
        HybridGraphReport().render(context(tmp_path, state))
        page = (tmp_path / GRAPH_FILENAME).read_text()
        assert "## Worst-covered instruments" in page
        universe = json.loads((FIXTURE / "universe.json").read_text())
        named = [name for name in universe["target_ids"] if f"`{name}`" in page]
        assert named

    def test_the_figures_are_linked_by_filename(self, tmp_path: Path, state) -> None:
        """
        So the page survives the directory being copied or served elsewhere.

        An absolute path would work on the machine that produced it and
        nowhere else, which is the opposite of what a report is for.
        """
        HybridGraphReport().render(context(tmp_path, state))
        page = (tmp_path / GRAPH_FILENAME).read_text()
        assert "![graph_degrees](graph_degrees.png)" in page
        assert str(tmp_path) not in page


@pytest.mark.usefixtures("requires_golden")
class TestCoverageFindings:
    """The two numbers that decide whether the graph is usable."""

    def test_a_healthy_graph_says_so(self, tmp_path: Path, state) -> None:
        """
        An explicit verdict, not a table the reader has to interpret.

        A report that only prints numbers leaves the judgement to
        whoever reads it, and the whole value here is that the person
        who knows the threshold is the person writing the report.
        """
        HybridGraphReport().render(context(tmp_path, state))
        page = (tmp_path / GRAPH_FILENAME).read_text()
        assert "Every instrument has at least one neighbour." in page

    def test_the_self_loop_does_not_count_as_coverage(self) -> None:
        """
        An isolated instrument scores zero, not one.

        Every node has a self-loop, and its weight says nothing about
        that node's relationship to the rest of the book. Counting it
        would give the instruments most in need of attention a perfect
        score.
        """
        only_self = SparseGraphState(
            indices=np.array([[0, 0], [1, 1]], dtype=np.int64),
            values=np.ones(2, dtype=np.float32),
            n_nodes=2,
            is_target=np.array([False, True]),
        )
        best = HybridGraphReport._best_weight_per_node(only_self)
        assert best.tolist() == [0.0, 0.0]


class TestSkipping:
    """What happens when the report cannot apply."""

    def test_another_model_s_state_is_skipped_with_a_reason(self, tmp_path: Path) -> None:
        """
        A deliberate skip, not an incidental one.

        The pipeline turns any exception into a skipped report, so an
        accidental ``AttributeError`` would also be survivable -- but it
        would leave a stack trace in a log instead of a sentence anyone
        can act on.
        """
        outcome = HybridGraphReport().render_safely(
            context(tmp_path, StandardisingState.fit(np.zeros((4, 2))))
        )
        assert not outcome.succeeded
        assert "HybridState" in str(outcome.skipped_reason)
```

---

## 7. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_state.py`

10469 bytes · SHA-256 `501cd328cd45e06b`

```python
"""
Tests for the flagship's fitted state.

Most of these defend against one failure mode: a state that loads
*partially* and produces a model that runs, trains, reports plausible
metrics, and is wrong. That was the original's behaviour with eleven loose
sidecar files, and it is the thing this module exists to make impossible.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import BundleError
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.features.encoder import (
    EntityEncoderState,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.features.graph import build_graph
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.spec import (
    AttributeEncoderSpec,
    GraphSpec,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.state import (
    HybridState,
    StandardScalerState,
    Universe,
)

#: Target P&L in currency, deliberately far from unit scale so that a
#: forgotten inverse transform cannot pass a tolerance by accident.
TARGET_SCALE = 50_000.0


@pytest.fixture
def attributes() -> Mapping[str, Sequence[Any]]:
    """Four instruments: two elementary, two target."""
    return {
        "trade_id": ["E1", "E2", "T1", "T2"],
        "moneyness": [0.95, 1.05, 0.98, 1.02],
        "yrs_to_maturity": [0.25, 1.00, 0.50, 1.50],
        "delta": [0.30, 0.55, 0.40, 0.60],
        "vega": [10.0, 15.0, 11.0, 18.0],
        "product_type": ["vanilla_option", "forward", "vanilla_option", "forward"],
        "product_subtype": ["european", "outright", "european", "outright"],
        "trade_type": ["elementary", "elementary", "target", "target"],
        "underlying_risk_factors": [["FX"], ["RATES"], ["FX", "RATES"], ["RATES"]],
    }


@pytest.fixture
def state(attributes: Mapping[str, Sequence[Any]]) -> HybridState:
    """Assemble a small but complete state."""
    generator = np.random.default_rng(0)
    encoder = EntityEncoderState.fit(attributes, spec=AttributeEncoderSpec())
    encoded = encoder.transform(attributes)
    return HybridState(
        feature_scaler=StandardScalerState.fit(generator.normal(size=(40, 2))),
        target_scaler=StandardScalerState.fit(generator.normal(scale=TARGET_SCALE, size=(40, 2))),
        selected_basis=("E2", "E1"),
        encoder=encoder,
        graph=build_graph(
            encoded,
            spec=GraphSpec(n_neighbours=2),
            is_target=np.array([False, False, True, True]),
        ),
        universe=Universe(elementary_ids=("E2", "E1"), target_ids=("T1", "T2")),
        elementary_indices=np.array([0, 1], dtype=np.int64),
        target_indices=np.array([2, 3], dtype=np.int64),
    )


class TestStandardScaler:
    """The hand-rolled replacement for a pickled scikit-learn scaler."""

    def test_the_statistics_are_held_in_float64(self) -> None:
        """
        A float32 mean over thousands of scenarios is measurably wrong.

        The network consumes float32, but the statistics it is standardised
        by should not be computed in it.
        """
        scaler = StandardScalerState.fit(np.ones((1000, 3), dtype=np.float32))
        assert scaler.centre.dtype == np.float64

    def test_the_transform_standardises(self) -> None:
        """Zero mean and unit scale, which is the stage's entire claim."""
        values = np.random.default_rng(0).normal(loc=7.0, scale=3.0, size=(200, 2))
        scaled = StandardScalerState.fit(values).transform(values)
        assert scaled.mean(axis=0) == pytest.approx(np.zeros(2), abs=1e-5)
        assert scaled.std(axis=0) == pytest.approx(np.ones(2), abs=1e-5)

    def test_the_inverse_round_trips(self) -> None:
        """
        Inverting a transform must return the input.

        The inverse is the only thing standing between a user and a metric
        in the wrong units, so it gets its own test rather than being
        covered incidentally.
        """
        values = np.random.default_rng(1).normal(loc=-4.0, scale=9.0, size=(50, 3))
        scaler = StandardScalerState.fit(values)
        assert scaler.inverse_transform(scaler.transform(values)) == pytest.approx(values, abs=1e-4)

    def test_a_constant_column_does_not_divide_by_zero(self) -> None:
        """
        A zero-variance column must not turn the feature matrix into NaN.

        It is ordinary: an instrument that did not trade, a flat risk
        factor.
        """
        values = np.column_stack([np.ones(20), np.arange(20.0)])
        scaled = StandardScalerState.fit(values).transform(values)
        assert np.isfinite(scaled).all()


class TestInverseTransformTargets:
    """The one method the base class refuses to give a default for."""

    def test_predictions_come_back_in_currency(self, state: HybridState) -> None:
        """
        The headline behaviour of the whole class.

        A prediction of zero in standardised space is the mean P&L, not
        zero P&L, and reporting it as the latter would misstate every error
        by the size of the book.
        """
        standardised = np.zeros((5, 2))
        recovered = state.inverse_transform_targets(standardised)
        assert np.abs(recovered).max() > 0.01 * TARGET_SCALE

    def test_it_round_trips_against_the_scaler(self, state: HybridState) -> None:
        """Inverting a transform must return the input."""
        original = np.random.default_rng(2).normal(scale=TARGET_SCALE, size=(10, 2))
        assert state.inverse_transform_targets(
            state.target_scaler.transform(original)
        ) == pytest.approx(original, rel=1e-5)

    def test_an_unscaled_target_passes_straight_through(self, state: HybridState) -> None:
        """
        An unscaled target must not be silently rescaled on the way out.

        Recorded on the state rather than inferred, so there is nothing to
        guess.
        """
        unscaled = dataclasses.replace(state, scale_targets=False)
        predictions = np.arange(10.0).reshape(5, 2)
        assert np.array_equal(unscaled.inverse_transform_targets(predictions), predictions)


class TestRoundTrip:
    """Saving and loading the whole state."""

    def test_everything_survives(self, state: HybridState, tmp_path) -> None:
        """Each component, checked individually so a failure names itself."""
        state.save(tmp_path)
        restored = HybridState.load(tmp_path)

        assert restored.selected_basis == state.selected_basis
        assert restored.universe == state.universe
        assert restored.scale_targets == state.scale_targets
        assert np.array_equal(restored.elementary_indices, state.elementary_indices)
        assert np.array_equal(restored.target_indices, state.target_indices)
        assert np.array_equal(restored.target_scaler.centre, state.target_scaler.centre)
        assert np.array_equal(restored.graph.indices, state.graph.indices)
        assert restored.encoder.numeric_names == state.encoder.numeric_names

    def test_the_basis_order_survives(self, state: HybridState, tmp_path) -> None:
        """
        ``selected_basis`` is a sequence that looks like a set.

        Its order is the column order of the feature matrix, so a load that
        sorted it -- or round-tripped it through a set -- would feed the
        saved weights their columns transposed. The fixture's basis is
        deliberately not in sorted order, so sorting would be visible here.
        """
        assert state.selected_basis != tuple(sorted(state.selected_basis))
        state.save(tmp_path)
        assert HybridState.load(tmp_path).selected_basis == state.selected_basis

    def test_a_reloaded_state_inverts_identically(self, state: HybridState, tmp_path) -> None:
        """The round trip is only worth anything if the predictions match."""
        state.save(tmp_path)
        predictions = np.random.default_rng(3).normal(size=(8, 2))
        assert np.array_equal(
            HybridState.load(tmp_path).inverse_transform_targets(predictions),
            state.inverse_transform_targets(predictions),
        )

    def test_nothing_is_pickled(self, state: HybridState, tmp_path) -> None:
        """
        No component may reintroduce a pickled third-party object.

        Six of the original's eleven artifacts were pickles, which pinned
        every saved model to the scikit-learn and SciPy versions that wrote
        it.
        """
        state.save(tmp_path)
        suffixes = {path.suffix for path in tmp_path.rglob("*") if path.is_file()}
        assert suffixes <= {".npy", ".npz", ".json"}

    @pytest.mark.parametrize("removed", ["scalers.npz", "universe.json", "encoder", "graph"])
    def test_a_partial_state_is_refused(self, state: HybridState, tmp_path, removed: str) -> None:
        """
        The whole point of the class, stated once per component.

        Loading ten of eleven files gave the original a model that ran and
        was wrong. Every component must therefore be load-bearing, and the
        parametrisation is what proves none of them is quietly optional.
        """
        state.save(tmp_path)
        target = tmp_path / removed
        if target.is_dir():
            for path in sorted(target.rglob("*"), reverse=True):
                path.unlink()
            target.rmdir()
        else:
            target.unlink()

        with pytest.raises(BundleError, match=removed.rstrip("/")):
            HybridState.load(tmp_path)


class TestDescribe:
    """The run-report summary."""

    def test_it_reports_what_a_reader_would_check(self, state: HybridState) -> None:
        """
        The summary carries what a reader would actually check.

        The basis-selection count is the headline: a run that kept every
        instrument has not reduced anything, and that is the first symptom
        of a misconfigured variance threshold.
        """
        summary = state.describe()
        assert summary["n_elementary_selected"] == 2
        assert summary["n_targets"] == 2
        assert summary["n_graph_edges"] > 0

    def test_it_is_json_encodable(self, state: HybridState) -> None:
        """
        The summary goes into the run manifest.

        A NumPy integer in there would fail the write at the very end of a
        long training run.
        """
        json.dumps(state.describe())
```

---

## 8. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_universe.py`

4883 bytes · SHA-256 `9cb3857df194e430`

```python
"""
Tests for the instrument universe.

A universe is the contract between a matrix column and a real instrument. A
prediction is a vector of numbers; without a universe it is a vector of
numbers about nothing, and getting the ordering wrong does not raise -- it
produces plausible predictions attributed to the wrong instruments.

So the tests here are mostly about order being preserved and about lookups
failing loudly rather than returning something indexable.
"""

from __future__ import annotations

import pytest

from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.state import Universe


@pytest.fixture
def universe():
    """
    Provide a small universe with distinguishable identifiers.

    Returns
    -------
    Universe
        Three elementary instruments and two targets.
    """
    return Universe(elementary_ids=("e1", "e2", "e3"), target_ids=("t1", "t2"))


class TestCounts:
    """The sizes every downstream shape is built from."""

    def test_the_two_sides_are_counted_separately(self, universe):
        """A model's input width and output width are different numbers."""
        assert universe.n_elementary == 3
        assert universe.n_targets == 2

    def test_an_empty_side_is_allowed(self):
        """
        A universe with no targets is degenerate but representable.

        Refusing it here would move the error somewhere less informative:
        a data directory with no targets is a problem with the data, and the
        model's ``data.py``, which reads ``universe.json``, is where that
        reads clearly.
        """
        assert Universe(elementary_ids=("e1",), target_ids=()).n_targets == 0


class TestOrdering:
    """Order is the whole contract."""

    def test_identifiers_keep_the_order_they_were_given(self, universe):
        """
        Not sorted.

        Column order is decided by basis selection, and re-sorting here
        would silently reattribute every column to a different instrument.
        """
        assert universe.elementary_ids == ("e1", "e2", "e3")

    def test_the_combined_order_is_elementary_then_target(self, universe):
        """
        The row order of the combined attribute matrix.

        Concatenated here rather than at each call site, because two places
        agreeing by convention is a convention that eventually gets broken
        by someone who did not know it existed.
        """
        assert universe.instrument_ids == ("e1", "e2", "e3", "t1", "t2")

    def test_a_position_matches_the_combined_order(self, universe):
        """Positions are indices into `instrument_ids`, not into one side."""
        assert universe.position_of("e1") == 0
        assert universe.position_of("t1") == 3


class TestLookupFailure:
    """An absent instrument has to raise."""

    def test_an_unknown_instrument_raises(self, universe):
        """
        Rather than returning a sentinel.

        `-1` indexes the last row perfectly happily, and the result would
        be a prediction attributed to whichever instrument happened to be
        there -- wrong rather than absent.
        """
        with pytest.raises(KeyError):
            universe.position_of("absent")

    def test_the_error_says_how_big_the_universe_is(self, universe):
        """
        Context, because the usual cause is a mismatched snapshot.

        An identifier that was valid last week and is not today is a data
        question, and the sizes are the first clue.
        """
        with pytest.raises(KeyError, match="3 elementary and 2 target"):
            universe.position_of("absent")


class TestImmutability:
    """A universe is interpreted by everything downstream."""

    def test_a_universe_cannot_be_reassigned(self, universe):
        """
        Frozen, because reordering after a fit reattributes every index.

        A fitted state, a prediction and a report all read positions out of
        it, and none of them re-check.
        """
        with pytest.raises((AttributeError, TypeError)):
            universe.elementary_ids = ("x",)  # type: ignore[misc]

    def test_two_universes_with_the_same_contents_are_equal(self, universe):
        """
        Value semantics, so parity comparisons can use `==`.

        Two runs producing the same universe should compare equal without
        anyone writing a field-by-field check.
        """
        assert universe == Universe(elementary_ids=("e1", "e2", "e3"), target_ids=("t1", "t2"))

    def test_order_is_part_of_identity(self):
        """
        Two universes differing only in order are not equal.

        They describe different matrices, and anything that treated them as
        interchangeable would be exactly the bug this class guards.
        """
        assert Universe(elementary_ids=("a", "b"), target_ids=()) != Universe(
            elementary_ids=("b", "a"), target_ids=()
        )
```

---

## 9. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/test_visuals.py`

4792 bytes · SHA-256 `358c1bc5be3d1ee1`

```python
"""Tests for the model's own figures."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.features.graph import (
    SparseGraphState,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.visuals import (
    edge_weight_figure,
    neighbour_similarity_figure,
    node_degree_figure,
)

#: Instruments in the toy graph.
N_NODES = 6


def ring(n_nodes: int = N_NODES) -> SparseGraphState:
    """
    Build a row-normalised ring graph with self-loops.

    Parameters
    ----------
    n_nodes
        How many instruments.

    Returns
    -------
    SparseGraphState
        The graph.
    """
    pairs = [
        (node, neighbour)
        for node in range(n_nodes)
        for neighbour in (node, (node - 1) % n_nodes, (node + 1) % n_nodes)
    ]
    return SparseGraphState(
        indices=np.array(pairs, dtype=np.int64),
        values=np.full(len(pairs), 1.0 / 3.0, dtype=np.float32),
        n_nodes=n_nodes,
        is_target=np.array([False] * (n_nodes - 2) + [True] * 2),
    )


def empty(n_nodes: int = N_NODES) -> SparseGraphState:
    """
    Build a graph with no edges at all.

    Parameters
    ----------
    n_nodes
        How many instruments.

    Returns
    -------
    SparseGraphState
        The graph.
    """
    return SparseGraphState(
        indices=np.zeros((0, 2), dtype=np.int64),
        values=np.zeros(0, dtype=np.float32),
        n_nodes=n_nodes,
        is_target=np.zeros(n_nodes, dtype=np.bool_),
    )


class TestFigures:
    """Each factory builds a figure and writes nothing."""

    @pytest.mark.parametrize(
        "factory", [node_degree_figure, edge_weight_figure, neighbour_similarity_figure]
    )
    def test_a_figure_comes_back(self, factory) -> None:
        """
        The rule the whole visuals package obeys.

        A factory that also saved its output would need a separate
        variant for every caller: a report writer, a notebook, a
        dashboard and a test all want the figure and disagree about
        where it should go.
        """
        figure = factory(ring())
        assert isinstance(figure, Figure)
        assert figure.axes

    def test_the_degree_figure_marks_the_median(self) -> None:
        """
        So a reader can see immediately whether the graph is uniform.

        A fixed-neighbour graph is a vertical line and the median says
        nothing; the figure earns its place when a threshold was used
        and the distribution has a left tail.
        """
        axes = node_degree_figure(ring()).axes[0]
        assert axes.get_legend() is not None
        assert axes.get_xlabel() == "neighbours"

    def test_the_weight_figure_marks_the_uniform_level(self) -> None:
        """
        The reference the distribution should be compared against.

        Mass piled at ``1/degree`` means the kernel has flattened into
        an unweighted average, which is the failure the figure exists to
        make visible -- and it is invisible without the reference line.
        """
        axes = edge_weight_figure(ring()).axes[0]
        lines = [line for line in axes.get_lines() if line.get_linestyle() == "--"]
        assert lines

    def test_the_similarity_figure_is_sorted(self) -> None:
        """
        So the worst-covered instruments are at one end and easy to read.

        The question this figure answers is "how bad is the bottom of
        the distribution", and an unsorted scatter answers it far less
        directly.
        """
        axes = neighbour_similarity_figure(ring()).axes[0]
        values = axes.get_lines()[0].get_ydata()
        assert np.all(np.diff(values) >= 0)


class TestEmptyGraph:
    """What happens when there is nothing to plot."""

    @pytest.mark.parametrize("factory", [edge_weight_figure, neighbour_similarity_figure])
    def test_an_edgeless_graph_is_refused(self, factory) -> None:
        """
        Loudly, rather than by drawing an empty pair of axes.

        A graph with no edges is a far more serious finding than a
        missing figure, and an empty plot in a report reads as "nothing
        interesting here" rather than "the graph build failed".
        """
        with pytest.raises(ContractError, match="no edges"):
            factory(empty())

    def test_the_degree_figure_still_works(self) -> None:
        """
        Because a degree of zero is exactly what it should be showing.

        The other two have nothing to put on an axis; this one has the
        most important thing it could possibly report.
        """
        assert node_degree_figure(empty()).axes
```

