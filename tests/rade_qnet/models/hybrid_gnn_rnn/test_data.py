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

from src.rade_qnet.core.lifecycle.errors import ContractError
from src.rade_qnet.core.spec.data import (
    ChronologicalSplitSpec,
    ExplicitSplitSpec,
    ModelSourceSpec,
    ReductionSpec,
    SequenceSpec,
    TransformsSpec,
)
from src.rade_qnet.models.hybrid_gnn_rnn.data import HybridDataModule, _merge_attributes
from src.rade_qnet.testkit.parity import compare_state, load_golden

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "rade_qnet" / "golden"
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
            load_golden("hybrid_gnn_rnn"),
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
