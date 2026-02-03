"""
Portfolio builders for GNN-LSTM hybrid model.

Builds elementary (vanilla, digital) and target (exotic) FX option portfolios
separately using the library's Portfolio component, then extracts features and
constructs GNN inputs. Designed for modularity and reuse with existing pricing
infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.portfolio.core import Portfolio, Position, PortfolioResult
from src.portfolio.portfolio import PortfolioPricer
from src.pricers.registry import DefaultPricerRegistry

# FX option instruments
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.instruments.fx.options.digital import FxDigitalEuropeanOption
from src.instruments.fx.options.barrier import FxBarrierEuropeanOption
from src.instruments.fx.options.double_barrier import FxDoubleBarrierEuropeanOption
from src.instruments.fx.options.asian import FxAsianEuropeanOption
from src.instruments.fx.options.touch import FxTouchEuropeanOption


# ==============================================================================
# Default Market IDs for FX (EURUSD)
# ==============================================================================

DEFAULT_FX_SPOT_ID = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
DEFAULT_FX_VOL_ID = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")
DEFAULT_DOM_CURVE_ID = MarketId(asset_class="IR", mkt_type="CURVE", name="USD")
DEFAULT_FOR_CURVE_ID = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR")


# ==============================================================================
# Market Builder
# ==============================================================================

def build_fx_market(
    spot: float = 1.10,
    r_domestic: float = 0.05,
    r_foreign: float = 0.02,
    sigma: float = 0.15,
    asof: str = "2026-01-27",
    spot_id: MarketId = DEFAULT_FX_SPOT_ID,
    vol_id: MarketId = DEFAULT_FX_VOL_ID,
    dom_curve_id: MarketId = DEFAULT_DOM_CURVE_ID,
    for_curve_id: MarketId = DEFAULT_FOR_CURVE_ID,
) -> Market:
    """
    Build a minimal FX market for pricing.

    Parameters
    ----------
    spot : float
        FX spot rate (domestic per foreign, e.g. USD/EUR).
    r_domestic, r_foreign : float
        Flat continuously compounded rates.
    sigma : float
        Flat volatility.
    asof : str
        Valuation date.
    spot_id, vol_id, dom_curve_id, for_curve_id : MarketId
        Market identifiers.

    Returns
    -------
    Market
    """
    return Market(
        asof=asof,
        quotes={spot_id: Quote(value=float(spot))},
        curves={
            dom_curve_id: FlatZeroRateCurve(continuously_compounded_rate=float(r_domestic)),
            for_curve_id: FlatZeroRateCurve(continuously_compounded_rate=float(r_foreign)),
        },
        vols={vol_id: FlatVolSurface(sigma=float(sigma))},
    )


# ==============================================================================
# Elementary Portfolio Builder (vanilla + digital)
# ==============================================================================

def build_elementary_portfolio(
    n_vanilla: int = 500,
    n_digital: int = 500,
    spot: float = 1.10,
    strike_range: Tuple[float, float] = (0.85, 1.15),
    expiry_range: Tuple[float, float] = (0.1, 2.0),
    notional: float = 1_000_000.0,
    seed: Optional[int] = None,
    spot_id: MarketId = DEFAULT_FX_SPOT_ID,
    vol_id: MarketId = DEFAULT_FX_VOL_ID,
    dom_curve_id: MarketId = DEFAULT_DOM_CURVE_ID,
    for_curve_id: MarketId = DEFAULT_FOR_CURVE_ID,
) -> Portfolio:
    """
    Build a portfolio of elementary FX options (vanilla + digital).

    These are closed-form priceable instruments (BSM) that form the
    elementary trade set for the GNN-LSTM model.

    Parameters
    ----------
    n_vanilla : int
        Number of vanilla European options.
    n_digital : int
        Number of digital European options.
    spot : float
        Reference spot for strike generation (strike = spot * factor).
    strike_range : tuple
        (min_factor, max_factor) for strike = spot * uniform(min, max).
    expiry_range : tuple
        (min_expiry, max_expiry) in year fractions.
    notional : float
        Notional per instrument.
    seed : int, optional
        Random seed for reproducibility.
    spot_id, vol_id, dom_curve_id, for_curve_id : MarketId
        Market identifiers for the instruments.

    Returns
    -------
    Portfolio
        Portfolio of vanilla and digital FX options.
    """
    rng = np.random.default_rng(seed)
    positions: List[Position] = []

    # Vanilla options
    for i in range(n_vanilla):
        opt_type = "call" if rng.random() > 0.5 else "put"
        strike = float(spot * rng.uniform(*strike_range))
        expiry = float(rng.uniform(*expiry_range))

        inst = FxVanillaEuropeanOption(
            option_type=opt_type,
            notional=notional,
            strike=strike,
            expiry=expiry,
            spot_id=spot_id,
            vol_id=vol_id,
            domestic_curve_id=dom_curve_id,
            foreign_curve_id=for_curve_id,
        )
        positions.append(Position(position_id=f"elem_van_{i:04d}", instrument=inst, quantity=1.0))

    # Digital options
    for i in range(n_digital):
        opt_type = "call" if rng.random() > 0.5 else "put"
        strike = float(spot * rng.uniform(*strike_range))
        expiry = float(rng.uniform(*expiry_range))
        payout = 100.0 * (1.0 + rng.uniform(-0.2, 0.2))

        inst = FxDigitalEuropeanOption(
            option_type=opt_type,
            payoff="cash",
            payout_amount=payout,
            strike=strike,
            expiry=expiry,
            spot_id=spot_id,
            vol_id=vol_id,
            domestic_curve_id=dom_curve_id,
            foreign_curve_id=for_curve_id,
        )
        positions.append(Position(position_id=f"elem_dig_{i:04d}", instrument=inst, quantity=1.0))

    return Portfolio(positions=positions)


# ==============================================================================
# Target Portfolio Builder (exotics)
# ==============================================================================

def build_target_portfolio(
    n_barrier: int = 5,
    n_double_barrier: int = 5,
    n_asian: int = 5,
    n_touch: int = 5,
    spot: float = 1.10,
    strike_range: Tuple[float, float] = (0.90, 1.10),
    expiry_range: Tuple[float, float] = (0.25, 1.5),
    notional: float = 1_000_000.0,
    seed: Optional[int] = None,
    spot_id: MarketId = DEFAULT_FX_SPOT_ID,
    vol_id: MarketId = DEFAULT_FX_VOL_ID,
    dom_curve_id: MarketId = DEFAULT_DOM_CURVE_ID,
    for_curve_id: MarketId = DEFAULT_FOR_CURVE_ID,
) -> Portfolio:
    """
    Build a portfolio of target (exotic) FX options.

    These are the target trades whose P&L we want to predict from
    elementary trades using the GNN-LSTM model.

    Parameters
    ----------
    n_barrier : int
        Number of single-barrier options.
    n_double_barrier : int
        Number of double-barrier options.
    n_asian : int
        Number of Asian options.
    n_touch : int
        Number of touch options.
    spot : float
        Reference spot for strike/barrier generation.
    strike_range : tuple
        (min_factor, max_factor) for strike = spot * uniform(min, max).
    expiry_range : tuple
        (min_expiry, max_expiry) in year fractions.
    notional : float
        Notional per instrument.
    seed : int, optional
        Random seed.
    spot_id, vol_id, dom_curve_id, for_curve_id : MarketId
        Market identifiers.

    Returns
    -------
    Portfolio
        Portfolio of exotic FX options.
    """
    rng = np.random.default_rng(seed)
    positions: List[Position] = []

    # Single barrier options
    for i in range(n_barrier):
        opt_type = "call" if rng.random() > 0.5 else "put"
        strike = float(spot * rng.uniform(*strike_range))
        expiry = float(rng.uniform(*expiry_range))
        direction = "up" if rng.random() > 0.5 else "down"
        style = "knock_out" if rng.random() > 0.5 else "knock_in"
        if direction == "up":
            barrier = float(spot * rng.uniform(1.05, 1.25))
        else:
            barrier = float(spot * rng.uniform(0.75, 0.95))

        inst = FxBarrierEuropeanOption(
            option_type=opt_type,
            notional=notional,
            strike=strike,
            expiry=expiry,
            barrier_direction=direction,
            barrier_style=style,
            barrier_level=barrier,
            rebate_amount=0.0,
            spot_id=spot_id,
            vol_id=vol_id,
            domestic_curve_id=dom_curve_id,
            foreign_curve_id=for_curve_id,
        )
        positions.append(Position(position_id=f"tgt_bar_{i:04d}", instrument=inst, quantity=1.0))

    # Double barrier options
    for i in range(n_double_barrier):
        opt_type = "call" if rng.random() > 0.5 else "put"
        strike = float(spot * rng.uniform(0.95, 1.05))
        expiry = float(rng.uniform(*expiry_range))
        style = "knock_out" if rng.random() > 0.5 else "knock_in"
        lo = float(spot * rng.uniform(0.80, 0.95))
        hi = float(spot * rng.uniform(1.05, 1.20))
        if lo >= hi:
            lo, hi = hi, lo

        inst = FxDoubleBarrierEuropeanOption(
            option_type=opt_type,
            notional=notional,
            strike=strike,
            expiry=expiry,
            barrier_style=style,
            lower_barrier=lo,
            upper_barrier=hi,
            rebate_amount=0.0,
            spot_id=spot_id,
            vol_id=vol_id,
            domestic_curve_id=dom_curve_id,
            foreign_curve_id=for_curve_id,
        )
        positions.append(Position(position_id=f"tgt_dbar_{i:04d}", instrument=inst, quantity=1.0))

    # Asian options
    for i in range(n_asian):
        opt_type = "call" if rng.random() > 0.5 else "put"
        strike = float(spot * rng.uniform(*strike_range))
        expiry = float(rng.uniform(0.5, 1.5))
        avg_type = "arithmetic" if rng.random() > 0.5 else "geometric"

        inst = FxAsianEuropeanOption(
            option_type=opt_type,
            notional=notional,
            strike=strike,
            expiry=expiry,
            spot_id=spot_id,
            vol_id=vol_id,
            domestic_curve_id=dom_curve_id,
            foreign_curve_id=for_curve_id,
            averaging_type=avg_type,
        )
        positions.append(Position(position_id=f"tgt_asian_{i:04d}", instrument=inst, quantity=1.0))

    # Touch options
    for i in range(n_touch):
        direction = "up" if rng.random() > 0.5 else "down"
        style = "one_touch" if rng.random() > 0.5 else "no_touch"
        expiry = float(rng.uniform(0.25, 1.0))
        if direction == "up":
            barrier = float(spot * rng.uniform(1.05, 1.20))
        else:
            barrier = float(spot * rng.uniform(0.80, 0.95))
        payout = 100.0 * (1.0 + rng.uniform(-0.1, 0.1))

        inst = FxTouchEuropeanOption(
            touch_style=style,
            barrier_direction=direction,
            barrier_level=barrier,
            payout_amount=payout,
            notional=notional,
            expiry=expiry,
            spot_id=spot_id,
            vol_id=vol_id,
            domestic_curve_id=dom_curve_id,
            foreign_curve_id=for_curve_id,
        )
        positions.append(Position(position_id=f"tgt_touch_{i:04d}", instrument=inst, quantity=1.0))

    return Portfolio(positions=positions)


# ==============================================================================
# Feature Extraction
# ==============================================================================

def _get_strike(inst: Any) -> float:
    """Extract strike from instrument (touch uses barrier level)."""
    if hasattr(inst, "strike") and inst.strike is not None and inst.strike > 0:
        return float(inst.strike)
    if hasattr(inst, "barrier_level"):
        return float(inst.barrier_level)
    return 1.0


def _get_expiry(inst: Any) -> float:
    """Extract expiry (year fraction)."""
    return float(getattr(inst, "expiry", 1.0))


def _instrument_product_type(inst: Any) -> str:
    """Return product type string from instrument."""
    name = type(inst).__name__
    if "Fx" in name:
        return "fx_option"
    if "Equity" in name:
        return "equity_option"
    return "option"


def _instrument_subtype(inst: Any) -> str:
    """Return product subtype string from instrument type."""
    name = type(inst).__name__
    if "Vanilla" in name:
        return "vanilla"
    if "Digital" in name:
        return "digital"
    if "DoubleBarrier" in name:
        return "double_barrier"
    if "Barrier" in name:
        return "barrier"
    if "Asian" in name:
        return "asian"
    if "Touch" in name:
        return "touch"
    if "Lookback" in name:
        return "lookback"
    return "option"


@dataclass
class TradeFeatures:
    """
    Extracted features for trades in a portfolio.

    Attributes
    ----------
    position_ids : list[str]
        Position IDs in order.
    moneyness : np.ndarray
        Spot / strike per trade.
    yrs_to_maturity : np.ndarray
        Time to expiry per trade.
    delta : np.ndarray
        Delta per trade.
    vega : np.ndarray
        Vega per trade.
    pv : np.ndarray
        Present value per trade.
    product_type : list[str]
        E.g. "fx_option".
    product_subtype : list[str]
        E.g. "vanilla", "barrier", "asian".
    """
    position_ids: List[str]
    moneyness: np.ndarray
    yrs_to_maturity: np.ndarray
    delta: np.ndarray
    vega: np.ndarray
    pv: np.ndarray
    product_type: List[str]
    product_subtype: List[str]


def extract_trade_features(
    portfolio: Portfolio,
    portfolio_result: PortfolioResult,
    market: Market,
    spot_id: MarketId = DEFAULT_FX_SPOT_ID,
) -> TradeFeatures:
    """
    Extract trade features from a priced portfolio.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio of instruments.
    portfolio_result : PortfolioResult
        Result from PortfolioPricer.price().
    market : Market
        Market snapshot for spot lookup.
    spot_id : MarketId
        Spot market ID for moneyness calculation.

    Returns
    -------
    TradeFeatures
        Extracted features for all positions.
    """
    spot = float(market.quote(spot_id))
    n = len(portfolio.positions)

    position_ids: List[str] = []
    moneyness = np.zeros(n, dtype=np.float32)
    yrs_to_maturity = np.zeros(n, dtype=np.float32)
    delta = np.zeros(n, dtype=np.float32)
    vega = np.zeros(n, dtype=np.float32)
    pv = np.zeros(n, dtype=np.float32)
    product_type: List[str] = []
    product_subtype: List[str] = []

    # Build a lookup from position_id to result
    result_map = {r.position_id: r for r in portfolio_result.per_position}

    for i, pos in enumerate(portfolio.positions):
        inst = pos.instrument
        position_ids.append(pos.position_id)

        K = _get_strike(inst)
        moneyness[i] = spot / K if K > 0 else 1.0
        yrs_to_maturity[i] = _get_expiry(inst)
        product_type.append(_instrument_product_type(inst))
        product_subtype.append(_instrument_subtype(inst))

        result = result_map.get(pos.position_id)
        if result:
            pv[i] = result.pv
            delta[i] = result.greeks.get("delta", 0.0)
            vega[i] = result.greeks.get("vega", 0.0)

    return TradeFeatures(
        position_ids=position_ids,
        moneyness=moneyness,
        yrs_to_maturity=yrs_to_maturity,
        delta=delta,
        vega=vega,
        pv=pv,
        product_type=product_type,
        product_subtype=product_subtype,
    )


# ==============================================================================
# GNN Input Builder
# ==============================================================================

@dataclass
class GnnPortfolioData:
    """
    Container for GNN-LSTM inputs from portfolios.

    Attributes
    ----------
    trade_features : np.ndarray
        (n_trades, n_features) encoded feature matrix.
    adjacency_matrix : np.ndarray
        (n_trades, n_trades) row-normalised k-NN.
    pnl_history : np.ndarray
        (n_samples, n_timesteps, n_elementary).
    targets : np.ndarray
        (n_samples, n_targets).
    elementary_indices : np.ndarray
        Indices of elementary trades.
    target_indices : np.ndarray
        Indices of target trades.
    feature_names : list[str]
        Names of encoded features.
    elementary_portfolio : Portfolio
        Original elementary portfolio.
    target_portfolio : Portfolio
        Original target portfolio.
    """
    trade_features: np.ndarray
    adjacency_matrix: np.ndarray
    pnl_history: np.ndarray
    targets: np.ndarray
    elementary_indices: np.ndarray
    target_indices: np.ndarray
    feature_names: List[str]
    elementary_portfolio: Portfolio
    target_portfolio: Portfolio

    @property
    def n_elementary(self) -> int:
        return len(self.elementary_indices)

    @property
    def n_targets(self) -> int:
        return len(self.target_indices)

    @property
    def n_trades(self) -> int:
        return self.trade_features.shape[0]

    def to_gnn_inputs(self) -> Dict[str, np.ndarray]:
        """Package for HybridGnnRnn."""
        return {
            "trade_features": self.trade_features.astype(np.float32),
            "adjacency_matrix": self.adjacency_matrix.astype(np.float32),
            "pnl_history": self.pnl_history.astype(np.float32),
            "target_indices": self.target_indices.astype(np.int32),
            "elementary_indices": self.elementary_indices.astype(np.int32),
        }


def _encode_features(
    elem_features: TradeFeatures,
    target_features: TradeFeatures,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode trade features into a combined feature matrix.

    Simple encoding: numeric features + one-hot product subtype.
    For full encoder, use TradeAttributeEncoder.

    Returns
    -------
    features : np.ndarray
        (n_trades, n_features).
    feature_names : list[str]
    """
    n_elem = len(elem_features.position_ids)
    n_tgt = len(target_features.position_ids)
    n = n_elem + n_tgt

    # Numeric features
    moneyness = np.concatenate([elem_features.moneyness, target_features.moneyness])
    ttm = np.concatenate([elem_features.yrs_to_maturity, target_features.yrs_to_maturity])
    delta = np.concatenate([elem_features.delta, target_features.delta])
    vega = np.concatenate([elem_features.vega, target_features.vega])

    # One-hot encode product subtype
    all_subtypes = elem_features.product_subtype + target_features.product_subtype
    unique_subtypes = sorted(set(all_subtypes))
    subtype_to_idx = {s: i for i, s in enumerate(unique_subtypes)}
    subtype_onehot = np.zeros((n, len(unique_subtypes)), dtype=np.float32)
    for i, s in enumerate(all_subtypes):
        subtype_onehot[i, subtype_to_idx[s]] = 1.0

    features = np.column_stack([moneyness, ttm, delta, vega, subtype_onehot])
    feature_names = ["moneyness", "ttm", "delta", "vega"] + [f"subtype_{s}" for s in unique_subtypes]

    return features.astype(np.float32), feature_names


def _build_knn_adjacency(
    features: np.ndarray,
    k: int = 10,
    include_self: bool = True,
) -> np.ndarray:
    """
    Build row-normalised k-NN adjacency matrix.

    Parameters
    ----------
    features : np.ndarray
        (n, d) feature matrix.
    k : int
        Number of neighbours.
    include_self : bool
        Whether to include self-loops.

    Returns
    -------
    adj : np.ndarray
        (n, n) row-normalised adjacency.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix

    n = features.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean")
    nn.fit(features)
    _, indices = nn.kneighbors(features)

    rows, cols = [], []
    for i in range(n):
        for j in indices[i]:
            if j == i and not include_self:
                continue
            rows.append(i)
            cols.append(j)

    data = np.ones(len(rows), dtype=np.float32)
    adj_sparse = csr_matrix((data, (rows, cols)), shape=(n, n))

    row_sums = np.array(adj_sparse.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    adj_dense = adj_sparse.toarray() / row_sums[:, None]
    return adj_dense.astype(np.float32)


def _generate_synthetic_pnl(
    n_samples: int,
    n_timesteps: int,
    n_elementary: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic PnL history (random walk)."""
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, 1.0, (n_samples, n_timesteps, n_elementary))
    return np.cumsum(increments, axis=1).astype(np.float32)


def _generate_synthetic_targets(
    pnl_history: np.ndarray,
    n_targets: int,
    noise_std: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic targets (weighted sum of final elementary PnL + noise)."""
    rng = np.random.default_rng(seed)
    n_samples = pnl_history.shape[0]
    n_elementary = pnl_history.shape[2]

    final_pnl = pnl_history[:, -1, :]  # (n_samples, n_elementary)
    weights = rng.uniform(0.1, 1.0, (n_elementary, n_targets))
    weights = weights / weights.sum(axis=0, keepdims=True)

    targets = final_pnl @ weights + rng.normal(0.0, noise_std, (n_samples, n_targets))
    return targets.astype(np.float32)


def build_gnn_portfolio_data(
    elementary_portfolio: Portfolio,
    target_portfolio: Portfolio,
    elementary_result: PortfolioResult,
    target_result: PortfolioResult,
    market: Market,
    n_samples: int = 600,
    n_timesteps: int = 24,
    k_neighbours: int = 10,
    noise_std: float = 0.5,
    seed: Optional[int] = None,
    spot_id: MarketId = DEFAULT_FX_SPOT_ID,
) -> GnnPortfolioData:
    """
    Build GNN inputs from separately priced elementary and target portfolios.

    This combines features from both portfolios, builds the adjacency graph,
    and generates synthetic PnL/targets (for tutorial; production would use
    path-wise repricing).

    Parameters
    ----------
    elementary_portfolio : Portfolio
        Portfolio of elementary trades.
    target_portfolio : Portfolio
        Portfolio of target (exotic) trades.
    elementary_result : PortfolioResult
        Pricing result for elementary portfolio.
    target_result : PortfolioResult
        Pricing result for target portfolio.
    market : Market
        Market snapshot for feature extraction.
    n_samples : int
        Number of samples/scenarios.
    n_timesteps : int
        PnL history length.
    k_neighbours : int
        k for k-NN graph.
    noise_std : float
        Noise on synthetic targets.
    seed : int, optional
        Random seed.
    spot_id : MarketId
        Spot ID for moneyness.

    Returns
    -------
    GnnPortfolioData
    """
    # Extract features
    elem_features = extract_trade_features(elementary_portfolio, elementary_result, market, spot_id)
    target_features = extract_trade_features(target_portfolio, target_result, market, spot_id)

    # Encode features
    trade_features, feature_names = _encode_features(elem_features, target_features)

    # Build adjacency
    adjacency_matrix = _build_knn_adjacency(trade_features, k=k_neighbours, include_self=True)

    # Indices
    n_elem = len(elem_features.position_ids)
    n_tgt = len(target_features.position_ids)
    elementary_indices = np.arange(n_elem, dtype=np.int32)
    target_indices = np.arange(n_elem, n_elem + n_tgt, dtype=np.int32)

    # Synthetic PnL and targets
    pnl_history = _generate_synthetic_pnl(n_samples, n_timesteps, n_elem, seed=seed)
    targets = _generate_synthetic_targets(pnl_history, n_tgt, noise_std, seed=seed)

    return GnnPortfolioData(
        trade_features=trade_features,
        adjacency_matrix=adjacency_matrix,
        pnl_history=pnl_history,
        targets=targets,
        elementary_indices=elementary_indices,
        target_indices=target_indices,
        feature_names=feature_names,
        elementary_portfolio=elementary_portfolio,
        target_portfolio=target_portfolio,
    )


# ==============================================================================
# Train / Validation / Projection Split
# ==============================================================================

def train_val_projection_split(
    data: GnnPortfolioData,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    projection_ratio: float = 0.2,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Split samples into train, validation, and projection by index (time-ordered).

    Returns
    -------
    train_inputs, train_targets, val_inputs, val_targets, proj_inputs, proj_targets
    """
    n = data.pnl_history.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    def slice_data(start: int, length: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        pnl = data.pnl_history[start : start + length]
        tgt = data.targets[start : start + length]
        inputs = {
            "trade_features": np.tile(data.trade_features, (length, 1, 1)),
            "adjacency_matrix": np.tile(data.adjacency_matrix, (length, 1, 1)),
            "pnl_history": pnl,
            "target_indices": np.tile(data.target_indices, (length, 1)),
            "elementary_indices": np.tile(data.elementary_indices, (length, 1)),
        }
        return inputs, tgt.astype(np.float32)

    train_inputs, train_targets = slice_data(0, n_train)
    val_inputs, val_targets = slice_data(n_train, n_val)
    proj_inputs, proj_targets = slice_data(n_train + n_val, n - n_train - n_val)

    return train_inputs, train_targets, val_inputs, val_targets, proj_inputs, proj_targets


# ==============================================================================
# Convenience: Full Pipeline
# ==============================================================================

def build_fx_gnn_data(
    n_vanilla: int = 500,
    n_digital: int = 500,
    n_barrier: int = 5,
    n_double_barrier: int = 5,
    n_asian: int = 5,
    n_touch: int = 5,
    n_samples: int = 600,
    n_timesteps: int = 24,
    k_neighbours: int = 10,
    spot: float = 1.10,
    r_domestic: float = 0.05,
    r_foreign: float = 0.02,
    sigma: float = 0.15,
    noise_std: float = 0.5,
    seed: Optional[int] = None,
) -> GnnPortfolioData:
    """
    Full pipeline: build elementary and target portfolios, price, and create GNN data.

    This is a convenience function that combines:
    1. build_fx_market
    2. build_elementary_portfolio
    3. build_target_portfolio
    4. Price both with PortfolioPricer
    5. build_gnn_portfolio_data

    Parameters
    ----------
    n_vanilla, n_digital : int
        Elementary portfolio sizes.
    n_barrier, n_double_barrier, n_asian, n_touch : int
        Target portfolio sizes.
    n_samples, n_timesteps, k_neighbours : int
        GNN data parameters.
    spot, r_domestic, r_foreign, sigma : float
        Market parameters.
    noise_std : float
        Target noise.
    seed : int, optional
        Random seed.

    Returns
    -------
    GnnPortfolioData
    """
    # Build market
    market = build_fx_market(spot=spot, r_domestic=r_domestic, r_foreign=r_foreign, sigma=sigma)

    # Build portfolios separately
    elementary_portfolio = build_elementary_portfolio(
        n_vanilla=n_vanilla,
        n_digital=n_digital,
        spot=spot,
        seed=seed,
    )
    target_portfolio = build_target_portfolio(
        n_barrier=n_barrier,
        n_double_barrier=n_double_barrier,
        n_asian=n_asian,
        n_touch=n_touch,
        spot=spot,
        seed=seed,
    )

    # Price both
    registry = DefaultPricerRegistry().build()
    pricer = PortfolioPricer(pricer_registry=registry)

    elementary_result = pricer.price(elementary_portfolio, market)
    target_result = pricer.price(target_portfolio, market)

    # Build GNN data
    return build_gnn_portfolio_data(
        elementary_portfolio=elementary_portfolio,
        target_portfolio=target_portfolio,
        elementary_result=elementary_result,
        target_result=target_result,
        market=market,
        n_samples=n_samples,
        n_timesteps=n_timesteps,
        k_neighbours=k_neighbours,
        noise_std=noise_std,
        seed=seed,
    )
