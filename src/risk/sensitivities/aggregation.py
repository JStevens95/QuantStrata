"""Greeks aggregation: bucketing and risk-factor decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from src.marketdata.core.ids import MarketId

from src.risk.sensitivities.result import SensitivitiesReport, SensitivityRow


# Map greek name -> risk factor (spot, vol, rate)
_GREEK_TO_FACTOR: Mapping[str, str] = {
    "delta": "spot",
    "delta_spot": "spot",
    "gamma": "spot",
    "vega": "vol",
    "rho_domestic": "rate",
    "rho_foreign": "rate",
    "rho": "rate",
}


def _greek_to_factor(greek: str) -> str:
    return _GREEK_TO_FACTOR.get(greek, "other")


@dataclass(frozen=True, slots=True)
class GreeksSummary:
    """
    Aggregated Greeks: totals by greek, by risk factor, and optional per-market_id breakdown.

    Attributes
    ----------
    totals_by_greek : dict
        Total sensitivity per greek name (e.g. delta, vega).
    totals_by_factor : dict
        Total per risk factor: "spot", "vol", "rate", "other".
    per_market_id : list
        Per (market_id, greek_name) breakdown: list of (market_id or None, greek, value).
    """

    totals_by_greek: Dict[str, float]
    totals_by_factor: Dict[str, float]
    per_market_id: List[Tuple[Optional[MarketId], str, float]] = field(default_factory=list)

    def to_dicts(self) -> List[Mapping[str, object]]:
        """Per-market_id rows as list of dicts (for export)."""
        out: List[Mapping[str, object]] = []
        for mid, greek, value in self.per_market_id:
            out.append(
                {
                    "market_id": None if mid is None else str(mid),
                    "greek": greek,
                    "value": float(value),
                }
            )
        return out


def aggregate_sensitivities(
    report: SensitivitiesReport,
    *,
    by: str = "greek",
    include_per_market_id: bool = True,
) -> GreeksSummary:
    """
    Aggregate sensitivities by greek and/or risk factor.

    Parameters
    ----------
    report : SensitivitiesReport
        Output of compute_sensitivities.
    by : str
        "greek" (totals per greek), "risk_factor" (totals per spot/vol/rate), or both.
        Both are always computed; this flag is reserved for future options.
    include_per_market_id : bool
        If True, include per_market_id breakdown (market_id, greek, value).

    Returns
    -------
    GreeksSummary
    """
    totals_by_greek: Dict[str, float] = {}
    totals_by_factor: Dict[str, float] = {}
    per_market_id: List[Tuple[Optional[MarketId], str, float]] = []

    for row in report.rows:
        greek = str(row.key.greek)
        value = float(row.value)
        mid = row.key.market_id

        totals_by_greek[greek] = totals_by_greek.get(greek, 0.0) + value
        factor = _greek_to_factor(greek)
        totals_by_factor[factor] = totals_by_factor.get(factor, 0.0) + value

        if include_per_market_id:
            per_market_id.append((mid, greek, value))

    return GreeksSummary(
        totals_by_greek=totals_by_greek,
        totals_by_factor=totals_by_factor,
        per_market_id=per_market_id,
    )
