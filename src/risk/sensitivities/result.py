from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Mapping, Optional

from src.marketdata.ids import MarketId


@dataclass(frozen=True, slots=True)
class SensitivityKey:
    """
    Identifies a sensitivity in a future-proof way.

    Examples
    --------
    - ("delta", spot_id=EURUSD spot)
    - ("vega", vol_id=EURUSD vol)
    - ("rho", curve_id=USD.OIS)

    Future extensions can add:
    - tenor buckets (e.g., key.tenor)
    - vol buckets (expiry/strike)
    - risk class / qualifier tags
    """
    greek: str
    market_id: Optional[MarketId] = None

    # Optional bucketing hooks (unused in V1 but keeps the schema stable).
    tenor: Optional[float] = None
    expiry: Optional[float] = None
    strike: Optional[float] = None


@dataclass(frozen=True, slots=True)
class SensitivityRow:
    """
    One sensitivity value.

    Attributes
    ----------
    key:
        SensitivityKey describing (greek, market_id, optional buckets).
    value:
        Sensitivity value (e.g. dPV/dS, dPV/dr, dPV/dsigma).
    method:
        "analytic" or "fd_central".
    bump:
        Bump size used for FD (None for analytic).
    units:
        Human hint only (e.g., "per 1 spot", "per 1 vol", "per 1 rate").
    """
    key: SensitivityKey
    value: float
    method: str
    bump: Optional[float] = None
    units: str = ""


@dataclass(frozen=True, slots=True)
class SensitivitiesReport:
    """
    Lightweight report object (no pandas dependency).
    """
    rows: List[SensitivityRow]

    def to_dicts(self) -> List[Mapping[str, object]]:
        out: List[Mapping[str, object]] = []
        for r in self.rows:
            out.append(
                {
                    "greek": r.key.greek,
                    "market_id": None if r.key.market_id is None else str(r.key.market_id),
                    "tenor": r.key.tenor,
                    "expiry": r.key.expiry,
                    "strike": r.key.strike,
                    "value": float(r.value),
                    "method": str(r.method),
                    "bump": None if r.bump is None else float(r.bump),
                    "units": str(r.units),
                }
            )
        return out

    def to_csv(self) -> str:
        lines = ["greek,market_id,tenor,expiry,strike,value,method,bump,units"]
        for r in self.rows:
            mid = "" if r.key.market_id is None else str(r.key.market_id)
            tenor = "" if r.key.tenor is None else f"{float(r.key.tenor):.12g}"
            expiry = "" if r.key.expiry is None else f"{float(r.key.expiry):.12g}"
            strike = "" if r.key.strike is None else f"{float(r.key.strike):.12g}"
            bump = "" if r.bump is None else f"{float(r.bump):.12g}"
            lines.append(
                f"{r.key.greek},{mid},{tenor},{expiry},{strike},{float(r.value):.12g},{r.method},{bump},{r.units}"
            )
        return "\n".join(lines)

    def to_console(self, *, decimals: int = 6) -> str:
        if not self.rows:
            return "SensitivitiesReport(empty)"

        def fmt(x: float) -> str:
            if not math.isfinite(float(x)):
                return "nan" if math.isnan(float(x)) else ("+inf" if float(x) > 0 else "-inf")
            return f"{float(x):.{decimals}f}"

        headers = ["Greek", "MarketId", "Value", "Method", "Bump", "Units"]
        rows: List[List[str]] = []
        for r in self.rows:
            rows.append(
                [
                    str(r.key.greek),
                    "" if r.key.market_id is None else str(r.key.market_id),
                    fmt(r.value),
                    str(r.method),
                    "" if r.bump is None else fmt(r.bump),
                    str(r.units),
                ]
            )

        widths = [max(len(headers[i]), *(len(rr[i]) for rr in rows)) for i in range(len(headers))]
        header = " | ".join(f"{headers[i]:<{widths[i]}}" for i in range(len(headers)))
        sep = "-+-".join("-" * widths[i] for i in range(len(headers)))

        lines = [header, sep]
        for rr in rows:
            lines.append(" | ".join(f"{rr[i]:<{widths[i]}}" for i in range(len(headers))))
        return "\n".join(lines)