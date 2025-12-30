from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping


@dataclass(frozen=True, slots=True)
class AttributionRow:
    scenario: str
    pv_base: float
    pv_scn: float
    pnl: float
    contributions: Mapping[str, float]
    predicted_pnl: float
    residual: float
    rel_error: float


@dataclass(frozen=True, slots=True)
class AttributionReport:
    rows: List[AttributionRow]
    base_scenario_name: str = "BASE"

    def max_rel_error(self) -> float:
        errors = [r.rel_error for r in self.rows if r.scenario != self.base_scenario_name]
        return float(max(errors)) if errors else 0.0

    def to_console(self, *, decimals: int = 6) -> str:
        if not self.rows:
            return "AttributionReport(empty)"

        def fmt(x: float, signed: bool = False) -> str:
            if not math.isfinite(float(x)):
                return "nan" if math.isnan(float(x)) else ("+inf" if float(x) > 0 else "-inf")
            return f"{float(x):{('+' if signed else '')}.{decimals}f}"

        contrib_keys: List[str] = sorted({k for r in self.rows for k in r.contributions.keys()})
        headers = ["Scenario", "PnL", *contrib_keys, "Pred", "Residual", "RelErr"]

        table: List[List[str]] = []
        for r in self.rows:
            row = [str(r.scenario), fmt(r.pnl, signed=True)]
            for k in contrib_keys:
                row.append(fmt(float(r.contributions.get(k, 0.0)), signed=True))
            row += [fmt(r.predicted_pnl, signed=True), fmt(r.residual, signed=True), f"{r.rel_error:.3g}"]
            table.append(row)

        widths = [max(len(headers[i]), *(len(rr[i]) for rr in table)) for i in range(len(headers))]
        header_line = " | ".join(f"{headers[i]:<{widths[i]}}" for i in range(len(headers)))
        sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))

        lines = [header_line, sep_line]
        for rr in table:
            lines.append(" | ".join(f"{rr[i]:<{widths[i]}}" for i in range(len(headers))))
        return "\n".join(lines)

    def to_dicts(self) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for r in self.rows:
            out.append(
                {
                    "scenario": r.scenario,
                    "pv_base": float(r.pv_base),
                    "pv_scn": float(r.pv_scn),
                    "pnl": float(r.pnl),
                    "predicted_pnl": float(r.predicted_pnl),
                    "residual": float(r.residual),
                    "rel_error": float(r.rel_error),
                    "contributions": {str(k): float(v) for k, v in dict(r.contributions).items()},
                }
            )
        return out