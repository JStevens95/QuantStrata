"""VaR/CVaR summary report for front-office risk reporting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from src.risk.var.config import VarResult


@dataclass(frozen=True, slots=True)
class VarBreakdownRow:
    """One row of VaR factor breakdown (e.g. for parametric VaR)."""
    factor_name: str
    contribution: float


@dataclass(frozen=True, slots=True)
class VarSummaryReport:
    """
    VaR/CVaR summary report for orchestrators and export.

    Wraps VarResult with optional factor-level breakdown and consistent
    to_console / to_dicts / to_csv.
    """

    result: VarResult
    breakdown: List[VarBreakdownRow] = field(default_factory=list)

    def to_console(self, *, decimals: int = 6) -> str:
        """Pretty console output for VaR summary."""
        r = self.result

        def fmt(x: float) -> str:
            if not math.isfinite(float(x)):
                return "nan" if math.isnan(float(x)) else ("+inf" if float(x) > 0 else "-inf")
            return f"{float(x):.{decimals}f}"

        lines = [
            "VaR Summary",
            "-" * 40,
            f"Method:       {r.method}",
            f"Confidence:   {r.confidence:.2%}",
            f"Horizon:      {r.horizon_days} day(s)",
            f"VaR:          {fmt(r.var)}",
        ]
        if r.cvar is not None:
            lines.append(f"CVaR:         {fmt(r.cvar)}")
        if r.metadata:
            lines.append("Metadata:")
            for k, v in sorted(r.metadata.items()):
                lines.append(f"  {k}: {v}")
        if self.breakdown:
            lines.append("Breakdown:")
            for row in self.breakdown:
                lines.append(f"  {row.factor_name}: {fmt(row.contribution)}")
        return "\n".join(lines)

    def to_dicts(self) -> Mapping[str, object]:
        """Single nested dict for JSON/export."""
        r = self.result
        return {
            "method": r.method,
            "confidence": float(r.confidence),
            "horizon_days": int(r.horizon_days),
            "var": float(r.var),
            "cvar": None if r.cvar is None else float(r.cvar),
            "metadata": dict(r.metadata),
            "breakdown": [
                {"factor_name": row.factor_name, "contribution": float(row.contribution)}
                for row in self.breakdown
            ],
        }

    def to_csv(self) -> str:
        """CSV string: header + main row; optional breakdown section."""
        r = self.result
        lines = ["method,confidence,horizon_days,var,cvar"]
        cvar_str = "" if r.cvar is None else f"{float(r.cvar):.12g}"
        lines.append(f"{r.method},{r.confidence:.12g},{r.horizon_days},{float(r.var):.12g},{cvar_str}")
        if self.breakdown:
            lines.append("")
            lines.append("factor_name,contribution")
            for row in self.breakdown:
                lines.append(f"{row.factor_name},{float(row.contribution):.12g}")
        return "\n".join(lines)


def build_var_summary_report(
    result: VarResult,
    *,
    breakdown: Optional[List[VarBreakdownRow]] = None,
) -> VarSummaryReport:
    """Convenience: build VarSummaryReport from VarResult."""
    return VarSummaryReport(result=result, breakdown=breakdown or [])


__all__ = ["VarBreakdownRow", "VarSummaryReport", "build_var_summary_report"]
