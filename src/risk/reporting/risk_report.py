"""Combined front-office risk report: Greeks, VaR, attribution, scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from src.risk.attribution.report import AttributionReport
from src.risk.reporting.scenario_report import ScenarioReport
from src.risk.reporting.var_summary import VarSummaryReport
from src.risk.sensitivities.aggregation import GreeksSummary


@dataclass(frozen=True, slots=True)
class RiskReport:
    """
    Combined risk report for front-office use.

    All sections are optional; callers can build partial reports.
    Instrument-level breakdown is available via GreeksSummary.per_market_id
    and attribution contributions.
    """

    greeks_summary: Optional[GreeksSummary] = None
    var_summary: Optional[VarSummaryReport] = None
    attribution_report: Optional[AttributionReport] = None
    scenario_report: Optional[ScenarioReport] = None

    def to_console(self, *, decimals: int = 6) -> str:
        """Pretty console output: one section per present block."""
        sections: List[str] = []
        if self.greeks_summary is not None:
            g = self.greeks_summary
            lines = ["Greeks Summary", "-" * 40]
            for k, v in sorted(g.totals_by_greek.items()):
                lines.append(f"  {k}: {v:.{decimals}f}")
            lines.append("By factor:")
            for k, v in sorted(g.totals_by_factor.items()):
                lines.append(f"  {k}: {v:.{decimals}f}")
            sections.append("\n".join(lines))
        if self.var_summary is not None:
            sections.append(self.var_summary.to_console(decimals=decimals))
        if self.attribution_report is not None:
            sections.append("PnL Attribution\n" + "-" * 40 + "\n" + self.attribution_report.to_console(decimals=decimals))
        if self.scenario_report is not None:
            sections.append("Scenarios\n" + "-" * 40 + "\n" + self.scenario_report.to_console(pv_decimals=decimals, pnl_decimals=decimals))
        if not sections:
            return "RiskReport(empty)"
        return "\n\n".join(sections)

    def to_dicts(self) -> Dict[str, Any]:
        """Nested dict for JSON/export."""
        out: Dict[str, Any] = {}
        if self.greeks_summary is not None:
            out["greeks"] = {
                "totals_by_greek": dict(self.greeks_summary.totals_by_greek),
                "totals_by_factor": dict(self.greeks_summary.totals_by_factor),
                "per_market_id": self.greeks_summary.to_dicts(),
            }
        if self.var_summary is not None:
            out["var"] = self.var_summary.to_dicts()
        if self.attribution_report is not None:
            out["attribution"] = self.attribution_report.to_dicts()
        if self.scenario_report is not None:
            out["scenarios"] = self.scenario_report.to_dicts()
        return out

    def to_csv(self) -> str:
        """CSV string with sections (header rows for each block)."""
        blocks: List[str] = []
        if self.greeks_summary is not None:
            g = self.greeks_summary
            blocks.append("section=greeks\nmarket_id,greek,value")
            for row in g.to_dicts():
                mid = row.get("market_id")
                blocks.append(f"{mid},{row.get('greek')},{float(row.get('value', 0)):.12g}")
        if self.var_summary is not None:
            blocks.append("\nsection=var\n" + self.var_summary.to_csv())
        if self.attribution_report is not None:
            blocks.append("\nsection=attribution\nscenario,pnl,predicted_pnl,residual")
            for r in self.attribution_report.rows:
                blocks.append(f"{r.scenario},{r.pnl:.12g},{r.predicted_pnl:.12g},{r.residual:.12g}")
        if self.scenario_report is not None:
            blocks.append("\nsection=scenarios\n" + self.scenario_report.to_csv())
        return "\n".join(blocks) if blocks else ""


__all__ = ["RiskReport"]
