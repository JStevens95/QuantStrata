from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Mapping


@dataclass(frozen=True, slots=True)
class ScenarioRow:
    """
    One row of scenario results.

    Attributes
    ----------
    scenario:
        Scenario name (e.g., "BASE", "spot_up_1pct").
    pv:
        Portfolio PV under this scenario.
    pnl:
        Portfolio PnL vs base scenario (pv - pv_base).
    """
    scenario: str
    pv: float
    pnl: float


@dataclass(frozen=True, slots=True)
class ScenarioReport:
    """
    Report of portfolio PV/PnL across scenarios.

    This is intentionally lightweight and dependency-free (no pandas required).
    It is designed to be used in:
      - examples (pretty console output)
      - orchestrators (export)
      - unit tests (stable formatting and values)

    Notes
    -----
    You can extend this later to include:
      - per-position PV/PnL breakdown
      - greeks by scenario
      - aggregation by asset class / product type / book
    """
    rows: List[ScenarioRow]
    base_scenario: str = "BASE"

    @staticmethod
    def from_result(result, *, base_scenario: str = "BASE") -> "ScenarioReport":  # noqa: ANN001
        """
        Build a ScenarioReport from scenario_analysis output.

        Expected `result` interface
        --------------------------
        - result.scenario_names: Sequence[str]
        - result.pv: Sequence[float]
        - result.pnl: Sequence[float]

        This deliberately uses duck-typing so it remains compatible even if
        you later rename StressResult / ScenarioResult classes.
        """
        names = list(result.scenario_names)
        pv = list(result.pv)
        pnl = list(result.pnl)

        if len(names) != len(pv) or len(names) != len(pnl):
            raise ValueError("result fields must have the same length: scenario_names, pv, pnl.")

        rows = [ScenarioRow(scenario=str(n), pv=float(v), pnl=float(p)) for n, v, p in zip(names, pv, pnl)]
        return ScenarioReport(rows=rows, base_scenario=base_scenario)

    def to_dicts(self) -> List[Mapping[str, float | str]]:
        """
        Return list of dicts suitable for JSON export.
        """
        return [{"scenario": r.scenario, "pv": r.pv, "pnl": r.pnl} for r in self.rows]

    def to_csv(self) -> str:
        """
        Return a CSV string (header + rows).

        Notes
        -----
        We avoid locale-specific formatting for robustness.
        """
        lines = ["scenario,pv,pnl"]
        for r in self.rows:
            lines.append(f"{r.scenario},{r.pv:.12g},{r.pnl:.12g}")
        return "\n".join(lines)

    def to_console(self, *, pv_decimals: int = 6, pnl_decimals: int = 6) -> str:
        """
        Pretty console table output.

        Parameters
        ----------
        pv_decimals:
            Decimal places for PV.
        pnl_decimals:
            Decimal places for PnL.

        Returns
        -------
        str
            Multi-line formatted string suitable for print().
        """
        if not self.rows:
            return "ScenarioReport(empty)"

        scenario_width = max(len("Scenario"), max(len(r.scenario) for r in self.rows))
        pv_width = max(len("PV"), max(len(self._fmt_number(r.pv, pv_decimals)) for r in self.rows))
        pnl_width = max(len("PnL"), max(len(self._fmt_number(r.pnl, pnl_decimals, signed=True)) for r in self.rows))

        header = f"{'Scenario':<{scenario_width}} | {'PV':>{pv_width}} | {'PnL':>{pnl_width}}"
        sep = "-" * len(header)

        body_lines: List[str] = []
        for r in self.rows:
            pv_str = self._fmt_number(r.pv, pv_decimals)
            pnl_str = self._fmt_number(r.pnl, pnl_decimals, signed=True)
            body_lines.append(f"{r.scenario:<{scenario_width}} | {pv_str:>{pv_width}} | {pnl_str:>{pnl_width}}")

        return "\n".join([header, sep, *body_lines])

    @staticmethod
    def _fmt_number(x: float, decimals: int, signed: bool = False) -> str:
        """
        Format numeric values defensively for reporting.
        """
        if not math.isfinite(float(x)):
            return "nan" if math.isnan(float(x)) else ("+inf" if float(x) > 0 else "-inf")
        fmt = f"{{:{'+' if signed else ''}.{decimals}f}}"
        return fmt.format(float(x))