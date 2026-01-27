"""
End-to-end scenario risk pipeline.

This pipeline is intentionally small and deterministic:
- Fail fast if required inputs are missing (market + portfolio)
- Build scenario definitions from cfg.params
- Run the scenario engine (apply shocks + reprice)
- Write a scenario report as artifacts (CSV + JSON) if ArtifactStore.enable_save is True

Expected ctx.state inputs
-------------------------
- ctx.state["portfolio"]: Portfolio
- ctx.state["market"] or ctx.state["snapshot"]: Market (base snapshot)

Expected cfg.params structure (V1)
----------------------------------
params = {
  "risk": {
    "scenarios": [
      {"name": "spot_up_1pct", "type": "spot", "key": "FX.SPOT.EURUSD", "mode": "relative", "bump": 0.01},
      {"name": "vol_up_25bp",  "type": "vol",  "key": "FX.VOL.EURUSD",  "mode": "absolute", "bump": 0.0025},
      {"name": "usd_rate_up",  "type": "rate_parallel", "key": "IR.ZERO.USD", "mode": "absolute", "bump": 0.0001},
    ]
  }
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from src.orchestrator.core.context import Context
from src.orchestrator.core.state_keys import StateKeys as Keys
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step

# NOTE:
# Your repo already contains risk scenario modules (tests show this),
# so we import the "real" building blocks from src.risk.*.
# If the exact module path differs slightly, update these imports only.
from src.risk.scenarios.runner import ScenarioRunner  # type: ignore
from src.risk.scenarios.shocks import ParallelRateShock, SpotShock, VolShock  # type: ignore
from src.risk.reporting.scenario_report import ScenarioReport  # type: ignore

from src.marketdata.core.ids import MarketId  # type: ignore


# ---------------------------------------------------------------------
# Pipeline builder (registered by discovery)
# ---------------------------------------------------------------------

def build_pipeline(cfg: Any) -> Pipeline:
    """
    Build the risk.run_scenarios pipeline.

    Notes
    -----
    - The pipeline builder is intentionally "thin":
      it only defines the ordered steps and pipeline name.
    """
    return Pipeline(
        name="risk.run_scenarios",
        steps=(
            BuildScenarioPackStep(name="build_scenario_pack"),
            RunScenarioStep(name="run_scenarios"),
            WriteScenarioReportStep(name="write_scenario_report"),
        ),
    )


# ---------------------------------------------------------------------
# Internal helpers (small, explicit, and testable)
# ---------------------------------------------------------------------

def _require_dict(value: Any, *, where: str) -> Dict[str, Any]:
    """Validate that `value` is a dict and return a shallow copy for safe reads."""
    if not isinstance(value, dict):
        raise TypeError(f"{where} must be a dict, got {type(value).__name__}")
    return dict(value)


def _require_list(value: Any, *, where: str) -> List[Any]:
    """Validate that `value` is a list and return it as-is."""
    if not isinstance(value, list):
        raise TypeError(f"{where} must be a list, got {type(value).__name__}")
    return value


def _require_str(value: Any, *, where: str) -> str:
    """Coerce to a non-empty string."""
    s = str(value).strip()
    if not s:
        raise ValueError(f"{where} must be a non-empty string")
    return s


def _require_float(value: Any, *, where: str) -> float:
    """Coerce to float with a clean error message."""
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(f"{where} must be float-like") from exc


def _parse_market_id(value: Any, *, where: str) -> MarketId:
    """Parse a MarketId from either an existing MarketId or a string key."""
    if isinstance(value, MarketId):
        return value
    if isinstance(value, str) and value.strip():
        return MarketId.parse(value.strip())
    raise TypeError(f"{where} must be a MarketId or non-empty string")


def _get_base_market(state: Mapping[str, Any]) -> Any:
    """
    Resolve the base Market snapshot from ctx.state.

    We support both:
    - Keys.MARKET (preferred if you standardised it)
    - Keys.SNAPSHOT or raw "snapshot"/"market" aliases
    """
    # Preferred: use strongly typed Keys if present
    for key in (getattr(Keys, "MARKET", None), getattr(Keys, "SNAPSHOT", None)):
        if key is not None and key in state:
            return state[key]

    # Backwards compatible aliases
    if "market" in state:
        return state["market"]
    if "snapshot" in state:
        return state["snapshot"]

    raise KeyError("Missing base market snapshot in ctx.state (expected Keys.MARKET/Keys.SNAPSHOT or 'market'/'snapshot').")


def _get_portfolio(state: Mapping[str, Any]) -> Any:
    """Resolve the Portfolio from ctx.state (fail fast with the message your tests expect)."""
    if Keys.PORTFOLIO in state:
        return state[Keys.PORTFOLIO]
    if "portfolio" in state:
        return state["portfolio"]
    raise KeyError("Missing ctx.state['portfolio']. Provide a Portfolio before running risk pipeline.")


def _build_shock_from_cfg(item: Mapping[str, Any]) -> Tuple[str, Any]:
    """
    Build a concrete shock from config.

    Returns
    -------
    (scenario_name, shock_instance)
    """
    cfg = dict(item)  # copy for safety

    # Read/validate required fields
    name = _require_str(cfg.get("name"), where="params.risk.scenarios[].name")
    shock_type = _require_str(cfg.get("type"), where=f"params.risk.scenarios[{name}].type").lower()
    key = cfg.get("key")
    mode = _require_str(cfg.get("mode", "relative"), where=f"params.risk.scenarios[{name}].mode").lower()
    bump = _require_float(cfg.get("bump"), where=f"params.risk.scenarios[{name}].bump")

    # Parse market key
    mkt_id = _parse_market_id(key, where=f"params.risk.scenarios[{name}].key")

    # Construct the shock
    if shock_type == "spot":
        return name, SpotShock(mkt_id, bump=bump, mode=mode)
    if shock_type == "vol":
        return name, VolShock(mkt_id, bump=bump, mode=mode)
    if shock_type in {"rate_parallel", "parallel_rate", "rate"}:
        return name, ParallelRateShock(mkt_id, bump=bump, mode=mode)

    raise ValueError(
        f"Unsupported scenario type '{shock_type}' for scenario '{name}'. "
        "V1 supports: spot, vol, rate_parallel."
    )


# ---------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BuildScenarioPackStep(Step):
    """
    Parse cfg.params and build a scenario pack (list of named shocks).
    """

    def run(self, ctx: Context) -> Context:
        # Pull params dict from config
        params = _require_dict(getattr(ctx.cfg, "params", {}), where="RunConfig.params")

        # Extract risk config block (default empty dict)
        risk_cfg = _require_dict(params.get("risk", {}), where="params.risk")

        # Extract scenario list (default empty list)
        scenario_items = _require_list(risk_cfg.get("scenarios", []), where="params.risk.scenarios")

        # Build the pack as a list of (name, shock) tuples (simple and serializable)
        pack: List[Tuple[str, Any]] = []
        for raw in scenario_items:
            item_cfg = _require_dict(raw, where="params.risk.scenarios[]")
            pack.append(_build_shock_from_cfg(item_cfg))

        # Store in state for downstream steps
        ctx.state["scenario_pack"] = pack
        return ctx


@dataclass(frozen=True, slots=True)
class RunScenarioStep(Step):
    """
    Run the scenario engine (apply shocks + reprice portfolio).
    """

    def run(self, ctx: Context) -> Context:
        # Resolve required inputs
        portfolio = _get_portfolio(ctx.state)
        base_market = _get_base_market(ctx.state)

        # Resolve scenario pack built in prior step
        if "scenario_pack" not in ctx.state:
            raise KeyError("Missing ctx.state['scenario_pack']. Run BuildScenarioPackStep first.")
        pack: Sequence[Tuple[str, Any]] = ctx.state["scenario_pack"]

        # Build a runner (your risk package owns the pricing loop & pnl conventions)
        runner = ScenarioRunner()

        # Execute and store result (keep raw result object for flexibility)
        result = runner.run(portfolio=portfolio, base_market=base_market, scenarios=pack)
        ctx.state["scenario_result"] = result

        # Build a report object (or dict) for output + artifacts
        report = ScenarioReport.from_result(result)
        ctx.state["scenario_report"] = report

        return ctx


@dataclass(frozen=True, slots=True)
class WriteScenarioReportStep(Step):
    """
    Write scenario artifacts using ArtifactStore.

    This step should never crash if enable_save=False:
    ArtifactStore is responsible for no-op behaviour.
    """

    def run(self, ctx: Context) -> Context:
        # If no artifact store exists, there is nothing to write
        if ctx.artifact_store is None:
            return ctx

        # If the report doesn't exist, this is a pipeline wiring error
        if "scenario_report" not in ctx.state:
            raise KeyError("Missing ctx.state['scenario_report']. Run RunScenarioStep first.")

        report = ctx.state["scenario_report"]

        # Convert report to dicts for JSON + CSV (ScenarioReport already tested in your repo)
        report_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)

        # CSV rows: keep a stable header for diffs and tooling
        rows = report_dict.get("rows", [])
        header = ("scenario", "pv", "pnl")

        # Write artifacts under the run artifacts directory
        ctx.artifact_store.write_csv_rows("scenario_report.csv", header=header, rows=rows)
        ctx.artifact_store.write_json("scenario_report.json", report_dict)

        return ctx