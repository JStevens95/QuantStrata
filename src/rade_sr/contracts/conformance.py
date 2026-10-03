"""
Conformance checks for the rade_sr contracts.

Each ``check_*`` returns a :class:`CheckResult` and never raises — failures are
captured with an actionable message. ``run_all`` prints a summary table.
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Outcome of one conformance check."""
    name: str
    passed: bool
    detail: str = ""
    sub: List["CheckResult"] = field(default_factory=list)

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}" + (f" — {self.detail}" if self.detail else "")


# ─────────────────────────────────────────────────────────────────────────
# MarketDataClient
# ─────────────────────────────────────────────────────────────────────────

def check_market_data_client(client: Any, fx_pair: str = "EURUSD") -> CheckResult:
    """Verify a MarketDataClient loads + validates an FXAsset (and its IR deps).

    This exercises every ``get_fx_*`` and ``get_ir_*`` method, the asset
    ``_build_*`` logic, and the axis-alignment contracts in ``validate()``.
    """
    subs: List[CheckResult] = []

    # 1. method-presence + shape spot-checks (clearer errors than a deep stack)
    required = [
        "get_fx_spot", "get_fx_forward_points", "get_fx_atm_vol",
        "get_fx_smile_vol", "get_fx_shocks", "get_ir_curve", "get_ir_history",
        "get_ir_atm_vol", "get_ir_smile_vol", "get_ir_shocks",
    ]
    missing = [m for m in required if not callable(getattr(client, m, None))]
    subs.append(CheckResult(
        "methods present", not missing,
        "missing: " + ", ".join(missing) if missing else "all 10 methods present",
    ))

    spot_ok, spot_detail = _safe_shape_check(
        lambda: client.get_fx_spot(fx_pair),
        keys=("spot", "dates", "values"),
    )
    subs.append(CheckResult("get_fx_spot shape", spot_ok, spot_detail))

    shock_ok, shock_detail = _safe_shape_check(
        lambda: client.get_fx_shocks(fx_pair),
        keys=("spot_shocks", "vol_shocks", "vol_expiries", "vol_strikes",
              "domestic_rate_shocks", "foreign_rate_shocks"),
    )
    subs.append(CheckResult("get_fx_shocks shape", shock_ok, shock_detail))

    # 2. end-to-end load + validate (the real test)
    load_ok, load_detail = _safe_load_fx_asset(client, fx_pair)
    subs.append(CheckResult("FXAsset load + validate", load_ok, load_detail))

    passed = all(s.passed for s in subs)
    return CheckResult(
        f"MarketDataClient ({fx_pair})", passed,
        "" if passed else "see sub-checks", sub=subs,
    )


def _safe_shape_check(call, keys) -> tuple:
    try:
        out = call()
    except Exception as exc:  # noqa: BLE001
        return False, f"call raised: {exc}"
    if not isinstance(out, dict):
        return False, f"expected dict, got {type(out).__name__}"
    missing = [k for k in keys if k not in out]
    if missing:
        return False, f"missing keys: {missing}"
    return True, "ok"


def _safe_load_fx_asset(client: Any, fx_pair: str) -> tuple:
    try:
        from src.rade_sr.assets.fx import FXAsset
        from src.rade_sr.assets.types import AssetConfig

        asset = FXAsset(AssetConfig(
            asset_class="fx",
            asset_name=fx_pair,
            extra={"pair": fx_pair, "base_ccy": fx_pair[:3], "quote_ccy": fx_pair[3:6]},
        ))
        asset.load(client)
        ok = asset.validate()
        if not ok:
            return False, "asset.validate() returned False (check logs for axis mismatches)"
        n = asset.shocks.spot.shape[0] if asset.shocks is not None else 0
        return True, f"loaded + validated ({n} shock scenarios)"
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc}\n{traceback.format_exc(limit=3)}"


# ─────────────────────────────────────────────────────────────────────────
# PortfolioSource
# ─────────────────────────────────────────────────────────────────────────

def check_portfolio_source(source: Any, resolvers: Optional[dict] = None) -> CheckResult:
    """Verify a PortfolioSource yields ingestible attribute + PnL frames."""
    subs: List[CheckResult] = []
    try:
        attrs = source.load_attributes()
        pnl = source.load_pnl()
        subs.append(CheckResult(
            "load frames", True,
            f"attributes rows={len(attrs)}, pnl rows={len(pnl)}",
        ))
    except Exception as exc:  # noqa: BLE001
        return CheckResult("PortfolioSource", False, f"load failed: {exc}")

    try:
        from src.rade_sr.replication.ingest import ingest_portfolio

        inputs = ingest_portfolio(attrs, pnl, resolvers=resolvers)
        n_rf = len(inputs.asset_config)
        n_tr = len(inputs.attributes)
        tgt = None if inputs.target_pnl is None else inputs.target_pnl.shape
        subs.append(CheckResult(
            "ingest_portfolio", n_rf > 0 and n_tr > 0,
            f"{n_tr} trades, {n_rf} risk factors, target_pnl={tgt}",
        ))
    except Exception as exc:  # noqa: BLE001
        subs.append(CheckResult("ingest_portfolio", False, f"raised: {exc}"))

    passed = all(s.passed for s in subs)
    return CheckResult("PortfolioSource", passed, sub=subs)


# ─────────────────────────────────────────────────────────────────────────
# Pricer
# ─────────────────────────────────────────────────────────────────────────

def check_pricer(pricer: Any, market_data_client: Any, fx_pair: str = "EURUSD") -> CheckResult:
    """Verify a pricer returns a finite (n_trades, n_scenarios) PnL matrix.

    Loads one FX asset, generates a small trade set, and prices it.
    """
    try:
        import numpy as np

        from src.rade_sr.assets.fx import FXAsset
        from src.rade_sr.assets.types import AssetConfig
        from src.rade_sr.replication.trade_generator import RiskFactorAwareTradeGenerator

        asset = FXAsset(AssetConfig(
            asset_class="fx", asset_name=fx_pair,
            extra={"pair": fx_pair, "base_ccy": fx_pair[:3], "quote_ccy": fx_pair[3:6]},
        ))
        asset.load(market_data_client)

        gen = RiskFactorAwareTradeGenerator()
        trades = gen.generate({fx_pair: asset}, {
            "maturities": [0.25, 1.0],
            "fx": {"strike_pcts": [0.95, 1.0, 1.05]},
        })[fx_pair]

        matrix = pricer.price_trades(trades, asset)
        matrix = np.asarray(matrix)
        n_scen = asset.shocks.spot.shape[0]
        shape_ok = matrix.shape[0] == len(trades)
        finite = bool(np.isfinite(matrix).all())
        detail = (
            f"matrix={matrix.shape} (expected {len(trades)} trades x {n_scen} scen), "
            f"finite={finite}"
        )
        return CheckResult("Pricer (FX)", shape_ok and finite, detail)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Pricer (FX)", False, f"{exc}\n{traceback.format_exc(limit=3)}")


# ─────────────────────────────────────────────────────────────────────────
# Artifacts round-trip (rade_ml_pt handoff)
# ─────────────────────────────────────────────────────────────────────────

def check_artifacts_roundtrip(output_dir: str) -> CheckResult:
    """Read back one cluster's artifacts and verify the consumer-side contract."""
    try:
        import pickle

        import pandas as pd

        manifest_path = Path(output_dir) / "jobs.pkl"
        if not manifest_path.exists():
            return CheckResult("Artifacts round-trip", False, f"no manifest at {manifest_path}")
        with open(manifest_path, "rb") as f:
            manifest = pickle.load(f)
        if not manifest:
            return CheckResult("Artifacts round-trip", False, "empty manifest")

        info = manifest[0]["cluster_info"]
        elem_pnl = pd.read_parquet(info["elementary_pnl_path"])
        tgt_pnl = pd.read_parquet(info["target_pnl_path"])
        with open(info["elementary_attribs_path"], "rb") as f:
            elem_attr = pickle.load(f)
        with open(info["target_attribs_path"], "rb") as f:
            tgt_attr = pickle.load(f)

        checks = []
        # PnL orientation: rows = scenarios, columns = trades; column ids use '|'
        checks.append(("elem_pnl is DataFrame", isinstance(elem_pnl, pd.DataFrame)))
        checks.append(("elem column id has '|'",
                       len(elem_pnl.columns) > 0 and "|" in str(elem_pnl.columns[0])))
        checks.append(("attrs are dict[str,list]",
                       isinstance(elem_attr, dict) and isinstance(tgt_attr, dict)))
        checks.append(("attrs have trade_id + yrs_to_maturity",
                       "trade_id" in elem_attr and "yrs_to_maturity" in elem_attr))
        checks.append(("attr length matches pnl cols",
                       len(elem_attr.get("trade_id", [])) == elem_pnl.shape[1]))
        checks.append(("scenario rows align",
                       elem_pnl.shape[0] == tgt_pnl.shape[0]))

        failed = [n for n, ok in checks if not ok]
        passed = not failed
        detail = (
            f"elem_pnl={elem_pnl.shape}, tgt_pnl={tgt_pnl.shape}, "
            f"elem_attr_keys={len(elem_attr)}"
            + ("" if passed else f"; FAILED: {failed}")
        )
        return CheckResult(f"Artifacts round-trip ({manifest[0]['cluster_id']})", passed, detail)
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Artifacts round-trip", False, f"{exc}\n{traceback.format_exc(limit=3)}")


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────

def run_all(
    market_data_client: Any = None,
    portfolio_source: Any = None,
    pricer: Any = None,
    fx_pair: str = "EURUSD",
    artifacts_dir: Optional[str] = None,
    resolvers: Optional[dict] = None,
) -> List[CheckResult]:
    """Run every applicable check and print a summary. Returns the results."""
    results: List[CheckResult] = []
    if portfolio_source is not None:
        results.append(check_portfolio_source(portfolio_source, resolvers=resolvers))
    if market_data_client is not None:
        results.append(check_market_data_client(market_data_client, fx_pair=fx_pair))
    if pricer is not None and market_data_client is not None:
        results.append(check_pricer(pricer, market_data_client, fx_pair=fx_pair))
    if artifacts_dir is not None:
        results.append(check_artifacts_roundtrip(artifacts_dir))

    print("\n══ rade_sr conformance ════════════════════════════════════")
    for r in results:
        print(r)
        for s in r.sub:
            print(f"    └─ {s}")
    n_pass = sum(r.passed for r in results)
    print(f"── {n_pass}/{len(results)} top-level checks passed ──\n")
    return results
