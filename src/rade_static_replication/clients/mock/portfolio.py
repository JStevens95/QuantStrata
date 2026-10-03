"""
Mock portfolio client.

Generates a raw, *exploded* FX + Rates portfolio in the exact shape of the external
export (one row per trade × risk-type, notional row tagged ``AssetClass="ALL"``) plus
a scenario PnL frame. Replace with the Sage adapter in production.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.rade_static_replication.domain.contracts import RawPortfolio

_FX_PAIRS = ["EURUSD", "GBPUSD", "AUDUSD"]
_RATES_FACTORS = ["IR_CURVE_SWAP.USD", "IR_CURVE_SWAP.EUR", "IR_CURVE_SWAP.GBP"]


class MockPortfolioClient:
    """Produce a representative cross-asset desk portfolio."""

    def __init__(self, n_fx: int = 6, n_rates: int = 4, n_scenarios: int = 60, seed: int = 7) -> None:
        self.n_fx = n_fx
        self.n_rates = n_rates
        self.n_scenarios = n_scenarios
        self.rng = np.random.default_rng(seed)

    def load(self, cob_date: str) -> RawPortfolio:
        rows: List[dict] = []
        trade_ids: List[str] = []
        asset_of: dict[str, str] = {}
        tid = 0

        for i in range(self.n_fx):
            pair = _FX_PAIRS[i % len(_FX_PAIRS)]
            t = f"FX{tid:04d}"
            tid += 1
            trade_ids.append(t)
            asset_of[t] = "FX"
            bs = "Buy" if self.rng.random() > 0.5 else "Sell"
            mat = ["20270615", "20280615", "20290615"][i % 3]
            for rt, val, ac in [
                ("FXPV", self.rng.normal(0, 1e5), "FX"),
                ("FXPOS", self.rng.normal(0, 5e5), "FX"),
                ("Notional", abs(self.rng.normal(1e7, 2e6)), "ALL"),
            ]:
                rows.append({
                    "PTSDealNumber": t, "AssetClass": ac, "RiskType": rt,
                    "Sum_RiskValuesUSD_Net": val, "BuySellInd": bs,
                    "MaturityDate": mat, "DeskName": "G10_FX",
                    "Product": "VanillaOption", "ccy": pair, "RatesFactor": "",
                })

        for i in range(self.n_rates):
            factor = _RATES_FACTORS[i % len(_RATES_FACTORS)]
            t = f"IR{tid:04d}"
            tid += 1
            trade_ids.append(t)
            asset_of[t] = "RATES"
            bs = "Buy" if self.rng.random() > 0.5 else "Sell"
            mat = ["20300615", "20340615"][i % 2]
            for rt, val, ac in [
                ("IRPV", self.rng.normal(0, 2e5), "RATES"),
                ("Notional", abs(self.rng.normal(5e7, 1e7)), "ALL"),
            ]:
                rows.append({
                    "PTSDealNumber": t, "AssetClass": ac, "RiskType": rt,
                    "Sum_RiskValuesUSD_Net": val, "BuySellInd": bs,
                    "MaturityDate": mat, "DeskName": f"Rates_{factor.split('.')[-1]}",
                    "Product": "Swaption", "ccy": "", "RatesFactor": factor,
                })

        attributes = pd.DataFrame(rows)

        scen_cols = [f"scen_{i}" for i in range(self.n_scenarios)]
        pnl = self.rng.normal(0, 5e4, (len(trade_ids), self.n_scenarios))
        target = pd.DataFrame(pnl, index=trade_ids, columns=scen_cols)
        target.index.name = "trade_id"
        target["AssetClass"] = [asset_of[t] for t in trade_ids]

        return RawPortfolio(attributes=attributes, target_pnl=target, cob_date=cob_date, source="mock")
