# QuantStrata Quick Start

Get up and running with QuantStrata in a few minutes: install, create a venv, and run your first pricing example.

**Requirements:** Python 3.12+

---

## 1. Clone and setup

```bash
git clone https://github.com/quantstrata/quantstrata.git
cd quantstrata

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 2. First pricing example

Price a European FX vanilla option using Black–Scholes–Merton:

```python
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.interfaces import Quote
from src.marketdata.surfaces.vol_surface import FlatVolSurface
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer

spot_id = MarketId("FX", "SPOT", "EURUSD")
vol_id = MarketId("FX", "VOL", "EURUSD")
dom_curve_id = MarketId("IR", "CURVE", "USD.OIS")
for_curve_id = MarketId("IR", "CURVE", "EUR.OIS")

market = Market(
    asof="2026-01-27",
    quotes={spot_id: Quote(value=1.08)},
    curves={
        dom_curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.05),
        for_curve_id: FlatZeroRateCurve(continuously_compounded_rate=0.03),
    },
    vols={vol_id: FlatVolSurface(sigma=0.10)},
)

option = FxVanillaEuropeanOption(
    ccy_pair="EURUSD",
    option_type="call",
    strike=1.10,
    expiry=0.5,
    notional=1_000_000,
    spot_id=spot_id,
    vol_id=vol_id,
    domestic_curve_id=dom_curve_id,
    foreign_curve_id=for_curve_id,
)

pricer = FxEuropeanVanillaBsmPricer()
price = pricer.price(option, market)
greeks = pricer.greeks(option, market)

print(f"Price: {price:,.2f}")
print(f"Delta: {greeks.delta:.4f}, Vega: {greeks.vega:.2f}")
```

Run from the repo root (so `src` is on the path). You should see a price and Greeks.

---

## 3. Next steps

- **Full documentation:** [Documentation overview](README.md) — guides, reference, and project structure.
- **Interactive tutorials:** [Tutorials (Jupyter notebooks)](tutorials/README.md) — calibration, pricing, risk, and more.
- **Examples:** Scripts in `examples/` for market data, pricing, and risk.

For contributing and coding standards, see the [development](development/) docs.
