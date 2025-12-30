from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

from src.marketdata.market import Market
from src.instruments.fx.options.digital import EuropeanFxDigitalOption
from src.instruments.fx.options.vanilla import AmericanFxVanillaOption, EuropeanFxVanillaOption

from src.models.analytic.black_scholes.european import BlackScholesMertonEuropean


GreekName = Literal[
    "delta",
    "gamma",
    "vega",
    "rho_domestic",
    "rho_foreign",
]


def _rate_from_df(*, df: float, t: float) -> float:
    """
    Convert discount factor to continuously-compounded rate.

    df = exp(-rT)  =>  r = -ln(df)/T
    """
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


@dataclass(frozen=True, slots=True)
class FxEuropeanVanillaBsmPricer:
    """
    Adapter pricer: EuropeanFxVanillaOption -> BlackScholesMertonEuropean (generic carry).

    FX mapping
    ----------
    - r = r_d (domestic rate)
    - b = r_d - r_f  (cost-of-carry)
    - engine PV is "per 1 unit of foreign notional"
    - we multiply by notional_foreign to get domestic PV

    Greeks mapping (important!)
    ---------------------------
    The generic engine returns:
      - rho: dPV/dr holding b fixed
      - rho_carry: dPV/db holding r fixed  (name in engine may differ)

    For FX:
      b = r_d - r_f,  r = r_d

    Chain rule:
      dPV/dr_d = dPV/dr * (dr/dr_d) + dPV/db * (db/dr_d) = rho + rho_carry
      dPV/dr_f = dPV/db * (db/dr_f) = -rho_carry
    """

    # define model engine to use.
    engine: BlackScholesMertonEuropean = BlackScholesMertonEuropean()

    def price(self, trade: EuropeanFxVanillaOption, market: Market) -> float:
        # ---- Read market inputs ----
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        # Discount factors from curves (domestic and foreign).
        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        # Convert DFs -> continuous rates for the engine.
        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)

        # Generic carry parameter for the engine.
        b = float(r_d - r_f)

        # Implied vol from the surface.
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))

        # ---- Engine PV is per 1 unit foreign; scale by notional_foreign ----
        pv_per_unit = self.engine.price(
            option_type=trade.option_type,
            spot=S,
            strike=K,
            time_to_expiry=T,
            discount_rate=r_d,
            carry=b,
            sigma=sigma,
        )
        return float(trade.notional) * float(pv_per_unit)

    def greeks(self, trade: EuropeanFxVanillaOption, market: Market) -> Dict[GreekName, float]:
        # ---- Read market inputs ----
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)
        b = float(r_d - r_f)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))

        # ---- Engine greeks are per 1 unit foreign; scale by notional_foreign ----
        g = self.engine.greeks(
            option_type=trade.option_type,
            spot=S,
            strike=K,
            time_to_expiry=T,
            discount_rate=r_d,
            carry=b,
            sigma=sigma,
        )

        notional = float(trade.notional)

        delta = notional * float(g["delta"])
        gamma = notional * float(g["gamma"])
        vega = notional * float(g["vega"])

        # Engine provides:
        # - rho (wrt discount_rate r, holding carry fixed)
        # - rho_carry (wrt carry_rate b, holding r fixed)
        rho_r = notional * float(g["rho_discount"])
        rho_carry = notional * float(g["rho_carry"])

        rho_domestic = float(rho_r + rho_carry)
        rho_foreign = float(-rho_carry)

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "rho_domestic": float(rho_domestic),
            "rho_foreign": float(rho_foreign),
        }


@dataclass(frozen=True, slots=True)
class FxEuropeanDigitalBsmPricer:
    """
    Analytic BSM digital pricer for EuropeanFxDigitalOption.

    Implementation notes
    --------------------
    We reuse the same r_d, r_f mapping as vanilla:
      - r = r_d
      - b = r_d - r_f
      - df_d = exp(-r_d T)
      - df_f = exp(-r_f T)

    Digital payoff PV forms under generic carry (with df = exp(-rT)):
      cash-or-nothing:
        PV_call = payout * df * N(d2)
        PV_put  = payout * df * N(-d2)

      asset-or-nothing (asset = 1 unit of underlying):
        PV_call = payout * (S * exp((b-r)T)) * N(d1)
        PV_put  = payout * (S * exp((b-r)T)) * N(-d1)

    For FX "asset" means foreign units; PV is in domestic via S * df_f.
    """

    # define model engine to use.
    engine: BlackScholesMertonEuropean = BlackScholesMertonEuropean()

    def price(self, trade: EuropeanFxDigitalOption, market: Market) -> float:
        # ---- Read market inputs ----
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)
        b = float(r_d - r_f)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))

        # ---- Use engine internals consistently by calling its helpers via greeks/price ----
        # We call engine.greeks() to get d1/d2-related components indirectly is not ideal;
        # but we keep digital self-contained by recomputing d1/d2 locally here.
        if T == 0.0:
            # At expiry, PV is the payoff evaluated on S (deterministic).
            itm = (S > K) if trade.option_type == "call" else (S < K)
            if not itm:
                return 0.0
            if trade.payoff == "cash":
                return float(trade.payout_amount)
            # asset: pays payout_amount foreign units; worth payout_amount * S in domestic
            return float(trade.payout_amount) * float(S)

        # Compute d1/d2 for generic carry model:
        # d1 = (ln(S/K) + (b + 0.5*sigma^2)T)/(sigma*sqrtT), d2=d1-sigma*sqrtT
        sqrtT = math.sqrt(T)
        vol_sqrtT = sigma * sqrtT
        d1 = (math.log(S / K) + (b + 0.5 * sigma * sigma) * T) / vol_sqrtT
        d2 = d1 - vol_sqrtT

        # Standard normal CDF.
        from src.models.common.normal import std_norm_cdf  # local import to avoid cycles in some layouts

        payout = float(trade.payout_amount)

        # FX carry factor: exp((b-r)T) = exp((r_d-r_f-r_d)T) = exp(-r_f T) = df_f
        fwd_factor = float(math.exp((b - r_d) * T))

        if trade.payoff == "cash":
            # cash-or-nothing PV uses domestic discounting.
            if trade.option_type == "call":
                pv = df_d * payout * std_norm_cdf(d2)
            else:
                pv = df_d * payout * std_norm_cdf(-d2)
            return float(pv)

        # asset-or-nothing: pays payout foreign units; PV in domestic.
        # PV = payout * S * exp((b-r)T) * N(±d1) = payout * S * df_f * N(±d1)
        if trade.option_type == "call":
            pv = payout * (S * fwd_factor) * std_norm_cdf(d1)
        else:
            pv = payout * (S * fwd_factor) * std_norm_cdf(-d1)

        return float(pv)

    def greeks(self, trade: EuropeanFxDigitalOption, market: Market) -> Dict[GreekName, float]:
        """
        Optional: provide greeks so digitals integrate cleanly with PortfolioPricer/attribution.

        Notes
        -----
        Digitals have discontinuous payoffs; greeks can be large and sensitive.
        This is still useful for *small bump* validations.
        """
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)

        df_d = float(market.curve(trade.domestic_curve_id).df(T))
        df_f = float(market.curve(trade.foreign_curve_id).df(T))

        r_d = _rate_from_df(df=df_d, t=T)
        r_f = _rate_from_df(df=df_f, t=T)
        b = float(r_d - r_f)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))

        if T == 0.0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho_domestic": 0.0, "rho_foreign": 0.0}

        sqrtT = math.sqrt(T)
        vol_sqrtT = sigma * sqrtT
        d1 = (math.log(S / K) + (b + 0.5 * sigma * sigma) * T) / vol_sqrtT
        d2 = d1 - vol_sqrtT

        from src.models.common.normal import std_norm_cdf, std_norm_pdf

        payout = float(trade.payout_amount)
        fwd_factor = float(math.exp((b - r_d) * T))  # == df_f in FX mapping

        # Handy densities.
        nd1 = std_norm_pdf(d1)
        nd2 = std_norm_pdf(d2)

        # Start with PV and then compute sensitivities in closed form.
        pv = self.price(trade, market)

        # Delta/gamma/vega depend on payoff style.
        if trade.payoff == "cash":
            # PV = df_d * payout * N(±d2)
            sign = 1.0 if trade.option_type == "call" else -1.0

            # dN(sign*d2)/dS = n(d2) * sign * (dd2/dS), where dd2/dS = 1/(S*sigma*sqrtT)
            dd2_dS = 1.0 / (S * sigma * sqrtT)
            delta = df_d * payout * nd2 * sign * dd2_dS

            # Gamma: derivative of delta w.r.t. S (closed form)
            # This expands; keep explicit and readable:
            # delta = A * nd2 * (1/S) where A = df_d*payout*sign/(sigma*sqrtT)
            A = df_d * payout * sign / (sigma * sqrtT)
            # d(nd2)/dS = nd2 * (-d2) * dd2/dS
            dnd2_dS = nd2 * (-d2) * dd2_dS
            gamma = A * (dnd2_dS * (1.0 / S) + nd2 * (-1.0 / (S * S)))

            # Vega: derivative via d2
            # dN(sign*d2)/dsigma = n(d2) * sign * (dd2/dsigma)
            # dd2/dsigma = dd1/dsigma - sqrtT, and dd1/dsigma has known form:
            # dd1/dsigma = - (ln(S/K) + (b + 0.5*sigma^2)T) / (sigma^2*sqrtT) + sqrtT
            dd1_dsigma = -(math.log(S / K) + (b + 0.5 * sigma * sigma) * T) / (sigma * sigma * sqrtT) + sqrtT
            dd2_dsigma = dd1_dsigma - sqrtT
            vega = df_d * payout * nd2 * sign * dd2_dsigma

            # Rate greeks: for digitals we keep a simple chain-rule approach using engine’s rho pieces
            # by approximating rho_domestic/rho_foreign numerically would be another option,
            # but we keep V1 lightweight: reuse generic engine rho split.
            #
            # For now we provide 0 for rate greeks (safe default). You can upgrade later.
            rho_domestic = 0.0
            rho_foreign = 0.0

        else:
            # asset-or-nothing: PV = payout * S * fwd_factor * N(±d1)
            sign = 1.0 if trade.option_type == "call" else -1.0
            Nd1 = std_norm_cdf(sign * d1)

            # Delta: d/dS [ payout * S*fwd_factor*N(sign*d1) ]
            dd1_dS = 1.0 / (S * sigma * sqrtT)
            delta = payout * fwd_factor * (Nd1 + S * std_norm_pdf(sign * d1) * sign * dd1_dS)

            # Gamma: second derivative (keep explicit, readable)
            # Use product rule on delta expression:
            n_sd1 = std_norm_pdf(sign * d1)
            dn_sd1_dS = n_sd1 * (-(sign * d1)) * (sign * dd1_dS)  # derivative of pdf
            gamma = payout * fwd_factor * (
                (n_sd1 * sign * dd1_dS) + (n_sd1 * sign * dd1_dS) + (S * dn_sd1_dS * sign * dd1_dS) + (S * n_sd1 * sign * (-dd1_dS / S))
            )

            # Vega: via d1
            dd1_dsigma = -(math.log(S / K) + (b + 0.5 * sigma * sigma) * T) / (sigma * sigma * sqrtT) + sqrtT
            vega = payout * (S * fwd_factor) * n_sd1 * sign * dd1_dsigma

            rho_domestic = 0.0
            rho_foreign = 0.0

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "rho_domestic": float(rho_domestic),
            "rho_foreign": float(rho_foreign),
        }


@dataclass(frozen=True, slots=True)
class FxAmericanVanillaPlaceholderPricer:
    """
    Placeholder pricer for American FX vanilla.

    In V2/V3 you can replace this with:
    - tree pricer
    - finite difference (PDE/FD)
    - LSMC
    """
    def price(self, trade: AmericanFxVanillaOption, market: Market) -> float:
        raise NotImplementedError(
            "American FX vanilla pricing not implemented yet. "
            "Use a tree/FD/LSMC pricer and register it for AmericanFxVanillaOption."
        )

    def greeks(self, trade: AmericanFxVanillaOption, market: Market) -> Dict[str, float]:
        raise NotImplementedError(
            "American FX vanilla greeks not implemented yet."
        )