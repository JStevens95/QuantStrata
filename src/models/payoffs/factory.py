from __future__ import annotations

from typing import Union, TypeAlias
from dataclasses import dataclass

from src.models.payoffs.base import BasePayoff1D, BasePathPayoff1D
from src.models.payoffs.vanilla import VanillaPayoff
from src.models.payoffs.digital import DigitalCashPayoff, DigitalAssetPayoff
from src.models.payoffs.barrier import SingleBarrierPayoff
from src.models.payoffs.double_barrier import DoubleBarrierPayoff
from src.models.payoffs.asian import AsianPayoff
from src.models.payoffs.lookback import LookbackPayoff
from src.models.payoffs.touch import TouchPayoff

# Instruments
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.instruments.fx.options.digital import EuropeanFxDigitalOption
from src.instruments.fx.options.barrier import EuropeanFxBarrierOption
from src.instruments.fx.options.double_barrier import EuropeanFxDoubleBarrierOption
from src.instruments.fx.options.asian import EuropeanFxAsianOption
from src.instruments.fx.options.lookback import EuropeanFxLookbackOption
from src.instruments.fx.options.touch import EuropeanFxTouchOption

from src.models.payoffs.types import OptionType


# ======================================================================================
# Public payoff type (we ONLY traffic in the concrete base classes)
# ======================================================================================

Payoff1D: TypeAlias = Union[BasePayoff1D, BasePathPayoff1D]


# ======================================================================================
# Factory / router
# ======================================================================================

@dataclass(frozen=True, slots=True)
class PayoffFactory:
    """
    Central payoff router: instrument -> payoff object.

    Why this exists
    ---------------
    - Pricers should not hard-code product-specific payoff construction.
    - Adding a new product should usually mean:
        (1) implement payoff class
        (2) add ONE routing rule here
      and not touch every pricer.

    Contract
    --------
    - Terminal-only payoffs: subclass BasePayoff1D (terminal(spot))
    - Path-dependent payoffs: subclass BasePathPayoff1D (terminal_from_paths(paths))
    """

    def build(self, instrument: object) -> Payoff1D:
        """
        Build a payoff object for the given instrument.

        Raises
        ------
        TypeError
            If the instrument is not supported by the payoff library yet.
        ValueError
            If the instrument carries an unsupported payoff configuration.
        """
        # ---------------------------
        # FX European Vanilla
        # ---------------------------
        if isinstance(instrument, EuropeanFxVanillaOption):
            opt: OptionType = instrument.option_type
            return VanillaPayoff(option_type=opt, strike=float(instrument.strike))

        # ---------------------------
        # FX European Digital
        # ---------------------------
        if isinstance(instrument, EuropeanFxDigitalOption):
            opt: OptionType = instrument.option_type
            k = float(instrument.strike)
            payout = float(instrument.payout_amount)

            if instrument.payoff == "cash":
                return DigitalCashPayoff(option_type=opt, strike=k, cash=payout)
            if instrument.payoff == "asset":
                return DigitalAssetPayoff(option_type=opt, strike=k, asset_units=payout)

            raise ValueError(f"Unsupported digital payoff style: {instrument.payoff!r}")

        # ---------------------------
        # FX European Single Barrier (discrete monitoring, MC)
        # ---------------------------
        if isinstance(instrument, EuropeanFxBarrierOption):
            opt: OptionType = instrument.option_type
            return SingleBarrierPayoff(
                option_type=opt,
                strike=float(instrument.strike),
                barrier_direction=instrument.barrier_direction,  # type: ignore[arg-type]
                barrier_style=instrument.barrier_style,          # type: ignore[arg-type]
                barrier_level=float(instrument.barrier_level),
                rebate_amount=float(instrument.rebate_amount),
            )

        # ---------------------------
        # FX European Double Barrier (corridor option, path-dependent, MC)
        # ---------------------------
        if isinstance(instrument, EuropeanFxDoubleBarrierOption):
            opt: OptionType = instrument.option_type
            return DoubleBarrierPayoff(
                option_type=opt,
                strike=float(instrument.strike),
                barrier_style=instrument.barrier_style,  # type: ignore[arg-type]
                lower_barrier=float(instrument.lower_barrier),
                upper_barrier=float(instrument.upper_barrier),
                rebate_amount=float(instrument.rebate_amount),
            )

        # ---------------------------
        # FX European Asian (average price option, path-dependent, MC)
        # ---------------------------
        if isinstance(instrument, EuropeanFxAsianOption):
            opt: OptionType = instrument.option_type
            return AsianPayoff(
                option_type=opt,
                strike=float(instrument.strike),
                averaging_type=instrument.averaging_type,  # type: ignore[arg-type]
            )

        # ---------------------------
        # FX European Lookback (path extremum option, path-dependent, MC)
        # ---------------------------
        if isinstance(instrument, EuropeanFxLookbackOption):
            opt: OptionType = instrument.option_type
            return LookbackPayoff(
                option_type=opt,
                lookback_type=instrument.lookback_type,  # type: ignore[arg-type]
                strike=float(instrument.strike),
            )

        # ---------------------------
        # FX European Touch (binary barrier option, path-dependent, MC)
        # ---------------------------
        if isinstance(instrument, EuropeanFxTouchOption):
            return TouchPayoff(
                touch_style=instrument.touch_style,  # type: ignore[arg-type]
                barrier_direction=instrument.barrier_direction,  # type: ignore[arg-type]
                barrier_level=float(instrument.barrier_level),
                payout_amount=float(instrument.payout_amount),
            )

        # ---------------------------
        # Not supported yet
        # ---------------------------
        raise TypeError(f"No payoff mapping registered for instrument type: {type(instrument).__name__}")


# ======================================================================================
# Convenience singletons / helpers
# ======================================================================================

_DEFAULT_FACTORY = PayoffFactory()


def build_payoff_1d(instrument: object) -> Payoff1D:
    """Convenience wrapper around the default PayoffFactory."""
    return _DEFAULT_FACTORY.build(instrument)


def require_terminal_payoff(payoff: Payoff1D) -> BasePayoff1D:
    """
    Ensure the payoff is terminal-only.

    Use this in analytic PDE / BSM pricers that *must not* accept path-dependent payoffs.
    """
    if isinstance(payoff, BasePayoff1D):
        return payoff
    raise TypeError(
        f"This pricer requires a terminal-only payoff (BasePayoff1D), but got: {type(payoff).__name__}"
    )


def require_path_payoff(payoff: Payoff1D) -> BasePathPayoff1D:
    """
    Ensure payoff is path-dependent (requires full paths).

    Use this in Monte Carlo pricers that need barrier/asian/lookback path logic.
    """
    if isinstance(payoff, BasePathPayoff1D):
        return payoff
    raise TypeError(
        f"This pricer requires a path-dependent payoff (BasePathPayoff1D), but got: {type(payoff).__name__}"
    )