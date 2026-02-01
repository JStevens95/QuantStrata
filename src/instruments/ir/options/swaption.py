# src/instruments/ir/options/swaption.py
"""
Swaption (Option on Interest Rate Swap) Instrument.

Mathematical Framework
----------------------
A swaption is an option to enter into an interest rate swap at a future date.

**Payer Swaption**: Right to enter a payer swap (pay fixed, receive floating)
    - Benefits when rates rise above the strike
    - Call option on the swap rate

**Receiver Swaption**: Right to enter a receiver swap (receive fixed, pay floating)
    - Benefits when rates fall below the strike
    - Put option on the swap rate

Payoff at Expiry
----------------
At option expiry T_opt, the holder can enter a swap starting at T_opt.

Payer Swaption Payoff:
    max(0, S - K) × A × N

Receiver Swaption Payoff:
    max(0, K - S) × A × N

Where:
- S = swap rate at expiry
- K = strike rate
- A = annuity (PV01) of the underlying swap
- N = notional

Bachelier (Normal) Pricing
--------------------------
Swaptions are commonly priced using the Bachelier (normal) model, especially
in negative/low rate environments:

Payer:   PV = A × N × [(F - K) N(d) + σ√T n(d)]
Receiver: PV = A × N × [(K - F) N(-d) + σ√T n(d)]

Where:
- F = forward swap rate
- σ = normal volatility (in same units as rate, e.g., 0.005 = 50bp)
- d = (F - K) / (σ√T)

Settlement Styles
-----------------
- **Cash Settlement**: Settle the option value in cash at expiry
- **Physical Settlement**: Actually enter the swap at expiry

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.marketdata.core.ids import MarketId
from src.instruments.ir.options.capfloor import DayCountConvention

from src.instruments.core.types import SwaptionType, SettlementStyle


# =============================================================================
# SIMPLE INSTRUMENT (Direct Parameters)
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrSwaptionEuropeanOptionSimple:
    """
    European swaption with direct parameter input.
    
    A swaption is an option to enter into an interest rate swap.
    
    Parameters
    ----------
    notional : float
        Notional principal of the underlying swap.
    strike : float
        Strike swap rate K (e.g., 0.05 for 5%).
    option_expiry : float
        Time to option expiry in years.
    swap_tenor : float
        Tenor of the underlying swap in years.
    forward_swap_rate : float
        Current forward swap rate F.
    annuity : float
        Annuity (PV01) of the underlying swap.
    vol : float
        Normal (Bachelier) volatility σ.
    discount_factor : float
        Discount factor to option expiry.
    swaption_type : SwaptionType
        "payer" (call on rate) or "receiver" (put on rate).
    settlement : SettlementStyle
        "cash" or "physical" settlement.
    
    Examples
    --------
    1Y into 5Y payer swaption (1Y5Y):
        >>> swaption = IrSwaptionEuropeanOptionSimple(
        ...     notional=10_000_000,
        ...     strike=0.04,              # 4% strike
        ...     option_expiry=1.0,        # 1 year to expiry
        ...     swap_tenor=5.0,           # 5 year underlying swap
        ...     forward_swap_rate=0.042,  # Current forward swap rate
        ...     annuity=4.5,              # PV01 of underlying
        ...     vol=0.0060,               # 60bp normal vol
        ...     discount_factor=0.95,
        ...     swaption_type="payer",
        ... )
    """
    notional: float
    strike: float
    option_expiry: float           # Time to option expiry
    swap_tenor: float              # Tenor of underlying swap
    forward_swap_rate: float       # F = forward swap rate
    annuity: float                 # A = PV01 of underlying swap
    vol: float                     # σ = normal vol (absolute, not %)
    discount_factor: float         # DF to expiry
    swaption_type: SwaptionType = "payer"
    settlement: SettlementStyle = "cash"
    
    def __post_init__(self) -> None:
        """Validate swaption inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.option_expiry) < 0.0:
            raise ValueError("option_expiry must be >= 0.")
        if float(self.swap_tenor) <= 0.0:
            raise ValueError("swap_tenor must be > 0.")
        if float(self.annuity) <= 0.0:
            raise ValueError("annuity must be > 0.")
        if float(self.vol) < 0.0:
            raise ValueError("vol must be >= 0.")
        if float(self.discount_factor) <= 0.0:
            raise ValueError("discount_factor must be > 0.")
        if self.swaption_type not in ("payer", "receiver"):
            raise ValueError(f"swaption_type must be 'payer' or 'receiver'; got {self.swaption_type}")
        if self.settlement not in ("cash", "physical"):
            raise ValueError(f"settlement must be 'cash' or 'physical'; got {self.settlement}")
    
    @property
    def tenor_description(self) -> str:
        """
        Standard tenor notation (e.g., "1Y5Y" for 1Y into 5Y).
        """
        opt_y = int(round(float(self.option_expiry)))
        swap_y = int(round(float(self.swap_tenor)))
        
        opt_str = f"{opt_y}Y" if opt_y >= 1 else f"{int(self.option_expiry * 12)}M"
        swap_str = f"{swap_y}Y" if swap_y >= 1 else f"{int(self.swap_tenor * 12)}M"
        
        return f"{opt_str}{swap_str}"
    
    @property
    def is_in_the_money(self) -> bool:
        """
        Check if swaption is ITM.
        
        Payer: ITM when F > K
        Receiver: ITM when F < K
        """
        if self.swaption_type == "payer":
            return float(self.forward_swap_rate) > float(self.strike)
        return float(self.forward_swap_rate) < float(self.strike)


# =============================================================================
# MARKET DATA INSTRUMENT (Lookup from Market)
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrSwaptionEuropeanOption:
    """
    European swaption with market data lookup.
    
    Parameters
    ----------
    notional : float
        Notional principal of the underlying swap.
    strike : float
        Strike swap rate K.
    option_expiry : float
        Time to option expiry in years.
    swap_start : float
        When the underlying swap starts (usually = option_expiry).
    swap_end : float
        When the underlying swap ends.
    fixed_frequency : float
        Fixed leg frequency of underlying swap.
    floating_frequency : float
        Floating leg frequency of underlying swap.
    fixed_day_count : DayCountConvention
        Day count for fixed leg.
    floating_day_count : DayCountConvention
        Day count for floating leg.
    swaption_type : SwaptionType
        "payer" or "receiver".
    settlement : SettlementStyle
        "cash" or "physical".
    curve_id : MarketId
        Market identifier for discount/forward curve.
    vol_id : MarketId
        Market identifier for swaption volatility surface.
    
    Examples
    --------
    1Y into 5Y payer swaption:
        >>> swaption = IrSwaptionEuropeanOption(
        ...     notional=10_000_000,
        ...     strike=0.04,
        ...     option_expiry=1.0,
        ...     swap_start=1.0,
        ...     swap_end=6.0,          # 1Y + 5Y = 6Y total
        ...     swaption_type="payer",
        ...     curve_id=MarketId("IR", "CURVE", "USD.SOFR"),
        ...     vol_id=MarketId("IR", "VOL", "USD.SWAPTION"),
        ... )
    """
    notional: float
    strike: float
    option_expiry: float
    swap_start: float
    swap_end: float
    fixed_frequency: float = 0.5       # Semi-annual
    floating_frequency: float = 0.25   # Quarterly
    fixed_day_count: DayCountConvention = "30/360"
    floating_day_count: DayCountConvention = "ACT/360"
    swaption_type: SwaptionType = "payer"
    settlement: SettlementStyle = "cash"
    curve_id: MarketId = field(default_factory=lambda: MarketId("IR", "CURVE", "UNKNOWN"))
    vol_id: MarketId = field(default_factory=lambda: MarketId("IR", "VOL", "UNKNOWN"))
    
    def __post_init__(self) -> None:
        """Validate swaption inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.option_expiry) < 0.0:
            raise ValueError("option_expiry must be >= 0.")
        if float(self.swap_start) < float(self.option_expiry):
            raise ValueError("swap_start must be >= option_expiry.")
        if float(self.swap_end) <= float(self.swap_start):
            raise ValueError("swap_end must be > swap_start.")
        if self.swaption_type not in ("payer", "receiver"):
            raise ValueError(f"swaption_type must be 'payer' or 'receiver'")
        if self.settlement not in ("cash", "physical"):
            raise ValueError(f"settlement must be 'cash' or 'physical'")
    
    @property
    def swap_tenor(self) -> float:
        """Tenor of the underlying swap in years."""
        return float(self.swap_end) - float(self.swap_start)
    
    @property
    def tenor_description(self) -> str:
        """Standard tenor notation (e.g., "1Y5Y")."""
        opt_y = float(self.option_expiry)
        swap_t = self.swap_tenor
        
        if opt_y >= 1:
            opt_str = f"{int(opt_y)}Y"
        else:
            opt_str = f"{int(opt_y * 12)}M"
        
        if swap_t >= 1:
            swap_str = f"{int(swap_t)}Y"
        else:
            swap_str = f"{int(swap_t * 12)}M"
        
        return f"{opt_str}{swap_str}"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "SwaptionType",
    "SettlementStyle",
    # Instruments
    "IrSwaptionEuropeanOption",
    "IrSwaptionEuropeanOptionSimple",
]
