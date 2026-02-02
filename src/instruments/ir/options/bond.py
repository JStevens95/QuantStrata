# src/instruments/ir/options/bond.py
"""
Bond Option Instruments.

Mathematical Framework
----------------------
A bond option is an option to buy (call) or sell (put) a bond at a future date.

**Call Option**: Right to buy the bond at strike price K
    - Benefits when bond prices rise (rates fall)
    - Payoff: max(B_T - K, 0)

**Put Option**: Right to sell the bond at strike price K
    - Benefits when bond prices fall (rates rise)
    - Payoff: max(K - B_T, 0)

Where:
- B_T = bond price at option expiry
- K = strike price

Black76 Pricing
---------------
Bond options are commonly priced using Black76 model on the forward bond price:

Call:  PV = DF × [F × N(d₁) - K × N(d₂)]
Put:   PV = DF × [K × N(-d₂) - F × N(-d₁)]

Where:
- F = forward bond price = B_0 × exp(r × T) - PV(coupons during option life)
- DF = discount factor to expiry
- d₁ = [ln(F/K) + σ²T/2] / (σ√T)
- d₂ = d₁ - σ√T
- σ = bond price volatility (log-normal)

Forward Bond Price
------------------
For a coupon bond with coupons C_i paid at times t_i < T (option expiry):

F = (B_0 - Σ C_i × DF(t_i)) / DF(T)

Or equivalently:
F = B_0 × exp(r × T) - FV(coupons)

Clean vs Dirty Pricing
----------------------
- Clean price: Excludes accrued interest
- Dirty price: Includes accrued interest (full price)
- Bond options typically settle on clean price

Greeks
------
- Delta: dPV/dF (sensitivity to forward bond price)
- Gamma: d²PV/dF² (convexity)
- Vega: dPV/dσ (sensitivity to volatility)
- Theta: dPV/dt (time decay)
- Rho: dPV/dr (sensitivity to rates)

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from src.marketdata.core.ids import MarketId
from src.instruments.core.types import OptionType


# =============================================================================
# SIMPLE INSTRUMENT (Direct Parameters)
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondEuropeanOptionSimple:
    """
    European bond option with direct parameter input.
    
    A bond option is an option on a bond price, priced using Black76
    on the forward bond price.
    
    Parameters
    ----------
    notional : float
        Number of bonds (or notional amount).
    strike : float
        Strike price K (bond price, e.g., 98.5).
    expiry : float
        Time to option expiry in years.
    forward_bond_price : float
        Forward bond price F.
    vol : float
        Bond price volatility σ (log-normal Black76 vol).
    discount_factor : float
        Discount factor DF(T) to option expiry.
    option_type : OptionType
        "call" (right to buy) or "put" (right to sell).
    
    Examples
    --------
    6-month call option on a 10-year bond:
        >>> option = IrBondEuropeanOptionSimple(
        ...     notional=1_000_000,         # $1M face value
        ...     strike=102.0,               # Strike at 102%
        ...     expiry=0.5,                 # 6 months to expiry
        ...     forward_bond_price=103.5,   # Forward at 103.5%
        ...     vol=0.08,                   # 8% bond price vol
        ...     discount_factor=0.975,
        ...     option_type="call",
        ... )
    
    Notes
    -----
    - Strike and forward are typically quoted as percentage of face value
    - Vol is the log-normal volatility of the bond price
    - For zero coupon bonds, forward = spot × exp(r×T)
    - For coupon bonds, forward accounts for coupons during option life
    """
    notional: float
    strike: float                   # K = strike bond price
    expiry: float                   # T = time to option expiry
    forward_bond_price: float       # F = forward bond price
    vol: float                      # σ = Black76 volatility
    discount_factor: float          # DF(T) = discount to expiry
    option_type: OptionType = "call"
    
    def __post_init__(self) -> None:
        """Validate bond option inputs."""
        # Notional must be non-zero.
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        
        # Strike must be positive.
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        
        # Expiry must be non-negative.
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")
        
        # Forward bond price must be positive.
        if float(self.forward_bond_price) <= 0.0:
            raise ValueError("forward_bond_price must be > 0.")
        
        # Volatility must be non-negative.
        if float(self.vol) < 0.0:
            raise ValueError("vol must be >= 0.")
        
        # Discount factor must be positive.
        if float(self.discount_factor) <= 0.0:
            raise ValueError("discount_factor must be > 0.")
        
        # Option type must be call or put.
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put'; got {self.option_type}")
    
    @property
    def is_in_the_money(self) -> bool:
        """
        Check if option is in-the-money.
        
        Call: ITM when F > K (bond price above strike)
        Put: ITM when F < K (bond price below strike)
        """
        if self.option_type == "call":
            return float(self.forward_bond_price) > float(self.strike)
        return float(self.forward_bond_price) < float(self.strike)
    
    @property
    def moneyness(self) -> float:
        """
        Moneyness ratio F/K.
        
        > 1: ITM call / OTM put
        < 1: OTM call / ITM put
        = 1: ATM
        """
        return float(self.forward_bond_price) / float(self.strike)
    
    @property
    def intrinsic_value(self) -> float:
        """
        Intrinsic value of the option.
        
        Call: max(F - K, 0)
        Put: max(K - F, 0)
        """
        F = float(self.forward_bond_price)
        K = float(self.strike)
        
        if self.option_type == "call":
            return max(F - K, 0.0)
        return max(K - F, 0.0)


# =============================================================================
# MARKET DATA INSTRUMENT (Lookup from Market)
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrBondEuropeanOption:
    """
    European bond option with market data lookup.
    
    Parameters
    ----------
    notional : float
        Number of bonds (or notional face value).
    strike : float
        Strike price K (as percentage of face, e.g., 102.0).
    expiry : float
        Time to option expiry in years.
    underlying_maturity : float
        Maturity of the underlying bond in years (from today).
    underlying_coupon_rate : float
        Coupon rate of underlying bond.
    underlying_coupon_frequency : int
        Coupon frequency (1, 2, 4, 12).
    option_type : OptionType
        "call" or "put".
    curve_id : MarketId
        Market identifier for the discount/forward curve.
    vol_id : MarketId
        Market identifier for bond option volatility.
    
    Examples
    --------
    6-month call on 10-year 5% coupon bond:
        >>> option = IrBondEuropeanOption(
        ...     notional=1_000_000,
        ...     strike=102.0,
        ...     expiry=0.5,
        ...     underlying_maturity=10.0,
        ...     underlying_coupon_rate=0.05,
        ...     underlying_coupon_frequency=2,
        ...     option_type="call",
        ...     curve_id=MarketId("IR", "CURVE", "USD.GOVT"),
        ...     vol_id=MarketId("IR", "VOL", "USD.BOND"),
        ... )
    
    Notes
    -----
    The pricer computes the forward bond price accounting for:
    - Current spot bond price from curve
    - Coupons received during option life
    - Accrued interest adjustments
    """
    notional: float
    strike: float
    expiry: float
    underlying_maturity: float
    underlying_coupon_rate: float = 0.0      # 0 = zero coupon
    underlying_coupon_frequency: int = 2     # Semi-annual default
    option_type: OptionType = "call"
    curve_id: MarketId = field(default_factory=lambda: MarketId("IR", "CURVE", "UNKNOWN"))
    vol_id: MarketId = field(default_factory=lambda: MarketId("IR", "VOL", "UNKNOWN"))
    
    def __post_init__(self) -> None:
        """Validate bond option inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")
        if float(self.underlying_maturity) <= float(self.expiry):
            raise ValueError("underlying_maturity must be > expiry.")
        if float(self.underlying_coupon_rate) < 0.0:
            raise ValueError("underlying_coupon_rate must be >= 0.")
        if self.underlying_coupon_frequency not in (1, 2, 4, 12):
            raise ValueError("underlying_coupon_frequency must be 1, 2, 4, or 12.")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put'")
    
    @property
    def is_zero_coupon(self) -> bool:
        """Check if underlying is a zero coupon bond."""
        return float(self.underlying_coupon_rate) == 0.0
    
    @property
    def underlying_remaining_maturity_at_expiry(self) -> float:
        """Maturity of underlying bond at option expiry."""
        return float(self.underlying_maturity) - float(self.expiry)
    
    @property
    def description(self) -> str:
        """Human-readable option description."""
        opt_str = f"{int(self.expiry * 12)}M" if self.expiry < 1 else f"{self.expiry:.1f}Y"
        bond_str = f"{self.underlying_maturity:.0f}Y"
        
        if self.is_zero_coupon:
            bond_type = "ZC"
        else:
            bond_type = f"{self.underlying_coupon_rate * 100:.1f}%"
        
        return f"{opt_str} {self.option_type} on {bond_str} {bond_type} bond @ {self.strike:.2f}"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "IrBondEuropeanOption",
    "IrBondEuropeanOptionSimple",
]
