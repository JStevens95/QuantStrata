# src/marketdata/curves/bootstrapper.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Mapping, Sequence, Tuple, Union

import numpy as np

from src.marketdata.curves.discount import ExtrapolationMode, ZeroRateDiscountCurve

DepositCompounding = Literal["simple", "continuous"]
InstrumentType = Literal["deposit", "swap"]


# -----------------------------------------------------------------------------
# Quotes / instruments (V2-lite)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DepositQuote:
    """
    Deposit quote used for the short end of a discount curve.

    Parameters
    ----------
    maturity:
        Year-fraction maturity T (> 0).
    rate:
        Quoted annualized rate r (can be negative, but be careful).
    compounding:
        - "simple"     : DF(T) = 1 / (1 + r*T)
        - "continuous" : DF(T) = exp(-r*T)

    Notes
    -----
    - This is a pragmatic V2-lite design: it avoids day-count calendars.
    - You can add day-count conventions later without changing the bootstrap API.
    """
    maturity: float
    rate: float
    compounding: DepositCompounding = "simple"

    @property
    def instrument_type(self) -> InstrumentType:
        return "deposit"

    def __post_init__(self) -> None:
        t = float(self.maturity)
        if not np.isfinite(t) or t <= 0.0:
            raise ValueError("DepositQuote.maturity must be finite and > 0.")
        if not np.isfinite(float(self.rate)):
            raise ValueError("DepositQuote.rate must be finite.")
        if self.compounding not in ("simple", "continuous"):
            raise ValueError("DepositQuote.compounding must be 'simple' or 'continuous'.")


@dataclass(frozen=True, slots=True)
class ParSwapQuote:
    """
    Par swap quote for discount-curve bootstrapping (single-curve identity).

    We assume a standard single-curve par swap PV identity:
        PV_float = 1 - DF(T)
        PV_fixed = R * sum_i alpha_i * DF(t_i)

    At par:
        R * sum_i alpha_i * DF(t_i) = 1 - DF(T)

    This works as a V2-lite proxy for OIS-style bootstrapping when you don't
    want full instrument modelling yet.

    Parameters
    ----------
    maturity:
        Final maturity T (> 0).
    fixed_rate:
        Par fixed rate R.
    pay_freq:
        Fixed leg payments per year (e.g. 1=annual, 2=semi, 4=quarterly).
        Default 1 (annual).

    schedule:
        Optional explicit payment times (year fractions). If provided, overrides pay_freq.

    Notes
    -----
    - This is intentionally simple and deterministic for tests/examples.
    - You can replace this later with real OIS instruments + accrual schedules.
    """
    maturity: float
    fixed_rate: float
    pay_freq: int = 1
    schedule: Tuple[float, ...] | None = None

    @property
    def instrument_type(self) -> InstrumentType:
        return "swap"

    def __post_init__(self) -> None:
        t = float(self.maturity)
        if not np.isfinite(t) or t <= 0.0:
            raise ValueError("ParSwapQuote.maturity must be finite and > 0.")
        if not np.isfinite(float(self.fixed_rate)):
            raise ValueError("ParSwapQuote.fixed_rate must be finite.")
        if self.pay_freq < 1:
            raise ValueError("ParSwapQuote.pay_freq must be >= 1.")
        if self.schedule is not None:
            sched = tuple(float(x) for x in self.schedule)
            if len(sched) == 0:
                raise ValueError("ParSwapQuote.schedule must be non-empty when provided.")
            if any((not np.isfinite(x) or x <= 0.0) for x in sched):
                raise ValueError("ParSwapQuote.schedule entries must be finite and > 0.")
            if any(b <= a for a, b in zip(sched, sched[1:])):
                raise ValueError("ParSwapQuote.schedule must be strictly increasing.")
            if abs(sched[-1] - t) > 1e-12:
                raise ValueError("ParSwapQuote.schedule[-1] must equal maturity.")


CurveInstrument = Union[DepositQuote, ParSwapQuote]


# -----------------------------------------------------------------------------
# Result object (helps tests & debugging)
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """
    Output of a curve bootstrap.

    Attributes
    ----------
    curve:
        The usable discount curve object (ZeroRateDiscountCurve).
    tenors:
        Sorted maturities T_i (year fractions).
    dfs:
        Discount factors DF(T_i).
    zero_rates:
        Continuous zero rates r(T_i) = -ln(DF)/T.
    """
    curve: ZeroRateDiscountCurve
    tenors: np.ndarray
    dfs: np.ndarray
    zero_rates: np.ndarray


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_zero_curve_from_zero_rates(
    *,
    tenors: np.ndarray,
    zero_rates: np.ndarray,
    extrapolation: ExtrapolationMode = "flat",
) -> ZeroRateDiscountCurve:
    """
    Convenience wrapper: build a ZeroRateDiscountCurve directly from (T, r(T)).

    Use this when you already have a term structure and don't need bootstrapping.
    """
    return ZeroRateDiscountCurve(tenors=np.asarray(tenors, float), zero_rates=np.asarray(zero_rates, float), extrapolation=extrapolation)


def bootstrap_discount_curve(
    *,
    instruments: Sequence[CurveInstrument],
    extrapolation: ExtrapolationMode = "flat",
    min_df: float = 1e-12,
) -> BootstrapResult:
    """
    Bootstrap a discount curve from a sequence of deposits and par swaps.

    Design intent (V2-lite)
    -----------------------
    - Enough realism to stop everything being "flat".
    - Deterministic and unit-test friendly.
    - No vendor dependency and no heavy date-schedule machinery.

    Rules
    -----
    - Instruments are bootstrapped in increasing maturity.
    - DepositQuote directly sets DF(T).
    - ParSwapQuote solves DF(T) assuming all earlier payment DFs are already known.

    Raises
    ------
    ValueError
        If instruments are inconsistent, out of order, or imply invalid discount factors.
    """
    if not instruments:
        raise ValueError("bootstrap_discount_curve: instruments must not be empty.")

    # Sort by maturity for deterministic behaviour.
    inst_sorted = sorted(instruments, key=lambda q: float(q.maturity))

    # Core state: discount factors at bootstrapped maturities.
    df_by_t: Dict[float, float] = {}

    for inst in inst_sorted:
        t = float(inst.maturity)

        if inst.instrument_type == "deposit":
            dep = inst  # type: ignore[assignment]
            df_t = _df_from_deposit(dep)  # includes validation
            df_by_t[t] = df_t
            continue

        if inst.instrument_type == "swap":
            swp = inst  # type: ignore[assignment]
            times, accruals = _swap_schedule_and_accruals(swp)
            df_t = _df_from_par_swap(maturity=t, fixed_rate=float(swp.fixed_rate), pay_times=times, accruals=accruals, df_by_t=df_by_t)
            df_by_t[t] = df_t
            continue

        raise ValueError(f"Unsupported instrument_type={getattr(inst, 'instrument_type', None)!r}")

    # Assemble outputs
    tenors = np.array(sorted(df_by_t.keys()), dtype=float)
    dfs = np.array([float(df_by_t[tt]) for tt in tenors], dtype=float)

    _validate_discount_factors(tenors=tenors, dfs=dfs, min_df=min_df)

    # Convert to continuous zero rates on the grid
    zero_rates = np.array([_zero_rate_from_df(t=float(tt), df=float(df)) for tt, df in zip(tenors, dfs)], dtype=float)

    curve = ZeroRateDiscountCurve(tenors=tenors, zero_rates=zero_rates, extrapolation=extrapolation)
    return BootstrapResult(curve=curve, tenors=tenors, dfs=dfs, zero_rates=zero_rates)


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------

def _df_from_deposit(dep: DepositQuote) -> float:
    t = float(dep.maturity)
    r = float(dep.rate)

    if dep.compounding == "continuous":
        df = math.exp(-r * t)
    else:
        # simple compounding
        denom = 1.0 + r * t
        if denom <= 0.0:
            raise ValueError(
                "Deposit implies non-positive discount factor under simple compounding: "
                f"1 + r*T <= 0 (r={r}, T={t})."
            )
        df = 1.0 / denom

    if not np.isfinite(df) or df <= 0.0:
        raise ValueError(f"Deposit implies invalid DF(T): DF={df} at T={t}.")
    return float(df)


def _swap_schedule_and_accruals(swp: ParSwapQuote) -> tuple[np.ndarray, np.ndarray]:
    t = float(swp.maturity)

    if swp.schedule is not None:
        times = np.asarray(list(swp.schedule), dtype=float).reshape(-1)
    else:
        # Regular schedule: [1/f, 2/f, ..., T] (append T if not exact multiple)
        f = int(swp.pay_freq)
        step = 1.0 / float(f)

        # up to but excluding t, then ensure final t is included.
        times = np.arange(step, t + 1e-12, step, dtype=float)
        if times.size == 0 or abs(times[-1] - t) > 1e-12:
            times = np.concatenate([times, np.array([t], dtype=float)])

    # accruals alpha_i = t_i - t_{i-1}, with t_0 = 0
    prev = 0.0
    accruals: List[float] = []
    for ti in times.tolist():
        ai = float(ti - prev)
        if ai <= 0.0:
            raise ValueError("Invalid swap schedule: non-positive accrual encountered.")
        accruals.append(ai)
        prev = float(ti)

    return times, np.asarray(accruals, dtype=float)


def _df_from_par_swap(
    *,
    maturity: float,
    fixed_rate: float,
    pay_times: np.ndarray,
    accruals: np.ndarray,
    df_by_t: Mapping[float, float],
) -> float:
    """
    Solve DF(T) from:
        R * sum_i alpha_i DF(t_i) = 1 - DF(T)

    Requires all DF(t_i) for i < last to already exist.
    """
    t = float(maturity)
    r = float(fixed_rate)

    if not np.isfinite(r):
        raise ValueError("ParSwapQuote.fixed_rate must be finite.")

    times = np.asarray(pay_times, dtype=float).reshape(-1)
    alphas = np.asarray(accruals, dtype=float).reshape(-1)

    if times.size == 0 or alphas.size != times.size:
        raise ValueError("Swap schedule/accruals must be same non-zero length.")
    if abs(float(times[-1]) - t) > 1e-12:
        raise ValueError("Swap schedule must end at maturity.")

    # Sum known PV of fixed leg excluding the final DF(T)
    # A_prev = sum_{i < n} alpha_i DF(t_i)
    if times.size == 1:
        a_prev = 0.0
        alpha_last = float(alphas[0])
    else:
        a_prev = 0.0
        for ti, ai in zip(times[:-1], alphas[:-1]):
            ti_f = float(ti)
            if ti_f not in df_by_t:
                raise ValueError(
                    "Cannot bootstrap swap because earlier payment DF is missing.\n"
                    f"  Missing DF at t={ti_f}\n"
                    f"  Swap maturity T={t}\n"
                    "  Tip: include deposits / shorter swaps so all earlier payment times are bootstrapped first."
                )
            a_prev += float(ai) * float(df_by_t[ti_f])
        alpha_last = float(alphas[-1])

    denom = 1.0 + r * alpha_last
    if denom <= 0.0:
        raise ValueError(
            "Swap bootstrap failed: denominator (1 + R*alpha_last) <= 0.\n"
            f"  R={r}, alpha_last={alpha_last}, denom={denom}"
        )

    numer = 1.0 - r * a_prev
    df_t = numer / denom

    if not np.isfinite(df_t) or df_t <= 0.0:
        raise ValueError(
            "Swap bootstrap produced invalid DF(T).\n"
            f"  T={t}, DF(T)={df_t}, R={r}, A_prev={a_prev}, alpha_last={alpha_last}"
        )

    return float(df_t)


def _zero_rate_from_df(*, t: float, df: float) -> float:
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError("df must be > 0.")
    return float(-math.log(df) / float(t))


def _validate_discount_factors(*, tenors: np.ndarray, dfs: np.ndarray, min_df: float) -> None:
    tenors = np.asarray(tenors, dtype=float).reshape(-1)
    dfs = np.asarray(dfs, dtype=float).reshape(-1)

    if tenors.size == 0:
        raise ValueError("No bootstrapped tenors.")
    if tenors.size != dfs.size:
        raise ValueError("tenors and dfs must have same length.")
    if np.any(~np.isfinite(tenors)) or np.any(~np.isfinite(dfs)):
        raise ValueError("tenors/dfs must be finite.")
    if np.any(np.diff(tenors) <= 0.0):
        raise ValueError("tenors must be strictly increasing.")
    if np.any(dfs <= float(min_df)):
        raise ValueError(f"Some discount factors are <= min_df={min_df}.")
    if np.any(dfs > 1.0 + 1e-10):
        # allow tiny numerical jitter but flag real issues
        raise ValueError("Some discount factors are > 1.0 (unexpected for standard discounting).")

    # Monotonic decreasing DF sanity (not strictly required if negative rates exist),
    # but we warn by raising only on strong violations.
    # If you want to support negative rates robustly, relax this further.
    if np.any(np.diff(dfs) > 1e-6):
        raise ValueError(
            "Discount factors increased materially with maturity (sanity check failed). "
            "If you expect negative rates, relax/remove this check."
        )