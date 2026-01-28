from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple, Union

import numpy as np

from src.marketdata.core.types import BootstrapEngine, ExtrapolationMode
from src.marketdata.curves.term_structure import ZeroRateCurve
from src.marketdata.integration.quantlib.context import (
    require_quantlib, to_ql_date, yearfrac_to_ql_date,
)
from src.marketdata.quotes.rates import (
    DepositQuote,
    FraQuote,
    ParSwapQuote,
)


# list of curve instruments for bootstrapping.
CurveInstrument = Union[DepositQuote, ParSwapQuote, FraQuote]


# -----------------------------------------------------------------------------
# Result object
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """
    Output of a curve bootstrap.

    Attributes
    ----------
    curve:
        Usable curve object (ZeroRateDiscountCurve).
    tenors:
        Sorted maturities (year-fractions).
    dfs:
        Discount factors DF(T).
    zero_rates:
        Continuous zero rates r(T) = -ln(DF)/T for T>0.
    """
    curve: ZeroRateCurve
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
) -> ZeroRateCurve:
    """
    Convenience wrapper: build a ZeroRateDiscountCurve directly from (T, r(T)).
    """
    return ZeroRateCurve(
        tenors=np.asarray(tenors, dtype=float),
        zero_rates=np.asarray(zero_rates, dtype=float),
        extrapolation=extrapolation,
    )


def bootstrap_discount_curve(
    *,
    instruments: Sequence[CurveInstrument],
    extrapolation: ExtrapolationMode = "flat",
    min_df: float = 1e-12,
    engine: BootstrapEngine = "native",
    asof: str = "2025-01-01",
    output_tenors: np.ndarray | None = None,
) -> BootstrapResult:
    """
    Bootstrap a discount curve from deposits + par swaps.

    engine:
      - "native"   : deterministic V2-lite bootstrap (this module)
      - "quantlib" : QuantLib RateHelper bootstrap, sampled into ZeroRateDiscountCurve
    """
    engine_s = str(engine).strip().lower()
    if engine_s == "native":
        return _bootstrap_discount_curve_native(
            instruments=instruments,
            extrapolation=extrapolation,
            min_df=min_df,
        )
    if engine_s == "quantlib":
        return _bootstrap_discount_curve_quantlib(
            instruments=instruments,
            extrapolation=extrapolation,
            min_df=min_df,
            asof=asof,
            output_tenors=output_tenors,
        )
    raise ValueError(f"Unknown engine={engine!r}. Expected 'native' or 'quantlib'.")


# -----------------------------------------------------------------------------
# Native backend
# -----------------------------------------------------------------------------


def _bootstrap_discount_curve_native(
    *,
    instruments: Sequence[CurveInstrument],
    extrapolation: ExtrapolationMode = "flat",
    min_df: float = 1e-12,
) -> BootstrapResult:
    if not instruments:
        raise ValueError("bootstrap_discount_curve: instruments must not be empty.")

    inst_sorted = sorted(instruments, key=lambda q: float(q.maturity))
    df_by_t: dict[float, float] = {}

    for inst in inst_sorted:
        t = float(inst.maturity)

        if isinstance(inst, DepositQuote):
            df_by_t[t] = _df_from_deposit(inst)
            continue

        if isinstance(inst, ParSwapQuote):
            times, accruals = _swap_schedule_and_accruals(inst)
            df_by_t[t] = _df_from_par_swap(
                maturity=t,
                fixed_rate=float(inst.fixed_rate),
                pay_times=times,
                accruals=accruals,
                df_by_t=df_by_t,
            )
            continue
        
        if isinstance(inst, FraQuote):
            df_by_t[t] = _df_from_fra(
                fra=inst,
                df_by_t=df_by_t,
            )
            continue

        raise TypeError(f"Unsupported instrument type: {type(inst).__name__}")

    tenors = np.array(sorted(df_by_t.keys()), dtype=float)
    dfs = np.array([float(df_by_t[tt]) for tt in tenors], dtype=float)

    _validate_discount_factors(tenors=tenors, dfs=dfs, min_df=min_df)

    zero_rates = np.array(
        [_zero_rate_from_df(t=float(tt), df=float(df)) for tt, df in zip(tenors, dfs)],
        dtype=float,
    )

    curve = ZeroRateCurve(tenors=tenors, zero_rates=zero_rates, extrapolation=extrapolation)
    return BootstrapResult(curve=curve, tenors=tenors, dfs=dfs, zero_rates=zero_rates)


def _df_from_fra(
    *,
    fra: FraQuote,
    df_by_t: Mapping[float, float],
) -> float:
    """
    Bootstrap discount factor from FRA quote.

    FRA par condition:
        DF(T_start) - DF(T_end) × (1 + R × α) = 0

    Solving for DF(T_end):
        DF(T_end) = DF(T_start) / (1 + R × α)

    Parameters
    ----------
    fra:
        FRA quote (from rates.py).
    df_by_t:
        Dictionary of known discount factors (must contain T_start).

    Returns
    -------
    float
        Discount factor DF(T_end).

    Raises
    ------
    ValueError
        If DF(T_start) is missing or if computed DF(T_end) is invalid.
    """
    t_start = float(fra.t_start)  # Use t_start from rates.py FraQuote
    t_end = float(fra.t_end)  # Use t_end from rates.py FraQuote
    r = float(fra.forward_rate)  # Use forward_rate from rates.py FraQuote
    alpha = float(fra.day_count_fraction)  # Use computed property

    # Check that DF(T_start) exists.
    if t_start not in df_by_t:
        raise ValueError(
            "Cannot bootstrap FRA because start date DF is missing.\n"
            f"  Missing DF at t={t_start}\n"
            f"  FRA end date T={t_end}\n"
            "  Tip: include deposits / earlier instruments so DF(T_start) is bootstrapped first."
        )

    df_start = float(df_by_t[t_start])
    if df_start <= 0.0 or not np.isfinite(df_start):
        raise ValueError(f"Invalid DF at FRA start date t={t_start}: df={df_start}")

    # FRA par condition: DF(T_start) = DF(T_end) × (1 + R × α)
    denom = 1.0 + r * alpha
    if denom <= 0.0:
        raise ValueError(
            "FRA bootstrap failed: denominator (1 + R*α) <= 0.\n"
            f"  R={r}, α={alpha}, denom={denom}"
        )

    df_end = df_start / denom

    if not np.isfinite(df_end) or df_end <= 0.0:
        raise ValueError(
            "FRA bootstrap produced invalid DF(T_end).\n"
            f"  T_start={t_start}, T_end={t_end}, DF(T_start)={df_start}, "
            f"R={r}, α={alpha}, DF(T_end)={df_end}"
        )

    # Sanity check: DF(T_end) should be <= DF(T_start) for positive rates.
    if df_end > df_start + 1e-6:
        raise ValueError(
            "FRA bootstrap produced increasing discount factor (implies negative forward rate).\n"
            f"  DF(T_start)={df_start}, DF(T_end)={df_end}\n"
            "  If you expect negative rates, relax/remove this check."
        )

    return float(df_end)


def _df_from_deposit(dep: DepositQuote) -> float:
    t = float(dep.t) # Use t from DepositQuote
    r = float(dep.rate)

    if dep.compounding == "continuous":
        df = math.exp(-r * t)
    else:
        denom = 1.0 + r * t
        if denom <= 0.0:
            raise ValueError(
                "Deposit implies non-positive DF under simple compounding: "
                f"1 + r*T <= 0 (r={r}, T={t})."
            )
        df = 1.0 / denom

    if not np.isfinite(df) or df <= 0.0:
        raise ValueError(f"Deposit implies invalid DF(T): DF={df} at T={t}.")
    return float(df)


def _swap_schedule_and_accruals(swp: ParSwapQuote) -> tuple[np.ndarray, np.ndarray]:
    t = float(swp.maturity) # Use maturity from ParSwapQuote

    if swp.schedule is not None:
        times = np.asarray(list(swp.schedule), dtype=float).reshape(-1)
    else:
        f = int(swp.pay_freq)  # Use pay_freq property (converts FixedFreq to int)
        step = 1.0 / float(f)

        times = np.arange(step, t + 1e-12, step, dtype=float)
        if times.size == 0 or abs(float(times[-1]) - t) > 1e-12:
            times = np.concatenate([times, np.array([t], dtype=float)])

    prev = 0.0
    accruals: list[float] = []
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
    t = float(maturity)
    r = float(fixed_rate)  # Use fixed_rate property (aliases par_rate)

    if not np.isfinite(r):
        raise ValueError("ParSwapQuote.fixed_rate must be finite.")

    times = np.asarray(pay_times, dtype=float).reshape(-1)
    alphas = np.asarray(accruals, dtype=float).reshape(-1)

    if times.size == 0 or alphas.size != times.size:
        raise ValueError("Swap schedule/accruals must be same non-zero length.")
    if abs(float(times[-1]) - t) > 1e-12:
        raise ValueError("Swap schedule must end at maturity.")

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
        raise ValueError("Some discount factors are > 1.0 (unexpected for standard discounting).")
    if np.any(np.diff(dfs) > 1e-6):
        raise ValueError(
            "Discount factors increased materially with maturity (sanity check failed). "
            "If you expect negative rates, relax/remove this check."
        )


# -----------------------------------------------------------------------------
# QuantLib backend
# -----------------------------------------------------------------------------


def _bootstrap_discount_curve_quantlib(
    *,
    instruments: Sequence[CurveInstrument],
    extrapolation: ExtrapolationMode,
    min_df: float,
    asof: str,
    output_tenors: np.ndarray | None,
) -> BootstrapResult:
    """
    QuantLib RateHelper bootstrap backend (V2 milestone).

    Notes
    -----
    - Instruments are expressed in year-fractions; we map these to Periods/Date offsets
      deterministically (T -> round(T*365) days).
    - This is appropriate for demos/tests and integration scaffolding.
    """
    ql = require_quantlib()

    if not instruments:
        raise ValueError("bootstrap_discount_curve(engine='quantlib'): instruments must not be empty.")

    asof_date = to_ql_date(asof)
    ql.Settings.instance().evaluationDate = asof_date

    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    bdc = ql.Unadjusted

    ts_handle = ql.RelinkableYieldTermStructureHandle()
    ibor_index = ql.USDLibor(ql.Period(3, ql.Months), ts_handle)

    # ------------------------------------------------------------------
    # Guard: QuantLib swap helpers can fail if evaluationDate is not a valid
    # fixing date for the chosen index calendar (e.g. holidays like Jan 1).
    # ------------------------------------------------------------------
    fixing_cal = ibor_index.fixingCalendar()
    if not fixing_cal.isBusinessDay(asof_date):
        # Suggest a usable fixing date (Following).
        suggested = fixing_cal.adjust(asof_date, ql.Following)

        # ql.Date has ISO() in Python bindings.
        suggested_iso = suggested.ISO() if hasattr(suggested, "ISO") else str(suggested)

        raise ValueError(
            "QuantLib bootstrap (engine='quantlib') requires 'asof' to be a valid business day "
            "for the index fixing calendar.\n"
            f"  asof={asof!r} is not a valid fixing date for index={ibor_index.name()!r}.\n"
            f"  Try a business day such as {suggested_iso} (Following)."
        )

    helpers: list[Any] = []

    for inst in instruments:
        if isinstance(inst, DepositQuote):
            tenor = _yearfrac_to_ql_period(ql, float(inst.t))  # Use t field from rates.py
            quote = ql.QuoteHandle(ql.SimpleQuote(float(inst.rate)))

            helpers.append(
                ql.DepositRateHelper(
                    quote,
                    tenor,
                    0,          # fixing days
                    calendar,
                    bdc,
                    False,      # end of month
                    day_count,
                )
            )
            continue

        if isinstance(inst, ParSwapQuote):
            swap_tenor = _yearfrac_to_ql_period(ql, float(inst.maturity))
            fixed_freq = _pay_freq_to_ql_frequency(ql, int(inst.pay_freq))
            quote = ql.QuoteHandle(ql.SimpleQuote(float(inst.fixed_rate)))

            helpers.append(
                ql.SwapRateHelper(
                    quote,
                    swap_tenor,
                    calendar,
                    fixed_freq,
                    bdc,
                    day_count,
                    ibor_index,
                )
            )
            continue

        if isinstance(inst, FraQuote):
            # FRA: map start/end dates to QuantLib dates
            start_date = yearfrac_to_ql_date(asof=asof_date, yearfrac=float(inst.t_start))
            end_date = yearfrac_to_ql_date(asof=asof_date, yearfrac=float(inst.t_end))
            quote = ql.QuoteHandle(ql.SimpleQuote(float(inst.forward_rate)))

            helpers.append(
                ql.FraRateHelper(
                    quote,
                    start_date,
                    end_date,
                    ibor_index,
                )
            )
            continue

        raise TypeError(f"Unsupported instrument type: {type(inst).__name__}")

    curve = ql.PiecewiseLogLinearDiscount(asof_date, helpers, day_count)
    curve.enableExtrapolation()
    ts_handle.linkTo(curve)

    tenors = _resolve_sampling_tenors(instruments=instruments, output_tenors=output_tenors)
    dfs = np.empty_like(tenors, dtype=float)
    zero_rates = np.empty_like(tenors, dtype=float)

    for i, t in enumerate(tenors.tolist()):
        if t <= 0.0:
            dfs[i] = 1.0
            zero_rates[i] = 0.0
            continue

        dt = yearfrac_to_ql_date(asof=asof_date, yearfrac=float(t))
        df = float(curve.discount(dt))

        if not np.isfinite(df) or df <= 0.0:
            raise ValueError(f"QuantLib produced invalid DF at t={t}: df={df}")

        dfs[i] = df
        zero_rates[i] = _zero_rate_from_df(t=float(t), df=float(df))

    _validate_discount_factors(tenors=tenors, dfs=dfs, min_df=min_df)

    curve_out = ZeroRateCurve(tenors=tenors, zero_rates=zero_rates, extrapolation=extrapolation)
    return BootstrapResult(curve=curve_out, tenors=tenors, dfs=dfs, zero_rates=zero_rates)


def _yearfrac_to_ql_period(ql: Any, t: float) -> Any:
    """
    Map year-fraction to an approximate QuantLib Period in days (deterministic V2-lite mapping).

    RateHelpers typically want Periods, not Dates. Keep this local unless/until you
    want to centralize Period mapping into integration/quantlib/context.py.
    """
    t = float(t)
    if t <= 0.0:
        return ql.Period(0, ql.Days)
    days = int(max(1, round(t * 365.0)))
    return ql.Period(days, ql.Days)


def _pay_freq_to_ql_frequency(ql: Any, pay_freq: int) -> Any:
    if pay_freq == 1:
        return ql.Annual
    if pay_freq == 2:
        return ql.Semiannual
    if pay_freq == 4:
        return ql.Quarterly
    if pay_freq == 12:
        return ql.Monthly
    return ql.Annual


def _resolve_sampling_tenors(
    *,
    instruments: Sequence[CurveInstrument],
    output_tenors: np.ndarray | None,
) -> np.ndarray:
    if output_tenors is None:
        tenors = _default_sampling_grid_from_instruments(instruments)
    else:
        tenors = np.asarray(output_tenors, dtype=float).reshape(-1)

    if tenors.size == 0:
        raise ValueError("output_tenors must be non-empty (or inferable).")

    tenors = np.unique(np.clip(tenors, 0.0, None))
    tenors.sort()

    if tenors[0] != 0.0:
        tenors = np.concatenate([np.array([0.0], dtype=float), tenors])

    return tenors


def _default_sampling_grid_from_instruments(instruments: Sequence[CurveInstrument]) -> np.ndarray:
    tenors: list[float] = [0.0]

    for inst in instruments:
        tenors.append(float(inst.maturity))

        if isinstance(inst, FraQuote):
            tenors.append(float(inst.t_start))  # Use t_start from rates.py

        if isinstance(inst, ParSwapQuote):
            if inst.schedule is not None:
                tenors.extend([float(x) for x in inst.schedule])
            else:
                T = float(inst.maturity)
                f = int(inst.pay_freq)
                if f > 0 and T > 0:
                    step = 1.0 / float(f)
                    times = np.arange(step, T + 1e-12, step, dtype=float)
                    tenors.extend(times.tolist())

    grid = np.unique(np.array([t for t in tenors if t >= 0.0], dtype=float))
    grid.sort()
    return grid