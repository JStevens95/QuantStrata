# Interest Rate Curve Fitness

This document describes the statistical tests implemented in
[`zar_curve_fitness.py`](zar_curve_fitness.py).  The tool evaluates
whether a zero-coupon curve time series is fit for use in market-risk
models (VaR, Expected Shortfall) and produces a formatted Excel report.

The intended workflow:

1. Run on **ZARONIA_SARB** — demonstrate that the SARB benchmark rate
   fails fitness checks.
2. Run on **ZARONIA_CAS** (JIBAR minus CAS 3M) — demonstrate that it
   passes.

To switch between curves, edit the CONFIG block at the top of the script:

```python
INPUT_PATH  = r"C:\Users\Joseph\data\zaronia_cas.csv"
CURVE_LABEL = "ZARONIA_CAS"
```

---

## Tests

Five tests producing seven flag conditions:

| # | Test | Question | Flag(s) |
|---|------|----------|---------|
| 1 | ADF Unit Root | Are daily returns stationary? | `ADF_NON_STATIONARY` if returns p > 0.05 |
| 2 | Zero-Return Frequency | Is this rate genuinely traded? | `ZERO_RETURN` if zero days > 5 %; `STALE_RUNS` if longest unchanged streak > 5 days |
| 3 | Ljung-Box (McLeod-Li) | Are squared returns serially independent? | `AUTOCORRELATION` if lag-10 p < 0.01 |
| 4 | Excess Kurtosis | Is the return distribution pathological? | `EXTREME_TAILS` if excess kurtosis > 30 |
| 5 | Unique Rate Granularity | Does the rate take enough distinct values? | `LOW_GRANULARITY` if unique ratio < 0.50 |

### Rating

| Flags | Rating | Meaning |
|-------|--------|---------|
| 0 | **PASS** | Fit for risk models |
| 1 | **CAUTION** | Usable with documented caveats |
| 2+ | **FAIL** | Not fit for purpose |

---

## Test 1 — ADF Unit Root

### What it tests

Whether rate levels contain a unit root (expected) and whether daily
changes are stationary (required).

Interest rate levels follow a near-random walk — they should be
non-stationary.  The first differences (daily changes) should be
stationary.  If daily changes are non-stationary, the time series
cannot be used as a risk factor in historical simulation VaR.

### Mathematics

The Augmented Dickey-Fuller test estimates:

    Δy_t = α + γ·y_{t-1} + Σ δ_j·Δy_{t-j} + ε_t

and tests H0: γ = 0 (unit root) against H1: γ < 0 (stationary).

The number of lags is selected by minimising AIC.  The test statistic
follows the Dickey-Fuller distribution; p-values use MacKinnon (1996)
critical values.

### Implementation

```python
from statsmodels.tsa.stattools import adfuller
adfuller(data, maxlag=10, regression="c", autolag="AIC")
```

### Flag condition

Returns p-value > 0.05 raises `ADF_NON_STATIONARY`.

### Expected results

| Series | Expected p-value | Interpretation |
|--------|-----------------|----------------|
| Levels | > 0.05 | Unit root present (normal for rates) |
| Returns | < 0.05 | Stationary (required for VaR) |

### What failure means

Non-stationary returns mean the risk factor has not been properly
differenced or contains a structural trend.  VaR models built on
non-stationary returns produce unstable, unreliable capital estimates.
Per BCBS 352 S33, risk factors must be stationary or stationarised.

### Reference

Dickey, D.A. and Fuller, W.A. (1979). *Distribution of the estimators
for autoregressive time series with a unit root.* JASA, 74(366),
427-431.

---

## Test 2 — Zero-Return Frequency

### What it tests

The percentage of business days on which the rate did not change at all
(daily change = exactly 0 bps), and the longest streak of consecutive
identical values.

### Mathematics

    zero_return_pct = count(|Δ_t| == 0) / N × 100

    max_consecutive_unchanged = max group size where value_t == value_{t-1}

### Implementation

```python
n_zeros = (daily_changes_bps == 0).sum()
zero_pct = n_zeros / len(daily_changes_bps) * 100

changed = rate_levels != rate_levels.shift()
groups = changed.cumsum()
max_run = rate_levels.groupby(groups).size().max()
```

### Flag conditions

- Zero-return % > 5.0 raises `ZERO_RETURN`.
- Max consecutive unchanged > 5 days raises `STALE_RUNS`.

### Expected results

| Curve type | Expected zero % | Expected max run |
|-----------|----------------|-----------------|
| Traded (JIBAR-CAS) | < 2% | 1–3 days |
| Administered (SARB) | 30–80% | 10–50+ days |

### What failure means

A high zero-return percentage is the strongest single indicator that a
rate is not reflecting continuous market activity.  It is either
administered (set periodically by a committee), stale (not updated), or
synthetic (derived from an infrequent fixing).

Long consecutive runs of identical values confirm step-function
behaviour.  This artificially suppresses volatility, inflates
autocorrelation, and produces a distribution dominated by zeros —
none of which are compatible with standard VaR assumptions.

---

## Test 3 — Ljung-Box Autocorrelation (McLeod-Li variant)

### What it tests

Whether the **squared** daily returns exhibit significant serial
autocorrelation at lags 5 and 10.  This is the McLeod-Li variant of
the Ljung-Box test, targeting dependence in the variance rather than
the mean.

Standard Ljung-Box on raw changes misses the step-function pattern of
administered rates because symmetric jumps with random timing produce
near-zero linear autocorrelation.  Squaring the changes converts the
problem: long runs of zero-variance days followed by isolated spikes
create strong autocorrelation in the squared series.

### Mathematics

The Ljung-Box statistic applied to the squared return series is:

    Q(k) = n·(n + 2) · Σ_{j=1}^{k} [ ρ²_j(Δ²) / (n - j) ]

where ρ_j(Δ²) is the sample autocorrelation of the squared changes at
lag j and n is the sample size.  Under H0 (no dependence in variance),
Q(k) follows a chi-squared distribution with k degrees of freedom.

### Implementation

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
squared = daily_changes_bps ** 2
acorr_ljungbox(squared, lags=[5, 10], return_df=True)
```

### Flag condition

Lag-10 p-value < 0.01 raises `AUTOCORRELATION`.

### Why lag 10

Lag 10 covers two full trading weeks.  Administered rates that update
weekly or fortnightly produce squared-return clustering at these lags.
Lag 5 is reported for transparency but not used for flagging — mild
short-lag variance dependence is common even in genuine traded rates.

### Why squared returns

A rate that is flat for 20 days then jumps has near-zero autocorrelation
in the raw changes (the jumps are isolated and symmetric).  But the
squared changes reveal the pattern clearly: 19 days of zero followed by
a spike, repeating.  This creates strong positive autocorrelation in the
squared series, which the McLeod-Li test reliably detects.

### Expected results

| Curve type | Expected lag-10 p | Interpretation |
|-----------|-------------------|----------------|
| Traded (JIBAR-CAS) | > 0.01 | No significant variance clustering |
| Administered (SARB) | ≪ 0.01 | Step-function variance pattern |

### What failure means

Autocorrelated squared returns mean the variance is predictable — the
time series alternates between periods of no change and periods of
movement.  This violates the i.i.d. assumption underpinning historical
simulation VaR and produces unstable risk estimates.

### Reference

McLeod, A.I. and Li, W.K. (1983). *Diagnostic checking ARMA time
series models using squared-residual autocorrelations.* Journal of Time
Series Analysis, 4(4), 269-273.  Ljung, G.M. and Box, G.E.P. (1978).
*On a measure of lack of fit in time series models.* Biometrika, 65(2),
297-303.  BCBS 352 S16.

---

## Test 4 — Excess Kurtosis

### What it tests

The skewness and excess kurtosis of the daily-change distribution.
A zero-dominated distribution (administered rate) produces extreme
kurtosis because almost all mass sits at zero with rare large jumps in
the tails.  A traded rate has moderate excess kurtosis from fat tails
but nothing pathological.

### Mathematics

Excess kurtosis (Fisher's definition):

    κ = E[(X - μ)⁴] / σ⁴ - 3

A normal distribution has κ = 0.  Financial returns are typically
leptokurtic (κ > 0).  Administered rates with 90 %+ zero days produce
κ in the range 50–200 because the distribution is a near-delta at zero
with extreme outliers.

Skewness:

    γ = E[(X - μ)³] / σ³

Reported for completeness (not flagged).

### Implementation

```python
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
scipy_kurtosis(daily_changes_bps, fisher=True)
scipy_skew(daily_changes_bps)
```

### Flag condition

Excess kurtosis > 30 raises `EXTREME_TAILS`.

### Expected results

| Curve type | Expected excess kurtosis | Interpretation |
|-----------|-------------------------|----------------|
| Traded (JIBAR-CAS) | 3–15 | Fat tails, typical for financial data |
| Administered (SARB) | 50–200+ | Pathological zero-dominated distribution |

### What failure means

Extreme kurtosis means the return distribution is incompatible with any
standard VaR methodology (parametric, historical, or Monte Carlo) without
heavy-tailed distributional corrections.  The presence of a dominant
zero-mass point means the rate is not continuously priced and cannot
reliably generate the P&L scenarios needed for capital calculation.

### Reference

DeCarlo, L.T. (1997). *On the meaning and use of kurtosis.*
Psychological Methods, 2(3), 292-307.

---

## Test 5 — Unique Rate Granularity

### What it tests

The number of distinct rate levels observed as a proportion of total
observations.  A continuously traded rate takes a near-unique value
every day (ratio close to 1.0).  An administered rate that only changes
at committee meetings takes a handful of values over years (ratio near
zero).

### Mathematics

    unique_rate_ratio = n_unique_levels / n_observations

### Implementation

```python
n_unique = rate_levels.nunique()
ratio = n_unique / len(rate_levels)
```

### Flag condition

Unique rate ratio < 0.50 raises `LOW_GRANULARITY`.

### Expected results

| Curve type | Expected ratio | Expected unique levels (2 yr) |
|-----------|---------------|------------------------------|
| Traded (JIBAR-CAS) | > 0.95 | ~500 out of ~520 |
| Administered (SARB) | < 0.05 | ~15–25 out of ~520 |

### What failure means

A low unique-rate ratio confirms that the rate takes only a small number
of discrete values, consistent with administered or committee-set
behaviour.  Such a rate cannot produce the continuous P&L distribution
expected by risk models.  It also implies that any Monte Carlo simulation
calibrated to this rate will be poorly specified.

---

## Input Format

CSV or XLSX file:

| Column A | Column B | Column C | ... |
|----------|----------|----------|-----|
| Date | ON | 1M | ... |
| 2020-01-02 | 0.0450 | 0.0455 | ... |

- Rates as **decimals** (0.0825 = 8.25%).
- One row per business day, sorted ascending.

## Output

Excel workbook with 2 tabs:

| Tab | Contents |
|-----|----------|
| Summary | Rating, flags, obs count, mean level, daily vol per tenor |
| Detail | ADF statistics, zero-return metrics, Ljung-Box results, kurtosis, and granularity per tenor |

## Quick Start

```bash
pip install pandas numpy scipy statsmodels openpyxl
```

Edit the config at the top of `zar_curve_fitness.py`:

```python
INPUT_PATH  = "path/to/zaronia_sarb.csv"
OUTPUT_DIR  = "path/to/output"
CURVE_LABEL = "ZARONIA_SARB"
```

Run:

```bash
python docs/zar_curve_fitness.py
```

Then edit CONFIG for the second curve and re-run:

```python
INPUT_PATH  = "path/to/zaronia_cas.csv"
CURVE_LABEL = "ZARONIA_CAS"
```

---

## Full Source Code

The complete, runnable script is embedded below for reference.  It is
identical to [`zar_curve_fitness.py`](zar_curve_fitness.py).

```python
#!/usr/bin/env python3
"""
Interest Rate Curve Fitness
============================

Evaluates whether a zero-coupon curve time series is fit for use in
market-risk models (VaR, Expected Shortfall).  Produces a per-tenor
summary and a formatted Excel report.

Five tests, each answering one question:

    1. ADF Unit Root          — Are daily returns stationary?
    2. Zero-Return %          — Is this rate genuinely traded?
    3. Ljung-Box (McLeod-Li)  — Are squared returns serially independent?
    4. Excess Kurtosis        — Is the return distribution pathological?
    5. Unique Rate Granularity — Does the rate take enough distinct values?

Requires Python >= 3.12

Usage
-----
    1. Edit the CONFIG section below.
    2. pip install pandas numpy statsmodels openpyxl
    3. python zar_curve_fitness.py
"""

from __future__ import annotations

import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG — edit these for your environment
# ─────────────────────────────────────────────────────────────────────────────

INPUT_PATH: str = r"C:\Users\Joseph\data\zaronia_sarb.csv"
OUTPUT_DIR: str = r"C:\Users\Joseph\output"
CURVE_LABEL: str = "ZARONIA_SARB"

THRESHOLDS: dict = {
    "adf_returns_p": 0.05,        # flag if returns p-value exceeds this
    "zero_return_pct": 5.0,       # flag if zero-return % exceeds this
    "max_consecutive_unchanged": 5,  # flag if longest unchanged run exceeds this
    "ljung_box_p": 0.01,          # flag if lag-10 p-value is below this
    "excess_kurtosis": 30.0,      # flag if excess kurtosis exceeds this
    "unique_rate_ratio": 0.50,    # flag if distinct-values / observations < this
    "fail_flag_count": 2,         # >= this many flags = FAIL
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_curve(file_path: str) -> pd.DataFrame:
    """Load a zero-coupon curve from CSV or Excel.

    Expected layout:
        Column A    — dates (any pandas-parseable format)
        Columns B…  — tenor labels (ON, 1M, 3M, … 30Y)
        Values      — zero-coupon rates as decimals (0.0825 = 8.25 %)

    Returns a DataFrame indexed by DatetimeIndex, sorted ascending,
    with duplicate dates removed (last kept) and all-NaN rows dropped.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    elif ext in (".xlsx", ".xls"):
        data = pd.read_excel(file_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported file type '{ext}'.  Use .csv or .xlsx.")

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    n_duplicates = data.index.duplicated().sum()
    if n_duplicates:
        warnings.warn(f"{n_duplicates} duplicate dates — keeping last.")
        data = data[~data.index.duplicated(keep="last")]

    n_before = len(data)
    data = data.dropna(how="all")
    if len(data) < n_before:
        warnings.warn(f"Dropped {n_before - len(data)} all-NaN rows.")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# 3. TESTS — one function per test, each returns a plain dict
# ─────────────────────────────────────────────────────────────────────────────

def test_adf_unit_root(
    rate_levels: np.ndarray,
    daily_changes_bps: np.ndarray,
) -> dict:
    """Augmented Dickey-Fuller unit root test on levels and returns.

    Expected behaviour for a genuine market rate:
        - Levels have a unit root  (p > 0.05 → non-stationary)
        - Returns are stationary   (p < 0.05 → no unit root)

    Returns dict with keys:
        adf_levels_tstat, adf_levels_pvalue,
        adf_returns_tstat, adf_returns_pvalue
    """
    adf_levels = adfuller(rate_levels, maxlag=10, regression="c", autolag="AIC")
    adf_returns = adfuller(daily_changes_bps, maxlag=10, regression="c", autolag="AIC")

    return {
        "adf_levels_tstat": round(float(adf_levels[0]), 3),
        "adf_levels_pvalue": round(float(adf_levels[1]), 4),
        "adf_returns_tstat": round(float(adf_returns[0]), 3),
        "adf_returns_pvalue": round(float(adf_returns[1]), 4),
    }


def test_zero_return_frequency(
    daily_changes_bps: pd.Series,
    rate_levels: pd.Series,
) -> dict:
    """Measure how often the rate does not move at all.

    A genuine market rate changes on most business days.  A high
    zero-return percentage indicates an administered or stale rate.

    Returns dict with keys:
        zero_return_pct, max_consecutive_unchanged
    """
    n_obs = len(daily_changes_bps)
    n_zeros = int((daily_changes_bps == 0).sum())
    zero_pct = round(n_zeros / n_obs * 100, 1)

    # Longest streak of identical consecutive values
    changed = rate_levels != rate_levels.shift()
    groups = changed.cumsum()
    max_run = int(rate_levels.groupby(groups).size().max())

    return {
        "zero_return_pct": zero_pct,
        "max_consecutive_unchanged": max_run,
    }


def test_ljung_box_autocorrelation(daily_changes_bps: np.ndarray) -> dict:
    """McLeod-Li test (Ljung-Box on squared returns) for serial dependence.

    Standard Ljung-Box on raw changes misses the step-function pattern
    of administered rates because symmetric jumps with random timing
    produce near-zero linear autocorrelation.  Squaring the changes
    converts the problem into variance-clustering detection: long runs
    of zero-variance days followed by isolated spikes create strong
    autocorrelation in the squared series.

    Tests at lags 5 and 10.  The lag-10 p-value is used for flagging.

    Returns dict with keys:
        lb_lag5_statistic, lb_lag5_pvalue,
        lb_lag10_statistic, lb_lag10_pvalue
    """
    squared = daily_changes_bps ** 2
    lb = acorr_ljungbox(squared, lags=[5, 10], return_df=True)

    return {
        "lb_lag5_statistic": round(float(lb.iloc[0]["lb_stat"]), 1),
        "lb_lag5_pvalue": float(lb.iloc[0]["lb_pvalue"]),
        "lb_lag10_statistic": round(float(lb.iloc[1]["lb_stat"]), 1),
        "lb_lag10_pvalue": float(lb.iloc[1]["lb_pvalue"]),
    }


def test_excess_kurtosis(daily_changes_bps: np.ndarray) -> dict:
    """Measure skewness and excess kurtosis of the daily-change distribution.

    A zero-dominated distribution (administered rate) produces extreme
    kurtosis (50–200+) because almost all mass sits at zero with rare
    large jumps in the tails.  A traded rate has moderate excess
    kurtosis (typically 3–15) from fat tails.

    Returns dict with keys:
        skewness, excess_kurtosis
    """
    return {
        "skewness": round(float(scipy_skew(daily_changes_bps)), 2),
        "excess_kurtosis": round(float(scipy_kurtosis(daily_changes_bps, fisher=True)), 1),
    }


def test_unique_rate_granularity(rate_levels: pd.Series) -> dict:
    """Count distinct rate levels as a proportion of total observations.

    A continuously traded rate takes a (near-)unique value every day,
    giving a ratio close to 1.0.  An administered rate that only changes
    at committee meetings takes a handful of values over years, giving a
    ratio near zero.

    Returns dict with keys:
        n_unique_levels, unique_rate_ratio
    """
    n_unique = int(rate_levels.nunique())
    ratio = round(n_unique / len(rate_levels), 3)

    return {
        "n_unique_levels": n_unique,
        "unique_rate_ratio": ratio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCORER — evaluate flags and assign a rating
# ─────────────────────────────────────────────────────────────────────────────

def score_tenor(test_results: dict, thresholds: dict) -> tuple[list[str], str]:
    """Evaluate flag conditions and return (flag_list, rating).

    Flags:
        ADF_NON_STATIONARY  — returns fail to reject unit root
        ZERO_RETURN         — too many days with zero change
        STALE_RUNS          — longest unchanged streak exceeds threshold
        AUTOCORRELATION     — squared returns are serially dependent
        EXTREME_TAILS       — excess kurtosis is pathologically high
        LOW_GRANULARITY     — too few distinct rate levels
    """
    flags: list[str] = []

    if test_results["adf_returns_pvalue"] > thresholds["adf_returns_p"]:
        flags.append("ADF_NON_STATIONARY")

    zero_pct = test_results["zero_return_pct"]
    if zero_pct > thresholds["zero_return_pct"]:
        flags.append(f"ZERO_RETURN={zero_pct:.0f}%")

    max_run = test_results["max_consecutive_unchanged"]
    if max_run > thresholds["max_consecutive_unchanged"]:
        flags.append(f"STALE_RUNS={max_run}d")

    if test_results["lb_lag10_pvalue"] < thresholds["ljung_box_p"]:
        flags.append("AUTOCORRELATION")

    kurt = test_results["excess_kurtosis"]
    if kurt > thresholds["excess_kurtosis"]:
        flags.append(f"EXTREME_TAILS=k{kurt:.0f}")

    if test_results["unique_rate_ratio"] < thresholds["unique_rate_ratio"]:
        flags.append("LOW_GRANULARITY")

    n = len(flags)
    if n >= thresholds["fail_flag_count"]:
        rating = "FAIL"
    elif n >= 1:
        rating = "CAUTION"
    else:
        rating = "PASS"

    return flags, rating


# ─────────────────────────────────────────────────────────────────────────────
# 5. ORCHESTRATOR — run tests on every tenor
# ─────────────────────────────────────────────────────────────────────────────

def analyse_curve(
    curve_data: pd.DataFrame,
    thresholds: dict,
) -> dict[str, dict]:
    """Run all tests on every tenor and return a dict of results.

    Keys are tenor names.  Each value is a flat dict containing every
    test metric plus flags and rating.
    """
    results: dict[str, dict] = {}

    for tenor in curve_data.columns:
        levels = curve_data[tenor].dropna()
        n_obs = len(levels) - 1

        if n_obs < 30:
            results[tenor] = {
                "n_observations": n_obs,
                "rating": "SKIP",
                "flags": ["INSUFFICIENT_DATA"],
                "n_flags": 1,
            }
            continue

        daily_changes_bps = levels.diff().dropna() * 10_000
        daily_vol_bps = round(float(daily_changes_bps.std()), 2)
        mean_level_pct = round(float(levels.mean() * 100), 3)

        adf = test_adf_unit_root(levels.values, daily_changes_bps.values)
        zr = test_zero_return_frequency(daily_changes_bps, levels)
        lb = test_ljung_box_autocorrelation(daily_changes_bps.values)
        kt = test_excess_kurtosis(daily_changes_bps.values)
        gr = test_unique_rate_granularity(levels)

        merged = {
            "n_observations": n_obs,
            "mean_level_pct": mean_level_pct,
            "daily_vol_bps": daily_vol_bps,
        }
        merged.update(adf)
        merged.update(zr)
        merged.update(lb)
        merged.update(kt)
        merged.update(gr)

        flags, rating = score_tenor(merged, thresholds)
        merged["flags"] = flags
        merged["n_flags"] = len(flags)
        merged["rating"] = rating

        results[tenor] = merged

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. REPORT — Excel workbook builder
# ─────────────────────────────────────────────────────────────────────────────

# Styles
_FONT_HEADER = Font(name="Arial", bold=True, size=10, color="FFFFFF")
_FILL_HEADER = PatternFill("solid", fgColor="2F5496")
_FONT_BODY = Font(name="Arial", size=10)
_FONT_TITLE = Font(name="Arial", bold=True, size=14, color="2F5496")
_FONT_NOTE = Font(name="Arial", size=9, color="808080", italic=True)
_BORDER = Border(
    left=Side("thin", color="D0D0D0"),
    right=Side("thin", color="D0D0D0"),
    top=Side("thin", color="D0D0D0"),
    bottom=Side("thin", color="D0D0D0"),
)

_RATING_STYLES: dict[str, tuple[PatternFill, Font]] = {
    "FAIL": (
        PatternFill("solid", fgColor="FFC7CE"),
        Font(name="Arial", size=10, color="9C0006", bold=True),
    ),
    "CAUTION": (
        PatternFill("solid", fgColor="FFEB9C"),
        Font(name="Arial", size=10, color="9C6500", bold=True),
    ),
    "PASS": (
        PatternFill("solid", fgColor="C6EFCE"),
        Font(name="Arial", size=10, color="006100", bold=True),
    ),
}


def _write_cell(ws, row: int, col: int, value, fmt: str | None = None):
    """Write a styled body cell and return it."""
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = _FONT_BODY
    cell.border = _BORDER
    cell.alignment = Alignment(horizontal="center")
    if fmt:
        cell.number_format = fmt
    return cell


def _write_header_row(ws, row: int, headers: list[str]) -> None:
    """Write a styled header row."""
    for col, label in enumerate(headers, 1):
        cell = ws.cell(row=row, column=col, value=label)
        cell.font = _FONT_HEADER
        cell.fill = _FILL_HEADER
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = _BORDER


def _style_rating(cell, rating: str) -> None:
    """Apply conditional colour to a rating cell."""
    fill, font = _RATING_STYLES.get(rating, _RATING_STYLES["PASS"])
    cell.fill = fill
    cell.font = font


def build_report(
    results: dict[str, dict],
    label: str,
    thresholds: dict,
) -> openpyxl.Workbook:
    """Build a 2-tab Excel report: Summary and Detail.

    Tab 1 — Summary:  one row per tenor with rating, flags, basic stats.
    Tab 2 — Detail:   all test metrics per tenor.
    """
    wb = openpyxl.Workbook()
    tenors = list(results.keys())
    ratings = Counter(r["rating"] for r in results.values())

    # ── TAB 1: SUMMARY ────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Summary"
    ws.sheet_properties.tabColor = "2F5496"

    ws.cell(row=1, column=1, value=f"{label} — Curve Fitness Summary").font = _FONT_TITLE
    ws.cell(
        row=2, column=1,
        value=(
            f"FAIL: {ratings.get('FAIL', 0)}   "
            f"CAUTION: {ratings.get('CAUTION', 0)}   "
            f"PASS: {ratings.get('PASS', 0)}"
        ),
    ).font = _FONT_NOTE

    headers = ["Tenor", "Rating", "Flags", "Obs", "Mean (%)", "Vol (bp/d)"]
    _write_header_row(ws, 4, headers)

    for i, tenor in enumerate(tenors):
        row = 5 + i
        res = results[tenor]
        _write_cell(ws, row, 1, tenor)
        _style_rating(_write_cell(ws, row, 2, res["rating"]), res["rating"])
        _write_cell(ws, row, 3, "; ".join(res["flags"]))
        _write_cell(ws, row, 4, res.get("n_observations", ""))
        _write_cell(ws, row, 5, res.get("mean_level_pct", ""), "0.000")
        _write_cell(ws, row, 6, res.get("daily_vol_bps", ""), "0.00")

    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 12
    ws.column_dimensions["C"].width = 40
    for letter in "DEF":
        ws.column_dimensions[letter].width = 14

    # ── TAB 2: DETAIL ─────────────────────────────────────────────────────
    ws2 = wb.create_sheet("Detail")
    ws2.sheet_properties.tabColor = "4FC3F7"

    ws2.cell(row=1, column=1, value=f"{label} — Test Detail").font = _FONT_TITLE
    ws2.cell(
        row=2, column=1,
        value=(
            f"ADF: returns p > {thresholds['adf_returns_p']} → flag.  "
            f"Zero-return: > {thresholds['zero_return_pct']}% → flag.  "
            f"LB: lag-10 p < {thresholds['ljung_box_p']} → flag.  "
            f"Kurtosis: > {thresholds['excess_kurtosis']} → flag.  "
            f"Unique ratio: < {thresholds['unique_rate_ratio']} → flag."
        ),
    ).font = _FONT_NOTE

    detail_headers = [
        "Tenor",
        "ADF Levels t", "ADF Levels p", "Levels Result",
        "ADF Returns t", "ADF Returns p", "Returns Result",
        "Zero Return %", "Max Consec. Unchanged",
        "LB(5) Stat", "LB(5) p", "LB(10) Stat", "LB(10) p",
        "Skewness", "Excess Kurt.",
        "Unique Levels", "Unique Ratio",
    ]
    _write_header_row(ws2, 4, detail_headers)

    sig_fill = PatternFill("solid", fgColor="FFC7CE")

    for i, tenor in enumerate(tenors):
        row = 5 + i
        res = results[tenor]

        _write_cell(ws2, row, 1, tenor)

        if res["rating"] == "SKIP":
            _write_cell(ws2, row, 2, "—")
            continue

        _write_cell(ws2, row, 2, res["adf_levels_tstat"], "+0.000")
        _write_cell(ws2, row, 3, res["adf_levels_pvalue"], "0.0000")
        levels_label = "Unit Root" if res["adf_levels_pvalue"] > 0.05 else "Stationary"
        _write_cell(ws2, row, 4, levels_label)

        _write_cell(ws2, row, 5, res["adf_returns_tstat"], "+0.000")
        _write_cell(ws2, row, 6, res["adf_returns_pvalue"], "0.0000")
        returns_stationary = res["adf_returns_pvalue"] < thresholds["adf_returns_p"]
        returns_label = "Stationary" if returns_stationary else "NON-STATIONARY"
        rc = _write_cell(ws2, row, 7, returns_label)
        if not returns_stationary:
            rc.fill = sig_fill

        zr = res["zero_return_pct"]
        zc = _write_cell(ws2, row, 8, zr, "0.0")
        if zr > thresholds["zero_return_pct"]:
            zc.fill = sig_fill

        _write_cell(ws2, row, 9, res["max_consecutive_unchanged"])

        _write_cell(ws2, row, 10, res["lb_lag5_statistic"], "0.0")
        _write_cell(ws2, row, 11, res["lb_lag5_pvalue"], "0.00E+00")
        _write_cell(ws2, row, 12, res["lb_lag10_statistic"], "0.0")
        lb10_p = res["lb_lag10_pvalue"]
        lb_cell = _write_cell(ws2, row, 13, lb10_p, "0.00E+00")
        if lb10_p < thresholds["ljung_box_p"]:
            lb_cell.fill = sig_fill

        _write_cell(ws2, row, 14, res["skewness"], "+0.00")
        kurt = res["excess_kurtosis"]
        kc = _write_cell(ws2, row, 15, kurt, "0.0")
        if kurt > thresholds["excess_kurtosis"]:
            kc.fill = sig_fill

        _write_cell(ws2, row, 16, res["n_unique_levels"])
        ur = res["unique_rate_ratio"]
        uc = _write_cell(ws2, row, 17, ur, "0.000")
        if ur < thresholds["unique_rate_ratio"]:
            uc.fill = sig_fill

    for col in range(1, len(detail_headers) + 1):
        ws2.column_dimensions[get_column_letter(col)].width = 18

    return wb


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN — CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {INPUT_PATH} ...")
    curve = load_curve(INPUT_PATH)
    print(f"  {curve.shape[0]} dates  x  {curve.shape[1]} tenors")

    print("Running tests ...")
    results = analyse_curve(curve, THRESHOLDS)

    ratings = Counter(r["rating"] for r in results.values())
    print(
        f"  FAIL={ratings.get('FAIL', 0)}  "
        f"CAUTION={ratings.get('CAUTION', 0)}  "
        f"PASS={ratings.get('PASS', 0)}"
    )
    for tenor in curve.columns:
        r = results[tenor]
        print(f"  {tenor:>6s}  [{r['rating']:>7s}]  {'; '.join(r['flags'])}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{CURVE_LABEL}_fitness.xlsx")

    print(f"Writing {output_path} ...")
    workbook = build_report(results, CURVE_LABEL, THRESHOLDS)
    workbook.save(output_path)
    print("Done.")
```
