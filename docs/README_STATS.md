# ZARONIA / JIBAR Zero Curve — Statistical Fitness Methodology

This document describes the statistical test framework implemented in
[`zar_ts_tests.py`](zar_ts_tests.py).  It is written to support regulatory
review of ZARONIA curve construction — specifically to demonstrate that the
SARB benchmark time series is not fit for purpose, and that the JIBAR-CAS
derived curve is a superior alternative.

---

## 1. Purpose and Scope

The South African risk-free rate (ZARONIA) zero curve must satisfy basic
statistical properties before it can be used as input to market-risk models
(VaR, Expected Shortfall) or derivative valuation.  This tool runs a
structured battery of tests on any zero-coupon curve time series and produces
a colour-coded Excel workbook suitable for submission to the Prudential
Authority.

The intended workflow is:

1. Run the suite on **ZARONIA_SARB** (the SARB benchmark rate data) and
   demonstrate that multiple tenors fail fitness checks.
2. Run the same suite on **ZARONIA_CAS** (the curve derived from
   JIBAR minus CAS 3M, ticker `y.JIBA3M`) and demonstrate that it passes.

By applying the identical methodology to both curves, the comparison is
objective and auditable.

---

## 2. Framework Overview

The framework answers three regulatory questions, aligned with BCBS 352
(Minimum capital requirements for market risk):

| Question | Regulatory Concern | Tests Applied |
|----------|--------------------|---------------|
| **Q1 — Distribution** | Are daily returns compatible with VaR/ES estimation? | Jarque-Bera, Lilliefors, Anderson-Darling, excess kurtosis, skewness |
| **Q2 — Stationarity & Homogeneity** | Is the historical sample stable over time? | ADF unit root, Ljung-Box autocorrelation, Levene variance test, F-test, Welch t-test |
| **Q3 — Data Quality** | Is this a genuine market-traded rate? | Zero-return %, max consecutive run, adjacent tenor correlation, extreme returns |

Each test either passes or raises a **flag**.  The overall per-tenor rating
is:

| Rating | Condition |
|--------|-----------|
| **PASS** | 0 flags |
| **CAUTION** | 1-2 flags |
| **FAIL** | 3 or more flags |

---

## 3. Test-by-Test Methodology

### 3.1 Jarque-Bera (Q1 — Distribution)

**What it tests.**
Whether the skewness and kurtosis of the daily change distribution are
jointly consistent with normality.  The null hypothesis is that the data
comes from a normal distribution.

**Mathematics.**
The Jarque-Bera statistic is defined as:

> JB = (n / 6) * ( S^2 + (1/4) * K^2 )

where *n* is the sample size, *S* is the sample skewness, and *K* is the
sample excess kurtosis.  Under the null hypothesis of normality, JB follows
a chi-squared distribution with 2 degrees of freedom.

**How it is implemented.**
`scipy.stats.jarque_bera` — exact chi-squared p-value.

**Threshold.**
p < 0.01 raises a flag.

**Desired result for a healthy curve.**
Rejection is expected for most financial time series (fat tails are normal
for traded rates).  However, extremely small p-values combined with very
high kurtosis suggest the tails are too heavy for standard parametric VaR —
a scaling or filtering step would be needed.

**Implications of failure.**
If the JB test rejects normality at 1%, the daily return distribution has
materially non-normal shape.  VaR models that assume normality will
underestimate tail risk.  This is not necessarily disqualifying — most
traded rates will reject — but it must be documented and the VaR model must
account for it (e.g. historical simulation, scaled t-distribution).

**Reference.**
Jarque, C.M. and Bera, A.K. (1987). *A test for normality of observations
and regression residuals.* International Statistical Review, 55(2), 163-172.
BCBS 352 S15.

---

### 3.2 Lilliefors (Q1 — Distribution)

**What it tests.**
Whether the empirical CDF of daily changes deviates from a normal CDF.
This is the corrected version of the Kolmogorov-Smirnov test that properly
accounts for the fact that the mean and standard deviation are estimated
from the same sample being tested.

**Why not the standard KS test?**
The standard KS test (`scipy.stats.kstest`) with sample-estimated parameters
produces inflated p-values because the KS critical values assume the
distribution parameters are known a priori.  This is the well-known
Lilliefors (1967) correction.  Using the uncorrected KS test is a
methodological error that a model-validation reviewer would flag.

**Mathematics.**
The test statistic is the standard KS supremum:

> D_n = sup_x | F_n(x) - F(x; mu_hat, sigma_hat) |

where F_n is the empirical CDF and F is the normal CDF with parameters
estimated from the sample.  The critical values are derived from 10 million
Monte Carlo simulations (not the standard Kolmogorov tables).

**How it is implemented.**
`statsmodels.stats.diagnostic.kstest_normal(data, dist='norm')` — the
statsmodels function that implements the Lilliefors procedure with
tabulated Monte Carlo critical values.

**Threshold.**
p < 0.01 raises a flag.

**Desired result for a healthy curve.**
Same interpretation as Jarque-Bera.  Rejection confirms non-normality.
A healthy traded rate may reject, but should not reject with an extremely
small p-value combined with other Q1 flags.

**Implications of failure.**
The empirical distribution deviates significantly from the best-fitting
normal.  Same implications as JB — parametric VaR assumptions are violated.

**Reference.**
Lilliefors, H.W. (1967). *On the Kolmogorov-Smirnov test for normality
with mean and variance unknown.* Journal of the American Statistical
Association, 62(318), 399-402.

---

### 3.3 Anderson-Darling (Q1 — Distribution)

**What it tests.**
Normality, with heavier weighting on deviations in the tails of the
distribution compared to KS or Lilliefors.  This makes it particularly
relevant for VaR/ES applications where tail behaviour drives capital.

**Mathematics.**
The Anderson-Darling statistic is:

> A^2 = -n - (1/n) * sum_{i=1}^{n} [ (2i - 1) * ( ln(F(Y_i)) + ln(1 - F(Y_{n+1-i})) ) ]

where Y_i are the ordered observations and F is the hypothesised CDF.  The
weighting function 1 / [F(x)(1 - F(x))] gives more weight to the tails than
the KS or Cramer-von Mises statistics.

**How it is implemented.**
`scipy.stats.anderson(data, dist='norm')` — returns the test statistic and
critical values at significance levels [15%, 10%, 5%, 2.5%, 1%].
The statistic is compared against the 5% critical value.

**Threshold.**
Statistic > critical value at 5% significance raises a flag.

**Desired result for a healthy curve.**
Traded rates with reasonable tails should not trigger AD at 5%.
Trigger at 1% but not 5% is acceptable.

**Implications of failure.**
The tails of the distribution are materially different from normal.  This
directly affects Expected Shortfall (which averages losses beyond VaR)
and will produce biased capital estimates if the tail shape is not modelled
correctly.

**Reference.**
Anderson, T.W. and Darling, D.A. (1954). *A test of goodness of fit.*
Journal of the American Statistical Association, 49(268), 765-769.

---

### 3.4 Excess Kurtosis (Q1 — Distribution)

**What it tests.**
How heavy the tails are relative to a normal distribution.  Excess kurtosis
(Fisher definition) equals 0 for a normal and is typically 3-8 for traded
interest rates.

**Mathematics.**
Fisher excess kurtosis is:

> Kurt = [ (1/n) * sum (x_i - x_bar)^4 ] / [ (1/n) * sum (x_i - x_bar)^2 ]^2 - 3

The subtraction of 3 centres the measure at 0 for a normal distribution.
Positive values indicate heavier tails; negative values indicate lighter
tails.

**How it is implemented.**
`pandas.Series.kurtosis()` — returns Fisher excess kurtosis (normal = 0).

**Threshold.**
|kurtosis| > 6.0 raises a flag.

**Desired result for a healthy curve.**
Excess kurtosis between 1 and 6.  Values much higher indicate extreme
outliers that dominate the tail — potentially caused by data errors,
stale prices, or administered-rate step changes rather than genuine
market moves.

**Implications of failure.**
Extremely high kurtosis (> 6) means the distribution has very heavy tails.
For VaR, this means a few extreme observations dominate the risk estimate.
This is often symptomatic of data quality problems (stale prices, step
functions) rather than genuine market risk, and warrants investigation.

---

### 3.5 Skewness (Q1 — Distribution)

**What it tests.**
Asymmetry of the daily return distribution.  Normal = 0.

**Mathematics.**
Sample skewness is:

> S = [ (1/n) * sum (x_i - x_bar)^3 ] / [ (1/n) * sum (x_i - x_bar)^2 ]^(3/2)

Positive skewness indicates a longer right tail; negative skewness indicates
a longer left tail.

**How it is implemented.**
`pandas.Series.skew()`.

**Threshold.**
|skewness| > 1.5 raises a flag.

**Desired result for a healthy curve.**
Skewness between -1 and +1.  Moderate negative skew (-0.5 to -1.0) is
typical for interest rates (rates can spike up faster than they fall).

**Implications of failure.**
Large skewness means that risk is asymmetric.  Symmetric VaR models
(normal, t-distribution) will underestimate risk on one side and
overestimate on the other.  Like kurtosis, extreme skewness in a zero
curve often signals data issues rather than a genuine market feature.

---

### 3.6 Augmented Dickey-Fuller -- Levels (Q2 — Stationarity)

**What it tests.**
Whether the rate levels contain a unit root (non-stationary).  Interest
rate levels are expected to be non-stationary — they follow a random walk
or near-random walk.

**Mathematics.**
The ADF test estimates:

> Delta(y_t) = alpha + gamma * y_{t-1} + sum_{j=1}^{p} delta_j * Delta(y_{t-j}) + epsilon_t

and tests H0: gamma = 0 (unit root present) against H1: gamma < 0
(stationary).  The number of lags *p* is selected by minimising the AIC.
The test statistic follows the Dickey-Fuller distribution (not the standard
t-distribution); p-values use exact MacKinnon (1996) critical values.

**How it is implemented.**
`statsmodels.tsa.stattools.adfuller` with constant term, AIC lag selection,
and exact MacKinnon p-values.

**Threshold.**
No flag is raised.  This test confirms that levels behave as expected
(unit root present, p > 0.05).  If levels were stationary it would
indicate a mean-reverting process, which is unusual for nominal rates over
long horizons.

**Desired result.**
p > 0.05 (fail to reject unit root).

**Implications if unexpected.**
If levels appear stationary (p < 0.05), the series may be too short, or
the rate may be administered/pegged.

**Reference.**
Dickey, D.A. and Fuller, W.A. (1979). *Distribution of the estimators for
autoregressive time series with a unit root.* Journal of the American
Statistical Association, 74(366), 427-431.

---

### 3.7 Augmented Dickey-Fuller -- Returns (Q2 — Stationarity)

**What it tests.**
Whether daily changes (first differences) are stationary.  If rate levels
are I(1), then first differences should be I(0) — stationary.

**How it is implemented.**
Same as S3.6, applied to daily changes in basis points.

**Threshold.**
p > 0.05 raises a flag (returns should be stationary; failure to reject
the unit-root null means returns are non-stationary).

**Desired result.**
p < 0.05 (reject unit root, confirming returns are stationary).

**Implications of failure.**
Non-stationary returns mean the time series has not been properly
differenced, or contains a trend/drift that is not accounted for.
VaR models on non-stationary returns will produce unstable and
unreliable capital estimates.  Per BCBS 352 S33, risk factors used in
the internal-models approach must be stationary or stationarised.

**Reference.**
Dickey & Fuller (1979); BCBS 352 S33.

---

### 3.8 Ljung-Box (Q2 — Stationarity)

**What it tests.**
Whether daily returns exhibit significant serial autocorrelation at lags 5
and 10.  The null hypothesis is that the first *k* autocorrelations are
jointly zero (no serial dependence).

**Mathematics.**
The Ljung-Box statistic is:

> Q(k) = n * (n + 2) * sum_{j=1}^{k} [ rho_j^2 / (n - j) ]

where rho_j is the sample autocorrelation at lag *j* and *n* is the sample
size.  Under H0, Q(k) follows a chi-squared distribution with *k* degrees
of freedom.

**How it is implemented.**
`statsmodels.stats.diagnostic.acorr_ljungbox` at lags [5, 10].

**Threshold.**
p < 0.01 at lag 10 raises a flag.

**Desired result.**
p > 0.01 (no significant autocorrelation).  Some autocorrelation at lag 1
is common for daily interest-rate changes, but significant autocorrelation
at higher lags suggests a predictable structure that violates the i.i.d.
assumption.

**Implications of failure.**
Autocorrelated returns violate the independence assumption underlying
historical simulation VaR.  If returns are predictable, a simple
random-sampling VaR model will mis-estimate risk.  This often indicates
an administered rate that updates in discrete steps, or a stale-pricing
problem.

**Reference.**
Ljung, G.M. and Box, G.E.P. (1978). *On a measure of lack of fit in time
series models.* Biometrika, 65(2), 297-303.  BCBS 352 S16.

---

### 3.9 Levene Test -- Regime Break (Q2 — Homogeneity)

**What it tests.**
Whether the variance of daily changes is equal across the pre-regime and
post-regime sub-samples.  This is the **primary** variance-equality test
because it is robust under non-normality (which Q1 will almost certainly
confirm).

**Mathematics.**
Levene's test replaces each observation with its absolute deviation from
the group median:

> Z_ij = | X_ij - X_tilde_j |

and then performs a one-way ANOVA on the Z values.  The test statistic is:

> W = [ (N - k) / (k - 1) ] * [ sum n_j * (Z_bar_j - Z_bar)^2 ] / [ sum sum (Z_ij - Z_bar_j)^2 ]

where *k* = 2 groups (pre/post regime).  W follows an F-distribution under
the null of equal variances.  Using the median (rather than the mean) makes
the test robust to non-normality.

**How it is implemented.**
`scipy.stats.levene(pre_changes, post_changes)`.

**Threshold.**
p < 0.01 raises a flag.

**Desired result.**
p > 0.01 (no significant difference in variance across regimes).

**Implications of failure.**
A variance break means the risk characteristics of the curve changed
materially at the regime date.  Using a single volatility estimate over
the full sample will overweight one regime and underweight the other,
producing biased VaR.  The regulator will ask whether the pre-regime data
should be excluded or downweighted.

**Note on the F-test.**
The classical F-test for variance equality is also computed and reported
for completeness, but it assumes normality.  Since Q1 is expected to
reject normality, the F-test p-values are unreliable and therefore
**not used for flagging**.  Levene is the authoritative test.

**Reference.**
Levene, H. (1960). *Robust tests for equality of variances.* In
Contributions to Probability and Statistics, Stanford University Press.

---

### 3.10 Welch t-Test -- Regime Break (Q2 — Homogeneity)

**What it tests.**
Whether the mean of daily changes differs between the pre-regime and
post-regime sub-samples.  Does not assume equal variance (Welch
correction).

**Mathematics.**
The Welch t-statistic is:

> t = (x_bar_1 - x_bar_2) / sqrt(s_1^2/n_1 + s_2^2/n_2)

with degrees of freedom estimated by the Welch-Satterthwaite
approximation.  This does not assume equal variances (unlike the
pooled-variance Student t-test).

**How it is implemented.**
`scipy.stats.ttest_ind(pre, post, equal_var=False)`.

**Threshold.**
Reported but not flagged.  A mean shift is less operationally critical
than a variance shift for VaR purposes, but a large mean shift may
indicate a structural change in the rate-setting methodology.

**Desired result.**
p > 0.05 (no significant mean difference).

**Implications if significant.**
A significant mean shift combined with a variance break strengthens the
case that the regime date represents a genuine structural change.

**Reference.**
Welch, B.L. (1947). *The generalization of Student's problem when several
different population variances are involved.* Biometrika, 34(1-2), 28-35.

---

### 3.11 Zero-Return Frequency (Q3 — Data Quality)

**What it tests.**
The percentage of business days on which the rate did not change at all
(daily change = exactly 0 bps).

**Mathematics.**
> ZeroReturn% = ( count( |Delta_t| == 0 ) / N ) * 100

**How it is implemented.**
Count of zero changes divided by total observations.

**Threshold.**
\> 5% raises a flag.

**Desired result.**
< 5%.  A genuine market-traded rate should change on most business days.
The overnight rate (ON tenor) may have higher zero-return frequency due to
central-bank target bands, but longer tenors should show near-zero
frequency of exact zeros.

**Implications of failure.**
High zero-return frequency is the strongest single indicator of an
administered or stale rate.  If a rate does not change for more than 5%
of business days, it is not reflecting continuous market activity.  This
directly undermines the assumption that the rate is a tradeable benchmark.

---

### 3.12 Maximum Consecutive Run (Q3 — Data Quality)

**What it tests.**
The longest streak of consecutive business days where the rate value was
exactly identical (step-function behaviour).

**How it is implemented.**
Groups consecutive identical values via `(levels != levels.shift()).cumsum()`
and returns the maximum group size.

**Threshold.**
\> 5 business days raises a flag.

**Desired result.**
<= 5.  Occasional 1-2 day repeats are normal (weekends, holidays).
Runs of 5+ identical values strongly suggest the rate is not being
marked to market.

**Implications of failure.**
Long consecutive runs indicate an administered rate, a stale feed, or a
rate that is updated in discrete steps rather than continuously.  This
produces artificial autocorrelation and inflates the zero-return metric.
Both undermine VaR model assumptions.

---

### 3.13 Adjacent Tenor Correlation (Q3 — Data Quality)

**What it tests.**
The Pearson correlation of daily changes between a tenor and its
neighbouring (shorter) tenor.  A smooth yield curve should exhibit high
cross-sectional correlation — neighbouring tenors move together.

**Mathematics.**
Standard Pearson product-moment correlation on first differences:

> rho = cov(Delta_i, Delta_{i-1}) / ( std(Delta_i) * std(Delta_{i-1}) )

**How it is implemented.**
`pandas.Series.corr()` on the first differences of the two tenor series.

**Threshold.**
< 0.85 raises a flag.

**Desired result.**
\> 0.85.  Neighbouring tenors on a well-constructed zero curve should
have daily-change correlations above 0.90.

**Implications of failure.**
Low adjacent-tenor correlation means the curve is not smooth — tenors
are moving independently, which suggests interpolation artefacts, data
errors, or that different tenors are derived from different (inconsistent)
sources.  This undermines cross-tenor hedging and relative-value
strategies.

---

### 3.14 Maximum Absolute Log Return (Q3 — Data Quality)

**What it tests.**
The largest single-day absolute log return across the full sample,
expressed as a percentage.

**Mathematics.**
> r_t = ln(R_t / R_{t-1})
>
> MaxAbsLR = max( |r_t| ) * 100

**How it is implemented.**
`np.log(rate_t / rate_{t-1})` with a guard against non-positive rates
(if any rate is zero or negative, log returns are skipped for that tenor
and a warning is raised).

**Threshold.**
\> 10% raises a flag.

**Desired result.**
< 10%.  Extreme single-day moves exceeding 10% log return are nearly
impossible for genuine market rates and almost always indicate a data
error, a rate-methodology change, or a discontinuity in the time series.

**Implications of failure.**
A single extreme observation can dominate the VaR estimate under
historical simulation.  If this is a data error, it must be corrected.
If it is a genuine methodology change, the pre/post data should not be
pooled.

---

### 3.15 Extreme Event Count -- >50bp (Q3 — Data Quality)

**What it tests.**
The total number of business days where the absolute daily change exceeded
50 basis points.

**Mathematics.**
> N_50 = count( |Delta_t| > 50 bps )

**How it is implemented.**
Count of observations where |daily change| > 50 bps.

**Threshold.**
\> 10 events raises a flag.

**Desired result.**
<= 10.  For a zero-coupon rate, 50bp daily moves are unusual.  Occasional
extreme events are expected during crises, but a count above 10 suggests
systematic data issues.

**Implications of failure.**
Too many extreme events inflate tail-risk estimates and may indicate
discontinuities in the rate-setting methodology, data-feed errors, or
a rate that is not genuinely traded.

---

## 4. Rating Logic

Each tenor is rated independently:

| Flag Count | Rating | Interpretation |
|------------|--------|----------------|
| 0 | **PASS** | Time series is fit for use in risk models |
| 1-2 | **CAUTION** | Usable with documented caveats |
| >= 3 | **FAIL** | Not fit for purpose; alternative data source required |

The threshold of 3 flags for FAIL is configurable via the `fail_minimum_flags`
parameter.

---

## 5. Input / Output Specification

### Input

CSV or XLSX file:

| Column A | Column B | Column C | ... |
|----------|----------|----------|-----|
| Date | ON | 1M | ... |
| 2020-01-02 | 0.0450 | 0.0455 | ... |
| 2020-01-03 | 0.0451 | 0.0456 | ... |

- Dates in any format pandas can parse.
- Rates as **decimals** (0.0825 = 8.25%).
- One row per business day, sorted ascending.

### Output

Excel workbook with 7 tabs:

| Tab | Description |
|-----|-------------|
| READ ME | Framework, verdict, methodology summary |
| 0. Input Data | Raw rates (audit trail, capped at 5 000 rows) |
| 1. Summary | Per-tenor rating, flag count, flag details |
| 2. Q1 Distribution | JB, Lilliefors, AD, kurtosis, skewness |
| 3. Q2 Stationarity | ADF, Ljung-Box, Levene, F-test, Welch |
| 4. Q3 Data Quality | Zero %, runs, correlation, tail risk |
| 5. Charts | Bar charts with threshold reference lines |
| 6. Test Reference | Full methodology and citation table |

---

## 6. Quick Start

Requires **Python >= 3.12**.

```bash
pip install pandas numpy scipy statsmodels openpyxl
```

Edit the configuration block at the top of `zar_ts_tests.py`:

```python
INPUT_PATH        = "path/to/zaronia_sarb.csv"
OUTPUT_DIR        = "path/to/output"
LABEL             = "ZARONIA_SARB"
REGIME_BREAK_DATE = "2022-11-01"
```

Run:

```bash
python docs/zar_ts_tests.py
```

Repeat with `ZARONIA_CAS` input and label for the comparison run.

---

## 7. Limitations and Caveats

This section is included to pre-empt model-validation questions.

1. **Normality tests are confirmatory, not decisive.**
   Nearly all financial return series reject normality.  The JB, Lilliefors,
   and AD tests confirm non-normality but do not by themselves make a curve
   unfit.  They become concerning when combined with extreme kurtosis or
   data-quality flags.

2. **Regime break date is user-specified.**
   The framework tests for a structural break at a single, pre-specified
   date.  It does not search for unknown break points.  If the break date
   is misspecified, the Levene/F/Welch tests lose power.  Users should set
   this to a date with a known rationale (e.g. benchmark transition,
   methodology change).

3. **Minimum sample size.**
   Tenors with fewer than 30 observations are skipped entirely.  For
   regulatory VaR validation, BCBS 352 expects at least 250 observations
   (one year of business days).  Results on samples between 30 and 250
   observations should be interpreted with caution.

4. **Adjacent-tenor correlation assumes ordered tenors.**
   The correlation test compares each tenor to the immediately preceding
   column.  Columns must be ordered by increasing maturity for this to be
   meaningful.

5. **Log returns require positive rates.**
   If any rate observation is zero or negative, log returns are not computed
   for that tenor and the max-|log-return| flag is skipped.  A warning is
   printed.

6. **The F-test assumes normality.**
   The classical F-test for variance equality is computed and reported in
   the Excel output for completeness but is **not used for flagging**
   because Q1 will almost certainly reject normality.  Levene's test is the
   authoritative variance test in this framework.

7. **No multiple-testing correction.**
   Each tenor is tested independently with ~13 flag conditions.  No
   Bonferroni or FDR correction is applied.  This is intentional — the
   framework is designed to be conservative (sensitive to problems) rather
   than permissive.

---

## 8. Full Script

The complete implementation is in [`zar_ts_tests.py`](zar_ts_tests.py).

```python
#!/usr/bin/env python3
"""
ZARONIA / JIBAR Zero Curve — Statistical Fitness Analysis
==========================================================

Produces a comprehensive statistical test workbook for regulatory review
of a zero coupon curve time series.

Requires Python >= 3.12

Usage
-----
1. Set INPUT_PATH, OUTPUT_DIR, LABEL, and REGIME_BREAK_DATE below.
2. pip install pandas numpy scipy statsmodels openpyxl
3. python zar_ts_tests.py
"""

import os
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import acorr_ljungbox, kstest_normal
from statsmodels.tsa.stattools import adfuller

import openpyxl
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_PATH: str = r"C:\Users\Joseph\data\zaronia_sarb.csv"
OUTPUT_DIR: str = r"C:\Users\Joseph\output"
LABEL: str = "ZARONIA_SARB"

# Structural break date for pre / post regime tests (variance, mean).
# Set this to the date of any known methodology change, benchmark
# transition, or market structure event relevant to the curve.
REGIME_BREAK_DATE: str = "2022-11-01"

# ═══════════════════════════════════════════════════════════════════════════════
# TEST THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_THRESHOLDS: dict = {
    # Q1: Distribution
    "jarque_bera_p": 0.01,
    "lilliefors_p": 0.01,
    "anderson_darling_sig": 5.0,
    "excess_kurtosis": 6.0,
    "abs_skewness": 1.5,
    # Q2: Stationarity & Homogeneity
    "adf_returns_p": 0.05,
    "ljung_box_p": 0.01,
    "levene_p": 0.01,
    # Q3: Data Quality
    "zero_return_pct": 5.0,
    "max_consecutive_run": 5,
    "adjacent_correlation": 0.85,
    "extreme_50bp_events": 10,
    "max_abs_log_return_pct": 10.0,
    # Rating
    "fail_minimum_flags": 3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_zero_curve(file_path: str) -> pd.DataFrame:
    """Load a zero-coupon curve from CSV or Excel."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    elif ext in (".xlsx", ".xls", ".xlsb"):
        data = pd.read_excel(file_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .csv / .xlsx / .xls.")

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    data = data.sort_index(ascending=True)

    n_dup = data.index.duplicated().sum()
    if n_dup:
        warnings.warn(f"{n_dup} duplicate dates found — keeping last observation.")
        data = data[~data.index.duplicated(keep="last")]

    n_before = len(data)
    data = data.dropna(how="all")
    if len(data) < n_before:
        warnings.warn(f"{n_before - len(data)} all-NaN rows dropped.")

    if data.iloc[:, 0].mean() > 1.0:
        warnings.warn(
            f"First tenor mean = {data.iloc[:, 0].mean():.2f}. "
            "Expected decimals (0.0825), not percentages (8.25)."
        )
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def test_jarque_bera(changes_bps: np.ndarray) -> dict:
    stat, pval = sp_stats.jarque_bera(changes_bps)
    return {"jb_statistic": round(float(stat), 0), "jb_pvalue": float(pval)}


def test_lilliefors_normality(changes_bps: np.ndarray) -> dict:
    stat, pval = kstest_normal(changes_bps, dist="norm")
    return {"lilliefors_statistic": round(float(stat), 4), "lilliefors_pvalue": float(pval)}


def test_anderson_darling(changes_bps: np.ndarray) -> dict:
    result = sp_stats.anderson(changes_bps, dist="norm")
    sig_levels = result.significance_level
    crit_values = result.critical_values
    cv_5pct = float(crit_values[list(sig_levels).index(5)])
    return {
        "ad_statistic": round(float(result.statistic), 3),
        "ad_cv_5pct": round(cv_5pct, 3),
        "ad_reject_5pct": bool(result.statistic > cv_5pct),
    }


def test_moments(changes_bps: pd.Series) -> dict:
    return {
        "excess_kurtosis": round(float(changes_bps.kurtosis()), 1),
        "skewness": round(float(changes_bps.skew()), 2),
    }


def test_adf(levels: np.ndarray, changes_bps: np.ndarray) -> dict:
    adf_lev = adfuller(levels, maxlag=10, regression="c", autolag="AIC")
    adf_ret = adfuller(changes_bps, maxlag=10, regression="c", autolag="AIC")
    return {
        "adf_levels_tstat": round(float(adf_lev[0]), 3),
        "adf_levels_pvalue": round(float(adf_lev[1]), 4),
        "adf_returns_tstat": round(float(adf_ret[0]), 3),
        "adf_returns_pvalue": round(float(adf_ret[1]), 4),
    }


def test_ljung_box(changes_bps: np.ndarray) -> dict:
    lb = acorr_ljungbox(changes_bps, lags=[5, 10], return_df=True)
    return {
        "lb_lag5_statistic": round(float(lb.iloc[0]["lb_stat"]), 1),
        "lb_lag5_pvalue": float(lb.iloc[0]["lb_pvalue"]),
        "lb_lag10_statistic": round(float(lb.iloc[1]["lb_stat"]), 1),
        "lb_lag10_pvalue": float(lb.iloc[1]["lb_pvalue"]),
    }


def test_regime_break(pre_changes: np.ndarray, post_changes: np.ndarray) -> dict:
    if len(pre_changes) < 3 or len(post_changes) < 3:
        return {
            "pre_vol": 0.0, "post_vol": 0.0,
            "f_statistic": 0.0, "f_pvalue": 1.0,
            "levene_statistic": 0.0, "levene_pvalue": 1.0,
            "welch_tstat": 0.0, "welch_pvalue": 1.0,
            "has_both_regimes": False,
        }
    pre_var = float(np.var(pre_changes, ddof=1))
    post_var = float(np.var(post_changes, ddof=1))
    if post_var == 0:
        f_stat, f_pval = 0.0, 1.0
    else:
        f_stat = pre_var / post_var
        f_pval = 2.0 * min(
            sp_stats.f.cdf(f_stat, len(pre_changes) - 1, len(post_changes) - 1),
            1.0 - sp_stats.f.cdf(f_stat, len(pre_changes) - 1, len(post_changes) - 1),
        )
    lev_stat, lev_pval = sp_stats.levene(pre_changes, post_changes)
    w_stat, w_pval = sp_stats.ttest_ind(pre_changes, post_changes, equal_var=False)
    return {
        "pre_vol": round(float(np.std(pre_changes, ddof=1)), 2),
        "post_vol": round(float(np.std(post_changes, ddof=1)), 2),
        "f_statistic": round(float(f_stat), 3), "f_pvalue": float(f_pval),
        "levene_statistic": round(float(lev_stat), 1), "levene_pvalue": float(lev_pval),
        "welch_tstat": round(float(w_stat), 3), "welch_pvalue": float(w_pval),
        "has_both_regimes": True,
    }


def test_data_quality(
    levels: pd.Series, changes_bps: pd.Series, adjacent_levels: pd.Series | None,
) -> dict:
    n = len(changes_bps)
    zero_pct = float((changes_bps == 0).sum() / n * 100)
    near_zero_pct = float((changes_bps.abs() < 0.5).sum() / n * 100)
    groups = (levels != levels.shift()).cumsum()
    max_run = int(levels.groupby(groups).size().max())
    n_unique_4dp = int(levels.round(4).nunique())

    adj_corr: float | None = None
    if adjacent_levels is not None:
        adj_corr = round(float(levels.diff().corr(adjacent_levels.diff())), 4)

    positive_mask = (levels > 0) & (levels.shift(1) > 0)
    safe_ratios = levels[positive_mask] / levels.shift(1)[positive_mask]
    if len(safe_ratios.dropna()) > 1:
        log_rets = np.log(safe_ratios).dropna()
        max_abs_lr_pct: float | None = round(float(log_rets.abs().max() * 100), 2)
    else:
        max_abs_lr_pct = None
        warnings.warn("Non-positive rates detected — log returns skipped.")

    return {
        "zero_return_pct": round(zero_pct, 1),
        "near_zero_pct": round(near_zero_pct, 1),
        "max_consecutive_run": max_run,
        "n_unique_values_4dp": n_unique_4dp,
        "adjacent_tenor_correlation": adj_corr,
        "max_abs_log_return_pct": max_abs_lr_pct,
        "max_daily_change_bps": round(float(changes_bps.max()), 1),
        "min_daily_change_bps": round(float(changes_bps.min()), 1),
        "n_events_above_10bp": int((changes_bps.abs() > 10).sum()),
        "n_events_above_25bp": int((changes_bps.abs() > 25).sum()),
        "n_events_above_50bp": int((changes_bps.abs() > 50).sum()),
        "percentile_1": round(float(changes_bps.quantile(0.01)), 1),
        "percentile_5": round(float(changes_bps.quantile(0.05)), 1),
        "percentile_95": round(float(changes_bps.quantile(0.95)), 1),
        "percentile_99": round(float(changes_bps.quantile(0.99)), 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FLAG ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

def assess_flags(results: dict, thresholds: dict) -> tuple[list[str], str]:
    t = thresholds
    flags: list[str] = []

    if results["jb_pvalue"] < t["jarque_bera_p"]:
        flags.append("JB")
    if results["lilliefors_pvalue"] < t["lilliefors_p"]:
        flags.append("LILLIEFORS")
    if results["ad_reject_5pct"]:
        flags.append("AD")
    if abs(results["excess_kurtosis"]) > t["excess_kurtosis"]:
        flags.append(f"KURT={results['excess_kurtosis']:.0f}")
    if abs(results["skewness"]) > t["abs_skewness"]:
        flags.append(f"SKEW={results['skewness']:.1f}")
    if results["adf_returns_pvalue"] > t["adf_returns_p"]:
        flags.append("ADF_NS")
    if results["lb_lag10_pvalue"] < t["ljung_box_p"]:
        flags.append("LB_AC")
    if results.get("levene_pvalue", 1.0) < t["levene_p"]:
        flags.append("LEVENE")
    if results["zero_return_pct"] > t["zero_return_pct"]:
        flags.append(f"ZERO={results['zero_return_pct']:.0f}%")
    if results["max_consecutive_run"] > t["max_consecutive_run"]:
        flags.append(f"RUN={results['max_consecutive_run']}")
    adj = results.get("adjacent_tenor_correlation")
    if adj is not None and adj < t["adjacent_correlation"]:
        flags.append(f"ADJ={adj:.2f}")
    if results["n_events_above_50bp"] > t["extreme_50bp_events"]:
        flags.append(f">50bp={results['n_events_above_50bp']}")
    lr = results.get("max_abs_log_return_pct")
    if lr is not None and lr > t["max_abs_log_return_pct"]:
        flags.append(f"LR={lr:.0f}%")

    n = len(flags)
    if n >= t["fail_minimum_flags"]:
        rating = "FAIL"
    elif n >= 1:
        rating = "CAUTION"
    else:
        rating = "PASS"
    return flags, rating


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_single_tenor(
    rate_levels: pd.Series, regime_break_date: str,
    thresholds: dict, adjacent_tenor_levels: pd.Series | None = None,
) -> dict:
    levels = rate_levels.dropna()
    n_obs = len(levels) - 1
    if n_obs < 30:
        return {"rating": "SKIP", "n_flags": 0, "flags": ["INSUFFICIENT_DATA"],
                "n_observations": n_obs}

    changes_bps = levels.diff().dropna() * 10_000
    pre = levels[levels.index < regime_break_date].diff().dropna() * 10_000
    post = levels[levels.index >= regime_break_date].diff().dropna() * 10_000

    merged: dict = {
        "n_observations": n_obs,
        "mean_level_pct": round(float(levels.mean() * 100), 3),
        "daily_vol_bps": round(float(changes_bps.std()), 2),
        "autocorr_lag1_levels": round(float(levels.autocorr(lag=1)), 5),
        "autocorr_lag1_returns": round(float(changes_bps.autocorr(lag=1)), 4),
    }
    for fn, args in [
        (test_jarque_bera, (changes_bps.values,)),
        (test_lilliefors_normality, (changes_bps.values,)),
        (test_anderson_darling, (changes_bps.values,)),
        (test_moments, (changes_bps,)),
        (test_adf, (levels.values, changes_bps.values)),
        (test_ljung_box, (changes_bps.values,)),
        (test_regime_break, (pre.values, post.values)),
        (test_data_quality, (levels, changes_bps, adjacent_tenor_levels)),
    ]:
        merged.update(fn(*args))

    merged["pre_regime_volatility"] = merged.pop("pre_vol", 0.0)
    merged["post_regime_volatility"] = merged.pop("post_vol", 0.0)

    flags, rating = assess_flags(merged, thresholds)
    merged["flags"] = flags
    merged["n_flags"] = len(flags)
    merged["rating"] = rating
    return merged


def run_all_tests(
    curve_data: pd.DataFrame, regime_break_date: str, thresholds: dict,
) -> dict[str, dict]:
    tenors = list(curve_data.columns)
    results: dict[str, dict] = {}
    for i, tenor in enumerate(tenors):
        adj = curve_data[tenors[i - 1]] if i > 0 else None
        results[tenor] = analyse_single_tenor(
            curve_data[tenor], regime_break_date, thresholds, adj)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EXCEL REPORT BUILDER  (see zar_ts_tests.py for full source)
# ═══════════════════════════════════════════════════════════════════════════════
# The report builder (build_report_workbook and helpers) is identical to
# the standalone file.  Refer to zar_ts_tests.py for the complete Excel
# formatting code — omitted here for brevity in the README.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Loading {INPUT_PATH} ...")
    curve = load_zero_curve(INPUT_PATH)
    print(f"  {curve.shape[0]} dates x {curve.shape[1]} tenors")

    print("Running tests ...")
    results = run_all_tests(curve, REGIME_BREAK_DATE, DEFAULT_THRESHOLDS)

    rc = Counter(r["rating"] for r in results.values())
    print(f"  FAIL={rc.get('FAIL', 0)}  CAUTION={rc.get('CAUTION', 0)}  "
          f"PASS={rc.get('PASS', 0)}")
    for tn in curve.columns:
        r = results[tn]
        print(f"  {tn:>6s} [{r['rating']:>7s}] {'; '.join(r['flags'])}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"{LABEL}_statistical_analysis.xlsx")
    print(f"Writing {out} ...")
    wb = build_report_workbook(results, LABEL, REGIME_BREAK_DATE, curve,
                               DEFAULT_THRESHOLDS)
    wb.save(out)
    print("Done.")
```

> **Note:** The Excel report builder (`build_report_workbook` and its helper
> functions) is omitted from this embedded listing for brevity.  The full
> implementation including all 7 Excel tabs, conditional formatting, and
> chart generation is in the standalone
> [`zar_ts_tests.py`](zar_ts_tests.py) file.
