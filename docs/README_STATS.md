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

The complete, runnable implementation.  This is identical to
[`zar_ts_tests.py`](zar_ts_tests.py) — copy either one.

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
    """Load a zero-coupon curve from CSV or Excel.

    Expected layout
    ---------------
    Column A   — dates (any pandas-parseable format)
    Columns B… — tenor labels (ON, 1M, 3M, … 30Y)
    Values     — zero-coupon rates as decimals (0.0825 = 8.25 %)
    """
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
        warnings.warn(f"{n_dup} duplicate dates — keeping last observation.")
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
# Each function encapsulates one statistical test and returns a plain dict.
# No function reads module-level thresholds — all config arrives via params.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Q1: Distribution ──────────────────────────────────────────────────────────

def test_jarque_bera(changes_bps: np.ndarray) -> dict:
    """Jarque-Bera joint normality test.  H0: data is normal."""
    stat, pval = sp_stats.jarque_bera(changes_bps)
    return {"jb_statistic": round(float(stat), 0), "jb_pvalue": float(pval)}


def test_lilliefors_normality(changes_bps: np.ndarray) -> dict:
    """Lilliefors test — corrected KS that accounts for estimating mu/sigma
    from the sample.  Uses statsmodels kstest_normal (10M Monte-Carlo CVs)."""
    stat, pval = kstest_normal(changes_bps, dist="norm")
    return {
        "lilliefors_statistic": round(float(stat), 4),
        "lilliefors_pvalue": float(pval),
    }


def test_anderson_darling(changes_bps: np.ndarray) -> dict:
    """Anderson-Darling — weights tail discrepancies more heavily than KS."""
    result = sp_stats.anderson(changes_bps, dist="norm")
    cv_5pct = float(result.critical_values[list(result.significance_level).index(5)])
    return {
        "ad_statistic": round(float(result.statistic), 3),
        "ad_cv_5pct": round(cv_5pct, 3),
        "ad_reject_5pct": bool(result.statistic > cv_5pct),
    }


def test_moments(changes_bps: pd.Series) -> dict:
    """Excess kurtosis (Fisher, normal = 0) and skewness."""
    return {
        "excess_kurtosis": round(float(changes_bps.kurtosis()), 1),
        "skewness": round(float(changes_bps.skew()), 2),
    }


# ── Q2: Stationarity & Homogeneity ───────────────────────────────────────────

def test_adf(levels: np.ndarray, changes_bps: np.ndarray) -> dict:
    """ADF unit-root test.  Levels expected non-stationary; returns stationary."""
    adf_lev = adfuller(levels, maxlag=10, regression="c", autolag="AIC")
    adf_ret = adfuller(changes_bps, maxlag=10, regression="c", autolag="AIC")
    return {
        "adf_levels_tstat": round(float(adf_lev[0]), 3),
        "adf_levels_pvalue": round(float(adf_lev[1]), 4),
        "adf_returns_tstat": round(float(adf_ret[0]), 3),
        "adf_returns_pvalue": round(float(adf_ret[1]), 4),
    }


def test_ljung_box(changes_bps: np.ndarray) -> dict:
    """Ljung-Box test for serial autocorrelation at lags 5 and 10."""
    lb = acorr_ljungbox(changes_bps, lags=[5, 10], return_df=True)
    return {
        "lb_lag5_statistic": round(float(lb.iloc[0]["lb_stat"]), 1),
        "lb_lag5_pvalue": float(lb.iloc[0]["lb_pvalue"]),
        "lb_lag10_statistic": round(float(lb.iloc[1]["lb_stat"]), 1),
        "lb_lag10_pvalue": float(lb.iloc[1]["lb_pvalue"]),
    }


def test_regime_break(
    pre_changes: np.ndarray,
    post_changes: np.ndarray,
) -> dict:
    """Pre / post regime tests: Levene (primary), F-test, Welch t."""
    if len(pre_changes) < 3 or len(post_changes) < 3:
        return {
            "pre_regime_volatility": 0.0,
            "post_regime_volatility": 0.0,
            "f_statistic": 0.0,
            "f_pvalue": 1.0,
            "levene_statistic": 0.0,
            "levene_pvalue": 1.0,
            "welch_tstat": 0.0,
            "welch_pvalue": 1.0,
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
    w_stat, w_pval = sp_stats.ttest_ind(
        pre_changes, post_changes, equal_var=False,
    )

    return {
        "pre_regime_volatility": round(float(np.std(pre_changes, ddof=1)), 2),
        "post_regime_volatility": round(float(np.std(post_changes, ddof=1)), 2),
        "f_statistic": round(float(f_stat), 3),
        "f_pvalue": float(f_pval),
        "levene_statistic": round(float(lev_stat), 1),
        "levene_pvalue": float(lev_pval),
        "welch_tstat": round(float(w_stat), 3),
        "welch_pvalue": float(w_pval),
    }


# ── Q3: Data Quality ─────────────────────────────────────────────────────────

def test_data_quality(
    levels: pd.Series,
    changes_bps: pd.Series,
    adjacent_levels: pd.Series | None,
) -> dict:
    """Descriptive data-quality metrics — no hypothesis test."""
    n = len(changes_bps)

    zero_pct = float((changes_bps == 0).sum() / n * 100)
    near_zero_pct = float((changes_bps.abs() < 0.5).sum() / n * 100)

    groups = (levels != levels.shift()).cumsum()
    max_run = int(levels.groupby(groups).size().max())
    n_unique_4dp = int(levels.round(4).nunique())

    adj_corr: float | None = None
    if adjacent_levels is not None:
        adj_corr = round(float(levels.diff().corr(adjacent_levels.diff())), 4)

    # Log returns — guarded against non-positive rates
    positive_mask = (levels > 0) & (levels.shift(1) > 0)
    safe_ratios = levels[positive_mask] / levels.shift(1)[positive_mask]
    if len(safe_ratios.dropna()) > 1:
        log_rets = np.log(safe_ratios).dropna()
        max_abs_lr_pct: float | None = round(
            float(log_rets.abs().max() * 100), 2,
        )
    else:
        max_abs_lr_pct = None
        warnings.warn(
            "Non-positive rates detected — log returns skipped for this tenor."
        )

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
# Separated from the test engine so that every flag criterion is auditable
# in one place.
# ═══════════════════════════════════════════════════════════════════════════════

def assess_flags(results: dict, thresholds: dict) -> tuple[list[str], str]:
    """Evaluate flag conditions across all three questions.

    Returns (flag_codes, rating).
    """
    t = thresholds
    flags: list[str] = []

    # Q1
    if results["jb_pvalue"] < t["jarque_bera_p"]:
        flags.append("JB")
    if results["lilliefors_pvalue"] < t["lilliefors_p"]:
        flags.append("LILLIEFORS")
    if results["ad_reject_5pct"]:
        flags.append("AD")
    kurt = results["excess_kurtosis"]
    if abs(kurt) > t["excess_kurtosis"]:
        flags.append(f"KURT={kurt:.0f}")
    skew = results["skewness"]
    if abs(skew) > t["abs_skewness"]:
        flags.append(f"SKEW={skew:.1f}")

    # Q2
    if results["adf_returns_pvalue"] > t["adf_returns_p"]:
        flags.append("ADF_NS")
    if results["lb_lag10_pvalue"] < t["ljung_box_p"]:
        flags.append("LB_AC")
    if results["levene_pvalue"] < t["levene_p"]:
        flags.append("LEVENE")

    # Q3
    zr = results["zero_return_pct"]
    if zr > t["zero_return_pct"]:
        flags.append(f"ZERO={zr:.0f}%")
    run = results["max_consecutive_run"]
    if run > t["max_consecutive_run"]:
        flags.append(f"RUN={run}")
    adj = results["adjacent_tenor_correlation"]
    if adj is not None and adj < t["adjacent_correlation"]:
        flags.append(f"ADJ={adj:.2f}")
    n50 = results["n_events_above_50bp"]
    if n50 > t["extreme_50bp_events"]:
        flags.append(f">50bp={n50}")
    lr = results["max_abs_log_return_pct"]
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
    rate_levels: pd.Series,
    regime_break_date: str,
    thresholds: dict,
    adjacent_tenor_levels: pd.Series | None = None,
) -> dict:
    """Run every test on one tenor and return a consolidated result dict.

    The returned dict contains every key the report builder and flag
    assessment need — nothing more, nothing less.
    """
    levels = rate_levels.dropna()
    n_obs = len(levels) - 1

    if n_obs < 30:
        return {
            "rating": "SKIP",
            "n_flags": 0,
            "flags": ["INSUFFICIENT_DATA"],
            "n_observations": n_obs,
        }

    changes_bps = levels.diff().dropna() * 10_000
    pre = levels[levels.index < regime_break_date].diff().dropna() * 10_000
    post = levels[levels.index >= regime_break_date].diff().dropna() * 10_000

    # Run individual tests — each returns a dict of result keys
    jb = test_jarque_bera(changes_bps.values)
    lf = test_lilliefors_normality(changes_bps.values)
    ad = test_anderson_darling(changes_bps.values)
    mom = test_moments(changes_bps)
    adf = test_adf(levels.values, changes_bps.values)
    lb = test_ljung_box(changes_bps.values)
    regime = test_regime_break(pre.values, post.values)
    dq = test_data_quality(levels, changes_bps, adjacent_tenor_levels)

    # Assemble into one flat dict — key names match what build_report_workbook reads
    merged: dict = {
        "n_observations": n_obs,
        "mean_level_pct": round(float(levels.mean() * 100), 3),
        "daily_vol_bps": round(float(changes_bps.std()), 2),
        "autocorr_lag1_levels": round(float(levels.autocorr(lag=1)), 5),
        "autocorr_lag1_returns": round(float(changes_bps.autocorr(lag=1)), 4),
    }
    for sub_result in [jb, lf, ad, mom, adf, lb, regime, dq]:
        merged.update(sub_result)

    # Assess flags
    flags, rating = assess_flags(merged, thresholds)
    merged["flags"] = flags
    merged["n_flags"] = len(flags)
    merged["rating"] = rating

    return merged


def run_all_tests(
    curve_data: pd.DataFrame,
    regime_break_date: str,
    thresholds: dict,
) -> dict[str, dict]:
    """Run the full test suite on every tenor in the curve."""
    tenors = list(curve_data.columns)
    results: dict[str, dict] = {}
    for i, tenor in enumerate(tenors):
        adj = curve_data[tenors[i - 1]] if i > 0 else None
        results[tenor] = analyse_single_tenor(
            rate_levels=curve_data[tenor],
            regime_break_date=regime_break_date,
            thresholds=thresholds,
            adjacent_tenor_levels=adj,
        )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EXCEL REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

# ── Reusable styles ───────────────────────────────────────────────────────────

_HEADER_FONT = Font(name="Arial", bold=True, size=10, color="FFFFFF")
_HEADER_FILL = PatternFill("solid", fgColor="2F5496")
_BODY_FONT = Font(name="Arial", size=10)
_NOTE_FONT = Font(name="Arial", size=9, color="808080", italic=True)
_TITLE_FONT = Font(name="Arial", bold=True, size=14, color="2F5496")
_SECTION_FONT = Font(name="Arial", bold=True, size=11, color="2F5496")
_THIN_BORDER = Border(
    left=Side("thin", color="D0D0D0"),
    right=Side("thin", color="D0D0D0"),
    top=Side("thin", color="D0D0D0"),
    bottom=Side("thin", color="D0D0D0"),
)
_SIG_FILL = PatternFill("solid", fgColor="FFC7CE")

_RATING_STYLES: dict[str, tuple[PatternFill, Font]] = {
    "FAIL": (PatternFill("solid", fgColor="FFC7CE"),
             Font(name="Arial", size=10, color="9C0006", bold=True)),
    "CAUTION": (PatternFill("solid", fgColor="FFEB9C"),
                Font(name="Arial", size=10, color="9C6500", bold=True)),
    "PASS": (PatternFill("solid", fgColor="C6EFCE"),
             Font(name="Arial", size=10, color="006100", bold=True)),
}

# ── Helper functions ──────────────────────────────────────────────────────────


def _header_row(ws, row: int, n_cols: int) -> None:
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = _THIN_BORDER


def _cell(
    ws, row: int, col: int, value,
    fmt: str | None = None,
    highlight: PatternFill | None = None,
):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = _BODY_FONT
    cell.border = _THIN_BORDER
    cell.alignment = Alignment(horizontal="center")
    if fmt:
        cell.number_format = fmt
    if highlight:
        cell.fill = highlight
    return cell


def _rating_style(cell, rating: str) -> None:
    fill, font = _RATING_STYLES.get(rating, _RATING_STYLES["PASS"])
    cell.fill = fill
    cell.font = font


def _bar_chart(
    ws, start: int, title: str, y_label: str,
    tenors: list[str], values: list[float],
    threshold: float | None = None,
) -> int:
    """Insert a bar chart with optional threshold line. Returns next free row."""
    ws.cell(row=start, column=1, value=title).font = _SECTION_FONT
    ws.cell(row=start + 1, column=1, value="Tenor")
    ws.cell(row=start + 1, column=2, value=y_label)
    n_dc = 2
    if threshold is not None:
        ws.cell(row=start + 1, column=3, value="Threshold")
        n_dc = 3
    _header_row(ws, start + 1, n_dc)

    for i, (tn, val) in enumerate(zip(tenors, values)):
        r = start + 2 + i
        for c in range(1, n_dc + 1):
            ws.cell(row=r, column=c).font = _BODY_FONT
            ws.cell(row=r, column=c).border = _THIN_BORDER
        ws.cell(row=r, column=1, value=tn)
        ws.cell(row=r, column=2, value=val)
        if threshold is not None:
            ws.cell(row=r, column=3, value=threshold).font = _NOTE_FONT

    chart = BarChart()
    chart.type = "col"
    chart.title = title
    chart.style = 10
    chart.width = 28
    chart.height = 14
    chart.y_axis.title = y_label

    val_ref = Reference(
        ws, min_col=2, max_col=2,
        min_row=start + 2, max_row=start + 1 + len(tenors),
    )
    cat_ref = Reference(
        ws, min_col=1,
        min_row=start + 2, max_row=start + 1 + len(tenors),
    )
    chart.add_data(val_ref)
    chart.set_categories(cat_ref)
    chart.series[0].title = openpyxl.chart.series.SeriesLabel(v=y_label)

    if threshold is not None:
        thr_ref = Reference(
            ws, min_col=3, max_col=3,
            min_row=start + 2, max_row=start + 1 + len(tenors),
        )
        chart.add_data(thr_ref)
        chart.series[1].title = openpyxl.chart.series.SeriesLabel(v="Threshold")
        chart.series[1].graphicalProperties.line.width = 20000

    ws.add_chart(chart, f"E{start}")
    return start + 2 + len(tenors) + 18


# ── Workbook assembly ─────────────────────────────────────────────────────────


def build_report_workbook(
    test_results: dict[str, dict],
    label: str,
    regime_break_date: str,
    input_data: pd.DataFrame,
    thresholds: dict,
) -> openpyxl.Workbook:
    """Build the complete 7-tab Excel report."""
    wb = openpyxl.Workbook()
    tenors = list(test_results.keys())
    n_tenors = len(tenors)
    ratings = Counter(r["rating"] for r in test_results.values())
    regime_yr = regime_break_date[:4]
    n_obs = len(input_data)
    d_from = input_data.index[0].strftime("%d-%b-%Y")
    d_to = input_data.index[-1].strftime("%d-%b-%Y")
    sub = f"{n_obs} business days  |  {n_tenors} tenors  |  {d_from} to {d_to}"
    t = thresholds
    SIG = _SIG_FILL

    # ── READ ME ───────────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "READ ME"
    ws.sheet_properties.tabColor = "333333"
    ws.column_dimensions["A"].width = 4
    ws.column_dimensions["B"].width = 110

    readme_lines = [
        (2, f"{label} — Statistical Fitness Assessment", _TITLE_FONT),
        (3, f"Generated {datetime.now().strftime('%d-%b-%Y %H:%M')}", _NOTE_FONT),
        (5, "DATASET", _SECTION_FONT),
        (6, f"Label: {label}", _BODY_FONT),
        (7, f"Observations: {n_obs} business days across {n_tenors} tenors", _BODY_FONT),
        (8, f"Date range: {d_from} to {d_to}", _BODY_FONT),
        (9, f"Regime break: {regime_break_date}", _BODY_FONT),
        (11, "VERDICT", _SECTION_FONT),
        (12, (f"FAIL: {ratings.get('FAIL', 0)}  |  "
              f"CAUTION: {ratings.get('CAUTION', 0)}  |  "
              f"PASS: {ratings.get('PASS', 0)}"), _BODY_FONT),
        (14, "FRAMEWORK", _SECTION_FONT),
        (15, "This assessment answers three regulatory questions:", _BODY_FONT),
        (17, "Q1. DISTRIBUTION", _SECTION_FONT),
        (18, "     Jarque-Bera, Lilliefors, Anderson-Darling, excess kurtosis, skewness.", _BODY_FONT),
        (19, "     Ref: BCBS 352 S15; Jarque & Bera (1987); Lilliefors (1967); Anderson & Darling (1954).", _NOTE_FONT),
        (21, "Q2. STATIONARITY & HOMOGENEITY", _SECTION_FONT),
        (22, f"     ADF, Ljung-Box, Levene / F-test / Welch (pre/post {regime_yr}).", _BODY_FONT),
        (23, "     Ref: BCBS 352 S33; Dickey & Fuller (1979); Ljung & Box (1978); Levene (1960).", _NOTE_FONT),
        (25, "Q3. DATA QUALITY", _SECTION_FONT),
        (26, "     Zero-return %, max consecutive run, adjacent tenor correlation, extreme returns.", _BODY_FONT),
        (28, "RATING", _SECTION_FONT),
        (29, f"FAIL >= {t['fail_minimum_flags']} flags  |  CAUTION 1-{t['fail_minimum_flags'] - 1}  |  PASS 0", _BODY_FONT),
        (31, "IMPLEMENTATIONS (exact — no approximations)", _SECTION_FONT),
        (32, "Jarque-Bera:       scipy.stats.jarque_bera", _BODY_FONT),
        (33, "Lilliefors:        statsmodels.stats.diagnostic.kstest_normal", _BODY_FONT),
        (34, "Anderson-Darling:  scipy.stats.anderson", _BODY_FONT),
        (35, "ADF:               statsmodels.tsa.stattools.adfuller", _BODY_FONT),
        (36, "Ljung-Box:         statsmodels.stats.diagnostic.acorr_ljungbox", _BODY_FONT),
        (37, "Levene:            scipy.stats.levene", _BODY_FONT),
        (38, "F-test:            scipy.stats.f (reported; assumes normality)", _BODY_FONT),
        (39, "Welch:             scipy.stats.ttest_ind(equal_var=False)", _BODY_FONT),
    ]
    for row, text, font in readme_lines:
        ws.cell(row=row, column=2, value=text).font = font

    # ── TAB 0: INPUT DATA ─────────────────────────────────────────────────
    ws0 = wb.create_sheet("0. Input Data")
    ws0.sheet_properties.tabColor = "A0A0A0"
    ws0.cell(row=1, column=1, value=f"{label} — Input Data").font = _TITLE_FONT
    ws0.cell(row=2, column=1, value=f"{sub}. Audit trail.").font = _NOTE_FONT

    ws0.cell(row=4, column=1, value="Date")
    for ci, tn in enumerate(input_data.columns):
        ws0.cell(row=4, column=2 + ci, value=tn)
    _header_row(ws0, 4, 1 + len(input_data.columns))

    cap = min(len(input_data), 5000)
    for ri in range(cap):
        er = 5 + ri
        dc = ws0.cell(row=er, column=1, value=input_data.index[ri])
        dc.font, dc.border, dc.number_format = _BODY_FONT, _THIN_BORDER, "DD-MMM-YYYY"
        for ci in range(len(input_data.columns)):
            vc = ws0.cell(row=er, column=2 + ci, value=input_data.iloc[ri, ci])
            vc.font, vc.border, vc.number_format = _BODY_FONT, _THIN_BORDER, "0.0000%"
    if len(input_data) > 5000:
        ws0.cell(
            row=5 + cap, column=1,
            value=f"... {len(input_data) - 5000} rows truncated.",
        ).font = _NOTE_FONT

    ws0.column_dimensions["A"].width = 14
    for ci in range(len(input_data.columns)):
        ws0.column_dimensions[get_column_letter(2 + ci)].width = 11

    # ── TAB 1: SUMMARY ────────────────────────────────────────────────────
    ws1 = wb.create_sheet("1. Summary")
    ws1.sheet_properties.tabColor = "2F5496"
    ws1.cell(row=1, column=1, value=f"{label} — Per-Tenor Rating Summary").font = _TITLE_FONT
    ws1.cell(row=2, column=1, value=sub).font = _NOTE_FONT

    sh = ["Tenor", "Rating", "# Flags", "Obs",
          "Mean (%)", "Vol (bp/d)", "Flag Details"]
    for ci, h in enumerate(sh, 1):
        ws1.cell(row=4, column=ci, value=h)
    _header_row(ws1, 4, len(sh))

    for ti, tn in enumerate(tenors):
        r = 5 + ti
        res = test_results[tn]
        _cell(ws1, r, 1, tn)
        _rating_style(_cell(ws1, r, 2, res["rating"]), res["rating"])
        _cell(ws1, r, 3, res["n_flags"])
        _cell(ws1, r, 4, res.get("n_observations", ""))
        _cell(ws1, r, 5, res.get("mean_level_pct", ""), "0.000")
        _cell(ws1, r, 6, res.get("daily_vol_bps", ""), "0.00")
        _cell(ws1, r, 7, "; ".join(res["flags"]))

    ws1.column_dimensions["A"].width = 8
    ws1.column_dimensions["B"].width = 10
    ws1.column_dimensions["G"].width = 65
    for cl in "CDEF":
        ws1.column_dimensions[cl].width = 16

    # ── TAB 2: Q1 DISTRIBUTION ────────────────────────────────────────────
    ws2 = wb.create_sheet("2. Q1 Distribution")
    ws2.sheet_properties.tabColor = "4FC3F7"
    ws2.cell(row=1, column=1, value="Q1: Distribution fitness for VaR/ES").font = _TITLE_FONT
    ws2.cell(row=2, column=1, value=sub).font = _NOTE_FONT

    dh = ["Tenor", "Excess Kurt.", "Skewness",
          "JB Stat", "JB p", "JB Result",
          "Lilliefors Stat", "Lilliefors p", "Lilliefors Result",
          "AD Stat", "AD CV (5%)", "AD Result"]
    for ci, h in enumerate(dh, 1):
        ws2.cell(row=4, column=ci, value=h)
    _header_row(ws2, 4, len(dh))

    for ti, tn in enumerate(tenors):
        r = 5 + ti
        res = test_results[tn]
        if res["rating"] == "SKIP":
            continue
        _cell(ws2, r, 1, tn)
        _cell(ws2, r, 2, res["excess_kurtosis"], "0.0",
              SIG if abs(res["excess_kurtosis"]) > t["excess_kurtosis"] else None)
        _cell(ws2, r, 3, res["skewness"], "+0.00",
              SIG if abs(res["skewness"]) > t["abs_skewness"] else None)
        _cell(ws2, r, 4, res["jb_statistic"], "#,##0")
        _cell(ws2, r, 5, res["jb_pvalue"], "0.00E+00")
        jbc = _cell(ws2, r, 6, "REJECT" if res["jb_pvalue"] < t["jarque_bera_p"] else "PASS")
        if res["jb_pvalue"] < t["jarque_bera_p"]:
            jbc.fill = SIG
        _cell(ws2, r, 7, res["lilliefors_statistic"], "0.0000")
        _cell(ws2, r, 8, res["lilliefors_pvalue"], "0.00E+00")
        lfc = _cell(ws2, r, 9, "REJECT" if res["lilliefors_pvalue"] < t["lilliefors_p"] else "PASS")
        if res["lilliefors_pvalue"] < t["lilliefors_p"]:
            lfc.fill = SIG
        _cell(ws2, r, 10, res["ad_statistic"], "0.000")
        _cell(ws2, r, 11, res["ad_cv_5pct"], "0.000")
        adc = _cell(ws2, r, 12, "REJECT" if res["ad_reject_5pct"] else "PASS")
        if res["ad_reject_5pct"]:
            adc.fill = SIG

    for c in range(1, len(dh) + 1):
        ws2.column_dimensions[get_column_letter(c)].width = 16

    # ── TAB 3: Q2 STATIONARITY ────────────────────────────────────────────
    ws3 = wb.create_sheet("3. Q2 Stationarity")
    ws3.sheet_properties.tabColor = "81C784"
    ws3.cell(row=1, column=1, value="Q2: Stationarity and homogeneity").font = _TITLE_FONT
    ws3.cell(row=2, column=1, value=sub).font = _NOTE_FONT

    # Part A: Unit Root & Autocorrelation
    ws3.cell(row=4, column=1, value="A. Unit Root & Autocorrelation").font = _SECTION_FONT

    ah = ["Tenor", "ADF Lev t", "ADF Lev p", "Level Result",
          "ADF Ret t", "ADF Ret p", "Return Result",
          "AC(1) Lev", "AC(1) Ret",
          "LB(5) Stat", "LB(5) p", "LB(10) Stat", "LB(10) p"]
    for ci, h in enumerate(ah, 1):
        ws3.cell(row=5, column=ci, value=h)
    _header_row(ws3, 5, len(ah))

    for ti, tn in enumerate(tenors):
        r = 6 + ti
        res = test_results[tn]
        if res["rating"] == "SKIP":
            continue
        _cell(ws3, r, 1, tn)
        _cell(ws3, r, 2, res["adf_levels_tstat"], "+0.000")
        _cell(ws3, r, 3, res["adf_levels_pvalue"], "0.0000")
        _cell(ws3, r, 4, "Unit Root" if res["adf_levels_pvalue"] > 0.05 else "Stationary")
        _cell(ws3, r, 5, res["adf_returns_tstat"], "+0.000")
        _cell(ws3, r, 6, res["adf_returns_pvalue"], "0.0000")
        rc = _cell(ws3, r, 7, "Stationary" if res["adf_returns_pvalue"] < 0.05 else "NON-STATIONARY")
        if res["adf_returns_pvalue"] > 0.05:
            rc.fill = SIG
        _cell(ws3, r, 8, res["autocorr_lag1_levels"], "0.00000")
        _cell(ws3, r, 9, res["autocorr_lag1_returns"], "0.0000")
        _cell(ws3, r, 10, res["lb_lag5_statistic"], "0.0")
        _cell(ws3, r, 11, res["lb_lag5_pvalue"], "0.00E+00")
        _cell(ws3, r, 12, res["lb_lag10_statistic"], "0.0")
        _cell(ws3, r, 13, res["lb_lag10_pvalue"], "0.00E+00")

    for c in range(1, 14):
        ws3.column_dimensions[get_column_letter(c)].width = 16

    # Part B: Regime Break
    rs = 6 + n_tenors + 2
    ws3.cell(row=rs, column=1, value=f"B. Regime Break (pre/post {regime_yr})").font = _SECTION_FONT
    ws3.cell(row=rs + 1, column=1, value="Levene = primary (robust). F-test = reported only (assumes normality).").font = _NOTE_FONT

    rh = ["Tenor", f"Pre-{regime_yr} Vol", f"Post-{regime_yr} Vol",
          "Levene Stat", "Levene p", "Levene Result",
          "F-stat", "F p", "Welch t", "Welch p"]
    for ci, h in enumerate(rh, 1):
        ws3.cell(row=rs + 2, column=ci, value=h)
    _header_row(ws3, rs + 2, len(rh))

    for ti, tn in enumerate(tenors):
        r = rs + 3 + ti
        res = test_results[tn]
        if res["rating"] == "SKIP":
            continue
        _cell(ws3, r, 1, tn)
        _cell(ws3, r, 2, res["pre_regime_volatility"], "0.00")
        _cell(ws3, r, 3, res["post_regime_volatility"], "0.00")
        _cell(ws3, r, 4, res["levene_statistic"], "0.0")
        _cell(ws3, r, 5, res["levene_pvalue"], "0.00E+00")
        lev_reject = res["levene_pvalue"] < t["levene_p"]
        lc = _cell(ws3, r, 6, "REJECT" if lev_reject else "PASS")
        if lev_reject:
            lc.fill = SIG
        _cell(ws3, r, 7, res["f_statistic"], "0.000")
        _cell(ws3, r, 8, res["f_pvalue"], "0.00E+00")
        _cell(ws3, r, 9, res["welch_tstat"], "+0.000")
        _cell(ws3, r, 10, res["welch_pvalue"], "0.00E+00")

    # ── TAB 4: Q3 DATA QUALITY ────────────────────────────────────────────
    ws4 = wb.create_sheet("4. Q3 Data Quality")
    ws4.sheet_properties.tabColor = "FFB74D"
    ws4.cell(row=1, column=1, value="Q3: Data quality and market-tradability").font = _TITLE_FONT
    ws4.cell(row=2, column=1, value=sub).font = _NOTE_FONT

    # Part A: Step Function
    ws4.cell(row=4, column=1, value="A. Step-Function Detection").font = _SECTION_FONT
    sth = ["Tenor", "Zero Return %", "Near-Zero %", "Max Consec. Run", "Unique (4dp)"]
    for ci, h in enumerate(sth, 1):
        ws4.cell(row=5, column=ci, value=h)
    _header_row(ws4, 5, len(sth))

    for ti, tn in enumerate(tenors):
        r = 6 + ti
        res = test_results[tn]
        if res["rating"] == "SKIP":
            continue
        _cell(ws4, r, 1, tn)
        _cell(ws4, r, 2, res["zero_return_pct"], "0.0",
              SIG if res["zero_return_pct"] > t["zero_return_pct"] else None)
        _cell(ws4, r, 3, res["near_zero_pct"], "0.0")
        _cell(ws4, r, 4, res["max_consecutive_run"],
              highlight=SIG if res["max_consecutive_run"] > t["max_consecutive_run"] else None)
        _cell(ws4, r, 5, res["n_unique_values_4dp"])
    for c in range(1, 6):
        ws4.column_dimensions[get_column_letter(c)].width = 22

    # Part B: Cross-Section
    cs = 6 + n_tenors + 2
    ws4.cell(row=cs, column=1, value="B. Cross-Sectional Consistency").font = _SECTION_FONT
    csh = ["Tenor", "Adjacent Tenor", "Daily Chg Corr", "Result"]
    for ci, h in enumerate(csh, 1):
        ws4.cell(row=cs + 1, column=ci, value=h)
    _header_row(ws4, cs + 1, len(csh))

    for ti, tn in enumerate(tenors):
        r = cs + 2 + ti
        res = test_results[tn]
        if res["rating"] == "SKIP":
            continue
        _cell(ws4, r, 1, tn)
        _cell(ws4, r, 2, tenors[ti - 1] if ti > 0 else "N/A")
        ac = res["adjacent_tenor_correlation"]
        if ac is not None:
            _cell(ws4, r, 3, ac, "0.0000",
                  SIG if ac < t["adjacent_correlation"] else None)
            adj_c = _cell(ws4, r, 4, "FAIL" if ac < t["adjacent_correlation"] else "PASS")
            if ac < t["adjacent_correlation"]:
                adj_c.fill = SIG
        else:
            _cell(ws4, r, 3, "N/A")
            _cell(ws4, r, 4, "N/A")

    # Part C: Tail Risk
    ts_ = cs + 2 + n_tenors + 2
    ws4.cell(row=ts_, column=1, value="C. Tail Risk & Extreme Returns").font = _SECTION_FONT
    th_ = ["Tenor", "Max Chg (bp)", "Min Chg (bp)", "Max |LR| %",
           ">10bp", ">25bp", ">50bp",
           "P1 (bp)", "P5 (bp)", "P95 (bp)", "P99 (bp)"]
    for ci, h in enumerate(th_, 1):
        ws4.cell(row=ts_ + 1, column=ci, value=h)
    _header_row(ws4, ts_ + 1, len(th_))

    for ti, tn in enumerate(tenors):
        r = ts_ + 2 + ti
        res = test_results[tn]
        if res["rating"] == "SKIP":
            continue
        _cell(ws4, r, 1, tn)
        _cell(ws4, r, 2, res["max_daily_change_bps"], "0.0")
        _cell(ws4, r, 3, res["min_daily_change_bps"], "0.0")
        lr_val = res["max_abs_log_return_pct"]
        _cell(ws4, r, 4, lr_val if lr_val is not None else "N/A",
              "0.00" if lr_val is not None else None,
              SIG if lr_val is not None and lr_val > t["max_abs_log_return_pct"] else None)
        _cell(ws4, r, 5, res["n_events_above_10bp"])
        _cell(ws4, r, 6, res["n_events_above_25bp"])
        _cell(ws4, r, 7, res["n_events_above_50bp"],
              highlight=SIG if res["n_events_above_50bp"] > t["extreme_50bp_events"] else None)
        _cell(ws4, r, 8, res["percentile_1"], "0.0")
        _cell(ws4, r, 9, res["percentile_5"], "0.0")
        _cell(ws4, r, 10, res["percentile_95"], "0.0")
        _cell(ws4, r, 11, res["percentile_99"], "0.0")

    for c in range(1, 12):
        ws4.column_dimensions[get_column_letter(c)].width = 14

    # ── TAB 5: CHARTS ─────────────────────────────────────────────────────
    ws5 = wb.create_sheet("5. Charts")
    ws5.sheet_properties.tabColor = "4DD0E1"
    ws5.cell(row=1, column=1, value=f"{label} — Visual Evidence").font = _TITLE_FONT

    cr = 3
    cr = _bar_chart(ws5, cr, "Excess Kurtosis by Tenor", "Kurtosis",
                    tenors, [test_results[x].get("excess_kurtosis", 0) for x in tenors],
                    t["excess_kurtosis"])
    cr = _bar_chart(ws5, cr, "Skewness by Tenor", "Skewness",
                    tenors, [test_results[x].get("skewness", 0) for x in tenors])
    cr = _bar_chart(ws5, cr, "Zero-Return Frequency", "Zero %",
                    tenors, [test_results[x].get("zero_return_pct", 0) for x in tenors],
                    t["zero_return_pct"])
    cr = _bar_chart(ws5, cr, "Max Consecutive Run", "Days",
                    tenors, [test_results[x].get("max_consecutive_run", 0) for x in tenors],
                    t["max_consecutive_run"])
    cr = _bar_chart(ws5, cr, "Adjacent Tenor Correlation", "Correlation",
                    tenors, [test_results[x].get("adjacent_tenor_correlation") or 0 for x in tenors],
                    t["adjacent_correlation"])
    cr = _bar_chart(ws5, cr, "Max |Log Return|", "%",
                    tenors, [test_results[x].get("max_abs_log_return_pct") or 0 for x in tenors],
                    t["max_abs_log_return_pct"])

    # ── TAB 6: TEST REFERENCE ─────────────────────────────────────────────
    ws6 = wb.create_sheet("6. Test Reference")
    ws6.sheet_properties.tabColor = "78909C"
    ws6.cell(row=1, column=1, value="Test Methodology, Thresholds & References").font = _TITLE_FONT
    ws6.cell(row=2, column=1, value="All hypothesis tests use exact library implementations.").font = _NOTE_FONT

    ws6.column_dimensions["A"].width = 4
    ws6.column_dimensions["B"].width = 10
    ws6.column_dimensions["C"].width = 22
    ws6.column_dimensions["D"].width = 60
    ws6.column_dimensions["E"].width = 18
    ws6.column_dimensions["F"].width = 38
    ws6.column_dimensions["G"].width = 40

    rch = ["Question", "Test", "Description", "Threshold", "Implementation", "Reference"]
    for ci, h in enumerate(rch, 2):
        ws6.cell(row=4, column=ci, value=h)
    _header_row(ws6, 4, 7)

    ref_data = [
        ("Q1", "Jarque-Bera", "Joint normality: JB = n/6 (S^2 + K^2/4).",
         f"p < {t['jarque_bera_p']}", "scipy.stats.jarque_bera",
         "Jarque & Bera (1987); BCBS 352 S15"),
        ("Q1", "Lilliefors", "Corrected KS (accounts for parameter estimation).",
         f"p < {t['lilliefors_p']}", "statsmodels kstest_normal",
         "Lilliefors (1967)"),
        ("Q1", "Anderson-Darling", "Tail-weighted normality test.",
         f"Stat > CV at {t['anderson_darling_sig']}%", "scipy.stats.anderson",
         "Anderson & Darling (1954)"),
        ("Q1", "Excess Kurtosis", "Tail heaviness (normal = 0).",
         f"> {t['excess_kurtosis']}", "pandas .kurtosis()",
         "VaR model validation"),
        ("Q1", "Skewness", "Asymmetry (normal = 0).",
         f"|skew| > {t['abs_skewness']}", "pandas .skew()",
         "VaR model validation"),
        ("Q2", "ADF (Levels)", "Unit root on levels (expected non-stationary).",
         "p > 0.05 (expected)", "statsmodels adfuller",
         "Dickey & Fuller (1979)"),
        ("Q2", "ADF (Returns)", "Unit root on changes (expected stationary).",
         f"p > {t['adf_returns_p']} -> flag", "statsmodels adfuller",
         "Dickey & Fuller (1979)"),
        ("Q2", "Ljung-Box", "Serial correlation at lags 5, 10.",
         f"p < {t['ljung_box_p']}", "statsmodels acorr_ljungbox",
         "Ljung & Box (1978); BCBS 352 S16"),
        ("Q2", "Levene", "Primary variance test (robust, non-normality safe).",
         f"p < {t['levene_p']}", "scipy.stats.levene",
         "Levene (1960)"),
        ("Q2", "F-test", "Variance ratio (assumes normality — reported only).",
         "Not flagged", "scipy.stats.f",
         "Two-sample F-test"),
        ("Q2", "Welch t-test", "Mean equality (unequal variance).",
         "Not flagged", "scipy.stats.ttest_ind",
         "Welch (1947)"),
        ("Q3", "Zero-Return %", "Days with exactly zero change.",
         f"> {t['zero_return_pct']}%", "pandas count",
         "Descriptive"),
        ("Q3", "Max Consec. Run", "Longest identical-value streak.",
         f"> {t['max_consecutive_run']} days", "pandas groupby",
         "Descriptive"),
        ("Q3", "Adjacent Corr.", "Daily-change correlation with neighbour.",
         f"< {t['adjacent_correlation']}", "pandas .corr()",
         "Cross-sectional check"),
        ("Q3", "Max |Log Return|", "Largest single-day |log return| (%).",
         f"> {t['max_abs_log_return_pct']}%", "numpy log",
         "Tail risk"),
        ("Q3", ">50bp Events", "Days with |change| > 50 bps.",
         f"> {t['extreme_50bp_events']}", "pandas count",
         "Tail risk"),
    ]

    for ri, (q, name, desc, thr, impl, ref) in enumerate(ref_data):
        r = 5 + ri
        ws6.cell(row=r, column=2, value=q).font = _BODY_FONT
        ws6.cell(row=r, column=2).border = _THIN_BORDER
        ws6.cell(row=r, column=3, value=name).font = Font(name="Arial", bold=True, size=10)
        ws6.cell(row=r, column=3).border = _THIN_BORDER
        dc = ws6.cell(row=r, column=4, value=desc)
        dc.font, dc.border = _BODY_FONT, _THIN_BORDER
        dc.alignment = Alignment(wrap_text=True)
        ws6.cell(row=r, column=5, value=thr).font = Font(name="Arial", size=10, color="FF0000")
        ws6.cell(row=r, column=5).border = _THIN_BORDER
        ws6.cell(row=r, column=6, value=impl).font = Font(name="Arial", size=9, color="2F5496")
        ws6.cell(row=r, column=6).border = _THIN_BORDER
        ws6.cell(row=r, column=7, value=ref).font = _NOTE_FONT
        ws6.cell(row=r, column=7).border = _THIN_BORDER

    return wb


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
    wb = build_report_workbook(
        results, LABEL, REGIME_BREAK_DATE, curve, DEFAULT_THRESHOLDS,
    )
    wb.save(out)
    print("Done.")
```
