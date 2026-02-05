"""
Advanced Correlation Module.

Provides sophisticated correlation structures for scenario generation:

1. **Static Correlation**
   - Cholesky decomposition (Gaussian)
   - Copulas (Gaussian, t, Clayton, Gumbel)

2. **Dynamic Correlation**
   - DCC-GARCH (Dynamic Conditional Correlation)
   - Regime-switching correlation

3. **Tail Dependence**
   - Student-t copula for tail dependence
   - Clayton copula for lower tail
   - Gumbel copula for upper tail

Mathematical Framework
----------------------
Copulas separate marginal distributions from dependence structure.
By Sklar's theorem, any multivariate distribution can be written as:

    F(x₁, ..., xₙ) = C(F₁(x₁), ..., Fₙ(xₙ))

where C is the copula and Fᵢ are marginal CDFs.

For financial risk:
- Gaussian copula: No tail dependence (underestimates joint crashes)
- Student-t copula: Symmetric tail dependence
- Clayton copula: Lower tail dependence (good for crashes)
- Gumbel copula: Upper tail dependence

Example
-------
>>> from src.marketdata.scenarios.correlation import (
...     StudentTCopula,
...     ClaytonCopula,
...     DynamicCorrelation,
... )
>>>
>>> # Student-t copula for tail dependence
>>> copula = StudentTCopula(correlation=corr_matrix, df=4)
>>> uniform_samples = copula.sample(n_scenarios=10000)
>>>
>>> # Convert to correlated normal
>>> from scipy.stats import norm
>>> z_correlated = norm.ppf(uniform_samples)
"""

from src.marketdata.scenarios.correlation.copulas import (
    Copula,
    GaussianCopula,
    StudentTCopula,
    ClaytonCopula,
    GumbelCopula,
)
from src.marketdata.scenarios.correlation.dynamic import (
    DynamicCorrelation,
    DCCConfig,
)

__all__ = [
    # Copulas
    "Copula",
    "GaussianCopula",
    "StudentTCopula",
    "ClaytonCopula",
    "GumbelCopula",
    # Dynamic correlation
    "DynamicCorrelation",
    "DCCConfig",
]
