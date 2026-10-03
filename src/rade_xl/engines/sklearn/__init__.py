"""
The scikit-learn engine.

Exists to make simple models genuinely cheap.  A ridge regression should cost
a few lines and still receive the full lifecycle -- leakage-aware splits,
versioned bundles, the standard metrics and reports, and fan-out across a job
set.  A framework that only pays off for large models is a framework people
work around.

Modules
-------
``engine.py``
    Wraps any estimator exposing ``fit`` and ``predict``, persisting it via
    ``joblib`` alongside the framework's own manifest.
``adapters.py``
    Turns a stream of batches into the single matrix a one-shot fit needs.
    Shared with the XGBoost engine, so there is one definition of how a
    source becomes a matrix and one definition of the row order that
    results.

Importing this package registers its components
-----------------------------------------------
Importing ``rade_xl.engines.sklearn`` registers :class:`SklearnEngine` under
the name ``"sklearn"``, which is what lets a specification name it as a
string.  Eager rather than lazy, for the reason given in the Torch engine's
package docstring: a registry populated only once somebody happens to have
imported the right module is the classic source of "no engine named
'sklearn'" from a configuration that is perfectly correct.
"""

from .engine import ENGINE_NAME, SklearnEngine

__all__ = ["ENGINE_NAME", "SklearnEngine"]
