"""
Tools that prove a model conforms to the framework's contracts.

A framework that only documents its contracts will see them violated.  This
package makes conformance executable: a model author imports a suite, points it
at their model, and gets a verdict.  It is shipped as part of the package (not
hidden in the test tree) because model authors outside this repository need it
too.

Modules
-------
``conformance.py``
    The contract suite.  Where ``isinstance`` against a protocol only proves a
    method exists, this calls it and checks the result: that a source is
    re-iterable and honest about its size, that a fitted state's inverse is a
    real inverse, that a bundle's splits do not overlap along the scenario
    axis, and that every declared capability behaves.  It deliberately does
    *not* flag entity-axis fitting over the full universe, which is legitimate.
    [Phase 1 delivered, extended per phase]
``fixtures.py``
    Small synthetic sources, specs, bundles and hooks.  Deterministic and fast,
    so a pipeline test needs no real data, no network and no training library.
    ``isolated_registries`` is the one every test that registers a component
    should use.  [Phase 1, delivered]
``parity.py``
    Compares a run against a golden fixture captured from the original
    implementation, at a tolerance chosen per level.  Exact for the fitted
    state and the batch contents, where a difference means different
    arithmetic; looser for the forward pass and the training curve, where
    reassociated floating point legitimately moves the last bits.  The effort
    is in the diagnostic rather than the verdict: a failure names the array,
    the worst element, both values and the mismatch count, because "arrays
    differ" across a refactor of several thousand lines is a day of
    bisection.  [Phase 0+3, delivered]

Not part of the production runtime
----------------------------------
Nothing in the framework's training path imports this package.  It may depend
on ``pytest``, which the rest of ``rade_xl`` may not.
"""

__all__: tuple[str, ...] = ()
