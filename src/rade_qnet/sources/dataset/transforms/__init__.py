"""
Fitted transforms, each with an explicit inverse.

Every transform here follows the same contract: it is fitted on training rows,
it reports the ``FittedState`` it produced, and -- if it touched the target --
it can invert a prediction back to the original units.  A transform that cannot
state its inverse cannot be applied to a target, because a metric computed in
transformed space is not the metric anyone asked for.

Modules
-------
``scaling.py``
    Standardisation and robust scaling, fitted on training rows only.  Owns
    the target inverse when it scaled the target.  [Phase 2]
``sequence.py``
    Rolling-window construction, with window boundaries respected at split
    edges.  Plain functions rather than a fitted state: windowing is fully
    determined by the specification, so there is nothing to fit.  [Phase 2]
``reduction.py``
    Dimensionality reduction, including basis selection.  Exposes an explicit
    ``fit_on`` choice (``train`` or ``all``) rather than silently fitting on
    every row, which is a leakage path in the implementation this framework
    replaces.  [Phase 2]
``encoding.py``
    Categorical and attribute encoding along the entity axis.  The one
    transform fitted over the full universe rather than training rows, for the
    reason given in its module docstring.  [Phase 2]
``composite.py``
    Composes the above into the single ``FittedState`` a bundle holds, and
    settles which part owns the target inverse.  [Phase 2]
"""

__all__: tuple[str, ...] = ()
