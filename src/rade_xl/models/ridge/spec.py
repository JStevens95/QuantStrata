"""
What a ridge regression needs to know about itself.

Every model package has a ``spec.py`` and it always means the same thing:
the Pydantic models that validate the open-ended blocks of a run
specification. For a tier 1 model that is one class validating
``model.params``. For a model with its own data build it is two, the second
validating ``source.params``.

Settings live here rather than beside the code that reads them so that a
reader can answer "what can I configure?" without reading any logic, and so
that a bad configuration is a parse error rather than a failure some minutes
into a run.
"""

from __future__ import annotations

from pydantic import Field

from ...core.spec.base import Spec

__all__ = ["RidgeSpec"]


class RidgeSpec(Spec):
    """
    Settings for the ridge regression.

    Parameters
    ----------
    alpha
        Strength of the L2 penalty. Larger shrinks the coefficients harder.
        Constrained positive: an alpha of zero is ordinary least squares,
        which on the near-collinear design a flattened sequence window
        produces is numerically unstable in a way that reads as enormous
        coefficients and a model excellent in sample and useless out of it.
    fit_intercept
        Whether to fit an intercept. Leave it on unless the target is
        already centred, because a model forced through the origin on
        data that is not centred spends its coefficients on the offset.
    """

    alpha: float = Field(default=1.0, gt=0.0)
    fit_intercept: bool = True
