"""
What a ridge regression computes.

Every model package has a ``model.py`` and it always answers the same
question: *what does this model compute?* Nothing in here knows that
``rade_xl`` exists -- no registry, no specification, no pipeline. That is
the point of the split. A reader who wants the mathematics opens this file
and finds only mathematics.

For a model with a custom architecture this file holds the network class.
For one built out of a library's estimator, as here, it holds the factory
that configures it. Either way the export is "the thing the model is", and
:mod:`~rade_xl.models.ridge.register` is what plugs it in.

Why ridge rather than ordinary least squares
---------------------------------------------
A flattened sequence window produces many correlated columns -- the same
quantity at successive lags. The ridge penalty is the standard fix for a
near-collinear design and costs one parameter, which is a better default
for a reference point than a model that sometimes explodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.linear_model import Ridge

if TYPE_CHECKING:
    from .spec import RidgeSpec

__all__ = ["build"]


def build(settings: RidgeSpec) -> Ridge:
    """
    Return an unfitted ridge regression configured by the settings.

    A plain function rather than a class, because there is no state to
    hold and nothing to override. The framework needs something it can
    call; it does not need an object.

    Parameters
    ----------
    settings
        The validated model settings.

    Returns
    -------
    sklearn.linear_model.Ridge
        Unfitted. The engine fits it in place.
    """
    return Ridge(alpha=settings.alpha, fit_intercept=settings.fit_intercept)
