"""
Ridge regression: the cheapest thing that can be called a model.

Why this package is the measure of the framework
-------------------------------------------------
Everything here is about thirty statements across three small files. If it
were two hundred, the framework would be charging a tax on simplicity that
no amount of capability elsewhere would repay -- a user with a linear model
would write forty lines of scikit-learn in a notebook instead, and then have
no bundle, no lineage, no leakage-aware split and no way to compare it to
anything.

The budget is enforced by ``test_each_model_is_within_its_tier_budget``
rather than hoped for, and the honest reading of a failure is that an
abstraction above this package has grown teeth.

What it demonstrates
--------------------
The tier 1 template, complete and unembellished:

``spec.py``
    What can be configured.
``model.py``
    What is computed.
``register.py``
    How it plugs in.

Copy these three files to start any new model. See
``docs/MODEL_IMPLEMENTATION.md`` for the procedure.

Importing this package registers the model
------------------------------------------
Importing ``rade_qnet.models.ridge`` registers it under the name ``"ridge"``,
which is what lets a specification name it as a string.
"""

from .register import RidgeModel
from .spec import RidgeSpec

__all__ = ["RidgeModel", "RidgeSpec"]
