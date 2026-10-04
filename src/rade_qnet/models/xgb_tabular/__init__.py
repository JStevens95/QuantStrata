"""
Gradient-boosted trees over a flattened feature table.

Tier
----
1 -- a library estimator, the framework's own data module, no custom state.
The same three files as :mod:`rade_qnet.models.ridge`, which is how you can
tell the template is doing its job.

Why it is here
--------------
It is the model most likely to beat the flagship. See
:mod:`rade_qnet.models.xgb_tabular.model` for why, and for what the
flattening of a sequence window costs it.

Importing this package registers the model under ``"xgb_tabular"``.
"""

from .register import XgbTabularModel
from .spec import XgbTabularSpec

__all__ = ["XgbTabularModel", "XgbTabularSpec"]
