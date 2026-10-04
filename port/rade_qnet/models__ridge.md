# `src/rade_qnet/models/ridge`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 40 | 1382 | `8e878ba14afd847c` |
| 2 | `data.py` | 70 | 2472 | `45111227031bd139` |
| 3 | `model.py` | 53 | 1769 | `86653e260ade081f` |
| 4 | `register.py` | 106 | 3612 | `1cacbb9686650b6a` |
| 5 | `spec.py` | 44 | 1532 | `968365b42f941d08` |

---

## 1. `src/rade_qnet/models/ridge/__init__.py`

1382 bytes · SHA-256 `8e878ba14afd847c`

```python
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
```

---

## 2. `src/rade_qnet/models/ridge/data.py`

2472 bytes · SHA-256 `45111227031bd139`

```python
"""
Where the ridge regression's data comes from, and what it requires of it.

Every model package has a ``data.py`` and it always answers two questions:

``REQUIRES``
    What the model consumes -- the input contract, checked against whatever
    the build produced before the model is constructed.
``data_module``
    Where the data comes from and how it is prepared.

Both live here, including when the answer is "the framework's own", because
the alternative is that a reader has to know which of two shapes a model is
before they know where to look.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.contract.requirement import InputRequirement
from ...sources.dataset.module import TabularDataModule

if TYPE_CHECKING:
    from ...core.spec.run import SupervisedRunSpec

__all__ = ["REQUIRES", "data_module"]

#: Ridge accepts anything.
#:
#: Not an omission. The sklearn adapter flattens every input it is handed
#: into one design matrix, so the number of blocks, their names and their
#: ranks genuinely do not change what this model computes -- a window of
#: four features over five steps and a flat row of twenty columns produce
#: the same fit.
#:
#: Declared explicitly rather than left unset, because "this model has no
#: constraints" and "nobody wrote the constraints down" look identical from
#: the outside and mean very different things. A reader of this line knows
#: the question was asked.
REQUIRES = InputRequirement.unconstrained()


def data_module(spec: SupervisedRunSpec) -> TabularDataModule:
    """
    Return the data module that builds this model's dataset.

    The framework's own, unmodified: a ridge regression reads a table, and
    a table is what :class:`~rade_qnet.sources.dataset.module.TabularDataModule`
    already knows how to load, split, scale, window and batch.

    A real deployment usually replaces this. The moment the data lives in
    a store rather than a CSV, the loader is yours -- and the change is to
    subclass ``DataModule`` here and return that instead, with nothing else
    in the package moving.

    Parameters
    ----------
    spec
        The validated run specification. Unused: the module reads its
        settings from ``spec.source`` when the framework calls it, which
        keeps this a pure constructor.

    Returns
    -------
    TabularDataModule
        The framework's own, with nothing added.
    """
    del spec
    return TabularDataModule()
```

---

## 3. `src/rade_qnet/models/ridge/model.py`

1769 bytes · SHA-256 `86653e260ade081f`

```python
"""
What a ridge regression computes.

Every model package has a ``model.py`` and it always answers the same
question: *what does this model compute?* Nothing in here knows that
``rade_qnet`` exists -- no registry, no specification, no pipeline. That is
the point of the split. A reader who wants the mathematics opens this file
and finds only mathematics.

For a model with a custom architecture this file holds the network class.
For one built out of a library's estimator, as here, it holds the factory
that configures it. Either way the export is "the thing the model is", and
:mod:`~rade_qnet.models.ridge.register` is what plugs it in.

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
```

---

## 4. `src/rade_qnet/models/ridge/register.py`

3612 bytes · SHA-256 `1cacbb9686650b6a`

```python
"""
How the ridge regression plugs into the framework.

Every model package has a ``register.py`` and it always does the same four
things: name the model, name its engine, bind the spec that validates its
settings and the contract its data must meet, and say how to build its data
and itself. It contains no
mathematics. A platform engineer maintaining the framework reads this file;
a quant arguing with the model reads ``model.py``; neither has to skim the
other's half.

Importing this module is what makes ``"ridge"`` resolvable by name in a run
specification. The package's ``__init__`` imports it for exactly that
reason, so a user who has imported the model package can name the model in
YAML without knowing this file exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.capability.supervised import SupervisedModel
from ...core.runtime.components import model

# Imported for its registration side effect: this is what puts the
# scikit-learn engine in the registry. The decorator below declares
# `engine="sklearn"`, and a declaration whose subject may or may not be
# registered depending on what else the process happened to import is the
# classic source of "no engine named 'sklearn'" from a correct
# specification. Defect 12 in the architecture notes.
from ...engines import sklearn as _engine  # noqa: F401
from .data import REQUIRES, data_module
from .model import build
from .spec import RidgeSpec

if TYPE_CHECKING:
    from sklearn.linear_model import Ridge

    from ...core.contract.signature import InputSignature
    from ...core.spec.run import SupervisedRunSpec
    from ...sources.dataset.module import TabularDataModule

__all__ = ["RidgeModel"]


@model("ridge", engine="sklearn")
class RidgeModel(SupervisedModel):
    """
    Framework declaration for the ridge regression.

    Attributes
    ----------
    requires
        What this model consumes, declared in ``data.py`` and checked
        against the data build before the model is constructed.
    spec
        Validates the ``model.params`` block of a run specification.
    """

    requires = REQUIRES
    spec = RidgeSpec

    def data_module(self, spec: SupervisedRunSpec) -> TabularDataModule:
        """
        Return the data module that builds this model's dataset.

        Delegated to ``data.py``, which is where every question about this
        model's data is answered -- including the requirement bound above.

        Parameters
        ----------
        spec
            The validated run specification.

        Returns
        -------
        TabularDataModule
            Whatever ``data.py`` builds.
        """
        return data_module(spec)

    def build_model(self, spec: SupervisedRunSpec, signature: InputSignature) -> Ridge:
        """
        Construct the unfitted estimator.

        A pure function of the spec, with no data in sight. That is what
        makes a bundle reloadable six months later: the saved signature
        plus the saved weights are sufficient to rebuild the identical
        object with no need to reproduce the dataset.

        Parameters
        ----------
        spec
            The validated run specification.
        signature
            The declared interface from the data build. Unused: a linear
            model takes its width from the matrix it is handed rather than
            being sized in advance.

        Returns
        -------
        sklearn.linear_model.Ridge
            Unfitted.
        """
        del signature
        return build(RidgeSpec.model_validate(dict(spec.model.params)))
```

---

## 5. `src/rade_qnet/models/ridge/spec.py`

1532 bytes · SHA-256 `968365b42f941d08`

```python
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
```

