# `src/rade_qnet/models/xgb_tabular`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 22 | 647 | `8a5273793dc9fa45` |
| 2 | `data.py` | 75 | 2787 | `cf2983dc854e17c0` |
| 3 | `model.py` | 72 | 2857 | `c7cf7a7b0632cefd` |
| 4 | `register.py` | 84 | 2508 | `292e398c993f0db6` |
| 5 | `spec.py` | 55 | 2332 | `446633bfa7a1d14b` |

---

## 1. `src/rade_qnet/models/xgb_tabular/__init__.py`

647 bytes · SHA-256 `8a5273793dc9fa45`

```python
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
```

---

## 2. `src/rade_qnet/models/xgb_tabular/data.py`

2787 bytes · SHA-256 `cf2983dc854e17c0`

```python
"""
Where the boosted-tree model's data comes from, and what it requires of it.

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
from ...sources.dataset.tabular import TabularDataModule

if TYPE_CHECKING:
    from ...core.spec.run import SupervisedRunSpec

__all__ = ["REQUIRES", "data_module"]

#: The boosted-tree model accepts anything.
#:
#: Not an omission. Every input is flattened into one design matrix before
#: it reaches the booster, so the number of blocks, their names and their
#: ranks do not change what this model computes.
#:
#: Worth knowing when reading a comparison against the flagship: a tree
#: handed a flattened window cannot tell that column 7 and column 19 are
#: the same quantity at different lags, and has to rediscover any such
#: relationship from the data. That is a real handicap on a sequential
#: problem and none at all otherwise -- and it is a consequence of this
#: declaration, not a detail of the engine.
#:
#: Declared explicitly rather than left unset, because "this model has no
#: constraints" and "nobody wrote the constraints down" look identical from
#: the outside and mean very different things. A reader of this line knows
#: the question was asked.
REQUIRES = InputRequirement.unconstrained()


def data_module(spec: SupervisedRunSpec) -> TabularDataModule:
    """
    Return the data module that builds this model's dataset.

    The framework's own, unmodified: a boosted-tree model reads a table, and
    a table is what :class:`~rade_qnet.sources.dataset.tabular.TabularDataModule`
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

## 3. `src/rade_qnet/models/xgb_tabular/model.py`

2857 bytes · SHA-256 `c7cf7a7b0632cefd`

```python
"""
What the boosted-tree model computes.

Why a tree baseline earns its place
------------------------------------
It is the model most likely to win. On tabular problems of the size a desk
actually has -- hundreds to tens of thousands of rows, dozens of features,
no spatial or sequential structure an architecture exploits -- boosted trees
beat neural networks routinely, and consistently enough that the burden of
proof sits with the network.

So this is not a straw man included for completeness. If the flagship cannot
beat it, that is the single most useful thing a comparison can report.

What it is handed, and why that matters when reading the result
-----------------------------------------------------------------
A sequence window arrives here flattened: one column per timestep-feature
pair. The tree therefore cannot know that column 7 and column 19 are the
same quantity at different lags, and has to rediscover any such relationship
from the data. That is a real handicap on a genuinely sequential problem and
no handicap at all on one where the recent past is just a set of numbers.

Which of those is true for a given book is exactly what comparing this
against ``lstm_tabular`` answers, and it is why both exist rather than one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Imported from the engine package rather than the library, because
# importing the package is what registers the engine -- and on macOS it is
# also what forces Torch's OpenMP runtime to load first. See
# `PHASE_6_ADDITIONAL_ENGINES.md` §8.2: the reverse order deadlocks with no
# error at all.
from ...engines.xgboost import BoosterModel

if TYPE_CHECKING:
    from .spec import XgbTabularSpec

__all__ = ["build"]


def build(settings: XgbTabularSpec) -> BoosterModel:
    """
    Return a holder for the booster training will produce.

    A booster is *returned* by ``xgboost.train`` rather than constructed and
    then fitted, so what can be built before training is an empty slot. That
    is what :class:`~rade_qnet.engines.xgboost.engine.BoosterModel` is: the
    framework needs an object to hand to ``prepare`` and to save at the end,
    and the library does not provide one until the fit is over.

    The holder is created empty because this model declares no settings of
    its own -- see :mod:`rade_qnet.models.xgb_tabular.spec` for why, and for
    what went wrong when it did. Everything the booster is configured with
    comes from the training spec, through the engine.

    Parameters
    ----------
    settings
        The validated model settings. Empty, and accepted anyway so that
        the signature does not have to change the day this model acquires
        one.

    Returns
    -------
    BoosterModel
        Unfitted, carrying no overrides.
    """
    del settings
    return BoosterModel()
```

---

## 4. `src/rade_qnet/models/xgb_tabular/register.py`

2508 bytes · SHA-256 `292e398c993f0db6`

```python
"""
How the boosted-tree model plugs into the framework.

Structurally identical to :mod:`rade_qnet.models.ridge.register`, which is the
point of having a template: a different engine and a different spec, in the
same four declarations and the same two methods. A reader who has
understood one model package has understood every tier 1 and tier 2 package
in the repository.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.authoring.supervised import SupervisedModel
from ...core.lifecycle.components import model
from .data import REQUIRES, data_module
from .model import build
from .spec import XgbTabularSpec

if TYPE_CHECKING:
    from ...core.contract.signature import InputSignature
    from ...core.spec.run import SupervisedRunSpec
    from ...engines.xgboost import BoosterModel
    from ...sources.dataset.tabular import TabularDataModule

__all__ = ["XgbTabularModel"]


@model("xgb_tabular", engine="xgboost")
class XgbTabularModel(SupervisedModel):
    """
    Framework declaration for the boosted-tree model.

    Attributes
    ----------
    requires
        What this model consumes, declared in ``data.py`` and checked
        against the data build before the model is constructed.
    spec
        Validates the ``model.params`` block of a run specification.
    """

    requires = REQUIRES
    spec = XgbTabularSpec

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

    def build_model(self, spec: SupervisedRunSpec, signature: InputSignature) -> BoosterModel:
        """
        Construct the unfitted booster holder.

        Parameters
        ----------
        spec
            The validated run specification.
        signature
            The declared interface from the data build. Unused: a tree
            takes its width from the matrix it is handed.

        Returns
        -------
        BoosterModel
            Unfitted.
        """
        del signature
        return build(XgbTabularSpec.model_validate(dict(spec.model.params)))
```

---

## 5. `src/rade_qnet/models/xgb_tabular/spec.py`

2332 bytes · SHA-256 `446633bfa7a1d14b`

```python
"""
What the boosted-tree model needs to know about itself: nothing.

An empty spec is a real and correct answer, not a placeholder
---------------------------------------------------------------
Two schemas describe a run: ``model.params``, validated here, and
``training``, validated by the engine's own
:class:`~rade_qnet.core.spec.training.XGBoostTrainingSpec`. The question
every model must answer is which settings belong on which side, and the
test is simple:

    Would changing it make this a *different model*, or the *same model
    trained differently*?

For this model the answer is always the second. It has no architecture of
its own -- it is the engine's booster over a flat table -- and
``XGBoostTrainingSpec`` already describes that booster completely: depth,
learning rate, subsample, regularisation, objective, round budget, early
stopping. There is nothing left for the model to declare.

So the class is empty, and the file still exists, because the layout is the
same at every tier and "this model has no settings" is worth saying once
and explicitly rather than leaving a reader to infer it from an absence.

Why this is not merely tidy
----------------------------
The earlier version of this file re-declared ``max_depth``,
``min_child_weight`` and ``reg_lambda`` with the same defaults as the
training spec. The engine merges the two at ``engine.py`` with
``{**booster_params(spec), **model.params}`` -- model last, model wins --
so a user who set ``training.max_depth: 12`` trained at depth 6 and was
told nothing. Both numbers were plausible, the run succeeded, and the only
symptom was a model slightly worse than it should have been.

That is the failure mode duplicated settings always have, and it is why
the ownership question above is the first one to answer when writing a
spec rather than a detail to settle later.
"""

from __future__ import annotations

from ...core.spec.base import Spec

__all__ = ["XgbTabularSpec"]


class XgbTabularSpec(Spec):
    """
    No model-level settings; the engine's training spec owns them all.

    Kept as a class rather than omitted so that the model's declaration in
    ``register.py`` reads the same as every other model's, and so that a
    specification carrying an unexpected ``model.params`` key is rejected
    rather than silently ignored.
    """
```

