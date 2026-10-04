# `src/rade_qnet/models/lstm_tabular`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 47 | 2125 | `8ed63db8039871d8` |
| 2 | `data.py` | 69 | 2412 | `338b3f7edf0d2b0e` |
| 3 | `model.py` | 204 | 6754 | `cac89769b93bf1b1` |
| 4 | `register.py` | 88 | 2742 | `6c27ef7ca1b2f4c6` |
| 5 | `spec.py` | 29 | 720 | `684004c8b0263ebe` |

---

## 1. `src/rade_qnet/models/lstm_tabular/__init__.py`

2125 bytes · SHA-256 `8ed63db8039871d8`

```python
"""
A recurrent network with no graph: the flagship's control group.

Tier
----
1 -- a custom architecture, but the framework's own data module and no
custom fitted state. A custom ``nn.Module`` does not by itself move a model
up a tier; needing your own *data build* does.

Why this is a measurement rather than a model
-----------------------------------------------
Nobody would deploy this. It is the most informative of the three baselines
anyway, because it is the only one that isolates a single variable.

The flagship fuses two streams: a graph that encodes *what each instrument
is*, and a recurrence that encodes *what the book just did*. This package is
the second stream alone, on the same data, through the same engine, with the
same loss. The gap between them is therefore an estimate of what the graph
contributes, and it is a number that has never been measured -- it is
assumed every time the graph is maintained, debugged, or justified to
somebody.

The ridge and tree baselines cannot answer that question, because they
differ from the flagship in several ways at once. A graph model that beats a
tree has shown only that *something* about it helps. A graph model that
beats this has shown that the graph does.

Why it is not simply the flagship with the graph switched off
---------------------------------------------------------------
A flag on ``HybridGnnRnn`` would be cheaper to write and worse in two ways.
It would put a branch in the flagship's forward pass that exists only for an
experiment, and every reader of that file afterwards would have to hold two
architectures in mind. And a disabled graph still leaves the fusion layer,
the target attention and their parameters in place, so the comparison would
measure "the graph's inputs" rather than "the graph".

A separate model keeps the flagship honest and makes the comparison mean
what it says.

Importing this package registers the model under ``"lstm_tabular"``.
"""

from .model import LstmTabular
from .register import LstmTabularModel
from .spec import LstmTabularSpec

__all__ = ["LstmTabular", "LstmTabularModel", "LstmTabularSpec"]
```

---

## 2. `src/rade_qnet/models/lstm_tabular/data.py`

2412 bytes · SHA-256 `338b3f7edf0d2b0e`

```python
"""
Where the recurrent baseline's data comes from, and what it requires of it.

This package is the reason the input contract exists, so the declaration
below is worth reading alongside :mod:`rade_qnet.core.contract.requirement`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.contract.requirement import InputRequirement, RequiredInput
from ...sources.dataset.module import TabularDataModule

if TYPE_CHECKING:
    from ...core.spec.run import SupervisedRunSpec

__all__ = ["REQUIRES", "data_module"]

#: Exactly one dynamic input, of rank 3 or rank 2, under any name.
#:
#: Each of those three clauses is load-bearing.
#:
#: *Exactly one*, because a recurrence reads one sequence. Before this was
#: declared, the forward pass scanned the batch and took the first tensor
#: it found, so a build emitting two feature blocks trained on whichever
#: one the dictionary happened to yield first -- reordering the build
#: changed the model and nothing reported it.
#:
#: *Rank 3 or 2*, because a window is ``(samples, timesteps, features)``
#: and a source with no sequence transform configured produces
#: ``(samples, features)``. The second is accepted and read as a window of
#: length one: that degrades this to a one-step recurrence, which is simply
#: a worse model and a better outcome than a run that cannot start on data
#: a user already has. Rank 4 is refused rather than quietly reshaped.
#:
#: *Any name*, because the input's name is the source's choice. A model
#: that hard-coded ``features`` would work against every fixture here and
#: fail against the first user whose block is called something else.
REQUIRES = InputRequirement(
    dynamic=(
        RequiredInput(
            rank=(3, 2),
            description="the feature window the recurrence reads",
        ),
    ),
)


def data_module(spec: SupervisedRunSpec) -> TabularDataModule:
    """
    Return the data module that builds this model's dataset.

    The framework's own. The sequence window comes from the source spec's
    transforms, so this model needs no data code of its own -- which is
    what keeps it tier 1 despite writing its own network.

    Parameters
    ----------
    spec
        The validated run specification. Unused.

    Returns
    -------
    TabularDataModule
        The framework's own, with nothing added.
    """
    del spec
    return TabularDataModule()
```

---

## 3. `src/rade_qnet/models/lstm_tabular/model.py`

6754 bytes · SHA-256 `cac89769b93bf1b1`

```python
"""
What the recurrent baseline computes: one recurrence and one linear head.

This file is pure PyTorch. It imports no registry, no run specification and
no pipeline, and it would work unchanged pasted into a notebook. That is
the tier 1 and tier 2 rule and it is worth stating plainly, because the
temptation in a model this small is to collapse everything into one file and
lose the property that the mathematics can be read, reviewed and unit-tested
without the framework in the way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from ...core.runtime.errors import ContractError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...core.contract.signature import InputSignature
    from .spec import LstmTabularSpec

__all__ = ["LstmTabular", "build"]

#: Rank of a batched sequence input: ``(samples, timesteps, features)``.
_SEQUENCE_RANK = 3


class LstmTabular(nn.Module):
    """
    One recurrence and one linear head.

    Sized from the signature at construction rather than lazily, for the
    reason the flagship states at length: a model with no parameters yet
    hands an optimiser an empty parameter group, which trains nothing while
    reporting a decreasing loss from the layers that did exist.

    Parameters
    ----------
    settings
        The architecture specification.
    n_features
        Width of the input's feature axis.
    """

    def __init__(self, settings: LstmTabularSpec, *, n_features: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=settings.units,
            num_layers=settings.layers,
            dropout=settings.dropout if settings.layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(settings.units, 1)

    def forward(self, **inputs: Tensor) -> Tensor:
        """
        Read the window and predict one value per sample.

        Takes its inputs as keywords because that is the engine's calling
        convention, and by name rather than position because the flagship
        does -- a model with four inputs cannot afford positional ones.
        Variadic rather than a fixed ``features=`` parameter, because the
        input's name is the source's choice and this baseline is meant to
        run against whatever a user already has.

        The target is not among them: the loader removes it before calling,
        so there is nothing here to filter out.

        Parameters
        ----------
        **inputs
            The batch's input tensors.

        Returns
        -------
        torch.Tensor
            Shape ``(samples, 1)``.

        Raises
        ------
        ContractError
            If the batch carries no recognisable sequence input.
        """
        sequence = _sequence_of(inputs)
        output, _ = self.rnn(sequence)
        # The last timestep only. A window exists to summarise the recent
        # past into the present, so the state after reading all of it is the
        # summary; pooling across timesteps would blur the ordering the
        # recurrence was chosen for.
        return self.head(output[:, -1, :])


def build(settings: LstmTabularSpec, signature: InputSignature) -> LstmTabular:
    """
    Return an untrained network sized for the declared interface.

    Takes the signature rather than a bare width so that the sizing rule
    lives beside the architecture it sizes. ``register.py`` stays free of
    anything that could be called a modelling decision.

    Parameters
    ----------
    settings
        The validated architecture settings.
    signature
        The declared interface from the data build.

    Returns
    -------
    LstmTabular
        Untrained.

    Raises
    ------
    ContractError
        If the signature declares no dynamic input.
    """
    return LstmTabular(settings, n_features=_feature_width(signature))


def _sequence_of(inputs: Mapping[str, Tensor]) -> Tensor:
    """
    Take the batch's single input and give it a time axis.

    The input is taken rather than searched for. ``REQUIRES`` in this
    package's ``data.py`` declares exactly one dynamic input, and the
    pipeline checks that before the model is built, so by the time a batch
    arrives there is nothing to choose between.

    That is a deliberate change from what this function used to do, which
    was to scan the batch and return the first tensor it found. Handed two
    feature blocks it trained on whichever one the dictionary yielded
    first: reordering the data build changed the model, with no exception,
    no warning and a perfectly plausible loss curve. The count is asserted
    here as well as declared, because a silent wrong answer is worth two
    lines of belt and braces.

    A two-dimensional input -- one row per sample, no window -- is read as
    a window of length one rather than refused, which is the other half of
    what ``REQUIRES`` declares.

    Parameters
    ----------
    inputs
        One batch's inputs.

    Returns
    -------
    torch.Tensor
        Shape ``(samples, timesteps, features)``.

    Raises
    ------
    ContractError
        If the batch does not carry exactly one input tensor.
    """
    tensors = [value for value in inputs.values() if isinstance(value, Tensor)]
    if len(tensors) != 1:
        raise ContractError(
            f"a recurrence reads one sequence, but the batch carries "
            f"{len(tensors)} input tensor(s): {sorted(inputs)}. This should "
            f"have been caught when the signature was checked against "
            f"REQUIRES -- see rade_qnet.models.lstm_tabular.data"
        )
    sequence = tensors[0]
    if sequence.dim() == _SEQUENCE_RANK:
        return sequence
    # Insert the axis at position one, not with ``atleast_3d``, which
    # appends it: a ``(samples, features)`` input would become
    # ``(samples, features, 1)``, i.e. one timestep per *feature* of a
    # single scalar. That is a valid shape the recurrence accepts and a
    # silently wrong model, so the axis is placed explicitly.
    return sequence.unsqueeze(1)


def _feature_width(signature: InputSignature) -> int:
    """
    Return the width of the signature's single dynamic input.

    Parameters
    ----------
    signature
        The declared interface.

    Returns
    -------
    int
        Size of the last axis.

    Raises
    ------
    ContractError
        If the signature declares no dynamic input.
    """
    for tensor in signature.dynamic.values():
        return int(tensor.shape[-1])
    raise ContractError(
        "the signature declares no dynamic input, so there is nothing for a recurrence to read"
    )
```

---

## 4. `src/rade_qnet/models/lstm_tabular/register.py`

2742 bytes · SHA-256 `6c27ef7ca1b2f4c6`

```python
"""
How the recurrent baseline plugs into the framework.

Same shape as the other two tier 1 packages, with a third engine. Note that
``build_model`` here forwards the signature where ridge discards it -- that
is the only material difference between a model that must be sized in
advance and one that is not, and it is a two-word difference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.capability.supervised import SupervisedModel
from ...core.runtime.components import model

# Imported for its registration side effect: this is what puts the Torch
# engine in the registry, so the `engine="torch"` declaration below resolves
# rather than depending on what else the process happened to import.
# Defect 12 in the architecture notes.
from ...engines import torch as _engine  # noqa: F401
from .data import REQUIRES, data_module
from .model import build
from .spec import LstmTabularSpec

if TYPE_CHECKING:
    from ...core.contract.signature import InputSignature
    from ...core.spec.run import SupervisedRunSpec
    from ...sources.dataset.module import TabularDataModule
    from .model import LstmTabular

__all__ = ["LstmTabularModel"]


@model("lstm_tabular", engine="torch")
class LstmTabularModel(SupervisedModel):
    """
    Framework declaration for the flagship's temporal stream, alone.

    Attributes
    ----------
    requires
        What this model consumes, declared in ``data.py`` and checked
        against the data build before the model is constructed.
    spec
        Validates the ``model.params`` block.
    """

    requires = REQUIRES
    spec = LstmTabularSpec

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

    def build_model(self, spec: SupervisedRunSpec, signature: InputSignature) -> LstmTabular:
        """
        Return a fully parameterised network.

        Parameters
        ----------
        spec
            The run specification, for the model's parameters.
        signature
            The declared interface, which supplies the input width.

        Returns
        -------
        LstmTabular
            Untrained.
        """
        settings = LstmTabularSpec.model_validate(dict(spec.model.params))
        return build(settings, signature)
```

---

## 5. `src/rade_qnet/models/lstm_tabular/spec.py`

720 bytes · SHA-256 `684004c8b0263ebe`

```python
"""What the recurrence needs to know about itself."""

from __future__ import annotations

from pydantic import Field

from ...core.spec.base import Spec

__all__ = ["LstmTabularSpec"]


class LstmTabularSpec(Spec):
    """
    Architecture settings for the recurrent baseline.

    Parameters
    ----------
    units
        Hidden width of the recurrence.
    layers
        How many stacked recurrent layers.
    dropout
        Dropout between layers. Ignored by PyTorch when ``layers`` is one,
        which is its own behaviour rather than something to work around.
    """

    units: int = Field(default=32, ge=1)
    layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
```

