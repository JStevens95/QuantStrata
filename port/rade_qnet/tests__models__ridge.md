# `tests/rade_qnet/models/ridge`

3 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 1 | 42 | `f92a7a31ce5f7c72` |
| 2 | `test_model.py` | 70 | 2668 | `b4bbdc31a30aaafd` |
| 3 | `test_register.py` | 95 | 3431 | `ce59e428145e17ed` |

---

## 1. `tests/rade_qnet/models/ridge/__init__.py`

42 bytes · SHA-256 `f92a7a31ce5f7c72`

```python
"""Mirrors ``rade_qnet.models.ridge``."""
```

---

## 2. `tests/rade_qnet/models/ridge/test_model.py`

2668 bytes · SHA-256 `b4bbdc31a30aaafd`

```python
"""
Unit tests for the ridge architecture, with no framework in sight.

These are the tests the four-file layout exists to make possible. Nothing
below constructs a run specification, touches the registry or starts an
engine, because ``model.py`` imports none of those -- which is a property
``test_model_layout.py`` enforces rather than hopes for.

The practical consequence is speed and blame. These run in milliseconds,
and a failure here is unambiguously about the model; a failure in
``test_register.py`` is unambiguously about the wiring.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rade_qnet.models.ridge.model import build
from src.rade_qnet.models.ridge.spec import RidgeSpec


class TestTheSpecRefusesNonsense:
    """Validation is the cheapest place to fail, so it should fail here."""

    def test_a_zero_penalty_is_rejected(self) -> None:
        """
        ``alpha`` must be strictly positive.

        Zero is ordinary least squares, which on the near-collinear design
        a flattened sequence window produces is unstable in a way that
        reads as a model excellent in sample and useless out of it. The
        constraint turns a subtle modelling failure into a parse error.
        """
        with pytest.raises(ValidationError):
            RidgeSpec(alpha=0.0)

    def test_the_defaults_are_usable_on_their_own(self) -> None:
        """
        An empty ``model.params`` block produces a working model.

        This is what lets a specification name ``ridge`` and nothing else.
        """
        assert RidgeSpec().alpha > 0


class TestBuildIsAPureFunctionOfTheSettings:
    """No data, no signature, no state: that is what makes bundles reloadable."""

    def test_the_settings_reach_the_estimator(self) -> None:
        """
        What the spec says is what the estimator gets.

        Worth asserting because the mapping is written out by hand, and a
        transposed pair of keyword arguments would train a perfectly
        plausible, differently configured model.
        """
        estimator = build(RidgeSpec(alpha=0.25, fit_intercept=False))
        assert estimator.alpha == 0.25
        assert estimator.fit_intercept is False

    def test_the_estimator_comes_back_unfitted(self) -> None:
        """
        Construction and fitting are separate, and the engine owns fitting.

        A ``build`` that fitted anything would make the model's behaviour
        depend on when it was called, and would put data inside a function
        the bundle reload path calls with no data available.
        """
        assert not hasattr(build(RidgeSpec()), "coef_")
```

---

## 3. `tests/rade_qnet/models/ridge/test_register.py`

3431 bytes · SHA-256 `ce59e428145e17ed`

```python
"""
Unit tests for the ridge wiring.

Separate from ``test_model.py`` for the same reason the source files are
separate: a failure here means the model is plugged in wrongly, and a
failure there means the model is wrong. Keeping them apart means the test
that fails tells you which.
"""

from __future__ import annotations

from src.rade_qnet.core.lifecycle.components import MODELS, get_model
from src.rade_qnet.core.spec.run import parse_run_spec
from src.rade_qnet.models.ridge.register import RidgeModel
from src.rade_qnet.models.ridge.spec import RidgeSpec
from src.rade_qnet.sources.dataset.tabular import TabularDataModule


def specification(path: str) -> dict:
    """
    Return a minimal valid run specification naming this model.

    Parameters
    ----------
    path
        Any path; nothing here reads the file.

    Returns
    -------
    dict
        A specification ready for :func:`~rade_qnet.core.spec.run.parse_run_spec`.
    """
    return {
        "task": "supervised",
        "model": {"name": "ridge", "params": {"alpha": 0.5}},
        "source": {"kind": "tabular", "path": path},
        "training": {"engine": "sklearn"},
        "reports": {"enabled": []},
        "hardware": {"device": "cpu"},
    }


class TestTheDeclaration:
    """The four things ``register.py`` is responsible for saying."""

    def test_the_model_resolves_by_name(self) -> None:
        """
        ``"ridge"`` in a specification finds this class.

        The import at the top of this module is what registers it, which
        is the whole discovery mechanism.
        """
        assert get_model("ridge") is RidgeModel

    def test_the_declared_engine_is_recorded_as_metadata(self) -> None:
        """
        The registry knows which engine the model needs.

        That is what lets a pipeline reject a specification pairing this
        model with the Torch engine before building anything, rather than
        failing partway through construction with a less obvious message.
        """
        assert MODELS.entry("ridge").metadata["engine"] == "sklearn"

    def test_the_spec_class_is_attached(self) -> None:
        """The open-ended ``model.params`` block has a validator."""
        assert RidgeModel.spec is RidgeSpec


class TestTheTwoMethods:
    """What the framework calls, and what it is entitled to get back."""

    def test_the_data_module_is_the_frameworks_own(self) -> None:
        """
        A tier 1 model adds no data code.

        If this ever returns a custom module, the model has moved to tier
        3 and needs a ``data.py`` -- which is a layout change, not a
        one-line change here.
        """
        spec = parse_run_spec(specification("ignored.csv"))
        assert isinstance(RidgeModel().data_module(spec), TabularDataModule)

    def test_params_from_the_specification_reach_the_estimator(self) -> None:
        """
        End to end through the wiring: YAML value to configured estimator.

        The unit tests in ``test_model.py`` prove ``build`` honours its
        settings; this proves the settings in the specification are the
        ones it is handed. Both can pass individually while the pair is
        broken, which is why this exists separately.
        """
        spec = parse_run_spec(specification("ignored.csv"))
        estimator = RidgeModel().build_model(spec, signature=None)  # type: ignore[arg-type]
        assert estimator.alpha == 0.5
```

