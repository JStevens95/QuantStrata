import pytest
import numpy as np

from src.marketdata.core.interfaces import Quote
from src.marketdata.quotes.factory import PositiveQuoteFactory, ScalarQuoteFactory


def test_scalar_quote_factory_accepts_python_scalars() -> None:
    f = ScalarQuoteFactory()
    q1 = f.build(1.23)
    q2 = f.build(5)

    assert isinstance(q1, Quote)
    assert isinstance(q2, Quote)
    assert q1.value == pytest.approx(1.23)
    assert q2.value == pytest.approx(5.0)


def test_scalar_quote_factory_accepts_numpy_scalar_blocks() -> None:
    f = ScalarQuoteFactory()

    q0 = f.build(np.array(0.12))                 # 0-d
    q1 = f.build(np.array([0.12], dtype=float))  # size-1 1D

    assert q0.value == pytest.approx(0.12)
    assert q1.value == pytest.approx(0.12)


def test_scalar_quote_factory_rejects_non_scalar_blocks() -> None:
    f = ScalarQuoteFactory()

    with pytest.raises(ValueError, match=r"Expected scalar params"):
        _ = f.build(np.array([0.1, 0.2], dtype=float))

    with pytest.raises(ValueError, match=r"Expected scalar params"):
        _ = f.build(np.array([[0.1, 0.2]], dtype=float))


def test_scalar_quote_factory_rejects_non_finite() -> None:
    f = ScalarQuoteFactory()

    with pytest.raises(ValueError, match=r"must be finite"):
        _ = f.build(float("nan"))

    with pytest.raises(ValueError, match=r"must be finite"):
        _ = f.build(float("inf"))

    with pytest.raises(ValueError, match=r"must be finite"):
        _ = f.build(np.array([np.nan], dtype=float))


def test_scalar_quote_factory_constraints() -> None:
    f = ScalarQuoteFactory(min_value=0.0, max_value=2.0, allow_negative=False)

    _ = f.build(0.0)
    _ = f.build(2.0)

    with pytest.raises(ValueError, match=r"must be >= 0"):
        _ = f.build(-1.0)


def test_positive_quote_factory_requires_strictly_positive() -> None:
    f = PositiveQuoteFactory()

    _ = f.build(1e-6)

    with pytest.raises(ValueError, match=r"must be >= 0"):
        _ = f.build(-1.0)

    with pytest.raises(ValueError, match=r"< min_value"):
        _ = f.build(0.0)


def test_scalar_quote_factory_min_value_branch() -> None:
    f = ScalarQuoteFactory(min_value=0.0, allow_negative=True)

    with pytest.raises(ValueError, match=r"< min_value"):
        _ = f.build(-1e-12)