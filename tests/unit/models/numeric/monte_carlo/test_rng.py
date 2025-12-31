from __future__ import annotations

import pytest
import numpy as np

from src.models.numeric.monte_carlo.rng import NormalRng


def test_standard_normals_shape_and_dtype() -> None:
    rng = NormalRng(seed=123)
    z = rng.standard_normals(n=5, d=3, dtype=np.float64)

    assert z.shape == (5, 3)
    assert z.dtype == np.float64


def test_standard_normals_reproducible_with_same_seed() -> None:
    rng1 = NormalRng(seed=7)
    rng2 = NormalRng(seed=7)

    z1 = rng1.standard_normals(n=100, d=2)
    z2 = rng2.standard_normals(n=100, d=2)

    assert np.array_equal(z1, z2)


def test_reseed_resets_sequence() -> None:
    rng = NormalRng(seed=1)
    z1 = rng.standard_normals(n=10, d=1)

    rng.reseed(1)
    z2 = rng.standard_normals(n=10, d=1)

    assert np.array_equal(z1, z2)


def test_antithetic_returns_even_row_count_and_pairs_cancel() -> None:
    rng = NormalRng(seed=42)

    # Request odd n => should round up to even.
    z = rng.standard_normals(n=5, d=4, antithetic=True)
    assert z.shape[0] % 2 == 0
    assert z.shape[1] == 4

    half = z.shape[0] // 2
    z1 = z[:half, :]
    z2 = z[half:, :]

    # Antithetic property: second half should be - first half.
    assert np.allclose(z2, -z1, rtol=0.0, atol=0.0)


def test_standard_normals_chunks_total_rows_and_remainder() -> None:
    rng = NormalRng(seed=123)

    n = 105
    d = 3
    chunk_size = 20

    chunks = list(rng.standard_normals_chunks(n=n, d=d, chunk_size=chunk_size, antithetic=False))
    assert len(chunks) > 1

    total = sum(c.shape[0] for c in chunks)
    assert total == n

    for c in chunks:
        assert c.shape[1] == d


def test_standard_normals_chunks_antithetic_even_chunks() -> None:
    rng = NormalRng(seed=999)

    n = 101  # odd on purpose
    d = 2
    chunk_size = 25

    chunks = list(rng.standard_normals_chunks(n=n, d=d, chunk_size=chunk_size, antithetic=True))

    # In antithetic mode, each produced chunk has even rows.
    for c in chunks:
        assert c.shape[0] % 2 == 0
        assert c.shape[1] == d


def test_invalid_inputs_raise() -> None:
    rng = NormalRng(seed=0)

    with pytest.raises(ValueError, match="n must be positive"):
        rng.standard_normals(n=0, d=1)

    with pytest.raises(ValueError, match="d must be positive"):
        rng.standard_normals(n=1, d=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        list(rng.standard_normals_chunks(n=10, d=1, chunk_size=0))