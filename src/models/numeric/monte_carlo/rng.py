from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class NormalRng:
    """
    Reproducible standard normal generator using NumPy's Generator (PCG64).

    Why this exists
    ---------------
    - Centralizes seeding and antithetic generation
    - Supports chunked generation to avoid large allocations

    Notes
    -----
    - `standard_normals(n, d)` returns shape (n, d) float64 by default.
    - `antithetic=True` returns an even number of rows; if `n` is odd,
      it is rounded up to the next even number.
    """

    seed: Optional[int] = None
    _gen: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._gen = np.random.default_rng(self.seed)

    def reseed(self, seed: Optional[int]) -> None:
        """Reset the underlying generator (useful for deterministic tests)."""
        self.seed = seed
        self._gen = np.random.default_rng(self.seed)

    def standard_normals(
        self,
        n: int,
        d: int = 1,
        *,
        antithetic: bool = False,
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        """
        Generate i.i.d. standard normal variates.

        Parameters
        ----------
        n:
            Number of samples (rows).
        d:
            Dimension per sample (columns).
        antithetic:
            If True, generate antithetic pairs (z, -z).
        dtype:
            Output dtype (float64 recommended for numerical stability).

        Returns
        -------
        ndarray shape (n, d) (or (n_even, d) if antithetic and n odd).
        """
        if n <= 0:
            raise ValueError("n must be positive.")
        if d <= 0:
            raise ValueError("d must be positive.")

        if not antithetic:
            z = self._gen.standard_normal(size=(n, d)).astype(dtype, copy=False)
            return z

        # Antithetic requires even count. If odd, round up.
        n_even = n if (n % 2 == 0) else (n + 1)
        half = n_even // 2

        z_half = self._gen.standard_normal(size=(half, d)).astype(dtype, copy=False)
        z = np.vstack((z_half, -z_half))
        return z

    def standard_normals_chunks(
        self,
        n: int,
        d: int = 1,
        *,
        chunk_size: int = 200_000,
        antithetic: bool = False,
        dtype: np.dtype = np.float64,
    ) -> Iterator[np.ndarray]:
        """
        Yield standard normals in chunks to limit peak memory.

        Each yielded chunk has shape (m, d). For antithetic=True, chunks will
        always have an even number of rows.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        remaining = int(n)
        while remaining > 0:
            m = min(remaining, chunk_size)
            z = self.standard_normals(m, d, antithetic=antithetic, dtype=dtype)
            yield z
            remaining -= m

    def state(self) -> Tuple[Optional[int], str]:
        """
        Small helper for debugging/logging: returns (seed, bitgen_name).
        """
        return self.seed, type(self._gen.bit_generator).__name__