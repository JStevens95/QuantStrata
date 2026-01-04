from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple


def _effective_n(n: int, *, antithetic: bool) -> int:
    """
    Compute the *effective* number of samples produced given (n, antithetic).

    - If antithetic=False: n_eff = n
    - If antithetic=True : n_eff is the next even integer >= n

    This avoids subtle drift when chunking (you must decide the global n_eff once).
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not antithetic:
        return int(n)
    return int(n if (n % 2 == 0) else (n + 1))


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
            Number of samples (rows) requested.
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

        Key correctness guarantee
        -------------------------
        The total number of rows yielded is exactly:
            n_eff = n                  (antithetic=False)
            n_eff = next_even(n)       (antithetic=True)

        This fixes the common bug where chunking with antithetic=True can silently
        produce *more* samples than intended if each chunk rounds itself up.

        Parameters
        ----------
        n:
            Requested number of samples.
        d:
            Dimension per sample.
        chunk_size:
            Target chunk size (rows). In antithetic mode, it must be >= 2 and
            chunks will be adjusted to be even-sized.
        antithetic:
            If True, generate antithetic pairs.
        dtype:
            Output dtype.

        Yields
        ------
        ndarray with shape (m, d) where sum(m) == n_eff.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if d <= 0:
            raise ValueError("d must be positive.")
        if antithetic and chunk_size < 2:
            raise ValueError("chunk_size must be >= 2 when antithetic=True (need even-sized chunks).")

        n_eff = _effective_n(int(n), antithetic=antithetic)
        remaining = int(n_eff)

        while remaining > 0:
            m = min(remaining, int(chunk_size))

            if antithetic:
                # Ensure even chunk sizes so the generator does not round-up per chunk.
                if m % 2 == 1:
                    # Because n_eff is even, remaining can never be 1 here.
                    # If m was odd, we can safely reduce by 1 to make it even.
                    m -= 1
                    if m == 0:
                        # Fallback defensive guard (should be unreachable).
                        m = 2

            z = self.standard_normals(m, d, antithetic=antithetic, dtype=dtype)
            # z.shape[0] should equal m (since m is already even in antithetic mode).
            yield z
            remaining -= int(z.shape[0])

    def state(self) -> Tuple[Optional[int], str]:
        """
        Small helper for debugging/logging: returns (seed, bitgen_name).
        """
        return self.seed, type(self._gen.bit_generator).__name__